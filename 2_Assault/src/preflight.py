"""Pre-training integration checks for the Assault DDQN pipeline."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch

from .agent import DDQNAgent
from .environment import create_assault_env
from .replay_buffer import ReplayBuffer
from .utils import get_runtime_info


@dataclass(frozen=True)
class PreflightReport:
    """Structured result for the DDQN pre-training gate."""

    passed: bool
    runtime: str
    device: str
    checks: Dict[str, bool]
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def ready_for_training(self) -> bool:
        """Whether the training loop may start."""
        return self.passed

    def as_dict(self) -> Dict[str, Any]:
        """Returns a notebook-friendly dictionary representation."""
        return {
            "passed": self.passed,
            "ready_for_training": self.ready_for_training,
            "runtime": self.runtime,
            "device": self.device,
            "checks": self.checks,
            "errors": self.errors,
            "details": self.details,
        }

    def format_summary(self) -> str:
        """Formats a compact preflight summary for notebooks and logs."""
        lines = [
            "===== DDQN PRE-FLIGHT =====",
            f"Runtime: {self.runtime}",
            f"Device: {self.device}",
        ]
        for name, passed in self.checks.items():
            suffix = "PASS" if passed else "FAIL"
            detail = self.details.get(name)
            if detail is None:
                lines.append(f"{name}: {suffix}")
            else:
                lines.append(f"{name}: {suffix} {detail}")
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in self.errors)
        lines.append("")
        lines.append(f"READY_FOR_TRAINING={self.ready_for_training}")
        return "\n".join(lines)


def run_preflight_checks(
    config: Mapping[str, Any],
    device: str | torch.device | None = None,
    env_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None,
) -> PreflightReport:
    """Runs fast integration checks before any training run starts.

    Args:
        config: Parsed project configuration.
        device: Optional PyTorch device override.
        env_factory: Optional environment factory used by tests to inject a
            controlled failure. Production calls use ``create_assault_env``.

    Returns:
        Structured preflight report. Any material check failure produces
        ``passed=False`` and the caller must not start training.
    """
    runtime_info = get_runtime_info()
    runtime = "Google Colab" if _running_in_colab() else "local"
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checks = {
        "Device": False,
        "Environment": False,
        "Observation": False,
        "QNetwork": False,
        "ReplayBuffer": False,
        "DDQN update": False,
        "Loss finite": False,
        "Target stable": False,
        "Target sync": False,
        "Save/load": False,
    }
    details: Dict[str, Any] = {"runtime_info": runtime_info}
    errors: List[str] = []
    env = None

    try:
        checks["Device"] = selected_device.type == "cpu" or torch.cuda.is_available()
        if not checks["Device"]:
            raise RuntimeError(f"Requested unavailable device: {selected_device}")

        factory = env_factory or _default_env_factory
        env = factory(config)
        checks["Environment"] = True

        observation, _ = env.reset(seed=int(config["reproducibility"]["seed"]))
        expected_shape = (
            int(config["network"]["input_channels"]),
            int(config["preprocessing"]["resize_height"]),
            int(config["preprocessing"]["resize_width"]),
        )
        if tuple(observation.shape) != expected_shape or observation.dtype != np.uint8:
            raise ValueError(f"Invalid observation contract: {observation.shape} {observation.dtype}")
        checks["Observation"] = True
        details["Observation"] = f"{tuple(observation.shape)} {observation.dtype}"

        agent = DDQNAgent(config, device=selected_device, seed=int(config["reproducibility"]["seed"]))
        with torch.no_grad():
            q_values = agent.online_network(torch.as_tensor(observation, device=agent.device).unsqueeze(0))
        if tuple(q_values.shape) != (1, int(config["network"]["num_actions"])) or not torch.isfinite(q_values).all():
            raise ValueError(f"Invalid QNetwork output: {tuple(q_values.shape)}")
        checks["QNetwork"] = True
        details["QNetwork"] = f"-> {tuple(q_values.shape)}"

        buffer = ReplayBuffer(capacity=max(8, int(config["replay_buffer"]["batch_size"])), seed=7)
        next_observation, reward, terminated, _, _ = env.step(int(env.action_space.sample()))
        for index in range(int(config["replay_buffer"]["batch_size"])):
            buffer.add(
                state=observation,
                action=index % int(config["network"]["num_actions"]),
                reward=float(reward),
                next_state=next_observation,
                done=bool(terminated),
            )
        batch = buffer.sample(int(config["replay_buffer"]["batch_size"]))
        checks["ReplayBuffer"] = True

        target_before = [parameter.detach().clone() for parameter in agent.target_network.parameters()]
        update_metrics = agent.update(batch)
        loss = float(update_metrics["loss"])
        checks["DDQN update"] = True
        details["DDQN update"] = f"loss={loss:.6f}"
        checks["Loss finite"] = bool(np.isfinite(loss))
        if not checks["Loss finite"]:
            raise RuntimeError(f"Non-finite preflight loss: {loss}")

        target_after = [parameter.detach().clone() for parameter in agent.target_network.parameters()]
        checks["Target stable"] = all(torch.equal(before, after) for before, after in zip(target_before, target_after))
        if not checks["Target stable"]:
            raise RuntimeError("Target network changed during DDQN update.")

        agent.sync_target_network()
        checks["Target sync"] = all(
            torch.equal(online, target)
            for online, target in zip(agent.online_network.parameters(), agent.target_network.parameters())
        )
        if not checks["Target sync"]:
            raise RuntimeError("Target network did not match Online after sync.")

        checkpoint_path: Optional[Path] = None
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "preflight_agent.pt"
            probe_state = torch.as_tensor(observation).unsqueeze(0)
            with torch.no_grad():
                expected_q = agent.online_network(probe_state.to(agent.device)).detach().cpu()
            agent.save(checkpoint_path)
            loaded_agent = DDQNAgent(config, device=selected_device, seed=999)
            loaded_agent.load(checkpoint_path)
            with torch.no_grad():
                actual_q = loaded_agent.online_network(probe_state.to(loaded_agent.device)).detach().cpu()
            checks["Save/load"] = torch.allclose(expected_q, actual_q, atol=1e-6)
            if not checks["Save/load"]:
                raise RuntimeError("Save/load did not restore equivalent Q-values.")
        details["Save/load"] = f"temporary_file_cleaned={checkpoint_path is not None and not checkpoint_path.exists()}"
    except Exception as exc:  # noqa: BLE001 - preflight reports material failures to the caller.
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            env.close()

    passed = all(checks.values()) and not errors
    return PreflightReport(
        passed=passed,
        runtime=runtime,
        device=str(selected_device),
        checks=checks,
        errors=errors,
        details=details,
    )


def _default_env_factory(config: Mapping[str, Any]) -> Any:
    return create_assault_env(dict(config), mode="train", seed=int(config["reproducibility"]["seed"]))


def _running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True
