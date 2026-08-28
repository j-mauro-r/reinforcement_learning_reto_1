"""DDQN agent core for Assault."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn

from .network import QNetwork
from .replay_buffer import ReplayBatch


class DDQNAgent:
    """Core DDQN implementation with online and target networks."""

    def __init__(
        self,
        config: Mapping[str, object],
        device: str | torch.device | None = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the DDQN agent.

        Args:
            config: Parsed project configuration containing ``network`` and
                ``agent`` sections.
            device: Optional PyTorch device. Defaults to CUDA when available.
            seed: Optional seed for Python, NumPy and PyTorch.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        network_config = config["network"]  # type: ignore[index]
        agent_config = config["agent"]  # type: ignore[index]
        self.input_channels = int(network_config["input_channels"])  # type: ignore[index]
        self.num_actions = int(network_config["num_actions"])  # type: ignore[index]
        self.gamma = float(agent_config["gamma"])  # type: ignore[index]
        self.learning_rate = float(agent_config["learning_rate"])  # type: ignore[index]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_network = QNetwork(self.input_channels, self.num_actions).to(self.device)
        self.target_network = QNetwork(self.input_channels, self.num_actions).to(self.device)
        self.sync_target_network()
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray | torch.Tensor, epsilon: float) -> int:
        """Selects an epsilon-greedy action.

        Args:
            state: Single state with shape ``(4, 84, 84)``.
            epsilon: Exploration probability in ``[0, 1]``.

        Returns:
            Valid action index in ``[0, num_actions - 1]``.

        Raises:
            ValueError: If epsilon is outside ``[0, 1]``.
        """
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if random.random() < float(epsilon):
            return random.randrange(self.num_actions)

        state_tensor = self._prepare_state(state).unsqueeze(0)
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def compute_ddqn_targets(self, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Computes DDQN targets using Online for selection and Target for evaluation.

        Args:
            rewards: Tensor of rewards shaped ``(batch,)``.
            next_states: Tensor of next states shaped ``(batch, 4, 84, 84)``.
            dones: Boolean or float tensor shaped ``(batch,)``.

        Returns:
            Target Q-values shaped ``(batch,)``.
        """
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            not_done = 1.0 - dones.to(dtype=torch.float32)
            return rewards + self.gamma * not_done * next_q_values

    def update(self, batch: ReplayBatch | Mapping[str, np.ndarray]) -> Dict[str, float]:
        """Runs one DDQN optimizer step on the online network.

        Args:
            batch: Replay batch returned by ``ReplayBuffer.sample`` or an
                equivalent mapping.

        Returns:
            Metrics containing at least finite ``loss``.
        """
        tensors = self._prepare_batch(batch)
        self.online_network.train()
        current_q = self.online_network(tensors["states"]).gather(1, tensors["actions"].unsqueeze(1)).squeeze(1)
        target_q = self.compute_ddqn_targets(
            rewards=tensors["rewards"],
            next_states=tensors["next_states"],
            dones=tensors["dones"],
        )
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu().item())}

    def sync_target_network(self) -> None:
        """Copies online network weights into the target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: str | Path) -> None:
        """Saves the HU003 agent state.

        Args:
            path: Destination file path.
        """
        checkpoint = {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "input_channels": self.input_channels,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
        }
        torch.save(checkpoint, Path(path))

    def load(self, path: str | Path) -> None:
        """Loads a HU003 agent state.

        Args:
            path: Checkpoint path created by ``save``.

        Raises:
            ValueError: If the checkpoint does not match this agent shape.
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if int(checkpoint["input_channels"]) != self.input_channels or int(checkpoint["num_actions"]) != self.num_actions:
            raise ValueError("Checkpoint network shape does not match this agent.")
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _prepare_state(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            state_tensor = state.to(self.device)
        else:
            state_tensor = torch.as_tensor(state, device=self.device)
        if state_tensor.shape != (self.input_channels, 84, 84):
            raise ValueError(f"Expected state shape ({self.input_channels}, 84, 84), got {tuple(state_tensor.shape)}.")
        return state_tensor

    def _prepare_batch(self, batch: ReplayBatch | Mapping[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        data = batch.as_dict() if isinstance(batch, ReplayBatch) else batch
        return {
            "states": torch.as_tensor(data["states"], device=self.device),
            "actions": torch.as_tensor(data["actions"], dtype=torch.int64, device=self.device),
            "rewards": torch.as_tensor(data["rewards"], dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(data["next_states"], device=self.device),
            "dones": torch.as_tensor(data["dones"], dtype=torch.float32, device=self.device),
        }
