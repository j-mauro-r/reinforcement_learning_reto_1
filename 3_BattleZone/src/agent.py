"""DQN agent core for BattleZone HU005."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.network import BattleZoneQNetwork
from src.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNUpdateResult:
    """Result of one controlled DQN optimizer step."""

    loss: float


class DQNAgent:
    """BattleZone DQN core with online/target networks and uniform replay."""

    def __init__(
        self,
        *,
        action_dim: int,
        state_shape: Tuple[int, ...],
        gamma: float,
        learning_rate: float,
        replay_buffer_capacity: int,
        batch_size: int,
        network_hidden_dim: int,
        network_conv_channels: Sequence[int],
        device: str,
    ) -> None:
        """Initializes DQN from explicit, externally supplied configuration."""
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if replay_buffer_capacity <= 0 or batch_size <= 0:
            raise ValueError("replay_buffer_capacity and batch_size must be positive.")

        self.action_dim = int(action_dim)
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.device = self._resolve_device(device)

        frame_stack, _, _, channels = self.state_shape
        network_kwargs = {
            "action_dim": self.action_dim,
            "frame_stack": frame_stack,
            "input_channels": channels,
            "hidden_dim": int(network_hidden_dim),
            "conv_channels": tuple(int(value) for value in network_conv_channels),
        }
        self.online_network = BattleZoneQNetwork(**network_kwargs).to(self.device)
        self.target_network = BattleZoneQNetwork(**network_kwargs).to(self.device)
        self.sync_target_network()

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=float(learning_rate),
        )
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            capacity=int(replay_buffer_capacity),
            state_shape=self.state_shape,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DQNAgent":
        """Builds the agent from ``battlezone_config.yaml``-equivalent data."""
        if config.get("algorithm") != "DQN":
            raise ValueError("BattleZone HU005 requires algorithm='DQN'.")
        dqn = config["dqn"]
        network = dqn["network"]
        return cls(
            action_dim=int(config["environment"]["expected_action_space_n"]),
            state_shape=tuple(config["validation"]["expected_final_shape"]),
            gamma=float(dqn["gamma"]),
            learning_rate=float(dqn["learning_rate"]),
            replay_buffer_capacity=int(dqn["replay_buffer"]["capacity"]),
            batch_size=int(dqn["batch_size"]),
            network_hidden_dim=int(network["hidden_dim"]),
            network_conv_channels=tuple(network["conv_channels"]),
            device=str(dqn["device"]),
        )

    def select_action(self, state: np.ndarray | Tensor, epsilon: float) -> int:
        """Selects an action using epsilon-greedy over the online network."""
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
        if np.random.rand() < float(epsilon):
            return int(np.random.randint(0, self.action_dim))

        state_tensor = self._to_observation_tensor(state)
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores one transition in uniform replay memory."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def sample_batch(self, batch_size: int | None = None) -> Dict[str, np.ndarray]:
        """Samples a uniform replay batch."""
        size = self.batch_size if batch_size is None else int(batch_size)
        return self.replay_buffer.sample(size)

    def compute_targets(self, batch: Dict[str, np.ndarray | Tensor]) -> Tensor:
        """Computes classic DQN targets using max Q from Target Network."""
        _, _, rewards, next_states, dones = self._prepare_batch(batch)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            return rewards + self.gamma * (1.0 - dones) * next_q_values

    def update(self, batch: Dict[str, np.ndarray | Tensor]) -> DQNUpdateResult:
        """Runs one optimizer update over a controlled batch."""
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)
        predicted_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = self.loss_fn(predicted_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return DQNUpdateResult(loss=float(loss.detach().item()))

    def sync_target_network(self) -> None:
        """Synchronizes target-network parameters from online network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def state_dict(self) -> Dict[str, Any]:
        """Returns the minimal restorable state required by HU005."""
        return {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gamma": self.gamma,
            "action_dim": self.action_dim,
            "state_shape": self.state_shape,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Loads state after validating immutable structural metadata."""
        expected = {
            "action_dim": self.action_dim,
            "state_shape": self.state_shape,
            "batch_size": self.batch_size,
        }
        incoming = {
            "action_dim": int(state["action_dim"]),
            "state_shape": tuple(state["state_shape"]),
            "batch_size": int(state["batch_size"]),
        }
        if incoming != expected:
            raise ValueError(
                f"Incompatible agent state. expected={expected}, incoming={incoming}."
            )

        self.gamma = float(state["gamma"])
        self.online_network.load_state_dict(state["online_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])

    def _prepare_batch(
        self,
        batch: Dict[str, np.ndarray | Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        states = self._to_observation_tensor(batch["states"])
        next_states = self._to_observation_tensor(batch["next_states"])
        actions = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.long).view(-1)
        rewards = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32).view(-1)
        dones = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32).view(-1)

        batch_size = states.shape[0]
        if next_states.shape[0] != batch_size:
            raise ValueError("next_states batch size does not match states batch size.")
        if actions.shape[0] != batch_size:
            raise ValueError("actions batch size does not match states batch size.")
        if rewards.shape[0] != batch_size:
            raise ValueError("rewards batch size does not match states batch size.")
        if dones.shape[0] != batch_size:
            raise ValueError("dones batch size does not match states batch size.")
        return states, actions, rewards, next_states, dones

    def _to_observation_tensor(self, observations: np.ndarray | Tensor) -> Tensor:
        tensor = observations if isinstance(observations, Tensor) else torch.as_tensor(observations)
        if tensor.ndim not in (4, 5):
            raise ValueError("Observations must be 4D or 5D tensors in HU003 layout.")
        return tensor.to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = str(device).strip().lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cpu":
            return torch.device("cpu")
        return torch.device(device)
