"""DDQN agent core for BattleZone HU005."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from src.network import BattleZoneQNetwork
from src.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DDQNUpdateResult:
    """Result of a single DDQN update step."""

    loss: float


class DDQNAgent:
    """BattleZone DDQN core with online/target networks and uniform replay."""

    def __init__(
        self,
        *,
        action_dim: int = 18,
        state_shape: Tuple[int, ...] = (4, 128, 128, 3),
        gamma: float = 0.99,
        learning_rate: float = 2.5e-4,
        replay_buffer_capacity: int = 128,
        batch_size: int = 8,
        device: str = "auto",
    ) -> None:
        """Initializes the DDQN agent core.

        Args:
            action_dim: Number of discrete actions.
            state_shape: HU003 state shape.
            gamma: Discount factor baseline for HU005.
            learning_rate: Optimizer learning rate baseline for HU005.
            replay_buffer_capacity: Capacity for uniform replay buffer.
            batch_size: Default sample size for replay-based updates.
            device: "auto", "cpu", or CUDA device string.
        """
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.action_dim = int(action_dim)
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.device = self._resolve_device(device)

        frame_stack, _, _, channels = self.state_shape
        self.online_network = BattleZoneQNetwork(
            action_dim=self.action_dim,
            frame_stack=frame_stack,
            input_channels=channels,
        ).to(self.device)
        self.target_network = BattleZoneQNetwork(
            action_dim=self.action_dim,
            frame_stack=frame_stack,
            input_channels=channels,
        ).to(self.device)
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

    def select_action(self, state: np.ndarray | Tensor, epsilon: float) -> int:
        """Selects an action using epsilon-greedy policy over online Q-values.

        Args:
            state: One HU003-formatted observation.
            epsilon: Exploration probability in ``[0, 1]``.

        Returns:
            Action index in ``[0, action_dim)``.

        Raises:
            ValueError: If epsilon is out of range.
        """
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
        """Samples a replay batch using uniform sampling."""
        size = self.batch_size if batch_size is None else int(batch_size)
        return self.replay_buffer.sample(size)

    def compute_targets(self, batch: Dict[str, np.ndarray | Tensor]) -> Tensor:
        """Computes DDQN targets for a batch.

        DDQN rule:
        1) select next action with online network;
        2) evaluate that action with target network;
        3) mask terminal transitions to remove bootstrap.
        """
        _, _, rewards, next_states, dones = self._prepare_batch(batch)
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            return rewards + self.gamma * (1.0 - dones) * next_q_values

    def update(self, batch: Dict[str, np.ndarray | Tensor]) -> DDQNUpdateResult:
        """Runs one optimizer update over a controlled batch."""
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)

        predicted_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = self.loss_fn(predicted_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return DDQNUpdateResult(loss=float(loss.detach().item()))

    def sync_target_network(self) -> None:
        """Synchronizes target network parameters from online network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def state_dict(self) -> Dict[str, Any]:
        """Returns minimal agent state required by HU005 save/load tests."""
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
        """Loads agent state exported by ``state_dict``."""
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

    def _resolve_device(self, device: str) -> torch.device:
        requested = str(device).strip().lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cpu":
            return torch.device("cpu")
        return torch.device(device)
