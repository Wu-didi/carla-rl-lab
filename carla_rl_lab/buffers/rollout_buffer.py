from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def generalized_advantage_estimate(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, np.ndarray]:
    """Compute GAE and bootstrapped returns for one ordered rollout."""

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not (len(rewards) == len(dones) == len(values)):
        raise ValueError("rewards, dones, and values must have equal lengths")

    advantages = np.zeros_like(rewards)
    next_value = float(last_value)
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * non_terminal - values[index]
        last_advantage = (
            delta + gamma * gae_lambda * non_terminal * last_advantage
        )
        advantages[index] = last_advantage
        next_value = float(values[index])

    return {
        "advantages": advantages.astype(np.float32),
        "returns": (advantages + values).astype(np.float32),
    }


class RolloutBuffer:
    """Ordered on-policy storage with GAE finalization."""

    def __init__(self, capacity: int, gamma: float, gae_lambda: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clear()

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.next_states: List[Optional[np.ndarray]] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        next_state: Optional[np.ndarray] = None,
    ) -> None:
        if len(self) >= self.capacity:
            raise RuntimeError("rollout buffer is full")
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.next_states.append(
            None if next_state is None else np.asarray(next_state, dtype=np.float32)
        )
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def batch(self, last_value: float = 0.0) -> Dict[str, Any]:
        if not self.states:
            raise ValueError("cannot finalize an empty rollout")
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        targets = generalized_advantage_estimate(
            rewards,
            dones,
            values,
            last_value,
            self.gamma,
            self.gae_lambda,
        )
        batch = {
            "states": np.asarray(self.states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "old_log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "advantages": targets["advantages"],
            "returns": targets["returns"],
            "last_value": np.float32(last_value),
        }
        if all(next_state is not None for next_state in self.next_states):
            batch["next_states"] = np.asarray(self.next_states, dtype=np.float32)
        return batch

    def __len__(self) -> int:
        return len(self.states)
