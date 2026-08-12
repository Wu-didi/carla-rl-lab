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
    episode_ends: Optional[np.ndarray] = None,
    next_values: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute GAE and bootstrapped returns for one ordered rollout."""

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not (len(rewards) == len(dones) == len(values)):
        raise ValueError("rewards, dones, and values must have equal lengths")
    episode_ends = (
        dones.copy()
        if episode_ends is None
        else np.asarray(episode_ends, dtype=np.float32).reshape(-1)
    )
    if len(episode_ends) != len(rewards):
        raise ValueError("episode_ends must have the same length as rewards")
    if next_values is not None:
        next_values = np.asarray(next_values, dtype=np.float32).reshape(-1)
        if len(next_values) != len(rewards):
            raise ValueError("next_values must have the same length as rewards")

    advantages = np.zeros_like(rewards)
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        if next_values is None:
            bootstrap_value = (
                float(last_value)
                if index == len(rewards) - 1
                else float(values[index + 1])
            )
        else:
            bootstrap_value = float(next_values[index])
        non_terminal = 1.0 - dones[index]
        delta = (
            rewards[index]
            + gamma * bootstrap_value * non_terminal
            - values[index]
        )
        last_advantage = (
            delta
            + gamma
            * gae_lambda
            * (1.0 - episode_ends[index])
            * last_advantage
        )
        advantages[index] = last_advantage

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
        self.terminals: List[bool] = []
        self.episode_ends: List[bool] = []
        self.values: List[float] = []
        self.next_values: List[Optional[float]] = []
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
        terminal: Optional[bool] = None,
        next_value: Optional[float] = None,
    ) -> None:
        if len(self) >= self.capacity:
            raise RuntimeError("rollout buffer is full")
        self.states.append(np.asarray(state))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.next_states.append(
            None if next_state is None else np.asarray(next_state)
        )
        self.rewards.append(float(reward))
        self.episode_ends.append(bool(done))
        self.terminals.append(bool(done) if terminal is None else bool(terminal))
        self.values.append(float(value))
        self.next_values.append(None if next_value is None else float(next_value))
        self.log_probs.append(float(log_prob))

    def batch(self, last_value: float = 0.0) -> Dict[str, Any]:
        if not self.states:
            raise ValueError("cannot finalize an empty rollout")
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.terminals, dtype=np.float32)
        episode_ends = np.asarray(self.episode_ends, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.zeros_like(values)
        for index in range(len(values)):
            if self.next_values[index] is not None:
                next_values[index] = float(self.next_values[index])
            elif index + 1 < len(values) and not self.episode_ends[index]:
                next_values[index] = values[index + 1]
            elif index == len(values) - 1 and not self.episode_ends[index]:
                next_values[index] = float(last_value)
        targets = generalized_advantage_estimate(
            rewards,
            dones,
            values,
            last_value,
            self.gamma,
            self.gae_lambda,
            episode_ends=episode_ends,
            next_values=next_values,
        )
        batch = {
            "states": np.asarray(self.states),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "rewards": rewards,
            "dones": dones,
            "episode_ends": episode_ends,
            "values": values,
            "old_log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "advantages": targets["advantages"],
            "returns": targets["returns"],
            "last_value": np.float32(last_value),
            "next_values": next_values,
        }
        if all(next_state is not None for next_state in self.next_states):
            batch["next_states"] = np.asarray(self.next_states)
        return batch

    def end_episode(self, next_value: float = 0.0, terminal: bool = False) -> None:
        """Cut GAE at the latest transition after an external env reset."""
        if not self.states:
            return
        self.episode_ends[-1] = True
        self.terminals[-1] = bool(terminal)
        self.next_values[-1] = float(next_value)

    def __len__(self) -> int:
        return len(self.states)
