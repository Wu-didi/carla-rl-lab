from __future__ import annotations

import collections
import random
from typing import Any, Deque, Dict, Mapping, Tuple

import numpy as np


Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]


class ReplayBuffer:
    """Simple off-policy replay buffer.

    It is intentionally small so SAC, TD3, and DDPG can share it without
    forcing a large storage abstraction on researchers.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> Dict[str, Any]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    def size(self) -> int:
        return len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.buffer.maxlen,
            "transitions": list(self.buffer),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_capacity = int(state.get("capacity", self.buffer.maxlen))
        if saved_capacity != self.buffer.maxlen:
            raise ValueError(
                "replay capacity mismatch: checkpoint={}, config={}".format(
                    saved_capacity, self.buffer.maxlen
                )
            )
        self.buffer.clear()
        for transition in state.get("transitions", []):
            self.buffer.append(transition)
