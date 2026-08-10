from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np


class BaseAgent(ABC):
    """Minimal interface used by trainers.

    Keep this interface intentionally small so researchers can edit actor,
    critic, loss, and update logic without fighting a large framework.
    """

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return an action for one vector observation."""

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run one optimizer update and return scalar logs."""

    @abstractmethod
    def save(self, directory: str, step_id: Optional[Union[int, str]] = None) -> None:
        """Save model state."""

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model state from a checkpoint path."""
