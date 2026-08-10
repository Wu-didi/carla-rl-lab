from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


class RewardTerm(ABC):
    """A single editable reward component."""

    name: str

    @abstractmethod
    def __call__(self, obs: Dict[str, Any], done: bool, info: Dict[str, Any]) -> float:
        raise NotImplementedError


@dataclass
class WeightedRewardTerm:
    name: str
    term: RewardTerm
    weight: float = 1.0


class RewardComposer:
    """Composable reward pipeline for CARLA tasks.

    The current CARLA environment still owns its legacy reward function. This
    composer is the v1 extension point for moving reward terms out of
    `carla_env.py` without changing algorithm code.
    """

    def __init__(self, terms: Optional[Iterable[WeightedRewardTerm]] = None):
        self.terms = list(terms or [])

    def __call__(
        self,
        obs: Dict[str, Any],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        info = info or {}
        logs: Dict[str, float] = {}
        total = 0.0
        for weighted in self.terms:
            raw_value = float(weighted.term(obs, done, info))
            value = weighted.weight * raw_value
            logs[f"reward/{weighted.name}"] = value
            total += value
        return total, logs
