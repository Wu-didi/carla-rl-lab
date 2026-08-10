from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

from carla_rl_lab.algorithms.base import BaseAgent


AgentFactory = Callable[[Any], BaseAgent]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    factory: AgentFactory
    family: str
    data_source: str
    runner: str
    action_space: str
    status: str
    description: str


_ALGORITHMS: Dict[str, AlgorithmSpec] = {}


def register_algorithm(spec: AlgorithmSpec) -> None:
    key = spec.name.lower()
    if key in _ALGORITHMS:
        raise ValueError(f"Algorithm already registered: {spec.name}")
    _ALGORITHMS[key] = spec


def get_algorithm(name: str) -> AlgorithmSpec:
    key = name.lower()
    if key not in _ALGORITHMS:
        available = ", ".join(list_algorithms())
        raise ValueError(f"Unknown algorithm '{name}'. Available algorithms: {available}")
    return _ALGORITHMS[key]


def create_agent(name: str, cfg: Any) -> BaseAgent:
    return get_algorithm(name).factory(cfg)


def list_algorithms() -> Iterable[str]:
    return sorted(_ALGORITHMS)
