from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    description: str
    seeds: Tuple[int, ...]
    env_overrides: Dict[str, Any]


_BENCHMARKS = {
    "lane_following_v0": BenchmarkSpec(
        name="lane_following_v0",
        description="Fixed Town05 lane-following protocol for initial algorithm comparisons.",
        seeds=(0, 1, 2, 3, 4),
        env_overrides={
            "town": "Town05",
            "number_of_vehicles": 50,
            "number_of_walkers": 0,
            "traffic": "off",
            "max_time_episode": 500,
            "desired_speed": 8.0,
            "view_mode": "none",
            "reward_profile": "research_v1",
        },
    )
}


def list_benchmarks() -> Tuple[str, ...]:
    return tuple(sorted(_BENCHMARKS))


def get_benchmark(name: str) -> BenchmarkSpec:
    try:
        return _BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(
            "Unknown benchmark '{}'. Available benchmarks: {}".format(
                name, ", ".join(list_benchmarks())
            )
        ) from exc


def apply_benchmark(cfg: Any, spec: BenchmarkSpec) -> Any:
    for key, value in spec.env_overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError("Benchmark override is not present in Config: {}".format(key))
        setattr(cfg, key, value)
    return cfg
