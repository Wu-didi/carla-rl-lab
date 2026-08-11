from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple


_SEEDS = (0, 1, 2, 3, 4)
_BASE_OVERRIDES = {
    "number_of_walkers": 0,
    "traffic": "off",
    "max_time_episode": 500,
    "desired_speed": 8.0,
    "view_mode": "none",
    "visualize_waypoints": False,
    "reward_profile": "research_v1",
    "weather": "ClearNoon",
}


def _benchmark(
    name: str,
    description: str,
    category: str,
    town: str,
    number_of_vehicles: int,
    **overrides: Any
) -> Dict[str, Any]:
    env_overrides = dict(_BASE_OVERRIDES)
    env_overrides.update(
        {
            "town": town,
            "number_of_vehicles": number_of_vehicles,
        }
    )
    env_overrides.update(overrides)
    horizon = int(env_overrides["max_time_episode"])
    return {
        "name": name,
        "description": description,
        "category": category,
        "seeds": _SEEDS,
        "success_reasons": ("timeout",),
        "success_criteria": {
            "min_horizon_fraction": 0.99,
            "min_distance_m": horizon * float(env_overrides.get("dt", 0.1)),
            "max_stationary_rate": 0.5,
        },
        "env_overrides": env_overrides,
    }


_BENCHMARKS = {
    "lane_following_empty_v0": _benchmark(
        "lane_following_empty_v0",
        "Town05 lane-following sanity check without dynamic traffic.",
        "control",
        "Town05",
        0,
    ),
    "lane_following_v0": _benchmark(
        "lane_following_v0",
        "Town05 lane following with moderate vehicle traffic and green lights.",
        "control",
        "Town05",
        50,
    ),
    "urban_traffic_v0": _benchmark(
        "urban_traffic_v0",
        "Town03 mixed urban traffic with active signals and pedestrians.",
        "traffic",
        "Town03",
        60,
        number_of_walkers=20,
        traffic="on",
        max_time_episode=750,
    ),
    "dense_traffic_v0": _benchmark(
        "dense_traffic_v0",
        "Town05 dense mixed traffic stress test under clear weather.",
        "traffic",
        "Town05",
        100,
        number_of_walkers=30,
        traffic="on",
        max_time_episode=750,
        desired_speed=7.0,
    ),
    "adverse_weather_v0": _benchmark(
        "adverse_weather_v0",
        "Town05 mixed traffic stress test in hard daytime rain.",
        "weather",
        "Town05",
        50,
        number_of_walkers=10,
        traffic="on",
        weather="HardRainNoon",
        max_time_episode=750,
        desired_speed=7.0,
    ),
    "town02_generalization_v0": _benchmark(
        "town02_generalization_v0",
        "Town02 held-out-map lane-following and traffic generalization test.",
        "generalization",
        "Town02",
        40,
        number_of_walkers=10,
        traffic="on",
    ),
}


_LIGHTWEIGHT_SUITE = (
    "lane_following_v0",
    "urban_traffic_v0",
    "dense_traffic_v0",
    "adverse_weather_v0",
    "town02_generalization_v0",
)


_BENCHMARK_SUITES = {
    "carla_lightweight_v0": _LIGHTWEIGHT_SUITE,
    # Kept as a compatibility alias. It is not an official CARLA paper suite.
    "carla_common_v0": _LIGHTWEIGHT_SUITE,
}


def list_benchmarks() -> Tuple[str, ...]:
    return tuple(sorted(_BENCHMARKS))


def get_benchmark(name: str) -> Dict[str, Any]:
    try:
        return deepcopy(_BENCHMARKS[name])
    except KeyError as exc:
        raise ValueError(
            "Unknown benchmark '{}'. Available benchmarks: {}".format(
                name, ", ".join(list_benchmarks())
            )
        ) from exc


def list_benchmark_suites() -> Tuple[str, ...]:
    return tuple(sorted(_BENCHMARK_SUITES))


def get_benchmark_suite(name: str) -> Tuple[str, ...]:
    try:
        return tuple(_BENCHMARK_SUITES[name])
    except KeyError as exc:
        raise ValueError(
            "Unknown benchmark suite '{}'. Available suites: {}".format(
                name, ", ".join(list_benchmark_suites())
            )
        ) from exc


def apply_benchmark(cfg: Any, benchmark: Dict[str, Any]) -> Any:
    for key, value in benchmark["env_overrides"].items():
        if not hasattr(cfg, key):
            raise AttributeError(
                "Benchmark override is not present in Config: {}".format(key)
            )
        setattr(cfg, key, value)
    return cfg
