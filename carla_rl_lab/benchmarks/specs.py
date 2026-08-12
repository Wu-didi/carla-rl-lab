from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

from carla_rl_lab.benchmarks.nocrash import bundled_route_file


_SEEDS = (0, 1, 2, 3, 4)
_BASE_OVERRIDES = {
    "number_of_walkers": 0,
    "traffic": "off",
    "max_time_episode": 500,
    "desired_speed": 8.0,
    "view_mode": "none",
    "visualize_waypoints": False,
    "reward_profile": "nocrash_v0",
    "weather": "ClearNoon",
}

_RLFOLD_MODEL_IMAGE_SIZE = 84
_RLFOLD_FRAME_STACK = 2
_RLFOLD_NUM_WAYPOINTS = 10
_RLFOLD_STATE_DIM = (
    3 * _RLFOLD_FRAME_STACK * _RLFOLD_MODEL_IMAGE_SIZE ** 2
    + 2 * _RLFOLD_NUM_WAYPOINTS
    + 2
)


def _nocrash_benchmark(
    name: str,
    town: str,
    traffic_density: str,
    vehicles: int,
    walkers: int,
    weather_group: str,
    route_mode: str,
    vehicle_range: Tuple[int, int] = (-1, -1),
    walker_range: Tuple[int, int] = (-1, -1),
) -> Dict[str, Any]:
    training = town == "Town01"
    fixed_routes = route_mode == "fixed"
    return {
        "name": name,
        "description": (
            "RLfOLD NoCrash CARLA 0.9.15 adaptation with {} traffic in {}."
        ).format(traffic_density, town),
        "category": "nocrash",
        "protocol_id": "rlfold_nocrash_0915_v0",
        "reference": {
            "project": "RLfOLD",
            "original_carla_version": "0.9.10.1",
            "adaptation_carla_version": "0.9.15",
        },
        "observation_profile": {
            "name": "rlfold_front_rgb_84_v0",
            "source": "front_rgb_256x256",
            "model_input": "front_rgb_84x84",
            "frame_stack": _RLFOLD_FRAME_STACK,
            "route_waypoints": _RLFOLD_NUM_WAYPOINTS,
            "measurements": ("normalized_speed", "previous_steer"),
        },
        "seeds": (0,),
        "success_reasons": ("route_completed",),
        "success_criteria": {
            "min_horizon_fraction": 0.0,
            "min_distance_m": 0.0,
            "max_stationary_rate": 1.0,
            "require_zero_collisions": fixed_routes,
        },
        "route_ids": tuple(range(25)) if fixed_routes else (),
        "weather_presets": (
            ("ClearNoon", "WetNoon", "HardRainNoon", "ClearSunset")
            if training
            else ("SoftRainSunset", "WetSunset")
        ),
        "traffic_density": traffic_density,
        "env_overrides": {
            "town": town,
            "number_of_vehicles": vehicles,
            "number_of_walkers": walkers,
            "min_number_of_vehicles": vehicle_range[0],
            "max_number_of_vehicles": vehicle_range[1],
            "min_number_of_walkers": walker_range[0],
            "max_number_of_walkers": walker_range[1],
            "traffic": "on",
            "max_time_episode": 1200,
            "desired_speed": 5.0,
            "view_mode": "none",
            "visualize_waypoints": False,
            "reward_profile": "nocrash_v0",
            "weather_group": weather_group,
            "route_file": bundled_route_file(town) if fixed_routes else "",
            "route_id": -1,
            "route_mode": route_mode,
            "action_mode": "target_speed_2d",
            "action_dim": 2,
            "observation_mode": "pixel_v1",
            "camera_layout": "front",
            "num_cameras": 1,
            "camera_sensor_width": 256,
            "camera_sensor_height": 256,
            "camera_fov": 90.0,
            "camera_location_x": 1.5,
            "camera_location_z": 2.4,
            "image_size": _RLFOLD_MODEL_IMAGE_SIZE,
            "frame_stack": _RLFOLD_FRAME_STACK,
            "max_waypoints": _RLFOLD_NUM_WAYPOINTS,
            "state_dim": _RLFOLD_STATE_DIM,
        },
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
    "nocrash_train_empty_v0": _nocrash_benchmark(
        "nocrash_train_empty_v0",
        "Town01",
        "empty",
        0,
        0,
        "nocrash_train",
        "endless",
    ),
    "nocrash_train_v0": _nocrash_benchmark(
        "nocrash_train_v0",
        "Town01",
        "variable",
        0,
        0,
        "nocrash_train",
        "endless",
        vehicle_range=(0, 150),
        walker_range=(0, 300),
    ),
    "nocrash_train_regular_v0": _nocrash_benchmark(
        "nocrash_train_regular_v0",
        "Town01",
        "regular",
        20,
        50,
        "nocrash_train",
        "endless",
    ),
    "nocrash_empty_v0": _nocrash_benchmark(
        "nocrash_empty_v0", "Town02", "empty", 0, 0, "nocrash_test", "fixed"
    ),
    "nocrash_regular_v0": _nocrash_benchmark(
        "nocrash_regular_v0", "Town02", "regular", 15, 50, "nocrash_test", "fixed"
    ),
    "nocrash_dense_v0": _nocrash_benchmark(
        "nocrash_dense_v0", "Town02", "dense", 70, 150, "nocrash_test", "fixed"
    ),
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
    "nocrash_0915_v0": (
        "nocrash_empty_v0",
        "nocrash_regular_v0",
        "nocrash_dense_v0",
    ),
    "rlfold_nocrash_0915_v0": (
        "nocrash_empty_v0",
        "nocrash_regular_v0",
        "nocrash_dense_v0",
    ),
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
    if hasattr(cfg, "benchmark_name"):
        cfg.benchmark_name = benchmark["name"]
    for key, value in benchmark["env_overrides"].items():
        if not hasattr(cfg, key):
            raise AttributeError(
                "Benchmark override is not present in Config: {}".format(key)
            )
        setattr(cfg, key, value)
    return cfg
