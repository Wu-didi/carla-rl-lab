from __future__ import annotations

from typing import Any

from carla_rl_lab.envs.carla_env import CarlaEnv
from carla_rl_lab.rewards import build_reward_profile


def make_carla_env(cfg: Any) -> CarlaEnv:
    reward_fn = build_reward_profile(cfg.reward_profile, cfg.desired_speed)
    return CarlaEnv(
        params={
            "number_of_vehicles": cfg.number_of_vehicles,
            "number_of_walkers": cfg.number_of_walkers,
            "dt": cfg.dt,
            "ego_vehicle_filter": cfg.ego_vehicle_filter,
            "surrounding_vehicle_spawned_randomly": cfg.surrounding_vehicle_spawned_randomly,
            "port": cfg.port,
            "town": cfg.town,
            "max_time_episode": cfg.max_time_episode,
            "max_waypoints": cfg.max_waypoints,
            "visualize_waypoints": cfg.visualize_waypoints,
            "desired_speed": cfg.desired_speed,
            "max_ego_spawn_times": cfg.max_ego_spawn_times,
            "view_mode": cfg.view_mode,
            "traffic": cfg.traffic,
            "weather": cfg.weather,
            "lidar_max_range": cfg.lidar_max_range,
            "max_nearby_vehicles": cfg.max_nearby_vehicles,
            "enable_risk_field": cfg.enable_risk_field,
            "risk_field_sectors": cfg.risk_field_sectors,
            "risk_field_alpha": cfg.risk_field_alpha,
            "state_dim": cfg.state_dim,
            "action_dim": cfg.action_dim,
            "action_bound": cfg.action_bound,
            "action_mode": cfg.action_mode,
            "max_walker_spawn_attempts": cfg.max_walker_spawn_attempts,
            "reward_fn": reward_fn,
            "seed": cfg.seed,
        }
    )
