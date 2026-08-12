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
            "min_number_of_vehicles": cfg.min_number_of_vehicles,
            "max_number_of_vehicles": cfg.max_number_of_vehicles,
            "min_number_of_walkers": cfg.min_number_of_walkers,
            "max_number_of_walkers": cfg.max_number_of_walkers,
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
            "observation_mode": cfg.observation_mode,
            "image_size": cfg.image_size,
            "frame_stack": cfg.frame_stack,
            "camera_layout": cfg.camera_layout,
            "num_cameras": cfg.num_cameras,
            "camera_sensor_width": cfg.camera_sensor_width,
            "camera_sensor_height": cfg.camera_sensor_height,
            "camera_fov": cfg.camera_fov,
            "camera_location_x": cfg.camera_location_x,
            "camera_location_z": cfg.camera_location_z,
            "route_file": cfg.route_file,
            "route_id": cfg.route_id,
            "route_mode": cfg.route_mode,
            "route_lookahead_m": cfg.route_lookahead_m,
            "route_sampling_resolution": cfg.route_sampling_resolution,
            "goal_tolerance": cfg.goal_tolerance,
            "weather_group": cfg.weather_group,
            "tm_port": cfg.tm_port,
            "blocked_seconds": cfg.blocked_seconds,
            "state_dim": cfg.state_dim,
            "action_dim": cfg.action_dim,
            "action_bound": cfg.action_bound,
            "action_mode": cfg.action_mode,
            "max_walker_spawn_attempts": cfg.max_walker_spawn_attempts,
            "reward_fn": reward_fn,
            "seed": cfg.seed,
        }
    )
