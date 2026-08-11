from __future__ import annotations

import math
import os
import queue
import random
import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import carla
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from carla_rl_lab.benchmarks.nocrash import load_nocrash_routes, weather_presets
from carla_rl_lab.envs.control import (
    TARGET_SPEED_2D,
    carla_action_to_policy,
    policy_action_to_carla,
    validate_action_spec,
)
from carla_rl_lab.observations import pixel_state_dim


class CarlaEnv(gym.Env):
    """CARLA environment for camera-based continuous-control research.

    Policy input is deliberately limited to RGB, route waypoints, speed, and
    steering. Simulator actor state is used only for reward and evaluation.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.number_of_vehicles = int(params["number_of_vehicles"])
        self.number_of_walkers = int(params["number_of_walkers"])
        self.dt = float(params["dt"])
        self.max_time_episode = int(params["max_time_episode"])
        self.max_waypoints = int(params["max_waypoints"])
        self.visualize_waypoints = bool(params["visualize_waypoints"])
        self.desired_speed = float(params["desired_speed"])
        self.view_mode = str(params["view_mode"])
        self.traffic = str(params["traffic"])
        self.weather = str(params.get("weather", "ClearNoon"))
        self.weather_group = str(params.get("weather_group", "fixed"))
        self.image_size = int(params.get("image_size", 84))
        self.frame_stack = int(params.get("frame_stack", 3))
        self.camera_fov = float(params.get("camera_fov", 90.0))
        self.route_lookahead_m = float(params.get("route_lookahead_m", 25.0))
        self.route_sampling_resolution = float(
            params.get("route_sampling_resolution", 2.0)
        )
        self.goal_tolerance = float(params.get("goal_tolerance", 4.0))
        self.route_mode = str(params.get("route_mode", "endless"))
        self.route_id = int(params.get("route_id", -1))
        self.blocked_seconds = float(params.get("blocked_seconds", 15.0))
        self.reward_fn = params.get("reward_fn")
        self.action_dim = int(params.get("action_dim", 2))
        self.action_bound = float(params.get("action_bound", 1.0))
        self.action_mode = str(params.get("action_mode", TARGET_SPEED_2D))
        self.max_walker_spawn_attempts = max(
            self.number_of_walkers,
            int(params.get("max_walker_spawn_attempts", 200)),
        )
        if params.get("observation_mode", "pixel_v1") != "pixel_v1":
            raise ValueError("Only observation_mode='pixel_v1' is supported")
        if self.route_mode not in ("endless", "fixed"):
            raise ValueError("route_mode must be 'endless' or 'fixed'")
        validate_action_spec(self.action_mode, self.action_dim)

        encoded_dim = pixel_state_dim(
            self.image_size, self.frame_stack, self.max_waypoints
        )
        self.state_dim = int(params.get("state_dim", encoded_dim))
        if self.state_dim != encoded_dim:
            raise ValueError(
                "state_dim={} does not match pixel_v1 schema ({})".format(
                    self.state_dim, encoded_dim
                )
            )

        self.action_space = spaces.Box(
            low=-self.action_bound,
            high=self.action_bound,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    0,
                    255,
                    shape=(3 * self.frame_stack, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
                "waypoints": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(2 * self.max_waypoints,),
                    dtype=np.float32,
                ),
                "vehicle_measurements": spaces.Box(
                    -1.0, 1.0, shape=(2,), dtype=np.float32
                ),
                "ego_state": spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype=np.float32
                ),
                "lane_info": spaces.Box(
                    -np.inf, np.inf, shape=(2,), dtype=np.float32
                ),
            }
        )
        self.policy_observation_space = spaces.Box(
            0, 255, shape=(self.state_dim,), dtype=np.uint8
        )

        print("Connecting to CARLA server...")
        self.client = carla.Client("localhost", int(params["port"]))
        self.client.set_timeout(20.0)
        self.world = self.client.load_world(str(params["town"]))
        self.world_map = self.world.get_map()
        self.tm_port = int(params.get("tm_port", 8000))
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self._original_settings = self.world.get_settings()
        self._original_synchronous = self._original_settings.synchronous_mode
        self._original_fixed_delta = self._original_settings.fixed_delta_seconds
        self.seed(params.get("seed"))

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self._seed % (2**31))

        self._vehicle_filter = str(params["ego_vehicle_filter"])
        self._random_surrounding_vehicles = bool(
            params["surrounding_vehicle_spawned_randomly"]
        )
        self._camera_transform = carla.Transform(
            carla.Location(
                x=float(params.get("camera_location_x", 1.5)),
                z=float(params.get("camera_location_z", 2.4)),
            )
        )
        route_file = str(params.get("route_file", ""))
        self._route_definitions = (
            load_nocrash_routes(route_file) if route_file else {}
        )
        self._spawn_points = list(self.world_map.get_spawn_points())
        if not self._spawn_points:
            raise RuntimeError("CARLA map has no vehicle spawn points")

        self.ego: Optional[carla.Vehicle] = None
        self.collision_sensor: Optional[carla.Sensor] = None
        self.rgb_camera: Optional[carla.Sensor] = None
        self.spawned_vehicles: List[carla.Vehicle] = []
        self.spawned_walkers: List[carla.Walker] = []
        self.walker_controllers: List[carla.WalkerAIController] = []
        self._camera_queue: queue.Queue = queue.Queue(maxsize=8)
        self._frames = deque(maxlen=self.frame_stack)
        self._latest_frame: Optional[np.ndarray] = None
        self._route: List[Tuple[Any, Any]] = []
        self._route_distances = np.zeros(1, dtype=np.float32)
        self._route_index = 0
        self._route_total_distance = 0.0
        self._destination: Optional[carla.Location] = None
        self._active_route_id = -1
        self._expert_agent = None
        self.last_reward_terms: Dict[str, float] = {}
        self._reset_episode_state()
        print("CARLA connection established")

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random, resolved_seed = seeding.np_random(seed)
        self._seed = int(resolved_seed)
        self._python_random = random.Random(self._seed % (2**32))
        if hasattr(self, "action_space"):
            self.action_space.seed(self._seed % (2**32))
        if hasattr(self, "traffic_manager"):
            self.traffic_manager.set_random_device_seed(self._seed % (2**31))
        if hasattr(self, "world") and hasattr(self.world, "set_pedestrians_seed"):
            self.world.set_pedestrians_seed(self._seed % (2**31))
        return [self._seed]

    def _reset_episode_state(self) -> None:
        self.time_step = 0
        self.total_step = getattr(self, "total_step", 0)
        self.termination_reason = None
        self._is_collision = False
        self._is_off_road = False
        self._collision_type = ""
        self._red_light_infraction = False
        self._red_light_ids = set()
        self._blocked_time = 0.0
        self._last_steer = 0.0
        self._pid_integral = 0.0
        self._pid_previous_error = 0.0
        self._route_completions = 0
        self.last_reward_terms = {}

    def _set_weather(self) -> None:
        choices = weather_presets(self.weather_group, self.weather)
        preset = self._python_random.choice(choices)
        self.weather = preset
        self.world.set_weather(getattr(carla.WeatherParameters, preset))

    def _choose_task(self) -> Tuple[carla.Transform, carla.Location]:
        if self._route_definitions:
            route_ids = sorted(self._route_definitions)
            selected = self.route_id
            if selected < 0:
                selected = self._python_random.choice(route_ids)
            if selected not in self._route_definitions:
                raise ValueError(
                    "route_id={} is not present in {}".format(
                        selected, sorted(self._route_definitions)
                    )
                )
            start, destination = self._route_definitions[selected]
            self._active_route_id = selected
            return start, destination.location

        start = self._python_random.choice(self._spawn_points)
        candidates = [
            item
            for item in self._spawn_points
            if item.location.distance(start.location) > 50.0
        ]
        destination = self._python_random.choice(candidates or self._spawn_points)
        self._active_route_id = -1
        return start, destination.location

    def _build_route(self, origin: carla.Location, destination: carla.Location) -> None:
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except ImportError as exc:
            raise ImportError(
                "CARLA navigation agents are missing. Add "
                "$CARLA_ROOT/PythonAPI/carla to PYTHONPATH."
            ) from exc

        planner = GlobalRoutePlanner(
            self.world_map, self.route_sampling_resolution
        )
        route = planner.trace_route(origin, destination)
        if len(route) < 2:
            raise RuntimeError("CARLA global planner returned an empty route")
        self._route = route
        self._route_index = 0
        locations = [item[0].transform.location for item in route]
        distances = [0.0]
        for previous, current in zip(locations[:-1], locations[1:]):
            distances.append(distances[-1] + previous.distance(current))
        self._route_distances = np.asarray(distances, dtype=np.float32)
        self._route_total_distance = max(float(distances[-1]), 1e-6)
        self._destination = destination

    def _spawn_ego(self, transform: carla.Transform) -> None:
        blueprints = list(self.world.get_blueprint_library().filter(self._vehicle_filter))
        if not blueprints:
            raise RuntimeError(
                "No ego blueprint matches '{}'".format(self._vehicle_filter)
            )
        blueprint = self._python_random.choice(blueprints)
        blueprint.set_attribute("role_name", "hero")
        if blueprint.has_attribute("color"):
            blueprint.set_attribute("color", "255,0,0")
        for attempt in range(int(self.params.get("max_ego_spawn_times", 200))):
            self.ego = self.world.try_spawn_actor(blueprint, transform)
            if self.ego is not None:
                return
            time.sleep(min(0.02 * (attempt + 1), 0.2))
        raise RuntimeError("Failed to spawn the ego vehicle")

    def _spawn_traffic(self) -> None:
        spawn_points = list(self._spawn_points)
        self._python_random.shuffle(spawn_points)
        library = self.world.get_blueprint_library()
        vehicle_blueprints = [
            blueprint
            for blueprint in library.filter("vehicle.*")
            if blueprint.has_attribute("number_of_wheels")
            and int(blueprint.get_attribute("number_of_wheels")) == 4
        ]
        for transform in spawn_points:
            if len(self.spawned_vehicles) >= self.number_of_vehicles:
                break
            if transform.location.distance(self.ego.get_location()) < 8.0:
                continue
            blueprint = (
                self._python_random.choice(vehicle_blueprints)
                if self._random_surrounding_vehicles
                else library.find("vehicle.tesla.model3")
            )
            blueprint.set_attribute("role_name", "autopilot")
            if blueprint.has_attribute("color"):
                colors = blueprint.get_attribute("color").recommended_values
                if colors:
                    blueprint.set_attribute("color", self._python_random.choice(colors))
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                self.spawned_vehicles.append(vehicle)
                vehicle.set_autopilot(True, self.tm_port)

    def _spawn_walkers(self) -> None:
        library = self.world.get_blueprint_library()
        walker_blueprints = list(library.filter("walker.pedestrian.*"))
        controller_blueprint = library.find("controller.ai.walker")
        attempts = 0
        while (
            len(self.spawned_walkers) < self.number_of_walkers
            and attempts < self.max_walker_spawn_attempts
        ):
            attempts += 1
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            blueprint = self._python_random.choice(walker_blueprints)
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(
                blueprint, carla.Transform(location)
            )
            if walker is None:
                continue
            controller = self.world.try_spawn_actor(
                controller_blueprint, carla.Transform(), attach_to=walker
            )
            if controller is None:
                walker.destroy()
                continue
            self.spawned_walkers.append(walker)
            self.walker_controllers.append(controller)
            controller.start()
            destination = self.world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(1.0 + self._python_random.random())
        if len(self.spawned_walkers) < self.number_of_walkers:
            warnings.warn(
                "Spawned {}/{} requested walkers".format(
                    len(self.spawned_walkers), self.number_of_walkers
                ),
                RuntimeWarning,
            )

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        self._is_collision = True
        type_id = event.other_actor.type_id
        if type_id.startswith("vehicle."):
            self._collision_type = "vehicle"
        elif type_id.startswith("walker."):
            self._collision_type = "pedestrian"
        else:
            self._collision_type = "layout"

    def _on_camera(self, image: carla.Image) -> None:
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4
        )
        rgb_chw = bgra[:, :, :3][:, :, ::-1].transpose(2, 0, 1).copy()
        item = (image.frame, rgb_chw)
        try:
            self._camera_queue.put_nowait(item)
        except queue.Full:
            try:
                self._camera_queue.get_nowait()
            except queue.Empty:
                pass
            self._camera_queue.put_nowait(item)

    def _spawn_sensors(self) -> None:
        library = self.world.get_blueprint_library()
        collision_bp = library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego
        )
        self.collision_sensor.listen(self._on_collision)

        camera_bp = library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_size))
        camera_bp.set_attribute("image_size_y", str(self.image_size))
        camera_bp.set_attribute("fov", str(self.camera_fov))
        camera_bp.set_attribute("sensor_tick", "0.0")
        self.rgb_camera = self.world.spawn_actor(
            camera_bp, self._camera_transform, attach_to=self.ego
        )
        self.rgb_camera.listen(self._on_camera)

    def _consume_camera(self, world_frame: int) -> None:
        deadline = time.time() + 10.0
        last_sensor_frame = None
        while time.time() < deadline:
            try:
                frame, image = self._camera_queue.get(
                    timeout=max(deadline - time.time(), 0.01)
                )
            except queue.Empty:
                break
            last_sensor_frame = frame
            if frame >= world_frame:
                self._latest_frame = image
                self._frames.append(image)
                return
        raise RuntimeError(
            "Timed out waiting for RGB frame {} (last sensor frame: {})".format(
                world_frame, last_sensor_frame
            )
        )

    def _tick(self) -> None:
        frame = self.world.tick()
        self._consume_camera(frame)

    def _configure_traffic_lights(self) -> None:
        for light in self.world.get_actors().filter("traffic.traffic_light*"):
            if self.traffic == "off":
                light.set_state(carla.TrafficLightState.Green)
                light.freeze(True)
            else:
                light.freeze(False)

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.seed(seed)
        self._destroy_episode_actors()
        self._reset_episode_state()
        self._set_weather()
        start, destination = self._choose_task()
        self._spawn_ego(start)
        self._build_route(start.location, destination)
        self._spawn_traffic()
        self._spawn_walkers()
        self._configure_traffic_lights()
        self._spawn_sensors()
        self._tick()
        while len(self._frames) < self.frame_stack:
            self._frames.appendleft(self._latest_frame.copy())
        return self._get_obs()

    def _target_speed_control(self, action: np.ndarray) -> carla.VehicleControl:
        unit = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1) / self.action_bound,
            -1.0,
            1.0,
        )
        target_speed = (float(unit[0]) + 1.0) * 0.5 * self.desired_speed
        current_speed = self._speed()
        error = (target_speed - current_speed) / max(self.desired_speed, 1e-6)
        self._pid_integral = float(
            np.clip(self._pid_integral + error * self.dt, -1.0, 1.0)
        )
        derivative = (error - self._pid_previous_error) / max(self.dt, 1e-6)
        self._pid_previous_error = error
        signal = error + 0.2 * self._pid_integral + 0.001 * derivative
        throttle = float(np.clip(signal, 0.0, 1.0))
        brake = float(np.clip(-signal, 0.0, 1.0))
        if target_speed < 0.05:
            throttle, brake = 0.0, 1.0
        return carla.VehicleControl(
            throttle=throttle, steer=float(unit[1]), brake=brake
        )

    def _apply_policy_action(self, action: np.ndarray) -> None:
        if self.action_mode == TARGET_SPEED_2D:
            control = self._target_speed_control(action)
        else:
            throttle, steer, brake = policy_action_to_carla(
                action, self.action_mode, self.action_bound
            )
            control = carla.VehicleControl(
                throttle=throttle, steer=steer, brake=brake
            )
        self.ego.apply_control(control)

    def _transition(self, action: np.ndarray):
        self._tick()
        self.time_step += 1
        self.total_step += 1
        self._update_spectator()
        obs = self._get_obs()
        done = self._terminal(obs)
        context = self._reward_context(obs)
        reward = self._get_reward(obs, done, context)
        cost = self._get_cost(obs)
        info = dict(context)
        info.update(
            {
                "termination_reason": self.termination_reason,
                "reward_terms": dict(self.last_reward_terms),
                "spawned_vehicles": len(self.spawned_vehicles),
                "spawned_walkers": len(self.spawned_walkers),
                "route_id": self._active_route_id,
                "route_completion": self.route_completion,
                "collision_type": self._collision_type,
                "weather": self.weather,
            }
        )
        return obs, reward, cost, done, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        validate_action_spec(self.action_mode, action.size)
        self._apply_policy_action(action)
        return self._transition(action)

    def set_ego_autopilot(self, enabled: bool = True) -> None:
        self._expert_agent = None
        if not enabled:
            self.ego.set_autopilot(False, self.tm_port)
            return
        try:
            from agents.navigation.behavior_agent import BehaviorAgent

            self._expert_agent = BehaviorAgent(self.ego, behavior="normal")
            self._expert_agent.set_destination(self._destination)
        except ImportError:
            self.ego.set_autopilot(True, self.tm_port)

    def step_sample(self):
        if self._expert_agent is not None:
            control = self._expert_agent.run_step()
            self.ego.apply_control(control)
        else:
            control = self.ego.get_control()
        if self.action_mode == TARGET_SPEED_2D:
            safe_speed = self._safe_desired_speed()
            speed_action = 2.0 * safe_speed / max(self.desired_speed, 1e-6) - 1.0
            action = np.asarray([speed_action, control.steer], dtype=np.float32)
        else:
            action = carla_action_to_policy(
                control.throttle,
                control.steer,
                control.brake,
                self.action_mode,
                self.action_bound,
            )
        obs, reward, cost, done, info = self._transition(action)
        return obs, reward, cost, done, info, action

    def _speed(self) -> float:
        velocity = self.ego.get_velocity()
        return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))

    def _advance_route(self) -> None:
        location = self.ego.get_location()
        end = min(self._route_index + 25, len(self._route))
        candidates = range(max(self._route_index - 2, 0), end)
        self._route_index = min(
            candidates,
            key=lambda index: self._route[index][0].transform.location.distance(
                location
            ),
        )

    @property
    def route_completion(self) -> float:
        if not self._route:
            return 0.0
        return float(
            np.clip(
                self._route_distances[self._route_index]
                / self._route_total_distance,
                0.0,
                1.0,
            )
        )

    def _route_waypoints(self, ego_transform: carla.Transform) -> np.ndarray:
        result = np.zeros((self.max_waypoints, 2), dtype=np.float32)
        yaw = math.radians(ego_transform.rotation.yaw)
        cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)
        for output_index in range(self.max_waypoints):
            route_index = min(
                self._route_index + output_index, len(self._route) - 1
            )
            location = self._route[route_index][0].transform.location
            dx = location.x - ego_transform.location.x
            dy = location.y - ego_transform.location.y
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy
            result[output_index] = np.clip(
                [local_x / self.route_lookahead_m, local_y / self.route_lookahead_m],
                -1.0,
                1.0,
            )
            if self.visualize_waypoints:
                self.world.debug.draw_point(
                    location + carla.Location(z=0.5),
                    size=0.08,
                    life_time=self.dt,
                    color=carla.Color(0, 255, 0),
                )
        return result

    def _lane_measurements(
        self, ego_transform: carla.Transform
    ) -> Tuple[float, float, float]:
        waypoint = self.world_map.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return 0.0, 0.0, 0.0
        lane_transform = waypoint.transform
        forward = lane_transform.get_forward_vector()
        right = np.asarray([-forward.y, forward.x], dtype=np.float32)
        offset = np.asarray(
            [
                ego_transform.location.x - lane_transform.location.x,
                ego_transform.location.y - lane_transform.location.y,
            ],
            dtype=np.float32,
        )
        signed_offset = float(np.dot(right, offset))
        route_yaw = self._route[self._route_index][0].transform.rotation.yaw
        heading = math.radians(ego_transform.rotation.yaw - route_yaw)
        heading = math.atan2(math.sin(heading), math.cos(heading))
        return float(waypoint.lane_width), signed_offset, heading

    def _get_obs(self) -> Dict[str, np.ndarray]:
        self._advance_route()
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        acceleration = self.ego.get_acceleration()
        angular_velocity = self.ego.get_angular_velocity()
        speed = self._speed()
        forward = transform.get_forward_vector()
        right = np.asarray([-forward.y, forward.x], dtype=np.float32)
        acceleration_xy = np.asarray([acceleration.x, acceleration.y], dtype=np.float32)
        acceleration_longitudinal = float(
            np.dot(np.asarray([forward.x, forward.y]), acceleration_xy)
        )
        acceleration_lateral = float(np.dot(right, acceleration_xy))
        lane_width, lane_offset, _ = self._lane_measurements(transform)
        control = self.ego.get_control()

        return {
            "image": np.concatenate(list(self._frames), axis=0),
            "waypoints": self._route_waypoints(transform).reshape(-1),
            "vehicle_measurements": np.asarray(
                [
                    np.clip(speed / max(self.desired_speed, 1e-6), 0.0, 1.0),
                    np.clip(control.steer, -1.0, 1.0),
                ],
                dtype=np.float32,
            ),
            "ego_state": np.asarray(
                [
                    transform.location.x,
                    transform.location.y,
                    math.radians(transform.rotation.yaw),
                    speed,
                    angular_velocity.z,
                    acceleration_longitudinal,
                    acceleration_lateral,
                ],
                dtype=np.float32,
            ),
            "lane_info": np.asarray([lane_width, lane_offset], dtype=np.float32),
        }

    def _safe_desired_speed(self) -> float:
        transform = self.ego.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)
        safe_speed = self.desired_speed
        actors: Sequence[carla.Actor] = list(
            self.world.get_actors().filter("vehicle.*")
        ) + list(self.world.get_actors().filter("walker.pedestrian.*"))
        for actor in actors:
            if actor.id == self.ego.id:
                continue
            location = actor.get_location()
            dx = location.x - transform.location.x
            dy = location.y - transform.location.y
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy
            lateral_limit = 2.5 if actor.type_id.startswith("vehicle.") else 3.0
            if 0.0 < local_x < 15.0 and abs(local_y) < lateral_limit:
                clearance = 8.0 if actor.type_id.startswith("vehicle.") else 6.0
                distance = max(local_x - clearance, 0.0)
                safe_speed = min(
                    safe_speed, self.desired_speed * np.clip(distance / 5.0, 0.0, 1.0)
                )
        if self.ego.is_at_traffic_light():
            state = self.ego.get_traffic_light_state()
            if state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                safe_speed = 0.0
        return float(safe_speed)

    def _detect_red_light_infraction(self, speed: float) -> bool:
        if not self.ego.is_at_traffic_light():
            return False
        state = self.ego.get_traffic_light_state()
        light = self.ego.get_traffic_light()
        if (
            light is not None
            and state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
            and speed > 1.0
            and light.id not in self._red_light_ids
        ):
            self._red_light_ids.add(light.id)
            self._red_light_infraction = True
            return True
        return False

    def _reward_context(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        _, _, heading_error = self._lane_measurements(self.ego.get_transform())
        steer = float(self.ego.get_control().steer)
        context = {
            "is_collision": self._is_collision,
            "is_off_road": self._is_off_road,
            "red_light_infraction": self._red_light_infraction,
            "safe_desired_speed": self._safe_desired_speed(),
            "heading_error": heading_error,
            "steer_delta": steer - self._last_steer,
            "desired_speed": self.desired_speed,
            "dt": self.dt,
        }
        self._last_steer = steer
        return context

    def _get_reward(
        self, obs: Dict[str, np.ndarray], done: bool, context: Dict[str, Any]
    ) -> float:
        if self.reward_fn is None:
            raise RuntimeError("A reward profile is required")
        reward, terms = self.reward_fn(obs, done, context)
        self.last_reward_terms = dict(terms)
        return float(reward)

    def _get_cost(self, obs: Dict[str, np.ndarray]) -> float:
        cost = 20.0 * float(self._is_collision or self._is_off_road)
        cost += 10.0 * float(self._red_light_infraction)
        speed = float(obs["ego_state"][3])
        if speed > self.desired_speed:
            cost += (speed - self.desired_speed) / max(self.desired_speed, 1e-6)
        return cost

    def _terminal(self, obs: Dict[str, np.ndarray]) -> bool:
        speed = float(obs["ego_state"][3])
        self._detect_red_light_infraction(speed)
        if self._is_collision:
            self.termination_reason = "collision"
            return True
        if self._red_light_infraction:
            self.termination_reason = "red_light"
            return True

        exact_waypoint = self.world_map.get_waypoint(
            self.ego.get_location(),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        lane_width, lateral_offset = obs["lane_info"]
        if exact_waypoint is None or (
            lane_width > 0.0 and abs(lateral_offset) > lane_width / 2.0 + 1.0
        ):
            self._is_off_road = True
            self.termination_reason = "lane_departure"
            return True

        _, _, heading = self._lane_measurements(self.ego.get_transform())
        if abs(heading) > math.pi / 2.0 and not exact_waypoint.is_junction:
            self._is_off_road = True
            self.termination_reason = "wrong_way"
            return True

        self._blocked_time = self._blocked_time + self.dt if speed < 0.1 else 0.0
        if self._blocked_time >= self.blocked_seconds:
            self.termination_reason = "blocked"
            return True

        reached = self.ego.get_location().distance(self._destination) < self.goal_tolerance
        if reached:
            self._route_completions += 1
            if self.route_mode == "fixed":
                self._route_index = len(self._route) - 1
                self.termination_reason = "route_completed"
                return True
            _, destination = self._choose_task()
            self._build_route(self.ego.get_location(), destination)
            if self._expert_agent is not None:
                self._expert_agent.set_destination(destination)

        if self.time_step >= self.max_time_episode:
            self.termination_reason = "timeout"
            return True
        return False

    def _update_spectator(self) -> None:
        if self.view_mode == "none":
            return
        transform = self.ego.get_transform()
        spectator = self.world.get_spectator()
        if self.view_mode == "top":
            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=40.0),
                    carla.Rotation(pitch=-90.0),
                )
            )
        elif self.view_mode == "follow":
            location = transform.transform(carla.Location(x=-6.0, z=3.0))
            spectator.set_transform(
                carla.Transform(
                    location,
                    carla.Rotation(pitch=-10.0, yaw=transform.rotation.yaw),
                )
            )

    def _destroy_actor(self, actor: Optional[carla.Actor]) -> None:
        if actor is None:
            return
        try:
            if actor.type_id.startswith("sensor."):
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        except (RuntimeError, AttributeError):
            pass

    def _destroy_episode_actors(self) -> None:
        for controller in self.walker_controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        for actor in (
            [self.rgb_camera, self.collision_sensor]
            + self.walker_controllers
            + self.spawned_walkers
            + self.spawned_vehicles
            + [self.ego]
        ):
            self._destroy_actor(actor)
        self.rgb_camera = None
        self.collision_sensor = None
        self.ego = None
        self.spawned_vehicles = []
        self.spawned_walkers = []
        self.walker_controllers = []
        self._expert_agent = None
        self._frames.clear()
        self._latest_frame = None
        self._camera_queue = queue.Queue(maxsize=8)

    def close(self) -> None:
        self._destroy_episode_actors()
        try:
            self.traffic_manager.set_synchronous_mode(False)
            settings = self.world.get_settings()
            settings.synchronous_mode = self._original_synchronous
            settings.fixed_delta_seconds = self._original_fixed_delta
            self.world.apply_settings(settings)
        except RuntimeError:
            pass
