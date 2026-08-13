from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np

from carla_rl_lab.envs.carla_env import CarlaEnv, carla


class FakeTrafficManager:
    def __init__(self):
        self.seed_value = None
        self.synchronous = None

    def set_random_device_seed(self, seed):
        self.seed_value = seed

    def set_synchronous_mode(self, enabled):
        self.synchronous = enabled


class FakeWorld:
    def __init__(self):
        self.pedestrian_seed = None
        self.applied_settings = None
        self.navigation_calls = 0
        self.events = []

    def set_pedestrians_seed(self, seed):
        self.pedestrian_seed = seed

    def apply_settings(self, settings):
        self.applied_settings = settings

    def get_random_location_from_navigation(self):
        self.navigation_calls += 1
        return None

    def get_blueprint_library(self):
        return FakeBlueprintLibrary()

    def try_spawn_actor(self, blueprint, transform, attach_to=None):
        if attach_to is None:
            return FakeWalker()
        return FakeWalkerController(self.events)

    def tick(self):
        self.events.append("tick")
        return 1


class FakeWalker:
    def destroy(self):
        return True


class FakeWalkerController:
    def __init__(self, events):
        self.events = events

    def start(self):
        self.events.append("start")

    def go_to_location(self, location):
        self.events.append("destination")

    def set_max_speed(self, speed):
        self.events.append("speed")


class FakeBlueprintLibrary:
    def filter(self, pattern):
        return [FakeBlueprint()]

    def find(self, name):
        return FakeBlueprint()


class FakeBlueprint:
    def has_attribute(self, name):
        return False


class EnvSafetyTest(unittest.TestCase):
    def test_env_seed_does_not_reset_algorithm_global_rng(self):
        random.seed(11)
        np.random.seed(11)
        expected = (random.random(), float(np.random.rand()))
        random.seed(11)
        np.random.seed(11)

        env = CarlaEnv.__new__(CarlaEnv)
        env.traffic_manager = FakeTrafficManager()
        env.world = FakeWorld()
        env.seed(123)

        actual = (random.random(), float(np.random.rand()))
        self.assertEqual(actual, expected)
        self.assertEqual(env.traffic_manager.seed_value, 123)
        self.assertEqual(env.world.pedestrian_seed, 123)

    def test_walker_spawn_has_a_hard_attempt_limit(self):
        env = CarlaEnv.__new__(CarlaEnv)
        env.world = FakeWorld()
        env.max_walker_spawn_attempts = 3
        env.number_of_walkers = 5
        env.spawned_walkers = []
        env.walker_controllers = []
        env._spawn_walkers()
        self.assertEqual(len(env.spawned_walkers), 0)
        self.assertEqual(env.world.navigation_calls, 3)

    def test_walker_controller_commands_follow_world_tick(self):
        env = CarlaEnv.__new__(CarlaEnv)
        env.world = FakeWorld()
        locations = iter([carla.Location(), carla.Location(x=1.0)])
        env.world.get_random_location_from_navigation = lambda: next(locations)
        env.max_walker_spawn_attempts = 1
        env.number_of_walkers = 1
        env.spawned_walkers = []
        env.walker_controllers = []
        env._python_random = random.Random(0)

        env._spawn_walkers()

        self.assertEqual(len(env.spawned_walkers), 1)
        self.assertEqual(
            env.world.events,
            ["tick", "start", "destination", "speed"],
        )


if __name__ == "__main__":
    unittest.main()
