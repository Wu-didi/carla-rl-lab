from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np

from carla_rl_lab.envs.carla_env import CarlaEnv


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

    def set_pedestrians_seed(self, seed):
        self.pedestrian_seed = seed

    def apply_settings(self, settings):
        self.applied_settings = settings

    def get_random_location_from_navigation(self):
        self.navigation_calls += 1
        return None


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
        env.walker_spawn_points = []
        self.assertEqual(env._spawn_walkers(5), 0)
        self.assertEqual(env.world.navigation_calls, 3)

    def test_world_and_traffic_manager_sync_together(self):
        env = CarlaEnv.__new__(CarlaEnv)
        env.settings = SimpleNamespace(synchronous_mode=False)
        env.world = FakeWorld()
        env.traffic_manager = FakeTrafficManager()
        env._set_synchronous_mode(True)
        self.assertTrue(env.settings.synchronous_mode)
        self.assertTrue(env.traffic_manager.synchronous)
        self.assertIs(env.world.applied_settings, env.settings)


if __name__ == "__main__":
    unittest.main()
