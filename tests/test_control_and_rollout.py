from __future__ import annotations

import unittest

import numpy as np

from carla_rl_lab.buffers import RolloutBuffer, generalized_advantage_estimate
from carla_rl_lab.envs.control import (
    carla_action_to_policy,
    policy_action_to_carla,
)


class ControlAndRolloutTest(unittest.TestCase):
    def test_signed_3d_action_is_backward_compatible(self):
        control = policy_action_to_carla(
            np.array([-0.5, 0.25, 0.75], dtype=np.float32), "signed_3d"
        )
        self.assertEqual(control, (0.0, 0.25, 0.75))
        encoded = carla_action_to_policy(*control, action_mode="signed_3d")
        np.testing.assert_allclose(encoded, [0.0, 0.25, 0.75])

    def test_longitudinal_2d_action_separates_throttle_and_brake(self):
        np.testing.assert_allclose(
            policy_action_to_carla(np.array([0.6, -0.2]), "longitudinal_2d"),
            (0.6, -0.2, 0.0),
        )
        np.testing.assert_allclose(
            policy_action_to_carla(np.array([-0.4, 0.2]), "longitudinal_2d"),
            (0.0, 0.2, 0.4),
        )
        np.testing.assert_allclose(
            carla_action_to_policy(0.0, 0.3, 0.7, "longitudinal_2d"),
            [-0.7, 0.3],
        )

    def test_action_mode_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            policy_action_to_carla(np.zeros(3), "longitudinal_2d")

    def test_gae_bootstraps_timeout_without_crossing_episode(self):
        targets = generalized_advantage_estimate(
            rewards=np.array([0.0], dtype=np.float32),
            dones=np.array([0.0], dtype=np.float32),
            values=np.array([1.0], dtype=np.float32),
            last_value=0.0,
            gamma=0.9,
            gae_lambda=0.95,
            episode_ends=np.array([1.0], dtype=np.float32),
            next_values=np.array([2.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(targets["advantages"][0]), 0.8, places=6)
        self.assertAlmostEqual(float(targets["returns"][0]), 1.8, places=6)

    def test_rollout_distinguishes_terminal_and_timeout(self):
        rollout = RolloutBuffer(2, gamma=0.9, gae_lambda=1.0)
        rollout.add(
            np.zeros(2),
            np.zeros(1),
            reward=0.0,
            done=True,
            value=1.0,
            log_prob=0.0,
            terminal=False,
            next_value=2.0,
        )
        batch = rollout.batch()
        self.assertEqual(float(batch["dones"][0]), 0.0)
        self.assertEqual(float(batch["episode_ends"][0]), 1.0)
        self.assertAlmostEqual(float(batch["returns"][0]), 1.8, places=6)


if __name__ == "__main__":
    unittest.main()
