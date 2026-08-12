from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from carla_rl_lab.config import Config
from scripts import train as off_policy_runner
from scripts import train_on_policy as on_policy_runner


def observation():
    return {
        "image": np.zeros((9, 84, 84), dtype=np.uint8),
        "waypoints": np.zeros(20, dtype=np.float32),
        "vehicle_measurements": np.zeros(2, dtype=np.float32),
        "ego_state": np.zeros(7, dtype=np.float32),
        "lane_info": np.zeros(2, dtype=np.float32),
    }


class FakeEnv:
    def __init__(self):
        self.steps = 0
        self.closed = False

    def reset(self, seed=None):
        return observation()

    def step(self, action):
        self.steps += 1
        return observation(), 1.0, 0.0, False, {"reward_terms": {}}

    def close(self):
        self.closed = True


class FakeAgent:
    def act(self, state):
        return np.zeros(2, dtype=np.float32)


class FailingEnv(FakeEnv):
    def step(self, action):
        self.steps += 1
        raise RuntimeError("simulator failure")


class FakeLogger:
    def __init__(self):
        self.run_record = {}

    def log(self, metrics, step):
        pass

    def log_image(self, name, image, step):
        pass

    def update_run_record(self, values):
        self.run_record.update(values)

    def finish(self, status, **details):
        self.run_record.update(details)
        self.run_record["status"] = status

    def close(self):
        pass


class TrainingRunnerTest(unittest.TestCase):
    def test_off_policy_warmup_samples_full_action_space(self):
        cfg = Config()
        cfg.action_dim = 2
        cfg.minimal_size = 10
        agent = FakeAgent()
        action, random_warmup = off_policy_runner.select_action(
            agent, cfg, np.zeros(cfg.state_dim, dtype=np.uint8), replay_size=0
        )
        self.assertTrue(random_warmup)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

        action, random_warmup = off_policy_runner.select_action(
            agent, cfg, np.zeros(cfg.state_dim, dtype=np.uint8), replay_size=10
        )
        self.assertFalse(random_warmup)
        self.assertTrue(np.array_equal(action, np.zeros(2, dtype=np.float32)))

    def test_off_policy_cli_overrides_research_budget(self):
        args = off_policy_runner.build_argparser().parse_args(
            [
                "--algo",
                "sac",
                "--total-timesteps",
                "123",
                "--checkpoint-interval",
                "25",
                "--minimal-size",
                "8",
                "--batch-size",
                "8",
                "--hidden-dim",
                "32",
                "--vehicles",
                "0",
                "--view-mode",
                "none",
                "--action-mode",
                "longitudinal_2d",
            ]
        )
        cfg = off_policy_runner.apply_overrides(Config(), args)
        self.assertEqual(cfg.total_timesteps, 123)
        self.assertEqual(cfg.checkpoint_interval, 25)
        self.assertEqual(cfg.number_of_vehicles, 0)
        self.assertEqual(cfg.hidden_dim, 32)
        self.assertEqual(cfg.action_dim, 2)
        self.assertEqual(cfg.target_entropy, -2.0)

    def test_online_demonstration_cli(self):
        args = off_policy_runner.build_argparser().parse_args(
            [
                "--algo",
                "sac",
                "--expert-dataset",
                "expert.npz",
                "--demo-pretrain-updates",
                "200",
                "--demo-bc-coef",
                "0.25",
            ]
        )
        cfg = off_policy_runner.apply_overrides(Config(), args)
        self.assertEqual(cfg.expert_dataset_path, "expert.npz")
        self.assertEqual(cfg.demo_pretrain_updates, 200)
        self.assertEqual(cfg.demo_bc_coef, 0.25)

    def test_on_policy_cli_overrides_same_config_instance(self):
        args = on_policy_runner.build_argparser().parse_args(
            [
                "--algo",
                "a2c",
                "--total-timesteps",
                "64",
                "--rollout-steps",
                "16",
                "--vehicles",
                "0",
                "--action-mode",
                "longitudinal_2d",
            ]
        )
        cfg = Config()
        configured = on_policy_runner.apply_overrides(cfg, args)
        self.assertIs(configured, cfg)
        self.assertEqual(cfg.algorithm, "a2c")
        self.assertEqual(cfg.total_timesteps, 64)
        self.assertEqual(cfg.rollout_steps, 16)
        self.assertEqual(cfg.action_dim, 2)

    def test_on_policy_target_speed_benchmark_uses_two_actions(self):
        args = on_policy_runner.build_argparser().parse_args(
            ["--algo", "ppo", "--benchmark", "nocrash_train_empty_v0"]
        )
        cfg = on_policy_runner.apply_overrides(Config(), args)
        self.assertEqual(cfg.town, "Town01")
        self.assertEqual(cfg.action_mode, "target_speed_2d")
        self.assertEqual(cfg.action_dim, 2)

    def test_off_policy_stops_and_checkpoints_at_step_budget(self):
        env = FakeEnv()
        cfg = Config()
        cfg.algorithm = "sac"
        cfg.action_mode = "longitudinal_2d"
        cfg.action_dim = 2
        cfg.total_timesteps = 3
        cfg.max_episodes = 10
        cfg.minimal_size = 10
        cfg.batch_size = 2
        cfg.checkpoint_interval = 2
        cfg.logger_backend = "none"

        with tempfile.TemporaryDirectory() as output_dir:
            cfg.run_name = output_dir
            with patch.object(off_policy_runner, "make_carla_env", return_value=env), patch.object(
                off_policy_runner, "make_agent", return_value=FakeAgent()
            ), patch.object(
                off_policy_runner,
                "build_experiment_logger",
                return_value=FakeLogger(),
            ), patch.object(
                off_policy_runner, "save_training_checkpoint"
            ) as save:
                off_policy_runner.train(cfg)

        self.assertEqual(env.steps, 3)
        self.assertTrue(env.closed)
        self.assertEqual([call[0][3] for call in save.call_args_list], [2, 3])

    def test_off_policy_stops_after_repeated_simulator_failures(self):
        env = FailingEnv()
        cfg = Config()
        cfg.total_timesteps = 3
        cfg.max_step_retries = 2
        cfg.logger_backend = "none"

        with tempfile.TemporaryDirectory() as output_dir:
            cfg.run_name = output_dir
            with patch.object(off_policy_runner, "make_carla_env", return_value=env), patch.object(
                off_policy_runner, "make_agent", return_value=FakeAgent()
            ), patch.object(
                off_policy_runner,
                "build_experiment_logger",
                return_value=FakeLogger(),
            ), patch.object(
                off_policy_runner.traceback, "print_exc"
            ):
                with self.assertRaisesRegex(RuntimeError, "2 consecutive times"):
                    off_policy_runner.train(cfg)

        self.assertEqual(env.steps, 2)
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
