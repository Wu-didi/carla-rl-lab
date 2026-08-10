from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from carla_rl_lab.algorithms import create_agent, get_algorithm, list_algorithms
from carla_rl_lab.benchmarks import get_benchmark, list_benchmarks
from carla_rl_lab.buffers import ReplayBuffer
from carla_rl_lab.evaluation import evaluate_benchmark
from carla_rl_lab.logging import ExperimentLogger
from carla_rl_lab.observations import encode_observation
from carla_rl_lab.rewards import build_reward_profile


def tiny_config(state_dim=8, network="SAC"):
    return SimpleNamespace(
        state_dim=state_dim,
        hidden_dim=16,
        action_dim=3,
        action_bound=1.0,
        device="cpu",
        gamma=0.99,
        tau=0.01,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        target_entropy=-3.0,
        network=network,
        exploration_noise=0.0,
        td3_policy_noise=0.2,
        td3_noise_clip=0.5,
        td3_policy_delay=2,
    )


def random_batch(batch_size, state_dim):
    return {
        "states": np.random.randn(batch_size, state_dim).astype(np.float32),
        "actions": np.random.uniform(-1.0, 1.0, (batch_size, 3)).astype(np.float32),
        "rewards": np.random.randn(batch_size).astype(np.float32),
        "next_states": np.random.randn(batch_size, state_dim).astype(np.float32),
        "dones": np.zeros(batch_size, dtype=np.float32),
    }


def vector_observation():
    return {
        "ego_state": np.zeros(9, dtype=np.float32),
        "lane_info": np.zeros(2, dtype=np.float32),
        "risk_field": np.zeros(12, dtype=np.float32),
        "lidar": np.zeros(240, dtype=np.float32),
        "waypoints": np.zeros(36, dtype=np.float32),
    }


class FakeAgent:
    def act(self, obs, deterministic=False):
        return np.zeros(3, dtype=np.float32)


class FakeEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return vector_observation()

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        info = {"termination_reason": "timeout" if done else None}
        return vector_observation(), 1.0, 0.25, done, info


class CoreSmokeTest(unittest.TestCase):
    def test_registry_and_off_policy_algorithms(self):
        self.assertEqual(list(list_algorithms()), ["ddpg", "sac", "td3"])
        batch = random_batch(4, 8)

        for name in list_algorithms():
            with self.subTest(algorithm=name):
                spec = get_algorithm(name)
                self.assertEqual(spec.data_source, "online")
                self.assertEqual(spec.family, "off_policy")
                self.assertEqual(spec.runner, "off_policy")

                cfg = tiny_config()
                agent = create_agent(name, cfg)
                action = agent.act(np.zeros(8, dtype=np.float32), deterministic=True)
                self.assertEqual(action.shape, (3,))
                self.assertTrue(np.isfinite(action).all())
                logs = agent.update(batch)
                self.assertTrue(logs)
                self.assertTrue(all(np.isfinite(float(value)) for value in logs.values()))

                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    agent.save(checkpoint_dir, "smoke")
                    checkpoint_path = os.path.join(
                        checkpoint_dir, "{}_ckpt_smoke.pt".format(name)
                    )
                    self.assertTrue(os.path.isfile(checkpoint_path))
                    restored = create_agent(name, cfg)
                    restored.load(checkpoint_path)
                    restored_action = restored.act(
                        np.zeros(8, dtype=np.float32), deterministic=True
                    )
                    self.assertEqual(restored_action.shape, (3,))

    def test_attention_sac(self):
        cfg = tiny_config(state_dim=299, network="Attention_SAC")
        agent = create_agent("sac", cfg)
        action = agent.act(np.zeros(299, dtype=np.float32), deterministic=True)
        self.assertEqual(action.shape, (3,))
        logs = agent.update(random_batch(2, 299))
        self.assertEqual(tuple(logs["attention_img"].shape), (1, 1, 37))

    def test_observation_and_replay_buffer(self):
        encoded = encode_observation(vector_observation(), expected_dim=299)
        self.assertEqual(encoded.shape, (299,))

        buffer = ReplayBuffer(4)
        for index in range(4):
            buffer.add(encoded, np.zeros(3), float(index), encoded, False)
        batch = buffer.sample(2)
        self.assertEqual(batch["states"].shape, (2, 299))
        self.assertEqual(batch["actions"].shape, (2, 3))

    def test_reward_profile_decomposition(self):
        obs = vector_observation()
        obs["ego_state"][3] = 8.0
        reward_fn = build_reward_profile("research_v1", desired_speed=8.0)
        reward, terms = reward_fn(
            obs,
            False,
            {"is_collision": False, "is_off_road": False},
        )
        self.assertAlmostEqual(reward, 10.0)
        self.assertIn("reward/speed_tracking", terms)
        self.assertIn("reward/collision", terms)

    def test_tensorboard_logger(self):
        with tempfile.TemporaryDirectory() as log_dir:
            logger = ExperimentLogger("tensorboard", log_dir, {"algorithm": "sac"})
            logger.log({"train/loss": 1.0}, step=0)
            logger.close()
            self.assertTrue(any(name.startswith("events.out.tfevents") for name in os.listdir(log_dir)))

    def test_benchmark_evaluator(self):
        self.assertIn("lane_following_v0", list_benchmarks())
        spec = get_benchmark("lane_following_v0")
        report = evaluate_benchmark(
            spec["name"],
            FakeEnv(),
            FakeAgent(),
            seeds=(0, 1),
            expected_dim=299,
        )
        self.assertEqual(report["summary"]["benchmark/success_rate"], 1.0)
        self.assertEqual(report["summary"]["benchmark/return_mean"], 2.0)
        self.assertEqual(report["summary"]["benchmark/cost_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
