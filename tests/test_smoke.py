from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from carla_rl_lab.algorithms import create_agent, get_algorithm, list_algorithms
from carla_rl_lab.benchmarks import (
    apply_benchmark,
    get_benchmark,
    get_benchmark_suite,
    get_paper_benchmark,
    inspect_route_file,
    list_benchmarks,
    list_benchmark_suites,
    list_paper_benchmarks,
    prepare_paper_benchmark,
    probe_paper_benchmark,
)
from carla_rl_lab.buffers import OfflineDataset, ReplayBuffer, RolloutBuffer
from carla_rl_lab.config import Config
from carla_rl_lab.evaluation import evaluate_benchmark, summarize_suite
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
        policy_lr=1e-3,
        max_grad_norm=0.5,
        entropy_coef=0.01,
        value_coef=0.5,
        ppo_clip=0.2,
        ppo_epochs=2,
        ppo_minibatch_size=4,
        gae_lambda=0.95,
        td3_bc_alpha=2.5,
        cql_alpha=1.0,
        cql_temperature=1.0,
        cql_num_random=2,
        offline_entropy_alpha=0.2,
        iql_expectile=0.7,
        iql_beta=3.0,
        iql_max_weight=100.0,
        discriminator_lr=1e-3,
        discriminator_updates=1,
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
        self.seed_calls = []
        self.dt = 0.1

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        self.step_count = 0
        return vector_observation()

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        info = {"termination_reason": "timeout" if done else None}
        obs = vector_observation()
        obs["ego_state"][0] = float(self.step_count)
        obs["ego_state"][3] = 1.0
        obs["lane_info"][1] = 0.2
        return obs, 1.0, 0.25, done, info


class CoreSmokeTest(unittest.TestCase):
    def test_registry_and_off_policy_algorithms(self):
        off_policy_algorithms = [
            name
            for name in list_algorithms()
            if get_algorithm(name).runner == "off_policy"
        ]
        self.assertEqual(off_policy_algorithms, ["ddpg", "sac", "td3"])
        batch = random_batch(4, 8)

        for name in off_policy_algorithms:
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

    def test_on_policy_algorithms(self):
        cfg = tiny_config()
        for name in ("a2c", "ppo"):
            with self.subTest(algorithm=name):
                spec = get_algorithm(name)
                self.assertEqual(spec.runner, "on_policy")
                agent = create_agent(name, cfg)
                rollout = RolloutBuffer(8, gamma=0.99, gae_lambda=0.95)
                for _ in range(8):
                    state = np.random.randn(8).astype(np.float32)
                    next_state = np.random.randn(8).astype(np.float32)
                    action, log_prob, value = agent.act_with_info(state)
                    rollout.add(
                        state,
                        action,
                        1.0,
                        False,
                        value,
                        log_prob,
                        next_state,
                    )
                logs = agent.update(rollout.batch(last_value=0.0))
                self.assertTrue(all(np.isfinite(float(value)) for value in logs.values()))
                self._assert_checkpoint_roundtrip(name, agent, cfg)

    def test_offline_algorithms(self):
        cfg = tiny_config()
        batch = random_batch(8, 8)
        for name in ("cql", "iql", "td3_bc"):
            with self.subTest(algorithm=name):
                spec = get_algorithm(name)
                self.assertEqual(spec.runner, "offline")
                agent = create_agent(name, cfg)
                logs = agent.update(batch)
                self.assertTrue(all(np.isfinite(float(value)) for value in logs.values()))
                self._assert_checkpoint_roundtrip(name, agent, cfg)

    def test_imitation_algorithms(self):
        cfg = tiny_config()
        expert_batch = random_batch(8, 8)
        bc_agent = create_agent("bc", cfg)
        bc_logs = bc_agent.update(expert_batch)
        self.assertTrue(all(np.isfinite(float(value)) for value in bc_logs.values()))
        self._assert_checkpoint_roundtrip("bc", bc_agent, cfg)

        for name in ("gail", "airl"):
            with self.subTest(algorithm=name):
                agent = create_agent(name, cfg)
                rollout = RolloutBuffer(8, gamma=0.99, gae_lambda=0.95)
                for index in range(8):
                    state = np.random.randn(8).astype(np.float32)
                    next_state = np.random.randn(8).astype(np.float32)
                    action, log_prob, value = agent.act_with_info(state)
                    rollout.add(
                        state,
                        action,
                        0.0,
                        False,
                        value,
                        log_prob,
                        next_state,
                    )
                batch = rollout.batch(last_value=0.0)
                batch["expert_states"] = expert_batch["states"]
                batch["expert_actions"] = expert_batch["actions"]
                batch["expert_next_states"] = expert_batch["next_states"]
                batch["expert_dones"] = expert_batch["dones"]
                logs = agent.update(batch)
                self.assertTrue(all(np.isfinite(float(value)) for value in logs.values()))
                self._assert_checkpoint_roundtrip(name, agent, cfg)

    def test_offline_dataset_roundtrip(self):
        source = random_batch(6, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "expert.npz")
            dataset = OfflineDataset(source, seed=7)
            dataset.save(path)
            restored = OfflineDataset.load(path, seed=7)
            self.assertEqual(len(restored), 6)
            self.assertEqual(restored.state_dim, 8)
            self.assertEqual(restored.action_dim, 3)
            self.assertEqual(restored.sample(4)["states"].shape, (4, 8))

    def _assert_checkpoint_roundtrip(self, name, agent, cfg):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            agent.save(checkpoint_dir, "smoke")
            checkpoint_path = os.path.join(
                checkpoint_dir, "{}_ckpt_smoke.pt".format(name)
            )
            self.assertTrue(os.path.isfile(checkpoint_path))
            restored = create_agent(name, cfg)
            restored.load(checkpoint_path)
            action = restored.act(np.zeros(8, dtype=np.float32), deterministic=True)
            self.assertEqual(action.shape, (3,))

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

    def test_research_v2_penalizes_idle_without_rewarding_early_failure(self):
        obs = vector_observation()
        reward_fn = build_reward_profile("research_v2", desired_speed=8.0)

        idle_reward, idle_terms = reward_fn(
            obs,
            False,
            {"is_collision": False, "is_off_road": False},
        )
        self.assertAlmostEqual(idle_reward, -0.2)
        self.assertAlmostEqual(idle_terms["reward/idle"], -0.2)

        obs["ego_state"][3] = 8.0
        moving_reward, moving_terms = reward_fn(
            obs,
            False,
            {"is_collision": False, "is_off_road": False},
        )
        self.assertAlmostEqual(moving_reward, 10.0)
        self.assertAlmostEqual(moving_terms["reward/progress"], 2.0)

        collision_reward, _ = reward_fn(
            obs,
            True,
            {"is_collision": True, "is_off_road": False},
        )
        self.assertAlmostEqual(collision_reward, -190.0)

    def test_tensorboard_logger(self):
        with tempfile.TemporaryDirectory() as log_dir:
            logger = ExperimentLogger("tensorboard", log_dir, {"algorithm": "sac"})
            logger.log({"train/loss": 1.0}, step=0)
            logger.close()
            self.assertTrue(any(name.startswith("events.out.tfevents") for name in os.listdir(log_dir)))
            self.assertTrue(os.path.isfile(os.path.join(log_dir, "run_config.json")))

    def test_benchmark_evaluator(self):
        self.assertEqual(
            list_benchmarks(),
            (
                "adverse_weather_v0",
                "dense_traffic_v0",
                "lane_following_empty_v0",
                "lane_following_v0",
                "town02_generalization_v0",
                "urban_traffic_v0",
            ),
        )
        self.assertEqual(
            list_benchmark_suites(),
            ("carla_common_v0", "carla_lightweight_v0"),
        )
        self.assertEqual(len(get_benchmark_suite("carla_common_v0")), 5)
        self.assertEqual(
            get_benchmark_suite("carla_common_v0"),
            get_benchmark_suite("carla_lightweight_v0"),
        )
        spec = get_benchmark("lane_following_v0")
        cfg = apply_benchmark(Config(), spec)
        self.assertEqual(cfg.weather, "ClearNoon")
        self.assertEqual(cfg.number_of_vehicles, 50)
        env = FakeEnv()
        report = evaluate_benchmark(
            spec["name"],
            env,
            FakeAgent(),
            seeds=(0, 1),
            expected_dim=299,
        )
        self.assertEqual(report["summary"]["benchmark/success_rate"], 0.0)
        self.assertEqual(report["summary"]["benchmark/return_mean"], 2.0)
        self.assertEqual(report["summary"]["benchmark/cost_mean"], 0.5)
        self.assertEqual(report["summary"]["benchmark/distance_mean_m"], 2.0)
        self.assertAlmostEqual(
            report["summary"]["benchmark/lane_offset_mean_m"], 0.2
        )
        self.assertEqual(env.seed_calls, [0, 1])
        suite_summary = summarize_suite({"first": report, "second": report})
        self.assertEqual(suite_summary["suite/success_rate"], 0.0)

    def test_paper_benchmark_registry_and_preflight(self):
        self.assertIn("town05_long", list_paper_benchmarks())
        self.assertIn("longest6_v2", list_paper_benchmarks())
        self.assertIn("bench2drive220", list_paper_benchmarks())
        self.assertEqual(get_paper_benchmark("bench2drive").expected_routes, 220)

        legacy = prepare_paper_benchmark("nocrash", environment={})
        self.assertFalse(legacy.ready)
        self.assertIn("legacy", legacy.errors[0].lower())

        with tempfile.TemporaryDirectory() as directory:
            carla_root = os.path.join(directory, "CARLA_0.9.10")
            leaderboard_root = os.path.join(directory, "leaderboard_root")
            scenario_runner_root = os.path.join(directory, "scenario_runner")
            routes = os.path.join(directory, "routes.xml")
            scenarios = os.path.join(directory, "scenarios.json")
            agent = os.path.join(directory, "agent.py")
            os.makedirs(os.path.join(carla_root, "PythonAPI", "carla", "dist"))
            os.makedirs(os.path.join(leaderboard_root, "leaderboard"))
            os.makedirs(os.path.join(scenario_runner_root, "srunner"))
            for path in (
                os.path.join(carla_root, "CarlaUE4.sh"),
                os.path.join(
                    leaderboard_root, "leaderboard", "leaderboard_evaluator.py"
                ),
                scenarios,
                agent,
            ):
                with open(path, "w") as output_file:
                    output_file.write("\n")
            with open(routes, "w") as output_file:
                output_file.write(
                    '<routes><route id="0" town="Town05">'
                    '<waypoint x="0" y="0" z="0"/>'
                    '</route></routes>'
                )

            manifest = inspect_route_file(routes)
            self.assertEqual(manifest["route_count"], 1)
            self.assertEqual(manifest["towns"], ["Town05"])

            launch = prepare_paper_benchmark(
                "town05_long",
                agent=agent,
                carla_root=carla_root,
                leaderboard_root=leaderboard_root,
                scenario_runner_root=scenario_runner_root,
                routes=routes,
                scenarios=scenarios,
                environment={},
            )
            self.assertTrue(launch.ready)
            self.assertIn("--trafficManagerPort=8000", launch.command)
            self.assertIn("--scenarios={}".format(scenarios), launch.command)
            self.assertTrue(launch.warnings)
            self.assertTrue(probe_paper_benchmark(launch).ready)

            bench_launch = prepare_paper_benchmark(
                "bench2drive220",
                agent=agent,
                carla_root=carla_root,
                leaderboard_root=leaderboard_root,
                scenario_runner_root=scenario_runner_root,
                routes=routes,
                route_subset="0",
                environment={},
            )
            self.assertTrue(bench_launch.ready)
            self.assertIn("--traffic-manager-port=8000", bench_launch.command)
            self.assertIn("--routes-subset=0", bench_launch.command)


if __name__ == "__main__":
    unittest.main()
