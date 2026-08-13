from __future__ import annotations

import os
import queue
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
from carla_rl_lab.envs.carla_env import CarlaEnv, _count_range
from carla_rl_lab.evaluation import evaluate_benchmark, summarize_suite
from carla_rl_lab.logging import ExperimentLogger
from carla_rl_lab.observations import encode_observation, pixel_state_dim
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
        image_size=32,
        frame_stack=1,
        max_waypoints=4,
    )


def random_batch(batch_size, state_dim):
    return {
        "states": np.random.randn(batch_size, state_dim).astype(np.float32),
        "actions": np.random.uniform(-1.0, 1.0, (batch_size, 3)).astype(np.float32),
        "rewards": np.random.randn(batch_size).astype(np.float32),
        "next_states": np.random.randn(batch_size, state_dim).astype(np.float32),
        "dones": np.zeros(batch_size, dtype=np.float32),
    }


def pixel_observation(image_size=8, frame_stack=1, max_waypoints=2):
    return {
        "image": np.zeros(
            (3 * frame_stack, image_size, image_size), dtype=np.uint8
        ),
        "waypoints": np.zeros(2 * max_waypoints, dtype=np.float32),
        "vehicle_measurements": np.zeros(2, dtype=np.float32),
        "ego_state": np.zeros(7, dtype=np.float32),
        "lane_info": np.zeros(2, dtype=np.float32),
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
        return pixel_observation()

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        info = {
            "termination_reason": "timeout" if done else None,
            "requested_vehicles": 12,
            "requested_walkers": 34,
            "spawned_vehicles": 11,
            "spawned_walkers": 31,
        }
        obs = pixel_observation()
        obs["ego_state"][0] = float(self.step_count)
        obs["ego_state"][3] = 1.0
        obs["lane_info"][1] = 0.2
        return obs, 1.0, 0.25, done, info


class FixedCollisionEnv(FakeEnv):
    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        info = {
            "termination_reason": "route_completed" if done else None,
            "route_completion": float(done),
            "collision_count": 1,
            "collision_counts": {"layout": 0, "vehicle": 1, "pedestrian": 0},
        }
        obs = pixel_observation()
        obs["ego_state"][0] = float(self.step_count)
        obs["ego_state"][3] = 1.0
        return obs, 1.0, 0.0, done, info


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

    def test_pixel_ppo(self):
        state_dim = pixel_state_dim(32, 1, 4)
        cfg = tiny_config(state_dim=state_dim, network="Pixel_SAC")
        cfg.action_dim = 3
        agent = create_agent("ppo", cfg)
        rollout = RolloutBuffer(4, gamma=0.99, gae_lambda=0.95)
        for _ in range(4):
            state = np.random.randint(0, 256, state_dim, dtype=np.uint8)
            action, log_prob, value = agent.act_with_info(state)
            rollout.add(state, action, 1.0, False, value, log_prob, state)
        batch = rollout.batch(last_value=0.0)
        self.assertEqual(batch["states"].dtype, np.uint8)
        logs = agent.update(batch)
        self.assertTrue(all(np.isfinite(float(value)) for value in logs.values()))
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            agent.save(checkpoint_dir, "smoke")
            restored = create_agent("ppo", cfg)
            restored.load(os.path.join(checkpoint_dir, "ppo_ckpt_smoke.pt"))
            self.assertEqual(
                restored.act(np.zeros(state_dim, dtype=np.uint8)).shape,
                (cfg.action_dim,),
            )

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

    def test_pixel_behavior_cloning(self):
        state_dim = pixel_state_dim(32, 1, 4)
        cfg = tiny_config(state_dim=state_dim, network="Pixel_SAC")
        agent = create_agent("bc", cfg)
        batch = random_batch(2, state_dim)
        batch["states"] = np.random.randint(
            0, 256, size=(2, state_dim), dtype=np.uint8
        )
        logs = agent.update(batch)
        self.assertTrue(np.isfinite(logs["bc_loss"]))
        self.assertEqual(
            agent.act(np.zeros(state_dim, dtype=np.uint8)).shape, (3,)
        )

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

    def test_pixel_sac(self):
        state_dim = pixel_state_dim(32, 1, 4)
        cfg = tiny_config(state_dim=state_dim, network="Pixel_SAC")
        agent = create_agent("sac", cfg)
        packed = np.zeros(state_dim, dtype=np.uint8)
        action = agent.act(packed, deterministic=True)
        self.assertEqual(action.shape, (3,))
        batch = random_batch(2, state_dim)
        batch["states"] = np.random.randint(
            0, 256, size=(2, state_dim), dtype=np.uint8
        )
        batch["next_states"] = np.random.randint(
            0, 256, size=(2, state_dim), dtype=np.uint8
        )
        logs = agent.update(batch)
        self.assertTrue(np.isfinite(logs["critic_1_loss"]))
        demo_update_logs = agent.update(
            batch, expert_batch=batch, bc_coef=0.25
        )
        self.assertTrue(np.isfinite(demo_update_logs["demo_bc_loss"]))
        adaptive_logs = agent.update(
            batch, expert_batch=batch, bc_coef=0.25, bc_mode="adaptive"
        )
        self.assertTrue(np.isfinite(adaptive_logs["demo_expert_advantage"]))
        self.assertTrue(np.isfinite(adaptive_logs["demo_critic_disagreement"]))
        self.assertGreaterEqual(adaptive_logs["demo_bc_weight_min"], 0.1)
        self.assertLessEqual(adaptive_logs["demo_bc_weight_max"], 2.0)
        demo_logs = agent.behavior_clone(batch, coefficient=0.5)
        self.assertTrue(np.isfinite(demo_logs["bc_loss"]))
        self.assertTrue(np.isfinite(demo_logs["bc_action_mae"]))

    def test_pixel_td3(self):
        state_dim = pixel_state_dim(32, 1, 4)
        cfg = tiny_config(state_dim=state_dim, network="Pixel_SAC")
        agent = create_agent("td3", cfg)
        packed = np.zeros(state_dim, dtype=np.uint8)
        action = agent.act(packed, deterministic=True)
        self.assertEqual(action.shape, (3,))
        batch = random_batch(2, state_dim)
        batch["states"] = np.random.randint(
            0, 256, size=(2, state_dim), dtype=np.uint8
        )
        batch["next_states"] = np.random.randint(
            0, 256, size=(2, state_dim), dtype=np.uint8
        )
        first_logs = agent.update(batch)
        second_logs = agent.update(batch)
        self.assertTrue(np.isfinite(first_logs["critic_1_loss"]))
        self.assertTrue(np.isfinite(second_logs["actor_loss"]))

    def test_default_front_camera_contract(self):
        cfg = Config()
        self.assertEqual(cfg.camera_layout, "front")
        self.assertEqual(cfg.num_cameras, 1)
        self.assertEqual(cfg.camera_sensor_width, 640)
        self.assertEqual(cfg.camera_sensor_height, 384)
        self.assertEqual(cfg.camera_fov, 120.0)
        self.assertEqual(cfg.camera_location_x, 1.5)
        self.assertEqual(cfg.camera_location_z, 2.5)
        self.assertEqual(
            cfg.state_dim,
            pixel_state_dim(
                cfg.image_size,
                cfg.frame_stack,
                cfg.max_waypoints,
            ),
        )

    def test_bench2drive_rl_camera_frame_is_resized_for_pixel_policy(self):
        env = CarlaEnv.__new__(CarlaEnv)
        env.image_size = 84
        env._camera_queue = queue.Queue(maxsize=8)
        bgra = np.zeros((384, 640, 4), dtype=np.uint8)
        bgra[:, :, 2] = 255
        image = SimpleNamespace(
            raw_data=bgra.tobytes(), height=384, width=640, frame=7
        )

        env._on_camera(image)
        frame, rgb = env._camera_queue.get_nowait()

        self.assertEqual(frame, 7)
        self.assertEqual(rgb.shape, (3, 84, 84))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.all(rgb[0] == 255))

    def test_observation_and_replay_buffer(self):
        expected_dim = pixel_state_dim(8, 1, 2)
        encoded = encode_observation(
            pixel_observation(), expected_dim=expected_dim
        )
        self.assertEqual(encoded.shape, (expected_dim,))
        self.assertEqual(encoded.dtype, np.uint8)

        buffer = ReplayBuffer(4)
        for index in range(4):
            buffer.add(encoded, np.zeros(3), float(index), encoded, False)
        batch = buffer.sample(2)
        self.assertEqual(batch["states"].shape, (2, expected_dim))
        self.assertEqual(batch["actions"].shape, (2, 3))

    def test_reward_profile_decomposition(self):
        obs = pixel_observation()
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
        obs = pixel_observation()
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
            logger.finish("completed", global_step=1)
            logger.close()
            self.assertTrue(any(name.startswith("events.out.tfevents") for name in os.listdir(log_dir)))
            self.assertTrue(os.path.isfile(os.path.join(log_dir, "run_config.json")))

    def test_benchmark_evaluator(self):
        self.assertEqual(
            set(list_benchmarks()),
            {
                "adverse_weather_v0",
                "dense_traffic_v0",
                "lane_following_empty_v0",
                "lane_following_v0",
                "nocrash_dense_v0",
                "nocrash_empty_v0",
                "nocrash_regular_v0",
                "nocrash_train_empty_v0",
                "nocrash_train_regular_v0",
                "nocrash_train_v0",
                "town02_generalization_v0",
                "urban_traffic_v0",
            },
        )
        self.assertEqual(
            list_benchmark_suites(),
            (
                "carla_common_v0",
                "carla_lightweight_v0",
                "nocrash_0915_v0",
                "rlfold_nocrash_0915_v0",
            ),
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
            expected_dim=pixel_state_dim(8, 1, 2),
        )
        self.assertEqual(report["summary"]["benchmark/success_rate"], 0.0)
        self.assertEqual(report["summary"]["benchmark/return_mean"], 2.0)
        self.assertEqual(report["summary"]["benchmark/cost_mean"], 0.5)
        self.assertEqual(report["summary"]["benchmark/distance_mean_m"], 2.0)
        self.assertAlmostEqual(
            report["summary"]["benchmark/lane_offset_mean_m"], 0.2
        )
        self.assertEqual(
            report["summary"]["benchmark/requested_vehicles_mean"], 12.0
        )
        self.assertEqual(
            report["summary"]["benchmark/spawned_walkers_mean"], 31.0
        )
        self.assertEqual(env.seed_calls, [0, 1])
        suite_summary = summarize_suite({"first": report, "second": report})
        self.assertEqual(suite_summary["suite/success_rate"], 0.0)

        resumed_env = FakeEnv()
        progress_lengths = []
        resumed = evaluate_benchmark(
            spec["name"],
            resumed_env,
            FakeAgent(),
            seeds=(0, 1),
            expected_dim=pixel_state_dim(8, 1, 2),
            initial_results=report["episodes"][:1],
            progress_callback=lambda results: progress_lengths.append(len(results)),
        )
        self.assertEqual(resumed_env.seed_calls, [1])
        self.assertEqual(progress_lengths, [2])
        self.assertEqual(resumed["episodes"], report["episodes"])

    def test_rlfold_protocol_uses_front_rgb_and_nocrash_success(self):
        train_spec = get_benchmark("nocrash_train_v0")
        train_cfg = apply_benchmark(Config(), train_spec)
        self.assertEqual(train_spec["protocol_id"], "rlfold_nocrash_0915_v0")
        self.assertEqual(train_cfg.town, "Town01")
        self.assertEqual(train_cfg.route_mode, "endless")
        self.assertEqual(train_cfg.camera_layout, "front")
        self.assertEqual(train_cfg.num_cameras, 1)
        self.assertEqual(train_cfg.camera_sensor_width, 256)
        self.assertEqual(train_cfg.image_size, 84)
        self.assertEqual(train_cfg.frame_stack, 2)
        self.assertEqual(train_cfg.min_number_of_vehicles, 0)
        self.assertEqual(train_cfg.max_number_of_vehicles, 150)
        self.assertEqual(train_cfg.min_number_of_walkers, 0)
        self.assertEqual(train_cfg.max_number_of_walkers, 300)

        env = FixedCollisionEnv()
        report = evaluate_benchmark(
            "nocrash_empty_v0",
            env,
            FakeAgent(),
            seeds=(0,),
            expected_dim=pixel_state_dim(8, 1, 2),
            route_limit=2,
            weather_limit=1,
        )
        self.assertEqual(env.seed_calls, [0, 1])
        self.assertEqual(report["summary"]["benchmark/collision_rate"], 1.0)
        self.assertEqual(report["summary"]["benchmark/success_rate"], 0.0)

    def test_traffic_count_range_validation(self):
        self.assertEqual(_count_range(20, -1, -1), (20, 20))
        self.assertEqual(_count_range(0, 0, 150), (0, 150))
        with self.assertRaisesRegex(ValueError, "minimum"):
            _count_range(0, 10, 5)

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
