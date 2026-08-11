from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from carla_rl_lab.algorithms import create_agent
from carla_rl_lab.buffers import OfflineDataset, ReplayBuffer
from carla_rl_lab.config import Config
from carla_rl_lab.utils import (
    apply_checkpoint_config,
    checkpoint_metadata,
    restore_training_state,
    save_training_checkpoint,
)


def checkpoint_config():
    cfg = SimpleNamespace(
        algorithm="sac",
        state_dim=4,
        hidden_dim=8,
        action_dim=3,
        action_bound=1.0,
        action_mode="signed_3d",
        device="cpu",
        gamma=0.99,
        tau=0.01,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        target_entropy=-3.0,
        network="SAC",
        max_waypoints=10,
        checkpoint_keep=2,
    )
    return cfg


class DatasetAndCheckpointTest(unittest.TestCase):
    def test_dataset_preserves_timeout_and_metadata(self):
        arrays = {
            "states": np.zeros((2, 4), dtype=np.float32),
            "actions": np.zeros((2, 2), dtype=np.float32),
            "rewards": np.ones(2, dtype=np.float32),
            "next_states": np.ones((2, 4), dtype=np.float32),
            "terminals": np.array([1.0, 0.0], dtype=np.float32),
            "timeouts": np.array([0.0, 1.0], dtype=np.float32),
            "episode_ids": np.array([0.0, 1.0], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dataset.npz")
            OfflineDataset(
                arrays,
                metadata={"collector": "autopilot", "action_mode": "longitudinal_2d"},
            ).save(path)
            dataset = OfflineDataset.load(path, seed=3)
            np.testing.assert_array_equal(dataset.arrays["dones"], [1.0, 0.0])
            np.testing.assert_array_equal(dataset.arrays["timeouts"], [0.0, 1.0])
            self.assertEqual(dataset.metadata["schema_version"], 2)
            self.assertEqual(dataset.metadata["collector"], "autopilot")

    def test_legacy_dataset_remains_loadable(self):
        arrays = {
            "states": np.zeros((2, 4), dtype=np.float32),
            "actions": np.zeros((2, 3), dtype=np.float32),
            "rewards": np.zeros(2, dtype=np.float32),
            "next_states": np.zeros((2, 4), dtype=np.float32),
            "dones": np.array([0.0, 1.0], dtype=np.float32),
        }
        dataset = OfflineDataset(arrays)
        np.testing.assert_array_equal(dataset.arrays["terminals"], arrays["dones"])
        np.testing.assert_array_equal(dataset.arrays["timeouts"], [0.0, 0.0])

    def test_pixel_dataset_preserves_uint8_states(self):
        arrays = {
            "states": np.zeros((2, 32), dtype=np.uint8),
            "actions": np.zeros((2, 2), dtype=np.float32),
            "rewards": np.zeros(2, dtype=np.float32),
            "next_states": np.ones((2, 32), dtype=np.uint8),
            "dones": np.zeros(2, dtype=np.float32),
        }
        dataset = OfflineDataset(arrays)
        self.assertEqual(dataset.arrays["states"].dtype, np.uint8)

    def test_checkpoint_contains_config_and_bounded_history(self):
        cfg = checkpoint_config()
        agent = create_agent("sac", cfg)
        with tempfile.TemporaryDirectory() as directory:
            for step in (10, 20, 30):
                save_training_checkpoint(
                    agent, cfg, directory, step, {"episode": step // 10}
                )
            last_path = os.path.join(directory, "sac_ckpt_last.pt")
            self.assertTrue(os.path.isfile(last_path))
            metadata = checkpoint_metadata(last_path)
            self.assertEqual(metadata["algorithm"], "sac")
            self.assertEqual(metadata["global_step"], 30)
            self.assertEqual(metadata["config"]["hidden_dim"], 8)
            restored_cfg = Config()
            apply_checkpoint_config(restored_cfg, last_path)
            self.assertEqual(restored_cfg.hidden_dim, 8)
            self.assertEqual(restore_training_state(last_path)["episode"], 3)
            with open(os.path.join(directory, "checkpoint_manifest.json")) as source:
                manifest = json.load(source)
            self.assertEqual(
                [item["global_step"] for item in manifest["checkpoints"]],
                [20, 30],
            )
            self.assertFalse(os.path.exists(os.path.join(directory, "sac_ckpt_10.pt")))

    def test_replay_buffer_can_be_restored(self):
        replay = ReplayBuffer(3)
        replay.add(np.zeros(2), np.zeros(1), 1.0, np.ones(2), False)
        restored = ReplayBuffer(3)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(len(restored), 1)
        self.assertEqual(float(restored.sample(1)["rewards"][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
