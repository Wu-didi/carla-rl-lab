from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.buffers import OfflineDataset
from carla_rl_lab.benchmarks import apply_benchmark, get_benchmark, list_benchmarks
from carla_rl_lab.config import Config
from carla_rl_lab.envs import ACTION_MODES, make_carla_env
from carla_rl_lab.observations import DEFAULT_FIELDS, encode_observation
from carla_rl_lab.rewards import list_reward_profiles
from carla_rl_lab.utils import set_seed
from carla_rl_lab.utils.provenance import (
    carla_versions,
    git_commit,
    jsonable_config,
    utc_timestamp,
)


def collect(cfg: Config, output_path: str, transition_count: int, policy: str) -> None:
    if transition_count <= 0:
        raise ValueError("--transitions must be positive")
    set_seed(cfg.seed)
    collection_started_at = utc_timestamp()
    source_commit = git_commit()
    arrays = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
        "terminals": [],
        "timeouts": [],
        "episode_ids": [],
        "costs": [],
    }
    env = None
    episode_id = 0
    try:
        env = make_carla_env(cfg)
        observation = env.reset(seed=cfg.seed + episode_id)
        if policy == "autopilot":
            env.set_ego_autopilot(True)

        while len(arrays["states"]) < transition_count:
            state = encode_observation(observation, cfg.state_dim)
            if policy == "autopilot":
                next_observation, reward, cost, done, info, action = env.step_sample()
            else:
                action = env.action_space.sample()
                next_observation, reward, cost, done, info = env.step(action)
            next_state = encode_observation(next_observation, cfg.state_dim)
            timeout = bool(done and info.get("termination_reason") == "timeout")
            terminal = bool(done and not timeout)
            arrays["states"].append(state)
            arrays["actions"].append(np.asarray(action, dtype=np.float32))
            arrays["rewards"].append(float(reward))
            arrays["next_states"].append(next_state)
            arrays["dones"].append(float(terminal))
            arrays["terminals"].append(float(terminal))
            arrays["timeouts"].append(float(timeout))
            arrays["episode_ids"].append(float(episode_id))
            arrays["costs"].append(float(cost))
            observation = next_observation

            collected = len(arrays["states"])
            if collected % 1000 == 0 or collected == transition_count:
                print("[Collect] transitions={}/{} episodes={}".format(
                    collected, transition_count, episode_id + 1
                ))
            if done and collected < transition_count:
                episode_id += 1
                observation = env.reset(seed=cfg.seed + episode_id)
                if policy == "autopilot":
                    env.set_ego_autopilot(True)

        metadata = {
            "schema_version": 2,
            "created_at": collection_started_at,
            "finished_at": utc_timestamp(),
            "git_commit": source_commit,
            "collector": policy,
            "carla_versions": carla_versions(env),
            "observation_fields": list(DEFAULT_FIELDS),
            "action_mode": cfg.action_mode,
            "dones_meaning": "true_terminal_only",
            "config": jsonable_config(cfg),
        }
        numeric_arrays = {}
        for name, values in arrays.items():
            dtype = np.uint8 if name in ("states", "next_states") else np.float32
            numeric_arrays[name] = np.asarray(values, dtype=dtype)
        dataset = OfflineDataset(numeric_arrays, metadata=metadata, seed=cfg.seed)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        dataset.save(output_path)
        print("Dataset -> {} ({} transitions)".format(
            os.path.abspath(output_path), len(dataset)
        ))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a versioned CARLA transition dataset"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--policy", choices=["autopilot", "random"], default="autopilot")
    parser.add_argument("--benchmark", choices=list(list_benchmarks()), default=None)
    parser.add_argument("--town", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--vehicles", dest="number_of_vehicles", type=int, default=None)
    parser.add_argument("--walkers", dest="number_of_walkers", type=int, default=None)
    parser.add_argument("--traffic", choices=["on", "off"], default=None)
    parser.add_argument("--view-mode", choices=["none", "top", "follow"], default=None)
    parser.add_argument("--action-mode", choices=ACTION_MODES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--reward",
        dest="reward_profile",
        choices=list(list_reward_profiles()),
        default=None,
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config()
    if args.benchmark:
        apply_benchmark(cfg, get_benchmark(args.benchmark))
    for name, value in vars(args).items():
        if (
            name not in ("output", "transitions", "policy", "benchmark")
            and value is not None
        ):
            setattr(cfg, name, value)
    cfg.action_dim = 3 if cfg.action_mode == "signed_3d" else 2
    print("[Config]", asdict(cfg))
    collect(cfg, args.output, args.transitions, args.policy)


if __name__ == "__main__":
    main()
