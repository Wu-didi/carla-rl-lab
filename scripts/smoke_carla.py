from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.config import Config
from carla_rl_lab.benchmarks import apply_benchmark, get_benchmark, list_benchmarks
from carla_rl_lab.envs import ACTION_MODES, make_carla_env
from carla_rl_lab.observations import encode_observation
from carla_rl_lab.utils.provenance import carla_versions


def run_smoke(cfg: Config, steps: int, frame_output: str = "") -> None:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    env = None
    try:
        env = make_carla_env(cfg)
        observation = env.reset(seed=cfg.seed)
        state = encode_observation(observation, cfg.state_dim)
        action = (
            np.array([0.2, 0.0], dtype=np.float32)
            if cfg.action_mode != "signed_3d"
            else np.array([0.2, 0.0, 0.0], dtype=np.float32)
        )
        total_reward = 0.0
        total_cost = 0.0
        completed_steps = 0
        termination_reason = None
        for _ in range(steps):
            observation, reward, cost, done, info = env.step(action)
            state = encode_observation(observation, cfg.state_dim)
            total_reward += float(reward)
            total_cost += float(cost)
            completed_steps += 1
            termination_reason = info.get("termination_reason")
            if done:
                break
        front_offset = 3 * cfg.num_cameras * (cfg.frame_stack - 1)
        current_rgb = observation["image"][front_offset : front_offset + 3]
        if frame_output:
            output_path = os.path.abspath(frame_output)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(current_rgb.transpose(1, 2, 0), mode="RGB").save(
                output_path
            )
        print(json.dumps({
            "status": "ok",
            "carla_versions": carla_versions(env),
            "town": cfg.town,
            "seed": cfg.seed,
            "state_dim": int(state.shape[0]),
            "camera_layout": cfg.camera_layout,
            "num_cameras": cfg.num_cameras,
            "action_mode": cfg.action_mode,
            "steps": completed_steps,
            "return": total_reward,
            "cost": total_cost,
            "termination_reason": termination_reason,
            "image_mean": float(current_rgb.mean()),
            "image_std": float(current_rgb.std()),
            "frame_output": os.path.abspath(frame_output) if frame_output else "",
        }, indent=2, sort_keys=True))
    finally:
        if env is not None:
            env.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real CARLA connection/reset/step smoke test"
    )
    parser.add_argument("--port", type=int, default=Config.port)
    parser.add_argument("--benchmark", choices=list(list_benchmarks()), default="")
    parser.add_argument("--town", default=Config.town)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--vehicles", dest="number_of_vehicles", type=int, default=0)
    parser.add_argument("--walkers", dest="number_of_walkers", type=int, default=0)
    parser.add_argument("--action-mode", choices=ACTION_MODES, default=Config.action_mode)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--frame-output", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config()
    for name, value in vars(args).items():
        if name not in ("steps", "benchmark", "frame_output"):
            setattr(cfg, name, value)
    if args.benchmark:
        apply_benchmark(cfg, get_benchmark(args.benchmark))
        cfg.port = args.port
    cfg.action_dim = 3 if cfg.action_mode == "signed_3d" else 2
    cfg.visualize_waypoints = False
    cfg.view_mode = "none"
    cfg.max_time_episode = max(args.steps + 1, 5)
    run_smoke(cfg, args.steps, args.frame_output)


if __name__ == "__main__":
    main()
