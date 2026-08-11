from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.config import Config
from carla_rl_lab.envs import ACTION_MODES, make_carla_env
from carla_rl_lab.observations import encode_observation
from carla_rl_lab.utils.provenance import carla_versions


def run_smoke(cfg: Config, steps: int) -> None:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    env = None
    try:
        env = make_carla_env(cfg)
        observation = env.reset(seed=cfg.seed)
        state = encode_observation(
            observation, cfg.state_dim, cfg.risk_field_sectors
        )
        action = (
            np.array([0.2, 0.0], dtype=np.float32)
            if cfg.action_mode == "longitudinal_2d"
            else np.array([0.2, 0.0, 0.0], dtype=np.float32)
        )
        total_reward = 0.0
        total_cost = 0.0
        completed_steps = 0
        termination_reason = None
        for _ in range(steps):
            observation, reward, cost, done, info = env.step(action)
            state = encode_observation(
                observation, cfg.state_dim, cfg.risk_field_sectors
            )
            total_reward += float(reward)
            total_cost += float(cost)
            completed_steps += 1
            termination_reason = info.get("termination_reason")
            if done:
                break
        print(json.dumps({
            "status": "ok",
            "carla_versions": carla_versions(env),
            "town": cfg.town,
            "seed": cfg.seed,
            "state_dim": int(state.shape[0]),
            "action_mode": cfg.action_mode,
            "steps": completed_steps,
            "return": total_reward,
            "cost": total_cost,
            "termination_reason": termination_reason,
        }, indent=2, sort_keys=True))
    finally:
        if env is not None:
            env.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real CARLA connection/reset/step smoke test"
    )
    parser.add_argument("--port", type=int, default=Config.port)
    parser.add_argument("--town", default=Config.town)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--vehicles", dest="number_of_vehicles", type=int, default=0)
    parser.add_argument("--walkers", dest="number_of_walkers", type=int, default=0)
    parser.add_argument("--action-mode", choices=ACTION_MODES, default=Config.action_mode)
    parser.add_argument("--seed", type=int, default=Config.seed)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config()
    for name, value in vars(args).items():
        if name != "steps":
            setattr(cfg, name, value)
    cfg.action_dim = 2 if cfg.action_mode == "longitudinal_2d" else 3
    cfg.visualize_waypoints = False
    cfg.view_mode = "none"
    cfg.max_time_episode = max(args.steps + 1, 5)
    run_smoke(cfg, args.steps)


if __name__ == "__main__":
    main()
