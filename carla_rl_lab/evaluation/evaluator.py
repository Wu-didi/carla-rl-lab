from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from carla_rl_lab.logging import ExperimentLogger
from carla_rl_lab.observations import encode_observation
from carla_rl_lab.utils import set_seed


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        raise ValueError("A benchmark requires at least one episode")

    returns = np.asarray([item["episode_return"] for item in results], dtype=np.float32)
    costs = np.asarray([item["episode_cost"] for item in results], dtype=np.float32)
    lengths = np.asarray([item["length"] for item in results], dtype=np.float32)
    speeds = np.asarray([item["mean_speed"] for item in results], dtype=np.float32)
    reasons = [item["termination_reason"] for item in results]
    count = float(len(results))
    off_road_reasons = {"off_road", "wrong_way", "lane_departure"}
    return {
        "benchmark/return_mean": float(returns.mean()),
        "benchmark/return_std": float(returns.std()),
        "benchmark/cost_mean": float(costs.mean()),
        "benchmark/length_mean": float(lengths.mean()),
        "benchmark/speed_mean": float(speeds.mean()),
        "benchmark/collision_rate": reasons.count("collision") / count,
        "benchmark/off_road_rate": sum(
            reason in off_road_reasons for reason in reasons
        )
        / count,
        "benchmark/success_rate": reasons.count("timeout") / count,
    }


def evaluate_benchmark(
    benchmark_name: str,
    env: Any,
    agent: Any,
    seeds: Iterable[int],
    expected_dim: int,
    risk_field_dim: int = 12,
    logger: Optional[ExperimentLogger] = None,
) -> Dict[str, Any]:
    results = []
    for episode_index, seed in enumerate(seeds):
        set_seed(seed)
        obs = env.reset()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        speeds = []
        length = 0
        info = {}

        while not done:
            obs_vector = encode_observation(obs, expected_dim, risk_field_dim)
            action = agent.act(obs_vector, deterministic=True)
            obs, reward, cost, done, info = env.step(action)
            total_reward += float(reward)
            total_cost += float(cost)
            speeds.append(float(obs["ego_state"][3]))
            length += 1

        result = {
            "seed": seed,
            "episode_return": total_reward,
            "episode_cost": total_cost,
            "length": length,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "termination_reason": str(info.get("termination_reason") or "unknown"),
        }
        results.append(result)
        if logger is not None:
            logger.log(
                {
                    "benchmark/episode_return": result["episode_return"],
                    "benchmark/episode_cost": result["episode_cost"],
                    "benchmark/episode_length": float(result["length"]),
                    "benchmark/mean_speed": result["mean_speed"],
                },
                episode_index,
            )

    summary = summarize_results(results)
    if logger is not None:
        logger.log(summary, len(results))
    return {"benchmark": benchmark_name, "episodes": results, "summary": summary}
