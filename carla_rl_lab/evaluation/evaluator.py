from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from carla_rl_lab.benchmarks import get_benchmark
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
    distances = np.asarray(
        [item.get("distance_m", 0.0) for item in results], dtype=np.float32
    )
    durations = np.asarray(
        [item.get("duration_s", 0.0) for item in results], dtype=np.float32
    )
    horizon_fractions = np.asarray(
        [item.get("horizon_fraction", 0.0) for item in results],
        dtype=np.float32,
    )
    lane_offsets = np.asarray(
        [item.get("mean_abs_lane_offset", 0.0) for item in results],
        dtype=np.float32,
    )
    overspeed_rates = np.asarray(
        [item.get("overspeed_rate", 0.0) for item in results], dtype=np.float32
    )
    stationary_rates = np.asarray(
        [item.get("stationary_rate", 0.0) for item in results], dtype=np.float32
    )
    reasons = [item["termination_reason"] for item in results]
    count = float(len(results))
    off_road_reasons = {"off_road", "wrong_way", "lane_departure"}
    collision_count = reasons.count("collision")
    off_road_count = sum(reason in off_road_reasons for reason in reasons)
    total_distance_km = max(float(distances.sum()) / 1000.0, 1e-3)
    return {
        "benchmark/return_mean": float(returns.mean()),
        "benchmark/return_std": float(returns.std()),
        "benchmark/cost_mean": float(costs.mean()),
        "benchmark/cost_std": float(costs.std()),
        "benchmark/length_mean": float(lengths.mean()),
        "benchmark/length_std": float(lengths.std()),
        "benchmark/speed_mean": float(speeds.mean()),
        "benchmark/distance_mean_m": float(distances.mean()),
        "benchmark/distance_std_m": float(distances.std()),
        "benchmark/duration_mean_s": float(durations.mean()),
        "benchmark/horizon_fraction_mean": float(horizon_fractions.mean()),
        "benchmark/horizon_fraction_std": float(horizon_fractions.std()),
        "benchmark/lane_offset_mean_m": float(lane_offsets.mean()),
        "benchmark/overspeed_rate": float(overspeed_rates.mean()),
        "benchmark/stationary_rate": float(stationary_rates.mean()),
        "benchmark/collision_rate": collision_count / count,
        "benchmark/off_road_rate": off_road_count / count,
        "benchmark/collisions_per_km": collision_count / total_distance_km,
        "benchmark/off_road_events_per_km": off_road_count / total_distance_km,
        "benchmark/success_rate": sum(
            bool(item.get("success", item["termination_reason"] == "timeout"))
            for item in results
        )
        / count,
    }


def summarize_suite(reports: Mapping[str, Dict[str, Any]]) -> Dict[str, float]:
    if not reports:
        raise ValueError("A benchmark suite requires at least one report")
    summaries = [report["summary"] for report in reports.values()]
    shared_keys = set(summaries[0])
    for summary in summaries[1:]:
        shared_keys.intersection_update(summary)
    return {
        "suite/{}".format(key.split("/", 1)[-1]): float(
            np.mean([summary[key] for summary in summaries])
        )
        for key in sorted(shared_keys)
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
    benchmark = get_benchmark(benchmark_name)
    env_overrides = benchmark["env_overrides"]
    horizon = int(env_overrides["max_time_episode"])
    dt = float(env_overrides.get("dt", getattr(env, "dt", 0.1)))
    desired_speed = float(env_overrides["desired_speed"])
    success_reasons = set(benchmark.get("success_reasons", ("timeout",)))
    success_criteria = benchmark.get("success_criteria", {})
    results = []
    for episode_index, seed in enumerate(seeds):
        set_seed(seed)
        if hasattr(env, "seed"):
            env.seed(seed)
        obs = env.reset()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        speeds = []
        lane_offsets = []
        overspeed_steps = 0
        stationary_steps = 0
        distance_m = 0.0
        last_position = np.asarray(obs["ego_state"][:2], dtype=np.float32)
        length = 0
        info = {}

        while not done:
            obs_vector = encode_observation(obs, expected_dim, risk_field_dim)
            action = agent.act(obs_vector, deterministic=True)
            obs, reward, cost, done, info = env.step(action)
            total_reward += float(reward)
            total_cost += float(cost)
            speed = float(obs["ego_state"][3])
            speeds.append(speed)
            lane_offsets.append(abs(float(obs["lane_info"][1])))
            overspeed_steps += int(speed > desired_speed)
            stationary_steps += int(speed < 0.1)
            position = np.asarray(obs["ego_state"][:2], dtype=np.float32)
            distance_m += float(np.linalg.norm(position - last_position))
            last_position = position
            length += 1

        termination_reason = str(info.get("termination_reason") or "unknown")
        horizon_fraction = min(float(length) / max(horizon, 1), 1.0)
        stationary_rate = float(stationary_steps) / max(length, 1)
        success = (
            termination_reason in success_reasons
            and horizon_fraction
            >= float(success_criteria.get("min_horizon_fraction", 0.0))
            and distance_m >= float(success_criteria.get("min_distance_m", 0.0))
            and stationary_rate
            <= float(success_criteria.get("max_stationary_rate", 1.0))
        )
        result = {
            "seed": seed,
            "episode_return": total_reward,
            "episode_cost": total_cost,
            "length": length,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "distance_m": distance_m,
            "duration_s": length * dt,
            "horizon_fraction": horizon_fraction,
            "mean_abs_lane_offset": float(np.mean(lane_offsets))
            if lane_offsets
            else 0.0,
            "overspeed_rate": float(overspeed_steps) / max(length, 1),
            "stationary_rate": stationary_rate,
            "termination_reason": termination_reason,
            "success": success,
        }
        results.append(result)
        if logger is not None:
            logger.log(
                {
                    "benchmark/episode_return": result["episode_return"],
                    "benchmark/episode_cost": result["episode_cost"],
                    "benchmark/episode_length": float(result["length"]),
                    "benchmark/mean_speed": result["mean_speed"],
                    "benchmark/distance_m": result["distance_m"],
                    "benchmark/horizon_fraction": result[
                        "horizon_fraction"
                    ],
                    "benchmark/lane_offset_mean_m": result[
                        "mean_abs_lane_offset"
                    ],
                    "benchmark/overspeed_rate": result["overspeed_rate"],
                    "benchmark/stationary_rate": result["stationary_rate"],
                    "benchmark/success": float(result["success"]),
                },
                episode_index,
            )

    summary = summarize_results(results)
    if logger is not None:
        logger.log(summary, len(results))
    return {
        "benchmark": benchmark_name,
        "protocol": benchmark,
        "episodes": results,
        "summary": summary,
    }
