from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from carla_rl_lab.logging import ExperimentLogger, NullLogger
from carla_rl_lab.utils import set_seed


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    episode_return: float
    episode_cost: float
    length: int
    mean_speed: float
    termination_reason: str


@dataclass(frozen=True)
class BenchmarkReport:
    benchmark: str
    episodes: List[EpisodeResult]
    summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "episodes": [asdict(episode) for episode in self.episodes],
            "summary": dict(self.summary),
        }


class BenchmarkEvaluator:
    def __init__(self, logger: Optional[ExperimentLogger] = None):
        self.logger = logger or NullLogger()

    def evaluate(
        self,
        benchmark_name: str,
        env: Any,
        agent: Any,
        observation_adapter: Any,
        seeds: Iterable[int],
    ) -> BenchmarkReport:
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
                obs_vector = observation_adapter.encode(obs)
                action = agent.act(obs_vector, deterministic=True)
                obs, reward, cost, done, info = env.step(action)
                total_reward += float(reward)
                total_cost += float(cost)
                speeds.append(float(obs["ego_state"][3]))
                length += 1

            result = EpisodeResult(
                seed=seed,
                episode_return=total_reward,
                episode_cost=total_cost,
                length=length,
                mean_speed=float(np.mean(speeds)) if speeds else 0.0,
                termination_reason=str(info.get("termination_reason") or "unknown"),
            )
            results.append(result)
            self.logger.log(
                {
                    "benchmark/episode_return": result.episode_return,
                    "benchmark/episode_cost": result.episode_cost,
                    "benchmark/episode_length": float(result.length),
                    "benchmark/mean_speed": result.mean_speed,
                },
                episode_index,
            )

        summary = self._summarize(results)
        self.logger.log(summary, len(results))
        return BenchmarkReport(benchmark_name, results, summary)

    @staticmethod
    def _summarize(results: List[EpisodeResult]) -> Dict[str, float]:
        if not results:
            raise ValueError("A benchmark requires at least one episode")
        returns = np.asarray([item.episode_return for item in results], dtype=np.float32)
        costs = np.asarray([item.episode_cost for item in results], dtype=np.float32)
        lengths = np.asarray([item.length for item in results], dtype=np.float32)
        speeds = np.asarray([item.mean_speed for item in results], dtype=np.float32)
        reasons = [item.termination_reason for item in results]
        count = float(len(results))
        off_road_reasons = {"off_road", "wrong_way", "lane_departure"}
        return {
            "benchmark/return_mean": float(returns.mean()),
            "benchmark/return_std": float(returns.std()),
            "benchmark/cost_mean": float(costs.mean()),
            "benchmark/length_mean": float(lengths.mean()),
            "benchmark/speed_mean": float(speeds.mean()),
            "benchmark/collision_rate": reasons.count("collision") / count,
            "benchmark/off_road_rate": sum(reason in off_road_reasons for reason in reasons) / count,
            "benchmark/success_rate": reasons.count("timeout") / count,
        }
