from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.algorithms import create_agent, list_algorithms
from carla_rl_lab.benchmarks import apply_benchmark, get_benchmark, list_benchmarks
from carla_rl_lab.config import Config
from carla_rl_lab.envs import make_carla_env
from carla_rl_lab.evaluation import evaluate_benchmark
from carla_rl_lab.logging import build_experiment_logger


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CarlaRLLab checkpoint")
    parser.add_argument(
        "--algorithm", "--algo", choices=list(list_algorithms()), default="sac"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--benchmark",
        choices=list(list_benchmarks()),
        default="lane_following_v0",
    )
    parser.add_argument(
        "--episodes", type=int, default=0, help="0 uses all benchmark seeds"
    )
    parser.add_argument("--port", type=int, default=Config.port)
    parser.add_argument(
        "--network", choices=["SAC", "Attention_SAC"], default=Config.network
    )
    parser.add_argument(
        "--logger",
        dest="logger_backend",
        choices=["tensorboard", "wandb", "both", "none"],
        default="tensorboard",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config()
    cfg.algorithm = args.algorithm
    cfg.port = args.port
    cfg.network = args.network
    cfg.logger_backend = args.logger_backend
    cfg.wandb_mode = args.wandb_mode

    benchmark = get_benchmark(args.benchmark)
    apply_benchmark(cfg, benchmark)
    benchmark_seeds = benchmark["seeds"]
    seeds = benchmark_seeds[:args.episodes] if args.episodes > 0 else benchmark_seeds

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
        "evaluations",
        benchmark["name"],
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = build_experiment_logger(cfg, output_dir, asdict(cfg))

    env = None
    try:
        env = make_carla_env(cfg)
        agent = create_agent(cfg.algorithm, cfg)
        agent.load(args.checkpoint)
        report = evaluate_benchmark(
            benchmark["name"],
            env,
            agent,
            seeds,
            expected_dim=cfg.state_dim,
            risk_field_dim=cfg.risk_field_sectors,
            logger=logger,
        )
        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, "w") as report_file:
            json.dump(report, report_file, indent=2)
        print(json.dumps(report["summary"], indent=2))
        print("Benchmark report -> {}".format(report_path))
    finally:
        if env is not None:
            env.close()
        logger.close()


if __name__ == "__main__":
    main()
