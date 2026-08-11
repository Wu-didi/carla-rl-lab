from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.algorithms import create_agent, list_algorithms
from carla_rl_lab.benchmarks import (
    apply_benchmark,
    get_benchmark,
    get_benchmark_suite,
    list_benchmarks,
    list_benchmark_suites,
)
from carla_rl_lab.config import Config
from carla_rl_lab.envs import make_carla_env
from carla_rl_lab.evaluation import evaluate_benchmark, summarize_suite
from carla_rl_lab.logging import build_experiment_logger


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CarlaRLLab checkpoint")
    parser.add_argument(
        "--algorithm", "--algo", choices=list(list_algorithms()), default="sac"
    )
    parser.add_argument("--checkpoint", required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--benchmark",
        choices=list(list_benchmarks()),
        default="",
    )
    target.add_argument(
        "--suite",
        choices=list(list_benchmark_suites()),
        default="",
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


def checkpoint_metadata(path: str):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": os.path.abspath(path),
        "sha256": digest.hexdigest(),
    }


def evaluate_one(args: argparse.Namespace, benchmark_name: str):
    cfg = Config()
    cfg.algorithm = args.algorithm
    cfg.port = args.port
    cfg.network = args.network
    cfg.logger_backend = args.logger_backend
    cfg.wandb_mode = args.wandb_mode

    benchmark = get_benchmark(benchmark_name)
    apply_benchmark(cfg, benchmark)
    benchmark_seeds = benchmark["seeds"]
    seeds = benchmark_seeds[:args.episodes] if args.episodes > 0 else benchmark_seeds

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
        "evaluations",
        benchmark_name,
        cfg.algorithm,
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
        report["algorithm"] = cfg.algorithm
        report["checkpoint"] = checkpoint_metadata(args.checkpoint)
        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, "w") as report_file:
            json.dump(report, report_file, indent=2)
        print(json.dumps(report["summary"], indent=2))
        print("Benchmark report -> {}".format(report_path))
        return report
    finally:
        if env is not None:
            env.close()
        logger.close()


def main() -> None:
    args = build_argparser().parse_args()
    benchmark_names = (
        get_benchmark_suite(args.suite)
        if args.suite
        else (args.benchmark or "lane_following_v0",)
    )
    reports = {
        benchmark_name: evaluate_one(args, benchmark_name)
        for benchmark_name in benchmark_names
    }
    if args.suite:
        suite_report = {
            "suite": args.suite,
            "benchmarks": reports,
            "summary": summarize_suite(reports),
        }
        suite_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts",
            "evaluations",
            args.suite,
            args.algorithm,
        )
        os.makedirs(suite_dir, exist_ok=True)
        suite_path = os.path.join(suite_dir, "report.json")
        with open(suite_path, "w") as report_file:
            json.dump(suite_report, report_file, indent=2)
        print(json.dumps(suite_report["summary"], indent=2))
        print("Suite report -> {}".format(suite_path))


if __name__ == "__main__":
    main()
