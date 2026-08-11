from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.benchmarks import (  # noqa: E402
    get_paper_benchmark,
    list_paper_benchmarks,
    prepare_paper_benchmark,
    probe_paper_benchmark,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight and run paper-standard CARLA benchmark evaluators"
    )
    parser.add_argument("--list", action="store_true", help="List supported protocols")
    parser.add_argument("--benchmark", choices=list(list_paper_benchmarks()))
    parser.add_argument("--agent", default="", help="Leaderboard-compatible agent .py")
    parser.add_argument("--agent-config", default="", help="Agent config or checkpoint path")
    parser.add_argument("--output", default="", help="Official evaluator JSON output")
    parser.add_argument("--carla-root", default="")
    parser.add_argument("--leaderboard-root", default="")
    parser.add_argument("--scenario-runner-root", default="")
    parser.add_argument("--routes", default="", help="Override official route XML")
    parser.add_argument("--scenarios", default="", help="Override scenario annotations")
    parser.add_argument("--python", dest="python_executable", default="")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--traffic-manager-seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--track", choices=("SENSORS", "MAP"), default="SENSORS")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--route-subset",
        default="",
        help="Bench2Drive route id/index list, useful for one-route smoke tests",
    )
    parser.add_argument("--gpu-rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="Require a CARLA server to be reachable during preflight",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the official evaluator; default is a dry-run preflight",
    )
    return parser


def print_catalog() -> None:
    for name in list_paper_benchmarks():
        spec = get_paper_benchmark(name)
        print(
            "{:<20} {:<20} CARLA {:<7} {}".format(
                spec.name, spec.status, spec.carla_version, spec.display_name
            )
        )


def main() -> None:
    args = build_argparser().parse_args()
    if args.list:
        print_catalog()
        return
    if not args.benchmark:
        raise SystemExit("--benchmark is required unless --list is used")

    launch = prepare_paper_benchmark(
        args.benchmark,
        agent=args.agent,
        agent_config=args.agent_config,
        output=args.output,
        carla_root=args.carla_root,
        leaderboard_root=args.leaderboard_root,
        scenario_runner_root=args.scenario_runner_root,
        routes=args.routes,
        scenarios=args.scenarios,
        python_executable=args.python_executable,
        host=args.host,
        port=args.port,
        traffic_manager_port=args.traffic_manager_port,
        traffic_manager_seed=args.traffic_manager_seed,
        repetitions=args.repetitions,
        track=args.track,
        timeout=args.timeout,
        route_subset=args.route_subset,
        gpu_rank=args.gpu_rank,
        resume=args.resume,
        check_server=args.check_server or args.run,
    )
    launch = probe_paper_benchmark(launch)
    print(json.dumps(launch.as_dict(), indent=2, ensure_ascii=False))
    if not launch.ready:
        raise SystemExit(2)
    if not args.run:
        return

    output_path = launch.paths["output"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    completed = subprocess.run(launch.command, env=launch.environment, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
