from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner

from carla_rl_lab.benchmarks import bundled_route_file, load_nocrash_routes
from carla_rl_lab.benchmarks.nocrash import trace_route_compat
from carla_rl_lab.utils.provenance import git_commit, git_is_dirty, utc_timestamp


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_length(route: Iterable[Tuple[Any, Any]]) -> float:
    locations = [item[0].transform.location for item in route]
    return sum(
        previous.distance(current)
        for previous, current in zip(locations[:-1], locations[1:])
    )


def _trace(
    planner: GlobalRoutePlanner,
    origin: carla.Location,
    destination: carla.Location,
    route_id: Any,
) -> Dict[str, Any]:
    native_route = planner.trace_route(origin, destination)
    route = native_route if len(native_route) >= 2 else trace_route_compat(
        planner, origin, destination
    )
    end_error = (
        route[-1][0].transform.location.distance(destination) if route else float("inf")
    )
    length_m = _route_length(route)
    valid = len(route) >= 2 and length_m > 1.0 and end_error <= 5.0
    return {
        "route_id": route_id,
        "planner_mode": "native" if len(native_route) >= 2 else "same_edge_cycle",
        "waypoints": len(route),
        "length_m": round(length_m, 3),
        "end_error_m": round(end_error, 3),
        "valid": valid,
    }


def _fixed_routes(
    planner: GlobalRoutePlanner, town: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    route_path = bundled_route_file(town)
    definitions = load_nocrash_routes(route_path)
    records = [
        _trace(planner, start.location, destination.location, route_id)
        for route_id, (start, destination) in sorted(definitions.items())
    ]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return (
        {
            "path": os.path.relpath(route_path, root),
            "sha256": _sha256(route_path),
            "expected_routes": 25,
            "actual_routes": len(definitions),
        },
        records,
    )


def _random_routes(
    planner: GlobalRoutePlanner,
    spawn_points: List[carla.Transform],
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for index in range(count):
        start = rng.choice(spawn_points)
        candidates = [
            point
            for point in spawn_points
            if point.location.distance(start.location) > 50.0
        ]
        if not candidates:
            records.append({"route_id": index, "valid": False, "error": "no_destination"})
            continue
        destination = rng.choice(candidates)
        records.append(
            _trace(planner, start.location, destination.location, index)
        )
    return records


def _atomic_json(path: str, payload: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".rlfold-preflight-", suffix=".tmp", dir=output_dir
    )
    try:
        with os.fdopen(descriptor, "w") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the RLfOLD NoCrash route protocol against CARLA"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--town", choices=("Town01", "Town02", "all"), default="all"
    )
    parser.add_argument("--sampling-resolution", type=float, default=2.0)
    parser.add_argument("--random-routes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="artifacts/preflight/rlfold_nocrash_0915_routes.json",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    towns = ("Town01", "Town02") if args.town == "all" else (args.town,)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": "rlfold_nocrash_0915_v0",
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "carla_versions": {
            "client": client.get_client_version(),
            "server": client.get_server_version(),
        },
        "sampling_resolution": args.sampling_resolution,
        "towns": {},
    }
    failures = 0
    for town in towns:
        world = client.load_world(town)
        world_map = world.get_map()
        planner = GlobalRoutePlanner(world_map, args.sampling_resolution)
        route_file, fixed = _fixed_routes(planner, town)
        random_records = (
            _random_routes(
                planner,
                list(world_map.get_spawn_points()),
                args.random_routes,
                args.seed,
            )
            if town == "Town01" and args.random_routes > 0
            else []
        )
        town_failures = sum(not item["valid"] for item in fixed + random_records)
        failures += town_failures
        report["towns"][town] = {
            "map": world_map.name,
            "route_file": route_file,
            "fixed_routes": fixed,
            "random_training_routes": random_records,
            "failures": town_failures,
        }
        print(
            "{}: fixed={}/{} random={}/{} failures={}".format(
                town,
                sum(item["valid"] for item in fixed),
                len(fixed),
                sum(item["valid"] for item in random_records),
                len(random_records),
                town_failures,
            )
        )
    report["status"] = "passed" if failures == 0 else "failed"
    report["failures"] = failures
    _atomic_json(args.output, report)
    print("Preflight report -> {}".format(os.path.abspath(args.output)))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
