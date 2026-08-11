from __future__ import annotations

import hashlib
import os
import re
import shlex
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PaperBenchmarkSpec:
    name: str
    display_name: str
    backend: str
    status: str
    description: str
    carla_version: str
    expected_routes: int
    metrics: Tuple[str, ...]
    route_candidates: Tuple[str, ...] = ()
    scenario_candidates: Tuple[str, ...] = ()
    references: Tuple[str, ...] = ()
    note: str = ""


@dataclass
class PaperBenchmarkLaunch:
    spec: PaperBenchmarkSpec
    command: Tuple[str, ...]
    environment: Dict[str, str]
    pythonpath_entries: Tuple[str, ...]
    paths: Dict[str, str]
    route_manifest: Dict[str, Any]
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.errors and bool(self.command)

    @property
    def command_string(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": asdict(self.spec),
            "ready": self.ready,
            "paths": dict(self.paths),
            "route_manifest": dict(self.route_manifest),
            "command": list(self.command),
            "command_string": self.command_string,
            "pythonpath_entries": list(self.pythonpath_entries),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


_PAPER_BENCHMARKS = {
    "corl2017": PaperBenchmarkSpec(
        name="corl2017",
        display_name="CARLA CoRL 2017",
        backend="legacy_08",
        status="legacy-only",
        description=(
            "Original CARLA benchmark with Straight, One Turn, Navigation, "
            "and Navigation with dynamic obstacles tasks."
        ),
        carla_version="0.8.2",
        expected_routes=0,
        metrics=("success_rate", "distance_to_goal", "infractions_per_km"),
        references=(
            "https://arxiv.org/abs/1711.03938",
            "https://carla.readthedocs.io/en/0.8.4/benchmark_start/",
        ),
        note="Requires the legacy CARLA 0.8.x client and driving-benchmarks API.",
    ),
    "nocrash": PaperBenchmarkSpec(
        name="nocrash",
        display_name="NoCrash",
        backend="legacy_08",
        status="legacy-only",
        description=(
            "Goal-directed Town01/Town02 evaluation under empty, regular, "
            "and dense traffic with train and test weather splits."
        ),
        carla_version="0.8.4",
        expected_routes=0,
        metrics=("success_rate",),
        references=(
            "https://openaccess.thecvf.com/content_ICCV_2019/html/"
            "Codevilla_Exploring_the_Limitations_of_Behavior_Cloning_for_"
            "Autonomous_Driving_ICCV_2019_paper.html",
        ),
        note="Requires the legacy NoCrash suite; CARLA 0.9.x results are not protocol-equivalent.",
    ),
    "town05_short": PaperBenchmarkSpec(
        name="town05_short",
        display_name="Town05 Short",
        backend="leaderboard1",
        status="runnable",
        description="Short Town05 routes used by TransFuser-era closed-loop papers.",
        carla_version="0.9.10",
        expected_routes=32,
        metrics=("driving_score", "route_completion", "infraction_score"),
        route_candidates=(
            "data/validation_routes/routes_town05_short.xml",
            "data/evaluation_routes/routes_town05_short.xml",
        ),
        scenario_candidates=("data/scenarios/town05_all_scenarios.json",),
        references=("https://github.com/autonomousvision/transfuser",),
    ),
    "town05_long": PaperBenchmarkSpec(
        name="town05_long",
        display_name="Town05 Long",
        backend="leaderboard1",
        status="runnable",
        description="Ten long Town05 routes used by TransFuser-era closed-loop papers.",
        carla_version="0.9.10",
        expected_routes=10,
        metrics=("driving_score", "route_completion", "infraction_score"),
        route_candidates=(
            "data/evaluation_routes/routes_town05_long.xml",
            "data/validation_routes/routes_town05_long.xml",
        ),
        scenario_candidates=("data/scenarios/town05_all_scenarios.json",),
        references=("https://github.com/autonomousvision/transfuser",),
    ),
    "longest6": PaperBenchmarkSpec(
        name="longest6",
        display_name="Longest6",
        backend="leaderboard1",
        status="route-file-required",
        description="Thirty-six long routes across six towns, commonly used after Town05 Long.",
        carla_version="0.9.10",
        expected_routes=36,
        metrics=("driving_score", "route_completion", "infraction_score"),
        route_candidates=(
            "data/longest6/longest6.xml",
            "data/longest6/routes_longest6.xml",
            "data/evaluation_routes/routes_longest6.xml",
            "data/longest6/00.xml",
        ),
        scenario_candidates=(
            "data/official/all_towns_traffic_scenarios_public.json",
            "data/scenarios/merged_all_towns_scenarios.json",
        ),
        references=("https://github.com/autonomousvision/transfuser",),
        note="Pass the exact route XML published with the paper being reproduced.",
    ),
    "longest6_v2": PaperBenchmarkSpec(
        name="longest6_v2",
        display_name="Longest6 v2",
        backend="bench2drive",
        status="route-file-required",
        description=(
            "CARLA 0.9.15 adaptation of the 36-route Longest6 benchmark using "
            "Leaderboard 2.x scenario logic."
        ),
        carla_version="0.9.15",
        expected_routes=36,
        metrics=("driving_score", "route_completion", "infraction_score"),
        route_candidates=(
            "data/longest6/longest6.xml",
            "data/longest6_v2/longest6.xml",
            "data/longest6_v2/routes_longest6.xml",
        ),
        references=("https://github.com/autonomousvision/carla_garage",),
        note="Longest6 v2 numbers are not directly comparable to Leaderboard 1.0 Longest6.",
    ),
    "carla_leaderboard1": PaperBenchmarkSpec(
        name="carla_leaderboard1",
        display_name="CARLA Leaderboard 1.x public routes",
        backend="leaderboard1",
        status="runnable",
        description="Official route-based Leaderboard evaluation with scripted traffic scenarios.",
        carla_version="0.9.10",
        expected_routes=0,
        metrics=("driving_score", "route_completion", "infraction_score"),
        route_candidates=(
            "data/official/routes_testing.xml",
            "data/official/routes_devtest.xml",
        ),
        scenario_candidates=("data/official/all_towns_traffic_scenarios_public.json",),
        references=(
            "https://github.com/carla-simulator/leaderboard",
            "https://leaderboard.carla.org/",
        ),
    ),
    "bench2drive220": PaperBenchmarkSpec(
        name="bench2drive220",
        display_name="Bench2Drive 220",
        backend="bench2drive",
        status="runnable",
        description=(
            "Official 220-route Bench2Drive split covering 44 interactive scenario types, "
            "23 weather conditions, and 12 towns."
        ),
        carla_version="0.9.15",
        expected_routes=220,
        metrics=(
            "driving_score",
            "route_completion",
            "infraction_score",
            "scenario_success_rate",
        ),
        route_candidates=("data/bench2drive220.xml",),
        references=(
            "https://arxiv.org/abs/2406.03877",
            "https://github.com/Thinklab-SJTU/Bench2Drive",
        ),
    ),
}

_ALIASES = {
    "bench2drive": "bench2drive220",
    "leaderboard1": "carla_leaderboard1",
    "town05-short": "town05_short",
    "town05-long": "town05_long",
    "longest6-v2": "longest6_v2",
}


def list_paper_benchmarks() -> Tuple[str, ...]:
    return tuple(sorted(_PAPER_BENCHMARKS))


def get_paper_benchmark(name: str) -> PaperBenchmarkSpec:
    key = _ALIASES.get(name.lower(), name.lower())
    try:
        return _PAPER_BENCHMARKS[key]
    except KeyError as exc:
        raise ValueError(
            "Unknown paper benchmark '{}'. Available benchmarks: {}".format(
                name, ", ".join(list_paper_benchmarks())
            )
        ) from exc


def inspect_route_file(path: str) -> Dict[str, Any]:
    route_path = Path(path)
    root = ET.parse(str(route_path)).getroot()
    routes = root.findall(".//route")
    towns = sorted({route.attrib.get("town", "") for route in routes if route.attrib.get("town")})
    scenario_types = set()
    weather_count = 0
    for route in routes:
        for scenario in route.findall(".//scenario"):
            scenario_type = scenario.attrib.get("type")
            if scenario_type:
                scenario_types.add(scenario_type)
        weather_count += len(route.findall(".//weather"))

    digest = hashlib.sha256()
    with route_path.open("rb") as route_file:
        for chunk in iter(lambda: route_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(route_path.resolve()),
        "sha256": digest.hexdigest(),
        "route_count": len(routes),
        "towns": towns,
        "embedded_scenario_types": sorted(scenario_types),
        "weather_entries": weather_count,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_root() -> Path:
    return _project_root().parent


def _first_path(
    explicit: str,
    environment_value: str,
    candidates: Sequence[Path],
) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if environment_value:
        return Path(environment_value).expanduser().resolve()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else None


def _normalise_leaderboard_root(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if (path / "leaderboard" / "leaderboard_evaluator.py").is_file():
        return path
    nested = path / "leaderboard"
    if (nested / "leaderboard" / "leaderboard_evaluator.py").is_file():
        return nested
    return path


def _root_candidates(spec: PaperBenchmarkSpec) -> Tuple[Tuple[Path, ...], Tuple[Path, ...]]:
    workspace = _workspace_root()
    if spec.backend == "bench2drive":
        leaderboard = (
            workspace / "simlingo" / "Bench2Drive" / "leaderboard",
            workspace / "offical_simligo" / "simlingo" / "Bench2Drive" / "leaderboard",
            workspace / "Bench2Drive" / "leaderboard",
        )
        scenario_runner = (
            workspace / "simlingo" / "Bench2Drive" / "scenario_runner",
            workspace / "offical_simligo" / "simlingo" / "Bench2Drive" / "scenario_runner",
            workspace / "Bench2Drive" / "scenario_runner",
        )
    else:
        leaderboard = (
            workspace / "LMDrive" / "leaderboard",
            workspace / "leaderboard",
        )
        scenario_runner = (
            workspace / "LMDrive" / "scenario_runner",
            workspace / "scenario_runner",
        )
    return leaderboard, scenario_runner


def _route_candidates(
    spec: PaperBenchmarkSpec, leaderboard_root: Optional[Path]
) -> Tuple[Path, ...]:
    candidates = []
    if leaderboard_root is not None:
        candidates.extend(leaderboard_root / relative for relative in spec.route_candidates)
    workspace = _workspace_root()
    if spec.name == "bench2drive220":
        candidates.extend(
            (
                workspace / "simlingo" / "leaderboard" / "data" / "bench2drive220.xml",
                workspace
                / "offical_simligo"
                / "simlingo"
                / "leaderboard"
                / "data"
                / "bench2drive220.xml",
            )
        )
    return tuple(candidates)


def _scenario_candidates(
    spec: PaperBenchmarkSpec, leaderboard_root: Optional[Path]
) -> Tuple[Path, ...]:
    if leaderboard_root is None:
        return ()
    return tuple(leaderboard_root / relative for relative in spec.scenario_candidates)


def _detected_version(carla_root: Optional[Path]) -> str:
    if carla_root is None:
        return ""
    match = re.search(r"(?<!\d)(0\.\d+\.\d+)(?!\d)", str(carla_root))
    return match.group(1) if match else ""


def _server_available(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def prepare_paper_benchmark(
    name: str,
    agent: str = "",
    agent_config: str = "",
    output: str = "",
    carla_root: str = "",
    leaderboard_root: str = "",
    scenario_runner_root: str = "",
    routes: str = "",
    scenarios: str = "",
    python_executable: str = "",
    host: str = "localhost",
    port: int = 2000,
    traffic_manager_port: int = 8000,
    traffic_manager_seed: int = 0,
    repetitions: int = 1,
    track: str = "SENSORS",
    timeout: float = 600.0,
    route_subset: str = "",
    gpu_rank: int = 0,
    resume: bool = False,
    check_server: bool = False,
    environment: Optional[Mapping[str, str]] = None,
) -> PaperBenchmarkLaunch:
    spec = get_paper_benchmark(name)
    source_environment = dict(os.environ if environment is None else environment)
    errors = []
    warnings = []

    if spec.backend == "legacy_08":
        errors.append(
            "{} is a CARLA {} legacy protocol and cannot be evaluated faithfully "
            "with the CARLA 0.9.x Leaderboard runner.".format(spec.display_name, spec.carla_version)
        )
        return PaperBenchmarkLaunch(
            spec=spec,
            command=(),
            environment={},
            pythonpath_entries=(),
            paths={},
            route_manifest={},
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    leaderboard_candidates, scenario_candidates = _root_candidates(spec)
    env_leaderboard = source_environment.get("LEADERBOARD_ROOT", "")
    if spec.backend == "bench2drive" and source_environment.get("BENCH2DRIVE_ROOT"):
        env_leaderboard = source_environment["BENCH2DRIVE_ROOT"]
    leaderboard_path = _normalise_leaderboard_root(
        _first_path(leaderboard_root, env_leaderboard, leaderboard_candidates)
    )
    scenario_runner_path = _first_path(
        scenario_runner_root,
        source_environment.get("SCENARIO_RUNNER_ROOT", ""),
        scenario_candidates,
    )
    workspace = _workspace_root()
    carla_candidates = (
        workspace.parent / "CARLA_0.9.15",
        workspace / "CARLA_0.9.15",
        workspace.parent / "CARLA_0.9.10.1",
        workspace / "CARLA_0.9.10.1",
    )
    carla_path = _first_path(
        carla_root, source_environment.get("CARLA_ROOT", ""), carla_candidates
    )
    route_path = _first_path(
        routes,
        source_environment.get("ROUTES", ""),
        _route_candidates(spec, leaderboard_path),
    )
    scenario_path = _first_path(
        scenarios,
        source_environment.get("SCENARIOS", ""),
        _scenario_candidates(spec, leaderboard_path),
    )
    agent_path = Path(agent).expanduser().resolve() if agent else None
    agent_config_path = Path(agent_config).expanduser().resolve() if agent_config else None
    output_path = Path(output).expanduser().resolve() if output else (
        _project_root() / "artifacts" / "paper_benchmarks" / spec.name / "results.json"
    )
    evaluator_path = (
        leaderboard_path / "leaderboard" / "leaderboard_evaluator.py"
        if leaderboard_path is not None
        else None
    )

    required_files = (
        ("CARLA executable", carla_path / "CarlaUE4.sh" if carla_path else None),
        ("Leaderboard evaluator", evaluator_path),
        ("route XML", route_path),
        ("agent", agent_path),
    )
    for label, path in required_files:
        if path is None or not path.is_file():
            errors.append("Missing {}: {}".format(label, path or "<not set>"))
    if scenario_runner_path is None or not (scenario_runner_path / "srunner").is_dir():
        errors.append("Missing ScenarioRunner root: {}".format(scenario_runner_path or "<not set>"))
    if spec.backend == "leaderboard1" and (scenario_path is None or not scenario_path.is_file()):
        errors.append("Missing scenario annotations: {}".format(scenario_path or "<not set>"))
    if agent_config_path is not None and not agent_config_path.exists():
        errors.append("Missing agent config: {}".format(agent_config_path))
    if repetitions < 1:
        errors.append("repetitions must be at least 1")
    if route_subset and spec.backend != "bench2drive":
        errors.append("--route-subset is only supported by the Bench2Drive evaluator")

    manifest: Dict[str, Any] = {}
    if route_path is not None and route_path.is_file():
        try:
            manifest = inspect_route_file(str(route_path))
            if spec.expected_routes and manifest["route_count"] != spec.expected_routes:
                warnings.append(
                    "Expected {} routes for {}, found {} in {}.".format(
                        spec.expected_routes,
                        spec.display_name,
                        manifest["route_count"],
                        route_path,
                    )
                )
        except (ET.ParseError, OSError) as exc:
            errors.append("Invalid route XML {}: {}".format(route_path, exc))

    detected_version = _detected_version(carla_path)
    if detected_version and detected_version != spec.carla_version:
        warnings.append(
            "{} is normally reported with CARLA {}, but CARLA {} was detected. "
            "Run with the paper's version for comparable numbers.".format(
                spec.display_name, spec.carla_version, detected_version
            )
        )
    if check_server and not _server_available(host, port):
        errors.append("No CARLA server is reachable at {}:{}".format(host, port))

    python_bin = python_executable or sys.executable
    command = []
    if evaluator_path is not None:
        command = [
            python_bin,
            "-u",
            str(evaluator_path),
            "--host={}".format(host),
            "--port={}".format(port),
            "--routes={}".format(route_path or ""),
            "--repetitions={}".format(repetitions),
            "--track={}".format(track),
            "--checkpoint={}".format(output_path),
            "--agent={}".format(agent_path or ""),
            "--agent-config={}".format(agent_config_path or ""),
            "--debug=0",
            "--timeout={}".format(timeout),
        ]
        if spec.backend == "bench2drive":
            command.extend(
                (
                    "--traffic-manager-port={}".format(traffic_manager_port),
                    "--traffic-manager-seed={}".format(traffic_manager_seed),
                    "--gpu-rank={}".format(gpu_rank),
                    "--debug-checkpoint={}".format(output_path.with_suffix(".live.txt")),
                )
            )
            if route_subset:
                command.append("--routes-subset={}".format(route_subset))
        else:
            command.extend(
                (
                    "--scenarios={}".format(scenario_path or ""),
                    "--trafficManagerPort={}".format(traffic_manager_port),
                    "--trafficManagerSeed={}".format(traffic_manager_seed),
                )
            )
        if resume:
            command.append("--resume=True")

    pythonpath = []
    if carla_path is not None:
        pythonpath.extend((carla_path / "PythonAPI", carla_path / "PythonAPI" / "carla"))
        egg_candidates = sorted(
            (carla_path / "PythonAPI" / "carla" / "dist").glob("carla-*.egg")
        )
        major_marker = "-py{}.".format(sys.version_info[0])
        compatible_eggs = [
            egg for egg in egg_candidates if major_marker in egg.name
        ]
        pythonpath.extend(compatible_eggs or egg_candidates)
    if leaderboard_path is not None:
        pythonpath.extend((leaderboard_path, leaderboard_path.parent))
    if scenario_runner_path is not None:
        pythonpath.append(scenario_runner_path)
    pythonpath.append(_project_root())
    pythonpath_entries = tuple(str(path) for path in pythonpath if path.exists())

    launch_environment = dict(source_environment)
    old_pythonpath = source_environment.get("PYTHONPATH", "")
    launch_environment["PYTHONPATH"] = os.pathsep.join(
        pythonpath_entries + ((old_pythonpath,) if old_pythonpath else ())
    )
    if carla_path is not None:
        launch_environment["CARLA_ROOT"] = str(carla_path)
    if leaderboard_path is not None:
        launch_environment["LEADERBOARD_ROOT"] = str(leaderboard_path)
    if scenario_runner_path is not None:
        launch_environment["SCENARIO_RUNNER_ROOT"] = str(scenario_runner_path)
    if spec.backend == "bench2drive":
        launch_environment["IS_BENCH2DRIVE"] = "1"

    paths = {
        "carla_root": str(carla_path or ""),
        "leaderboard_root": str(leaderboard_path or ""),
        "scenario_runner_root": str(scenario_runner_path or ""),
        "routes": str(route_path or ""),
        "scenarios": str(scenario_path or ""),
        "agent": str(agent_path or ""),
        "agent_config": str(agent_config_path or ""),
        "output": str(output_path),
    }
    return PaperBenchmarkLaunch(
        spec=spec,
        command=tuple(command),
        environment=launch_environment,
        pythonpath_entries=pythonpath_entries,
        paths=paths,
        route_manifest=manifest,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def probe_paper_benchmark(
    launch: PaperBenchmarkLaunch, timeout: float = 20.0
) -> PaperBenchmarkLaunch:
    """Import the selected official evaluator through its ``--help`` path."""

    evaluator = launch.paths.get("leaderboard_root", "")
    if not launch.command or not evaluator:
        return launch
    probe_command = (
        launch.command[0],
        "-u",
        os.path.join(evaluator, "leaderboard", "leaderboard_evaluator.py"),
        "--help",
    )
    try:
        completed = subprocess.run(
            probe_command,
            env=launch.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        launch.errors = launch.errors + (
            "Evaluator import smoke could not run: {}".format(exc),
        )
        return launch

    if completed.returncode != 0:
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        detail = lines[-1] if lines else "no diagnostic output"
        launch.errors = launch.errors + (
            "Evaluator import smoke failed (exit {}): {}".format(
                completed.returncode, detail
            ),
        )
    return launch
