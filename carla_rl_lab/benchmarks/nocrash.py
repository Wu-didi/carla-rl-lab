from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import carla
import networkx as nx


NOCRASH_TRAIN_WEATHERS = (
    "ClearNoon",
    "WetNoon",
    "HardRainNoon",
    "ClearSunset",
)
NOCRASH_TEST_WEATHERS = ("SoftRainSunset", "WetSunset")


def weather_presets(group: str, fixed_weather: str) -> List[str]:
    if group == "fixed":
        return [fixed_weather]
    if group == "nocrash_train":
        return list(NOCRASH_TRAIN_WEATHERS)
    if group == "nocrash_test":
        return list(NOCRASH_TEST_WEATHERS)
    raise ValueError(
        "Unknown weather_group '{}'; choose fixed, nocrash_train, or nocrash_test".format(
            group
        )
    )


def bundled_route_file(town: str) -> str:
    return os.path.join(
        os.path.dirname(__file__), "assets", "nocrash", town, "routes.xml"
    )


def load_nocrash_routes(
    path: str,
) -> Dict[int, Tuple[carla.Transform, carla.Transform]]:
    route_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(route_path):
        raise FileNotFoundError("NoCrash route file not found: {}".format(route_path))
    root = ET.parse(route_path).getroot()
    routes = {}
    for route in root.findall("route"):
        points = route.findall("./ego_vehicle/waypoint")
        if len(points) != 2:
            raise ValueError(
                "NoCrash route {} must contain exactly two endpoints".format(
                    route.attrib.get("id", "?")
                )
            )
        transforms = []
        for point in points:
            transforms.append(
                carla.Transform(
                    carla.Location(
                        x=float(point.attrib["x"]),
                        y=float(point.attrib["y"]),
                        z=float(point.attrib["z"]),
                    ),
                    carla.Rotation(
                        pitch=float(point.attrib.get("pitch", 0.0)),
                        yaw=float(point.attrib.get("yaw", 0.0)),
                        roll=float(point.attrib.get("roll", 0.0)),
                    ),
                )
            )
        routes[int(route.attrib["id"])] = (transforms[0], transforms[1])
    if not routes:
        raise ValueError("No routes found in {}".format(route_path))
    return routes


def trace_route_compat(
    planner: Any, origin: carla.Location, destination: carla.Location
) -> List[Tuple[Any, Any]]:
    """Trace a route, including CARLA's same-edge wraparound corner case.

    CARLA 0.9.15 returns a one-waypoint route when origin and destination are
    on the same graph edge but the destination lies behind the origin. In that
    case the valid route must finish the edge, traverse the shortest directed
    cycle, and re-enter the edge from its beginning.
    """

    route = planner.trace_route(origin, destination)
    if len(route) >= 2:
        return route

    start_edge = planner._localize(origin)
    end_edge = planner._localize(destination)
    if start_edge is None or end_edge is None or start_edge != end_edge:
        return route

    cycle = nx.astar_path(
        planner._graph,
        source=start_edge[1],
        target=end_edge[0],
        heuristic=planner._distance_heuristic,
        weight="length",
    )
    try:
        from agents.navigation.local_planner import RoadOption
    except ImportError as exc:
        raise ImportError(
            "CARLA navigation agents are missing. Add "
            "$CARLA_ROOT/PythonAPI/carla to PYTHONPATH."
        ) from exc
    node_route = [start_edge[0]] + cycle + [end_edge[1]]
    original_path_search = planner._path_search
    planner._path_search = lambda _origin, _destination: node_route
    planner._previous_decision = RoadOption.VOID
    planner._intersection_end_node = -1
    try:
        wrapped_route = planner.trace_route(origin, destination)
    finally:
        planner._path_search = original_path_search
    return wrapped_route
