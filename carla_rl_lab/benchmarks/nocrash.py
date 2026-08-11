from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import carla


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
