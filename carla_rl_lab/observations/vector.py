from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


PIXEL_FIELDS: Tuple[str, ...] = ("image", "waypoints", "vehicle_measurements")
DEFAULT_FIELDS = PIXEL_FIELDS


def pixel_state_dim(image_size: int, frame_stack: int, num_waypoints: int) -> int:
    """Return the packed size of RGB frames, route points, speed, and steer."""

    return 3 * int(frame_stack) * int(image_size) ** 2 + 2 * int(num_waypoints) + 2


def encode_observation(
    obs_dict: Dict[str, np.ndarray],
    expected_dim: int = 0,
    fields: Tuple[str, ...] = DEFAULT_FIELDS,
) -> np.ndarray:
    """Pack the policy observation as uint8 for memory-efficient replay.

    Images stay in [0, 255]. Route coordinates are expected in [-1, 1].
    Vehicle measurements contain normalized speed in [0, 1] and steer in
    [-1, 1]. Telemetry fields such as global pose are deliberately excluded.
    """

    missing = [field for field in fields if field not in obs_dict]
    if missing:
        raise KeyError(
            "Missing observation fields {}. Available fields: {}".format(
                missing, ", ".join(sorted(obs_dict))
            )
        )

    image = np.asarray(obs_dict["image"], dtype=np.uint8).reshape(-1)
    waypoints = np.asarray(obs_dict["waypoints"], dtype=np.float32).reshape(-1)
    measurements = np.asarray(
        obs_dict["vehicle_measurements"], dtype=np.float32
    ).reshape(-1)
    if measurements.shape != (2,):
        raise ValueError(
            "vehicle_measurements must contain [normalized_speed, steer]"
        )

    route_bytes = np.rint((np.clip(waypoints, -1.0, 1.0) + 1.0) * 127.5)
    speed_byte = np.rint(np.clip(measurements[0], 0.0, 1.0) * 255.0)
    steer_byte = np.rint((np.clip(measurements[1], -1.0, 1.0) + 1.0) * 127.5)
    auxiliary = np.concatenate(
        [route_bytes, np.asarray([speed_byte, steer_byte], dtype=np.float32)]
    ).astype(np.uint8)
    packed = np.concatenate([image, auxiliary]).astype(np.uint8, copy=False)

    if expected_dim and packed.shape[0] != expected_dim:
        raise ValueError(
            "Observation dim mismatch: expected {}, got {}".format(
                expected_dim, packed.shape[0]
            )
        )
    return packed
