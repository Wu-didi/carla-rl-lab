from __future__ import annotations

from typing import Tuple

import numpy as np


SIGNED_3D = "signed_3d"
LONGITUDINAL_2D = "longitudinal_2d"
ACTION_MODES = (SIGNED_3D, LONGITUDINAL_2D)


def validate_action_spec(action_mode: str, action_dim: int) -> None:
    expected_dim = 3 if action_mode == SIGNED_3D else 2
    if action_mode not in ACTION_MODES:
        raise ValueError(
            "unknown action_mode '{}'; choose from {}".format(
                action_mode, ", ".join(ACTION_MODES)
            )
        )
    if int(action_dim) != expected_dim:
        raise ValueError(
            "action_mode '{}' requires action_dim={}, got {}".format(
                action_mode, expected_dim, action_dim
            )
        )


def policy_action_to_carla(
    action: np.ndarray, action_mode: str = SIGNED_3D, action_bound: float = 1.0
) -> Tuple[float, float, float]:
    """Convert a bounded policy action into throttle, steer, and brake."""

    array = np.asarray(action, dtype=np.float32).reshape(-1)
    validate_action_spec(action_mode, array.size)
    if action_bound <= 0.0:
        raise ValueError("action_bound must be positive")
    unit = np.clip(array / float(action_bound), -1.0, 1.0)
    if action_mode == LONGITUDINAL_2D:
        longitudinal, steer = unit
        throttle = max(float(longitudinal), 0.0)
        brake = max(float(-longitudinal), 0.0)
    else:
        throttle = max(float(unit[0]), 0.0)
        steer = float(unit[1])
        brake = max(float(unit[2]), 0.0)
    return throttle, steer, brake


def carla_action_to_policy(
    throttle: float,
    steer: float,
    brake: float,
    action_mode: str = SIGNED_3D,
    action_bound: float = 1.0,
) -> np.ndarray:
    """Encode CARLA control values in the policy action representation."""

    if action_bound <= 0.0:
        raise ValueError("action_bound must be positive")
    throttle = float(np.clip(throttle, 0.0, 1.0))
    steer = float(np.clip(steer, -1.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    if action_mode == LONGITUDINAL_2D:
        validate_action_spec(action_mode, 2)
        unit = np.array([throttle - brake, steer], dtype=np.float32)
    else:
        validate_action_spec(action_mode, 3)
        unit = np.array([throttle, steer, brake], dtype=np.float32)
    return unit * float(action_bound)
