from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


DEFAULT_FIELDS: Tuple[str, ...] = (
    "ego_state",
    "lane_info",
    "risk_field",
    "lidar",
    "waypoints",
)


def encode_observation(
    obs_dict: Dict[str, np.ndarray],
    expected_dim: int = 0,
    risk_field_dim: int = 12,
    fields: Tuple[str, ...] = DEFAULT_FIELDS,
) -> np.ndarray:
    """Flatten a CARLA observation dictionary and validate its size."""

    parts = []
    for field in fields:
        if field == "risk_field" and field not in obs_dict:
            value = np.zeros(risk_field_dim, dtype=np.float32)
        elif field not in obs_dict:
            available = ", ".join(sorted(obs_dict))
            raise KeyError(
                "Missing observation field '{}'. Available fields: {}".format(
                    field, available
                )
            )
        else:
            value = obs_dict[field]
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))

    vector = np.concatenate(parts).astype(np.float32)
    if expected_dim and vector.shape[0] != expected_dim:
        raise ValueError(
            "Observation dim mismatch: expected {}, got {}. Fields={}".format(
                expected_dim, vector.shape[0], list(fields)
            )
        )
    return vector
