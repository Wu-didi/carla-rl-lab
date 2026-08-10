from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


DEFAULT_FIELDS: Tuple[str, ...] = (
    "ego_state",
    "lane_info",
    "risk_field",
    "lidar",
    "waypoints",
)


@dataclass(frozen=True)
class VectorObservationAdapter:
    """Convert CARLA observation dictionaries into flat vectors.

    The adapter owns observation ordering and shape validation. Keeping this
    outside the trainer makes policy/network experiments less error-prone.
    """

    expected_dim: int
    fields: Tuple[str, ...] = DEFAULT_FIELDS
    risk_field_dim: int = 12

    def encode(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for field in self.fields:
            value = self._get_field(obs_dict, field)
            parts.append(np.asarray(value, dtype=np.float32).reshape(-1))

        vector = np.concatenate(parts).astype(np.float32)
        if self.expected_dim and vector.shape[0] != self.expected_dim:
            raise ValueError(
                f"Observation dim mismatch: expected {self.expected_dim}, got {vector.shape[0]}. "
                f"Fields={list(self.fields)}"
            )
        return vector

    def _get_field(self, obs_dict: Dict[str, np.ndarray], field: str) -> np.ndarray:
        if field == "risk_field" and field not in obs_dict:
            return np.zeros(self.risk_field_dim, dtype=np.float32)
        if field not in obs_dict:
            available = ", ".join(sorted(obs_dict))
            raise KeyError(f"Missing observation field '{field}'. Available fields: {available}")
        return obs_dict[field]


def convert_obs_dict_to_vector(obs_dict: Dict[str, np.ndarray], expected_dim: int = 0) -> np.ndarray:
    """Backward-compatible helper for legacy scripts."""

    return VectorObservationAdapter(expected_dim=expected_dim).encode(obs_dict)
