from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


SCHEMA_VERSION = 2
TRANSITION_FIELDS = ("states", "actions", "rewards", "next_states", "dones")
EXPERT_FIELDS = ("states", "actions")
OPTIONAL_FIELDS = ("terminals", "timeouts", "episode_ids", "costs")


class OfflineDataset:
    """Validated in-memory view of an offline transition or expert dataset.

    Files use NumPy's portable ``.npz`` format. Offline RL requires all five
    transition fields; behavior cloning may use a state/action-only dataset.
    """

    def __init__(
        self,
        arrays: Mapping[str, Any],
        require_transitions: bool = True,
        seed: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        had_terminals = "terminals" in arrays
        had_metadata = metadata is not None or "metadata_json" in arrays
        required = TRANSITION_FIELDS[:4] if require_transitions else EXPERT_FIELDS
        missing = [name for name in required if name not in arrays]
        if require_transitions and "dones" not in arrays and "terminals" not in arrays:
            missing.append("dones or terminals")
        if missing:
            raise ValueError("dataset is missing fields: {}".format(", ".join(missing)))

        self.arrays: Dict[str, np.ndarray] = {}
        for name in TRANSITION_FIELDS + OPTIONAL_FIELDS:
            if name in arrays:
                dtype = None if name in ("states", "next_states") else np.float32
                self.arrays[name] = np.asarray(arrays[name], dtype=dtype)

        size = int(self.arrays["states"].shape[0])
        if size == 0:
            raise ValueError("dataset must contain at least one sample")
        for name, values in self.arrays.items():
            if values.shape[0] != size:
                raise ValueError(
                    "dataset field '{}' has {} rows; expected {}".format(
                        name, values.shape[0], size
                    )
                )

        if self.arrays["states"].ndim != 2 or self.arrays["actions"].ndim != 2:
            raise ValueError("states and actions must be rank-2 arrays")
        if require_transitions:
            if self.arrays["next_states"].shape != self.arrays["states"].shape:
                raise ValueError("next_states must have the same shape as states")
            self.arrays["rewards"] = self.arrays["rewards"].reshape(-1)
            if "terminals" not in self.arrays:
                self.arrays["terminals"] = self.arrays["dones"].reshape(-1).copy()
            else:
                self.arrays["terminals"] = self.arrays["terminals"].reshape(-1)
            if "timeouts" not in self.arrays:
                self.arrays["timeouts"] = np.zeros(size, dtype=np.float32)
            else:
                self.arrays["timeouts"] = self.arrays["timeouts"].reshape(-1)
            # Algorithms bootstrap across time-limit truncations, but not true terminals.
            self.arrays["dones"] = self.arrays["terminals"].copy()

        for name in ("states", "actions", "rewards", "next_states"):
            if name in self.arrays and not np.isfinite(self.arrays[name]).all():
                raise ValueError("dataset field '{}' contains NaN or Inf".format(name))

        raw_metadata = arrays.get("metadata_json")
        if metadata is None and raw_metadata is not None:
            try:
                metadata = json.loads(str(np.asarray(raw_metadata).reshape(()).item()))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("invalid metadata_json in dataset") from exc
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("schema_version", SCHEMA_VERSION if had_metadata else 1)
        self.metadata.setdefault("size", size)
        self.metadata.setdefault("state_dim", int(self.arrays["states"].shape[1]))
        self.metadata.setdefault("action_dim", int(self.arrays["actions"].shape[1]))
        if require_transitions:
            self.metadata.setdefault(
                "dones_meaning",
                "true_terminal_only" if had_terminals else "terminal_or_timeout_unknown",
            )

        for name, actual in (
            ("size", size),
            ("state_dim", int(self.arrays["states"].shape[1])),
            ("action_dim", int(self.arrays["actions"].shape[1])),
        ):
            if name in self.metadata and int(self.metadata[name]) != actual:
                raise ValueError(
                    "dataset metadata {}={} does not match arrays ({})".format(
                        name, self.metadata[name], actual
                    )
                )

        self.require_transitions = bool(require_transitions)
        self.rng = np.random.RandomState(seed)

    @classmethod
    def load(
        cls,
        path: str,
        require_transitions: bool = True,
        seed: Optional[int] = None,
    ) -> "OfflineDataset":
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        return cls(arrays, require_transitions=require_transitions, seed=seed)

    def save(self, path: str) -> None:
        if not path.lower().endswith(".npz"):
            raise ValueError("offline dataset path must end with .npz")
        payload = dict(self.arrays)
        payload["metadata_json"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True), dtype=np.str_
        )
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".dataset-", suffix=".npz", dir=os.path.dirname(os.path.abspath(path))
        )
        os.close(descriptor)
        try:
            np.savez_compressed(temporary_path, **payload)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def sample(
        self, batch_size: int, fields: Optional[Iterable[str]] = None
    ) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected = tuple(fields) if fields is not None else tuple(self.arrays)
        unknown = [name for name in selected if name not in self.arrays]
        if unknown:
            raise ValueError("unknown dataset fields: {}".format(", ".join(unknown)))
        indices = self.rng.randint(0, len(self), size=int(batch_size))
        return {name: self.arrays[name][indices] for name in selected}

    @property
    def state_dim(self) -> int:
        return int(self.arrays["states"].shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.arrays["actions"].shape[1])

    def __len__(self) -> int:
        return int(self.arrays["states"].shape[0])
