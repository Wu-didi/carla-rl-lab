from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


TRANSITION_FIELDS = ("states", "actions", "rewards", "next_states", "dones")
EXPERT_FIELDS = ("states", "actions")


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
    ) -> None:
        required = TRANSITION_FIELDS if require_transitions else EXPERT_FIELDS
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError("dataset is missing fields: {}".format(", ".join(missing)))

        self.arrays: Dict[str, np.ndarray] = {}
        for name in TRANSITION_FIELDS:
            if name in arrays:
                self.arrays[name] = np.asarray(arrays[name], dtype=np.float32)

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
            self.arrays["dones"] = self.arrays["dones"].reshape(-1)

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
        np.savez_compressed(path, **self.arrays)

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

