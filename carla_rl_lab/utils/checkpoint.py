from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from carla_rl_lab.utils.provenance import git_commit, jsonable_config, utc_timestamp


CHECKPOINT_FORMAT_VERSION = 1
def torch_load(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_path(directory: str, algorithm: str, step_id: Any) -> str:
    return os.path.join(
        directory, "{}_ckpt_{}.pt".format(algorithm, step_id)
    )


def checkpoint_metadata(path: str) -> Dict[str, Any]:
    payload = torch_load(path, map_location="cpu")
    return dict(payload.get("_carla_rl_lab", {}))


def apply_checkpoint_config(
    cfg: Any,
    path: str,
    fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    metadata = checkpoint_metadata(path)
    saved_config = metadata.get("config", {})
    excluded = {"device", "use_pretrained_model", "pretrained_model_path"}
    selected = (
        tuple(name for name in saved_config if name not in excluded)
        if fields is None
        else tuple(fields)
    )
    for name in selected:
        if name in saved_config and hasattr(cfg, name):
            setattr(cfg, name, saved_config[name])
    return metadata


def restore_training_state(path: str, restore_rng: bool = True) -> Dict[str, Any]:
    payload = torch_load(path, map_location="cpu")
    if restore_rng:
        rng = payload.get("_rng_state", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda"):
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return dict(payload.get("_trainer_state", {}))


def save_training_checkpoint(
    agent: Any,
    cfg: Any,
    directory: str,
    global_step: int,
    trainer_state: Optional[Mapping[str, Any]] = None,
) -> str:
    """Save an immutable checkpoint, update ``last``, and write a manifest."""

    os.makedirs(directory, exist_ok=True)
    algorithm = str(cfg.algorithm)
    step = int(global_step)
    agent.save(directory, step)
    path = checkpoint_path(directory, algorithm, step)
    if not os.path.isfile(path):
        raise FileNotFoundError("agent did not create expected checkpoint: {}".format(path))

    payload = torch_load(path, map_location="cpu")
    payload["_carla_rl_lab"] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": algorithm,
        "global_step": step,
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "config": jsonable_config(cfg),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        },
    }
    payload["_trainer_state"] = dict(trainer_state or {})
    payload["_rng_state"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    _atomic_torch_save(payload, path)

    last_path = checkpoint_path(directory, algorithm, "last")
    _atomic_copy(path, last_path)
    digest = _sha256(path)
    _update_manifest(
        directory,
        algorithm,
        path,
        last_path,
        step,
        digest,
        max(1, int(getattr(cfg, "checkpoint_keep", 5))),
    )
    return path


def _atomic_torch_save(payload: Mapping[str, Any], path: str) -> None:
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".checkpoint-", suffix=".tmp", dir=os.path.dirname(path)
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_copy(source: str, destination: str) -> None:
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".checkpoint-copy-",
        suffix=".tmp",
        dir=os.path.dirname(destination),
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_manifest(
    directory: str,
    algorithm: str,
    path: str,
    last_path: str,
    step: int,
    digest: str,
    keep: int,
) -> None:
    manifest_path = os.path.join(directory, "checkpoint_manifest.json")
    manifest: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": algorithm,
        "checkpoints": [],
    }
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r") as manifest_file:
                previous = json.load(manifest_file)
            if previous.get("algorithm") == algorithm:
                manifest = previous
        except (OSError, ValueError):
            pass

    records = [
        record
        for record in manifest.get("checkpoints", [])
        if int(record.get("global_step", -1)) != step
    ]
    records.append(
        {
            "global_step": step,
            "file": os.path.basename(path),
            "sha256": digest,
        }
    )
    records.sort(key=lambda item: int(item["global_step"]))
    while len(records) > keep:
        removed = records.pop(0)
        removed_path = os.path.join(directory, removed["file"])
        if os.path.isfile(removed_path) and os.path.abspath(removed_path) != os.path.abspath(path):
            os.unlink(removed_path)
    manifest.update(
        {
            "updated_at": utc_timestamp(),
            "latest": os.path.basename(last_path),
            "checkpoints": records,
        }
    )
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".manifest-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
        os.replace(temporary_path, manifest_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
