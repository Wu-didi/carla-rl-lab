from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: str = "") -> str:
    cwd = path or os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_is_dirty(path: str = "") -> bool:
    cwd = path or os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


def runtime_environment() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }


def jsonable_config(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        values = asdict(cfg)
    else:
        values = dict(vars(cfg))
    return json.loads(json.dumps(values, default=str))


def carla_versions(env: Any) -> Dict[str, str]:
    versions = {"client": "unknown", "server": "unknown"}
    try:
        versions["client"] = str(env.client.get_client_version())
    except (AttributeError, RuntimeError):
        pass
    try:
        versions["server"] = str(env.client.get_server_version())
    except (AttributeError, RuntimeError):
        pass
    return versions
