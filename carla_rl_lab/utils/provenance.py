from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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
