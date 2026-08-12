from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional

from carla_rl_lab.utils.provenance import (
    git_commit,
    git_is_dirty,
    runtime_environment,
    utc_timestamp,
)


class ExperimentLogger:
    """A thin owner for optional TensorBoard and W&B handles."""

    def __init__(
        self,
        backend: str,
        log_dir: str,
        config: Dict[str, Any],
        project: str = "carla-rl-lab",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        wandb_mode: str = "offline",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.run_record_path = os.path.join(log_dir, "run_config.json")
        self.run_record = {
            "schema_version": 2,
            "status": "created",
            "created_at": utc_timestamp(),
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "command": list(sys.argv),
            "cwd": os.getcwd(),
            "runtime": runtime_environment(),
            "config": config,
        }
        _write_json(self.run_record_path, self.run_record)
        self.writer = None
        self.wandb = None
        self.wandb_run = None
        backend = backend.lower()
        if backend not in ("none", "tensorboard", "wandb", "both"):
            raise ValueError("Unknown logger backend: {}".format(backend))

        if backend in ("tensorboard", "both"):
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
            config_text = json.dumps(config, indent=2, default=str)
            self.writer.add_text("config", "```json\n{}\n```".format(config_text), 0)

        if backend in ("wandb", "both"):
            try:
                import wandb
            except ImportError as exc:
                self.close()
                raise RuntimeError(
                    "W&B logging requested but 'wandb' is not installed. "
                    "Install it with: pip install -r requirements-wandb.txt"
                ) from exc
            self.wandb = wandb
            self.wandb_run = wandb.init(
                project=project,
                entity=entity or None,
                name=name or None,
                config=config,
                mode=wandb_mode,
                dir=log_dir,
            )

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.writer is not None:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(metric_name, float(value), step)
        if self.wandb_run is not None:
            self.wandb_run.log(dict(metrics), step=step)

    def log_image(self, name: str, image: Any, step: int) -> None:
        if self.writer is not None:
            self.writer.add_image(name, image, global_step=step)
        if self.wandb_run is not None:
            self.wandb_run.log({name: self.wandb.Image(image)}, step=step)

    def update_run_record(self, values: Dict[str, Any]) -> None:
        self.run_record.update(values)
        self.run_record["updated_at"] = utc_timestamp()
        _write_json(self.run_record_path, self.run_record)

    def finish(self, status: str, **details: Any) -> None:
        payload = dict(details)
        payload["status"] = status
        payload["finished_at"] = utc_timestamp()
        self.update_run_record(payload)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".run-config-", suffix=".tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True, default=str)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def action_metrics(actions: Any) -> Dict[str, float]:
    import numpy as np

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        return {}
    names = (
        ("longitudinal", "steer")
        if values.shape[1] == 2
        else ("throttle", "steer", "brake")
    )
    metrics = {}
    for index, name in enumerate(names):
        metrics["action/{}_mean".format(name)] = float(values[:, index].mean())
        metrics["action/{}_std".format(name)] = float(values[:, index].std())
        metrics["action/{}_min".format(name)] = float(values[:, index].min())
        metrics["action/{}_max".format(name)] = float(values[:, index].max())
    return metrics


def build_experiment_logger(
    cfg: Any, log_dir: str, config: Dict[str, Any]
) -> ExperimentLogger:
    return ExperimentLogger(
        backend=cfg.logger_backend,
        log_dir=log_dir,
        config=config,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        wandb_mode=cfg.wandb_mode,
    )
