from __future__ import annotations

import json
from typing import Any, Dict, Optional


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

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


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
