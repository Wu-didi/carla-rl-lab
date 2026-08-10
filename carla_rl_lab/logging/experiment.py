from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional


class ExperimentLogger(ABC):
    @abstractmethod
    def log(self, metrics: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_image(self, name: str, image: Any, step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class NullLogger(ExperimentLogger):
    def log(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_image(self, name: str, image: Any, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(ExperimentLogger):
    def __init__(self, log_dir: str, config: Dict[str, Any]):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)
        self.writer.add_text("config", "```json\n{}\n```".format(json.dumps(config, indent=2, default=str)), 0)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, float(value), step)

    def log_image(self, name: str, image: Any, step: int) -> None:
        self.writer.add_image(name, image, global_step=step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class WandbLogger(ExperimentLogger):
    def __init__(
        self,
        project: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        entity: Optional[str] = None,
        mode: str = "offline",
        log_dir: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requested but 'wandb' is not installed. "
                "Install it with: pip install wandb"
            ) from exc

        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity or None,
            name=name or None,
            config=config,
            mode=mode,
            dir=log_dir,
        )

    def log(self, metrics: Dict[str, float], step: int) -> None:
        self.run.log(dict(metrics), step=step)

    def log_image(self, name: str, image: Any, step: int) -> None:
        self.run.log({name: self.wandb.Image(image)}, step=step)

    def close(self) -> None:
        self.run.finish()


class CompositeLogger(ExperimentLogger):
    def __init__(self, loggers: Iterable[ExperimentLogger]):
        self.loggers = list(loggers)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log(metrics, step)

    def log_image(self, name: str, image: Any, step: int) -> None:
        for logger in self.loggers:
            logger.log_image(name, image, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def build_experiment_logger(cfg: Any, log_dir: str, config: Dict[str, Any]) -> ExperimentLogger:
    backend = cfg.logger_backend.lower()
    if backend == "none":
        return NullLogger()

    loggers = []
    if backend in ("tensorboard", "both"):
        loggers.append(TensorBoardLogger(log_dir, config))
    if backend in ("wandb", "both"):
        loggers.append(
            WandbLogger(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.run_name,
                mode=cfg.wandb_mode,
                config=config,
                log_dir=log_dir,
            )
        )
    if not loggers:
        raise ValueError("Unknown logger backend: {}".format(cfg.logger_backend))
    if len(loggers) == 1:
        return loggers[0]
    return CompositeLogger(loggers)
