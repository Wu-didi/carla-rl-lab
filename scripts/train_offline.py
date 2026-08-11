from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.algorithms import create_agent, get_algorithm, list_algorithms
from carla_rl_lab.buffers import OfflineDataset
from carla_rl_lab.config import Config
from carla_rl_lab.logging import build_experiment_logger
from carla_rl_lab.utils import (
    apply_checkpoint_config,
    checkpoint_metadata,
    restore_training_state,
    save_training_checkpoint,
    set_seed,
)


def offline_algorithms():
    return [name for name in list_algorithms() if get_algorithm(name).runner == "offline"]


def train(cfg: Config) -> None:
    if get_algorithm(cfg.algorithm).runner != "offline":
        raise ValueError("{} does not use the offline runner".format(cfg.algorithm))
    if not cfg.dataset_path:
        raise ValueError("--dataset is required")
    if cfg.offline_updates <= 0:
        raise ValueError("offline_updates must be positive")
    if cfg.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    set_seed(cfg.seed)
    dataset = OfflineDataset.load(cfg.dataset_path, seed=cfg.seed)
    if not cfg.use_pretrained_model:
        cfg.state_dim = dataset.state_dim
        cfg.action_dim = dataset.action_dim
        cfg.action_mode = dataset.metadata.get(
            "action_mode",
            "longitudinal_2d" if dataset.action_dim == 2 else cfg.action_mode,
        )
        source_config = dataset.metadata.get("config", {})
        cfg.risk_field_sectors = int(
            source_config.get("risk_field_sectors", cfg.risk_field_sectors)
        )
        cfg.max_waypoints = int(source_config.get("max_waypoints", cfg.max_waypoints))
    if dataset.state_dim != cfg.state_dim or dataset.action_dim != cfg.action_dim:
        raise ValueError(
            "dataset dimensions are state_dim={}, action_dim={}; config expects {}, {}".format(
                dataset.state_dim, dataset.action_dim, cfg.state_dim, cfg.action_dim
            )
        )
    dataset_action_mode = dataset.metadata.get("action_mode")
    if dataset_action_mode and dataset_action_mode != cfg.action_mode:
        raise ValueError(
            "dataset action_mode={} does not match config {}".format(
                dataset_action_mode, cfg.action_mode
            )
        )
    print("[Config]", asdict(cfg))

    run_name = cfg.run_name or "{}_seed{}".format(cfg.algorithm, cfg.seed)
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
        "runs",
        run_name,
    )
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = build_experiment_logger(cfg, log_dir, asdict(cfg))
    try:
        agent = create_agent(cfg.algorithm, cfg)
        start_update = 1
        if cfg.use_pretrained_model:
            agent.load(cfg.pretrained_model_path)
            metadata = checkpoint_metadata(cfg.pretrained_model_path)
            trainer_state = restore_training_state(cfg.pretrained_model_path)
            start_update = int(metadata.get("global_step", 0)) + 1
            if "dataset_rng_state" in trainer_state:
                dataset.rng.set_state(trainer_state["dataset_rng_state"])
        logger.log(
            {
                "dataset/size": float(len(dataset)),
                "dataset/terminal_rate": float(dataset.arrays["terminals"].mean()),
                "dataset/timeout_rate": float(dataset.arrays["timeouts"].mean()),
            },
            max(0, start_update - 1),
        )
        for update in range(start_update, cfg.offline_updates + 1):
            losses = agent.update(dataset.sample(cfg.batch_size))
            logger.log(
                {"train/{}".format(name): float(value) for name, value in losses.items()},
                update,
            )
            if update % cfg.checkpoint_interval == 0 or update == cfg.offline_updates:
                save_training_checkpoint(
                    agent,
                    cfg,
                    checkpoint_dir,
                    update,
                    {
                        "dataset_metadata": dataset.metadata,
                        "dataset_rng_state": dataset.rng.get_state(),
                    },
                )
                print("[Update {:07d}] algorithm={}".format(update, cfg.algorithm))
    finally:
        logger.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CarlaRLLab offline RL trainer")
    parser.add_argument("--algorithm", "--algo", choices=offline_algorithms(), default=None)
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--updates", dest="offline_updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--logger", dest="logger_backend", choices=["tensorboard", "wandb", "both", "none"], default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
    )
    parser.add_argument("--checkpoint", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config()
    cfg.algorithm = "td3_bc"
    if args.checkpoint:
        apply_checkpoint_config(cfg, args.checkpoint)
        saved_algorithm = checkpoint_metadata(args.checkpoint).get("algorithm")
        if args.algorithm and saved_algorithm and args.algorithm != saved_algorithm:
            raise ValueError("algorithm does not match checkpoint metadata")
    for name, value in vars(args).items():
        if name != "checkpoint" and value is not None:
            setattr(cfg, name, value)
    cfg.pretrained_model_path = args.checkpoint
    cfg.use_pretrained_model = bool(args.checkpoint)
    train(cfg)


if __name__ == "__main__":
    main()
