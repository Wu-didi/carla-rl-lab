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
from carla_rl_lab.utils import set_seed


def offline_algorithms():
    return [name for name in list_algorithms() if get_algorithm(name).runner == "offline"]


def train(cfg: Config) -> None:
    if get_algorithm(cfg.algorithm).runner != "offline":
        raise ValueError("{} does not use the offline runner".format(cfg.algorithm))
    if not cfg.dataset_path:
        raise ValueError("--dataset is required")
    set_seed(cfg.seed)
    dataset = OfflineDataset.load(cfg.dataset_path, seed=cfg.seed)
    if dataset.state_dim != cfg.state_dim or dataset.action_dim != cfg.action_dim:
        raise ValueError(
            "dataset dimensions are state_dim={}, action_dim={}; config expects {}, {}".format(
                dataset.state_dim, dataset.action_dim, cfg.state_dim, cfg.action_dim
            )
        )

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
        if cfg.use_pretrained_model:
            agent.load(cfg.pretrained_model_path)
        for update in range(1, cfg.offline_updates + 1):
            losses = agent.update(dataset.sample(cfg.batch_size))
            logger.log(
                {"train/{}".format(name): float(value) for name, value in losses.items()},
                update,
            )
            if update % cfg.checkpoint_interval == 0 or update == cfg.offline_updates:
                agent.save(checkpoint_dir, "last")
                print("[Update {:07d}] algorithm={}".format(update, cfg.algorithm))
    finally:
        logger.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CarlaRLLab offline RL trainer")
    parser.add_argument("--algorithm", "--algo", choices=offline_algorithms(), default="td3_bc")
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--updates", dest="offline_updates", type=int, default=Config.offline_updates)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--checkpoint-interval", type=int, default=Config.checkpoint_interval)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--logger", dest="logger_backend", choices=["tensorboard", "wandb", "both", "none"], default=Config.logger_backend)
    parser.add_argument("--run-name", default=Config.run_name)
    parser.add_argument("--checkpoint", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config()
    for name, value in vars(args).items():
        if name != "checkpoint":
            setattr(cfg, name, value)
    cfg.pretrained_model_path = args.checkpoint
    cfg.use_pretrained_model = bool(args.checkpoint)
    print("[Config]", asdict(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
