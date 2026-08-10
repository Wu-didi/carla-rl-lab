from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.algorithms import create_agent, get_algorithm, list_algorithms
from carla_rl_lab.buffers import ReplayBuffer
from carla_rl_lab.config import Config
from carla_rl_lab.envs import make_carla_env
from carla_rl_lab.logging import ExperimentLogger, build_experiment_logger
from carla_rl_lab.observations import encode_observation
from carla_rl_lab.rewards import list_reward_profiles
from carla_rl_lab.utils import set_seed


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs_dir(cfg: Config) -> str:
    run_name = cfg.run_name or "{}_seed{}".format(cfg.algorithm, cfg.seed)
    return os.path.join(project_root(), "artifacts", "runs", run_name)


def make_agent(cfg: Config):
    spec = get_algorithm(cfg.algorithm)
    if spec.runner != "off_policy":
        raise ValueError(
            "Algorithm '{}' requires runner='{}', but this script provides "
            "runner='off_policy'.".format(cfg.algorithm, spec.runner)
        )
    return create_agent(cfg.algorithm, cfg)


def log_losses(
    logger: ExperimentLogger,
    losses: Dict[str, Any],
    step: int,
    log_attention_image: bool,
) -> None:
    metrics = {}
    for name, value in losses.items():
        if name == "attention_img" and log_attention_image:
            logger.log_image("attention/alpha_heatmap", value, step)
        elif name != "attention_img":
            metrics["train/{}".format(name)] = float(value)
    if metrics:
        logger.log(metrics, step)


def log_action_metrics(
    logger: ExperimentLogger, actions: List[np.ndarray], step: int
) -> None:
    if not actions:
        return
    action_array = np.stack(actions)
    metrics = {}
    for index, name in enumerate(("throttle", "steer", "brake")):
        values = action_array[:, index]
        metrics["action/{}_mean".format(name)] = float(values.mean())
        metrics["action/{}_std".format(name)] = float(values.std())
        metrics["action/{}_min".format(name)] = float(values.min())
        metrics["action/{}_max".format(name)] = float(values.max())
    logger.log(metrics, step)


def save_checkpoint(cfg: Config, agent: Any, step: int) -> None:
    checkpoint_dir = os.path.join(runs_dir(cfg), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "gstep_last.txt"), "w") as step_file:
        step_file.write(str(step))
    agent.save(checkpoint_dir, step_id="last")


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    log_dir = runs_dir(cfg)
    os.makedirs(log_dir, exist_ok=True)
    logger = build_experiment_logger(cfg, log_dir, asdict(cfg))
    print("Experiment logs -> {} ({})".format(log_dir, cfg.logger_backend))

    env = None
    try:
        env = make_carla_env(cfg)
        agent = make_agent(cfg)
        replay_buffer = ReplayBuffer(cfg.buffer_size)

        if cfg.use_pretrained_model:
            if not os.path.isfile(cfg.pretrained_model_path):
                raise FileNotFoundError(
                    "Checkpoint not found: {}".format(cfg.pretrained_model_path)
                )
            agent.load(cfg.pretrained_model_path)
            print("Loaded model {}".format(cfg.pretrained_model_path))

        global_step = 0
        for episode in range(cfg.max_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_cost = 0.0
            episode_steps = 0
            episode_actions = []

            while not done:
                obs_vector = encode_observation(
                    obs, cfg.state_dim, cfg.risk_field_sectors
                )
                action = agent.act(obs_vector)
                episode_actions.append(action)

                try:
                    next_obs, reward, cost, done, info = env.step(action)
                except Exception:
                    traceback.print_exc()
                    print("CARLA step failed; resetting environment")
                    obs = env.reset()
                    continue

                next_obs_vector = encode_observation(
                    next_obs, cfg.state_dim, cfg.risk_field_sectors
                )
                replay_buffer.add(
                    obs_vector, action, reward, next_obs_vector, done
                )

                ready_size = max(cfg.minimal_size, cfg.batch_size)
                if replay_buffer.size() >= ready_size and cfg.train_every_step:
                    losses = agent.update(replay_buffer.sample(cfg.batch_size))
                    log_losses(
                        logger, losses, global_step, cfg.log_attention_image
                    )

                reward_terms = {
                    name: float(value)
                    for name, value in info.get("reward_terms", {}).items()
                }
                if reward_terms:
                    logger.log(reward_terms, global_step)

                obs = next_obs
                episode_reward += float(reward)
                episode_cost += float(cost)
                episode_steps += 1
                global_step += 1

            logger.log(
                {
                    "episode/reward": episode_reward,
                    "episode/cost": episode_cost,
                    "episode/length": float(episode_steps),
                    "episode/index": float(episode),
                },
                global_step,
            )
            log_action_metrics(logger, episode_actions, global_step)
            save_checkpoint(cfg, agent, global_step)
            print(
                "[Episode {:03d}] Reward={:.2f} Steps={} gstep={}".format(
                    episode, episode_reward, episode_steps, global_step
                )
            )
    finally:
        if env is not None:
            try:
                env.close()
                print("Cleared all CARLA actors")
            except Exception:
                traceback.print_exc()
        logger.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CarlaRLLab off-policy trainer")
    parser.add_argument(
        "--algorithm",
        "--algo",
        dest="algorithm",
        default=Config.algorithm,
        choices=list(list_algorithms()),
    )
    parser.add_argument("--town", default=Config.town)
    parser.add_argument("--port", type=int, default=Config.port)
    parser.add_argument(
        "--network", default=Config.network, choices=["SAC", "Attention_SAC"]
    )
    parser.add_argument(
        "--max-episodes",
        "--max_episodes",
        type=int,
        default=Config.max_episodes,
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--reward",
        dest="reward_profile",
        default=Config.reward_profile,
        choices=list(list_reward_profiles()),
    )
    parser.add_argument(
        "--logger",
        dest="logger_backend",
        default=Config.logger_backend,
        choices=["tensorboard", "wandb", "both", "none"],
    )
    parser.add_argument("--run-name", default=Config.run_name)
    parser.add_argument(
        "--wandb-mode",
        default=Config.wandb_mode,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--checkpoint",
        default=Config.pretrained_model_path,
        help="Optional algorithm checkpoint to resume from",
    )
    return parser


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.algorithm = args.algorithm
    cfg.town = args.town
    cfg.port = args.port
    cfg.network = args.network
    cfg.max_episodes = args.max_episodes
    cfg.seed = args.seed
    cfg.reward_profile = args.reward_profile
    cfg.logger_backend = args.logger_backend
    cfg.run_name = args.run_name
    cfg.wandb_mode = args.wandb_mode
    cfg.pretrained_model_path = args.checkpoint
    cfg.use_pretrained_model = bool(args.checkpoint)
    return cfg


def main() -> None:
    args = build_argparser().parse_args()
    cfg = apply_overrides(Config(), args)
    print("[Config]", asdict(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
