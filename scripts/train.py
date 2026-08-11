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
from carla_rl_lab.envs import ACTION_MODES, make_carla_env
from carla_rl_lab.logging import ExperimentLogger, build_experiment_logger
from carla_rl_lab.observations import encode_observation
from carla_rl_lab.rewards import list_reward_profiles
from carla_rl_lab.utils import (
    apply_checkpoint_config,
    checkpoint_metadata,
    restore_training_state,
    save_training_checkpoint,
    set_seed,
)
from carla_rl_lab.utils.provenance import carla_versions


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


def off_policy_algorithms():
    return [
        name for name in list_algorithms()
        if get_algorithm(name).runner == "off_policy"
    ]


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
    names = (
        ("longitudinal", "steer")
        if action_array.shape[1] == 2
        else ("throttle", "steer", "brake")
    )
    for index, name in enumerate(names):
        values = action_array[:, index]
        metrics["action/{}_mean".format(name)] = float(values.mean())
        metrics["action/{}_std".format(name)] = float(values.std())
        metrics["action/{}_min".format(name)] = float(values.min())
        metrics["action/{}_max".format(name)] = float(values.max())
    logger.log(metrics, step)


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
        global_step = 0
        start_episode = 0

        if cfg.use_pretrained_model:
            if not os.path.isfile(cfg.pretrained_model_path):
                raise FileNotFoundError(
                    "Checkpoint not found: {}".format(cfg.pretrained_model_path)
                )
            agent.load(cfg.pretrained_model_path)
            metadata = checkpoint_metadata(cfg.pretrained_model_path)
            trainer_state = restore_training_state(cfg.pretrained_model_path)
            global_step = int(metadata.get("global_step", 0))
            start_episode = int(trainer_state.get("episode", -1)) + 1
            replay_state = trainer_state.get("replay_buffer")
            if replay_state is not None:
                replay_buffer.load_state_dict(replay_state)
            print("Loaded model {}".format(cfg.pretrained_model_path))

        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        last_checkpoint_step = global_step
        for episode in range(start_episode, cfg.max_episodes):
            obs = env.reset(seed=cfg.seed + episode)
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
                terminal = bool(
                    done and info.get("termination_reason") != "timeout"
                )
                replay_buffer.add(
                    obs_vector, action, reward, next_obs_vector, terminal
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
            if (
                global_step - last_checkpoint_step >= cfg.checkpoint_interval
                or episode + 1 >= cfg.max_episodes
            ):
                trainer_state = {
                    "episode": episode,
                    "carla_versions": carla_versions(env),
                }
                if cfg.checkpoint_replay_buffer:
                    trainer_state["replay_buffer"] = replay_buffer.state_dict()
                save_training_checkpoint(
                    agent, cfg, checkpoint_dir, global_step, trainer_state
                )
                last_checkpoint_step = global_step
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
        default=None,
        choices=off_policy_algorithms(),
    )
    parser.add_argument("--town", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--network", default=None, choices=["SAC", "Attention_SAC"]
    )
    parser.add_argument("--action-mode", choices=ACTION_MODES, default=None)
    parser.add_argument(
        "--max-episodes",
        "--max_episodes",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--reward",
        dest="reward_profile",
        default=None,
        choices=list(list_reward_profiles()),
    )
    parser.add_argument(
        "--logger",
        dest="logger_backend",
        default=None,
        choices=["tensorboard", "wandb", "both", "none"],
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--checkpoint",
        default=Config.pretrained_model_path,
        help="Optional algorithm checkpoint to resume from",
    )
    return parser


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.checkpoint:
        apply_checkpoint_config(cfg, args.checkpoint)
        saved_algorithm = checkpoint_metadata(args.checkpoint).get("algorithm")
        if args.algorithm and saved_algorithm and args.algorithm != saved_algorithm:
            raise ValueError(
                "--algorithm={} does not match checkpoint algorithm={}".format(
                    args.algorithm, saved_algorithm
                )
            )
    for name in (
        "algorithm",
        "town",
        "port",
        "network",
        "action_mode",
        "max_episodes",
        "seed",
        "reward_profile",
        "logger_backend",
        "run_name",
        "wandb_mode",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.action_mode is not None:
        cfg.action_dim = 2 if cfg.action_mode == "longitudinal_2d" else 3
        if cfg.algorithm == "sac":
            cfg.target_entropy = -float(cfg.action_dim)
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
