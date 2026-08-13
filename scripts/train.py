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
from carla_rl_lab.buffers import OfflineDataset, ReplayBuffer
from carla_rl_lab.benchmarks import apply_benchmark, get_benchmark, list_benchmarks
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
from carla_rl_lab.utils.provenance import carla_versions, file_sha256, git_is_dirty


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


def load_expert_dataset(cfg: Config):
    if not cfg.expert_dataset_path:
        if cfg.demo_pretrain_updates > 0 or cfg.demo_bc_coef > 0.0:
            raise ValueError(
                "--expert-dataset is required when demonstration learning is enabled"
            )
        return None, None
    if cfg.algorithm != "sac":
        raise ValueError("online demonstration learning currently supports SAC only")
    dataset = OfflineDataset.load(
        cfg.expert_dataset_path, require_transitions=False, seed=cfg.seed
    )
    if dataset.state_dim != cfg.state_dim or dataset.action_dim != cfg.action_dim:
        raise ValueError("expert dataset dimensions do not match Config")
    action_mode = dataset.metadata.get("action_mode")
    if action_mode and action_mode != cfg.action_mode:
        raise ValueError("expert dataset action_mode does not match Config")
    record = {
        "path": os.path.relpath(
            os.path.abspath(cfg.expert_dataset_path), project_root()
        ),
        "sha256": file_sha256(cfg.expert_dataset_path),
        "metadata": dataset.metadata,
    }
    return dataset, record


def select_action(agent: Any, cfg: Config, obs_vector: np.ndarray, replay_size: int):
    if replay_size < cfg.minimal_size:
        action = np.random.uniform(
            -cfg.action_bound, cfg.action_bound, size=cfg.action_dim
        )
        return action.astype(np.float32), True
    return agent.act(obs_vector), False


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


def save_checkpoint(
    agent,
    cfg: Config,
    checkpoint_dir: str,
    global_step: int,
    episode: int,
    env,
    replay_buffer: ReplayBuffer,
    expert_dataset_record=None,
) -> None:
    trainer_state = {
        "episode": episode,
        "carla_versions": carla_versions(env),
    }
    if cfg.checkpoint_replay_buffer:
        trainer_state["replay_buffer"] = replay_buffer.state_dict()
    if expert_dataset_record is not None:
        trainer_state["expert_dataset"] = expert_dataset_record
    save_training_checkpoint(
        agent, cfg, checkpoint_dir, global_step, trainer_state
    )


def train(cfg: Config) -> None:
    if cfg.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if cfg.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if cfg.max_step_retries <= 0:
        raise ValueError("max_step_retries must be positive")
    if cfg.demo_pretrain_updates < 0:
        raise ValueError("demo_pretrain_updates cannot be negative")
    if cfg.demo_bc_coef < 0.0:
        raise ValueError("demo_bc_coef cannot be negative")
    if cfg.demo_bc_mode not in ("fixed", "adaptive"):
        raise ValueError("demo_bc_mode must be 'fixed' or 'adaptive'")
    if cfg.demo_q_temperature <= 0.0:
        raise ValueError("demo_q_temperature must be positive")
    if cfg.demo_advantage_beta < 0.0:
        raise ValueError("demo_advantage_beta cannot be negative")
    if cfg.demo_uncertainty_beta < 0.0:
        raise ValueError("demo_uncertainty_beta cannot be negative")
    if not 0.0 <= cfg.demo_bc_weight_min <= cfg.demo_bc_weight_max:
        raise ValueError("invalid adaptive demonstration weight bounds")
    if cfg.actor_update_mode not in ("standard", "confidence"):
        raise ValueError("actor_update_mode must be 'standard' or 'confidence'")
    if cfg.actor_uncertainty_beta < 0.0:
        raise ValueError("actor_uncertainty_beta cannot be negative")
    if not 0.0 <= cfg.actor_confidence_min <= 1.0:
        raise ValueError("actor_confidence_min must be in [0, 1]")
    if cfg.require_clean_git and git_is_dirty(project_root()):
        raise RuntimeError(
            "Public runs require a clean git worktree; commit or stash changes first"
        )
    set_seed(cfg.seed)
    expert_dataset, expert_dataset_record = load_expert_dataset(cfg)
    log_dir = runs_dir(cfg)
    os.makedirs(log_dir, exist_ok=True)
    logger = build_experiment_logger(cfg, log_dir, asdict(cfg))
    print("Experiment logs -> {} ({})".format(log_dir, cfg.logger_backend))

    env = None
    global_step = 0
    last_episode = -1
    try:
        env = make_carla_env(cfg)
        logger.update_run_record(
            {"status": "running", "carla_versions": carla_versions(env)}
        )
        agent = make_agent(cfg)
        replay_buffer = ReplayBuffer(cfg.buffer_size)
        start_episode = 0

        if expert_dataset_record is not None:
            logger.update_run_record({"expert_dataset": expert_dataset_record})

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
            else:
                print(
                    "Checkpoint has no replay buffer; online data collection "
                    "will warm up again before updates resume."
                )
            print("Loaded model {}".format(cfg.pretrained_model_path))

        if cfg.demo_pretrain_updates > 0 and not cfg.use_pretrained_model:
            print(
                "Pretraining SAC actor from {} expert samples for {} updates".format(
                    len(expert_dataset), cfg.demo_pretrain_updates
                )
            )
            for update_index in range(1, cfg.demo_pretrain_updates + 1):
                losses = agent.behavior_clone(
                    expert_dataset.sample(
                        cfg.batch_size, fields=("states", "actions")
                    )
                )
                logger.log(
                    {
                        "pretrain/{}".format(name): float(value)
                        for name, value in losses.items()
                    },
                    update_index,
                )
                if update_index % 500 == 0:
                    print(
                        "[Demo pretrain {}/{}] bc_loss={:.5f}".format(
                            update_index,
                            cfg.demo_pretrain_updates,
                            losses["bc_loss"],
                        )
                    )

        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        last_checkpoint_step = global_step
        last_episode = start_episode - 1
        for episode in range(start_episode, cfg.max_episodes):
            if global_step >= cfg.total_timesteps:
                break
            last_episode = episode
            obs = env.reset(seed=cfg.seed + episode)
            done = False
            episode_reward = 0.0
            episode_cost = 0.0
            episode_steps = 0
            episode_actions = []
            episode_random_actions = 0
            consecutive_step_failures = 0
            info: Dict[str, Any] = {}

            while not done and global_step < cfg.total_timesteps:
                obs_vector = encode_observation(obs, cfg.state_dim)
                action, random_warmup = select_action(
                    agent, cfg, obs_vector, replay_buffer.size()
                )
                episode_random_actions += int(random_warmup)

                try:
                    next_obs, reward, cost, done, info = env.step(action)
                except Exception as exc:
                    traceback.print_exc()
                    consecutive_step_failures += 1
                    if consecutive_step_failures >= cfg.max_step_retries:
                        raise RuntimeError(
                            "CARLA step failed {} consecutive times".format(
                                consecutive_step_failures
                            )
                        ) from exc
                    print("CARLA step failed; resetting environment")
                    obs = env.reset(seed=cfg.seed + episode)
                    continue
                consecutive_step_failures = 0
                episode_actions.append(action)

                next_obs_vector = encode_observation(next_obs, cfg.state_dim)
                terminal = bool(
                    done and info.get("termination_reason") != "timeout"
                )
                replay_buffer.add(
                    obs_vector, action, reward, next_obs_vector, terminal
                )
                obs = next_obs
                episode_reward += float(reward)
                episode_cost += float(cost)
                episode_steps += 1
                global_step += 1

                ready_size = max(cfg.minimal_size, cfg.batch_size)
                if replay_buffer.size() >= ready_size and cfg.train_every_step:
                    if expert_dataset is not None and cfg.demo_bc_coef > 0.0:
                        losses = agent.update(
                            replay_buffer.sample(cfg.batch_size),
                            expert_batch=expert_dataset.sample(
                                cfg.batch_size, fields=("states", "actions")
                            ),
                            bc_coef=cfg.demo_bc_coef,
                            bc_mode=cfg.demo_bc_mode,
                        )
                    else:
                        losses = agent.update(
                            replay_buffer.sample(cfg.batch_size)
                        )
                    log_losses(
                        logger, losses, global_step, cfg.log_attention_image
                    )

                reward_terms = {
                    name: float(value)
                    for name, value in info.get("reward_terms", {}).items()
                }
                if reward_terms:
                    logger.log(reward_terms, global_step)

                if global_step - last_checkpoint_step >= cfg.checkpoint_interval:
                    save_checkpoint(
                        agent,
                        cfg,
                        checkpoint_dir,
                        global_step,
                        episode,
                        env,
                        replay_buffer,
                        expert_dataset_record,
                    )
                    last_checkpoint_step = global_step

            logger.log(
                {
                    "episode/reward": episode_reward,
                    "episode/cost": episode_cost,
                    "episode/length": float(episode_steps),
                    "episode/index": float(episode),
                    "episode/truncated_by_budget": float(not done),
                    "traffic/requested_vehicles": float(
                        info.get("requested_vehicles", 0)
                    ),
                    "traffic/spawned_vehicles": float(
                        info.get("spawned_vehicles", 0)
                    ),
                    "traffic/requested_walkers": float(
                        info.get("requested_walkers", 0)
                    ),
                    "traffic/spawned_walkers": float(
                        info.get("spawned_walkers", 0)
                    ),
                    "events/collisions": float(info.get("collision_count", 0)),
                    "events/red_lights": float(info.get("red_light_count", 0)),
                    "episode/route_completion": float(
                        info.get("route_completion", 0.0)
                    ),
                    "exploration/random_warmup_rate": float(
                        episode_random_actions
                    )
                    / max(episode_steps, 1),
                },
                global_step,
            )
            log_action_metrics(logger, episode_actions, global_step)
            print(
                "[Episode {:03d}] Reward={:.2f} Steps={} gstep={}".format(
                    episode, episode_reward, episode_steps, global_step
                )
            )
        if global_step > 0 and global_step != last_checkpoint_step:
            save_checkpoint(
                agent,
                cfg,
                checkpoint_dir,
                global_step,
                last_episode,
                env,
                replay_buffer,
                expert_dataset_record,
            )
        logger.finish(
            "completed", global_step=global_step, last_episode=last_episode
        )
    except BaseException as exc:
        logger.finish(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            global_step=global_step,
            last_episode=last_episode,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
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
    parser.add_argument("--benchmark", choices=list(list_benchmarks()), default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--minimal-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--expert-dataset", dest="expert_dataset_path", default=None
    )
    parser.add_argument("--demo-pretrain-updates", type=int, default=None)
    parser.add_argument("--demo-bc-coef", type=float, default=None)
    parser.add_argument(
        "--demo-bc-mode", choices=["fixed", "adaptive"], default=None
    )
    parser.add_argument("--demo-q-temperature", type=float, default=None)
    parser.add_argument("--demo-advantage-beta", type=float, default=None)
    parser.add_argument("--demo-uncertainty-beta", type=float, default=None)
    parser.add_argument("--demo-bc-weight-min", type=float, default=None)
    parser.add_argument("--demo-bc-weight-max", type=float, default=None)
    parser.add_argument(
        "--actor-update-mode",
        choices=["standard", "confidence"],
        default=None,
    )
    parser.add_argument("--actor-uncertainty-beta", type=float, default=None)
    parser.add_argument("--actor-confidence-min", type=float, default=None)
    parser.add_argument(
        "--vehicles", dest="number_of_vehicles", type=int, default=None
    )
    parser.add_argument(
        "--walkers", dest="number_of_walkers", type=int, default=None
    )
    parser.add_argument(
        "--view-mode", choices=["none", "top", "follow"], default=None
    )
    parser.add_argument("--traffic", choices=["on", "off"], default=None)
    parser.add_argument("--max-time-episode", type=int, default=None)
    parser.add_argument(
        "--network", default=None, choices=["SAC", "Pixel_SAC"]
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
    parser.add_argument(
        "--checkpoint-replay-buffer",
        action="store_true",
        default=None,
        help="Store replay data in checkpoints for exact online resume",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        default=None,
        help="Refuse to start when the git worktree has uncommitted changes",
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
    if args.benchmark:
        apply_benchmark(cfg, get_benchmark(args.benchmark))
    for name in (
        "algorithm",
        "town",
        "port",
        "total_timesteps",
        "checkpoint_interval",
        "minimal_size",
        "batch_size",
        "buffer_size",
        "hidden_dim",
        "expert_dataset_path",
        "demo_pretrain_updates",
        "demo_bc_coef",
        "demo_bc_mode",
        "demo_q_temperature",
        "demo_advantage_beta",
        "demo_uncertainty_beta",
        "demo_bc_weight_min",
        "demo_bc_weight_max",
        "actor_update_mode",
        "actor_uncertainty_beta",
        "actor_confidence_min",
        "number_of_vehicles",
        "number_of_walkers",
        "view_mode",
        "traffic",
        "max_time_episode",
        "network",
        "action_mode",
        "max_episodes",
        "seed",
        "reward_profile",
        "logger_backend",
        "run_name",
        "wandb_mode",
        "checkpoint_replay_buffer",
        "require_clean_git",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.action_mode is not None:
        cfg.action_dim = 3 if cfg.action_mode == "signed_3d" else 2
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
