from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.algorithms import create_agent, get_algorithm, list_algorithms
from carla_rl_lab.buffers import RolloutBuffer
from carla_rl_lab.benchmarks import apply_benchmark, get_benchmark, list_benchmarks
from carla_rl_lab.config import Config
from carla_rl_lab.envs import ACTION_MODES, make_carla_env
from carla_rl_lab.logging import action_metrics, build_experiment_logger
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


def on_policy_algorithms():
    return [name for name in list_algorithms() if get_algorithm(name).runner == "on_policy"]


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(cfg: Config) -> None:
    if get_algorithm(cfg.algorithm).runner != "on_policy":
        raise ValueError("{} does not use the on_policy runner".format(cfg.algorithm))
    if cfg.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if cfg.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if cfg.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if cfg.max_step_retries <= 0:
        raise ValueError("max_step_retries must be positive")
    set_seed(cfg.seed)
    run_name = cfg.run_name or "{}_seed{}".format(cfg.algorithm, cfg.seed)
    log_dir = os.path.join(project_root(), "artifacts", "runs", run_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = build_experiment_logger(cfg, log_dir, asdict(cfg))
    env = None
    try:
        env = make_carla_env(cfg)
        agent = create_agent(cfg.algorithm, cfg)
        global_step = 0
        update_index = 0
        episode_index = 0
        if cfg.use_pretrained_model:
            agent.load(cfg.pretrained_model_path)
            metadata = checkpoint_metadata(cfg.pretrained_model_path)
            trainer_state = restore_training_state(cfg.pretrained_model_path)
            global_step = int(metadata.get("global_step", 0))
            update_index = int(trainer_state.get("update_index", 0))
            episode_index = int(trainer_state.get("episode_index", 0))

        observation = env.reset(seed=cfg.seed + episode_index)
        state = encode_observation(observation, cfg.state_dim)
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        last_done = False
        consecutive_step_failures = 0

        while global_step < cfg.total_timesteps:
            rollout_size = min(cfg.rollout_steps, cfg.total_timesteps - global_step)
            rollout = RolloutBuffer(rollout_size, cfg.gamma, cfg.gae_lambda)
            for _ in range(rollout_size):
                action, log_prob, value = agent.act_with_info(state)
                try:
                    next_observation, reward, cost, done, info = env.step(action)
                except Exception as exc:
                    traceback.print_exc()
                    consecutive_step_failures += 1
                    if consecutive_step_failures >= cfg.max_step_retries:
                        raise RuntimeError(
                            "CARLA step failed {} consecutive times".format(
                                consecutive_step_failures
                            )
                        ) from exc
                    rollout.end_episode(next_value=agent.value(state), terminal=False)
                    last_done = True
                    episode_index += 1
                    observation = env.reset(seed=cfg.seed + episode_index)
                    state = encode_observation(observation, cfg.state_dim)
                    episode_return = 0.0
                    episode_cost = 0.0
                    episode_length = 0
                    continue
                consecutive_step_failures = 0
                next_state = encode_observation(next_observation, cfg.state_dim)
                terminal = bool(
                    done and info.get("termination_reason") != "timeout"
                )
                timeout_value = (
                    agent.value(next_state) if done and not terminal else None
                )
                rollout.add(
                    state,
                    action,
                    reward,
                    done,
                    value,
                    log_prob,
                    next_state=next_state,
                    terminal=terminal,
                    next_value=timeout_value,
                )
                global_step += 1
                episode_return += float(reward)
                episode_cost += float(cost)
                episode_length += 1
                last_done = bool(done)
                state = next_state

                reward_terms = info.get("reward_terms", {})
                if reward_terms:
                    logger.log(reward_terms, global_step)
                if done:
                    logger.log(
                        {
                            "episode/reward": episode_return,
                            "episode/cost": episode_cost,
                            "episode/length": float(episode_length),
                            "episode/index": float(episode_index),
                        },
                        global_step,
                    )
                    episode_index += 1
                    episode_return = 0.0
                    episode_cost = 0.0
                    episode_length = 0
                    observation = env.reset(seed=cfg.seed + episode_index)
                    state = encode_observation(observation, cfg.state_dim)
                if global_step >= cfg.total_timesteps:
                    break

            if len(rollout) == 0:
                continue
            last_value = 0.0 if last_done else agent.value(state)
            batch = rollout.batch(last_value)
            losses = agent.update(batch)
            logger.log(
                {"train/{}".format(name): float(value) for name, value in losses.items()},
                global_step,
            )
            logger.log(action_metrics(batch["actions"]), global_step)
            update_index += 1
            if (
                global_step % cfg.checkpoint_interval < len(rollout)
                or global_step >= cfg.total_timesteps
            ):
                save_training_checkpoint(
                    agent,
                    cfg,
                    checkpoint_dir,
                    global_step,
                    {
                        "update_index": update_index,
                        "episode_index": episode_index if last_done else episode_index + 1,
                        "carla_versions": carla_versions(env),
                    },
                )
            print(
                "[Update {:05d}] algorithm={} steps={} rollout={}".format(
                    update_index, cfg.algorithm, global_step, len(rollout)
                )
            )
    finally:
        if env is not None:
            env.close()
        logger.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CarlaRLLab on-policy trainer")
    parser.add_argument("--algorithm", "--algo", choices=on_policy_algorithms(), default=None)
    parser.add_argument("--benchmark", choices=list(list_benchmarks()), default=None)
    parser.add_argument("--town", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--ppo-minibatch-size", type=int, default=None)
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
    parser.add_argument("--action-mode", choices=ACTION_MODES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reward", dest="reward_profile", choices=list(list_reward_profiles()), default=None)
    parser.add_argument("--logger", dest="logger_backend", choices=["tensorboard", "wandb", "both", "none"], default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
    )
    parser.add_argument("--checkpoint", default="")
    return parser


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.algorithm = "ppo"
    if args.checkpoint:
        apply_checkpoint_config(cfg, args.checkpoint)
        saved_algorithm = checkpoint_metadata(args.checkpoint).get("algorithm")
        if args.algorithm and saved_algorithm and args.algorithm != saved_algorithm:
            raise ValueError("algorithm does not match checkpoint metadata")
    if args.benchmark:
        apply_benchmark(cfg, get_benchmark(args.benchmark))
    for name, value in vars(args).items():
        if name not in ("benchmark", "checkpoint") and value is not None:
            setattr(cfg, name, value)
    if args.action_mode is not None:
        cfg.action_dim = 3 if cfg.action_mode == "signed_3d" else 2
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
