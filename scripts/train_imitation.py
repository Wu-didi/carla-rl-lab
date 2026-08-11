from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.algorithms import create_agent, get_algorithm, list_algorithms
from carla_rl_lab.buffers import OfflineDataset, RolloutBuffer
from carla_rl_lab.config import Config
from carla_rl_lab.envs import make_carla_env
from carla_rl_lab.logging import build_experiment_logger
from carla_rl_lab.observations import encode_observation
from carla_rl_lab.rewards import list_reward_profiles
from carla_rl_lab.utils import set_seed


def imitation_algorithms():
    return [name for name in list_algorithms() if get_algorithm(name).runner == "imitation"]


def output_paths(cfg: Config):
    run_name = cfg.run_name or "{}_seed{}".format(cfg.algorithm, cfg.seed)
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
        "runs",
        run_name,
    )
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return log_dir, checkpoint_dir


def train_bc(cfg: Config, dataset: OfflineDataset, logger, checkpoint_dir: str) -> None:
    agent = create_agent("bc", cfg)
    if cfg.use_pretrained_model:
        agent.load(cfg.pretrained_model_path)
    for update in range(1, cfg.imitation_updates + 1):
        losses = agent.update(dataset.sample(cfg.batch_size, ("states", "actions")))
        logger.log(
            {"train/{}".format(name): float(value) for name, value in losses.items()},
            update,
        )
        if update % cfg.checkpoint_interval == 0 or update == cfg.imitation_updates:
            agent.save(checkpoint_dir, "last")
            print("[Update {:07d}] algorithm=bc".format(update))


def train_adversarial(
    cfg: Config, dataset: OfflineDataset, logger, checkpoint_dir: str
) -> None:
    env = None
    try:
        env = make_carla_env(cfg)
        agent = create_agent(cfg.algorithm, cfg)
        if cfg.use_pretrained_model:
            agent.load(cfg.pretrained_model_path)
        observation = env.reset()
        state = encode_observation(
            observation, cfg.state_dim, cfg.risk_field_sectors
        )
        global_step = 0
        update_index = 0
        last_done = False

        while global_step < cfg.total_timesteps:
            rollout_size = min(cfg.rollout_steps, cfg.total_timesteps - global_step)
            rollout = RolloutBuffer(rollout_size, cfg.gamma, cfg.gae_lambda)
            for _ in range(rollout_size):
                action, log_prob, value = agent.act_with_info(state)
                try:
                    next_observation, reward, cost, done, info = env.step(action)
                except Exception:
                    traceback.print_exc()
                    observation = env.reset()
                    state = encode_observation(
                        observation, cfg.state_dim, cfg.risk_field_sectors
                    )
                    continue
                next_state = encode_observation(
                    next_observation, cfg.state_dim, cfg.risk_field_sectors
                )
                rollout.add(
                    state,
                    action,
                    reward,
                    done,
                    value,
                    log_prob,
                    next_state=next_state,
                )
                state = next_state
                last_done = bool(done)
                global_step += 1
                if done:
                    observation = env.reset()
                    state = encode_observation(
                        observation, cfg.state_dim, cfg.risk_field_sectors
                    )
                if global_step >= cfg.total_timesteps:
                    break

            if len(rollout) == 0:
                continue
            last_value = 0.0 if last_done else agent.value(state)
            batch = rollout.batch(last_value)
            expert_fields = ["states", "actions"]
            if cfg.algorithm == "airl":
                expert_fields.append("next_states")
            expert_batch = dataset.sample(len(rollout), expert_fields)
            for name, values in expert_batch.items():
                batch["expert_{}".format(name)] = values
            losses = agent.update(batch)
            logger.log(
                {"train/{}".format(name): float(value) for name, value in losses.items()},
                global_step,
            )
            update_index += 1
            if (
                global_step % cfg.checkpoint_interval < len(rollout)
                or global_step >= cfg.total_timesteps
            ):
                agent.save(checkpoint_dir, "last")
            print(
                "[Update {:05d}] algorithm={} steps={}".format(
                    update_index, cfg.algorithm, global_step
                )
            )
    finally:
        if env is not None:
            env.close()


def train(cfg: Config) -> None:
    if get_algorithm(cfg.algorithm).runner != "imitation":
        raise ValueError("{} does not use the imitation runner".format(cfg.algorithm))
    if not cfg.expert_dataset_path:
        raise ValueError("--expert-dataset is required")
    set_seed(cfg.seed)
    dataset = OfflineDataset.load(
        cfg.expert_dataset_path,
        require_transitions=cfg.algorithm == "airl",
        seed=cfg.seed,
    )
    if dataset.state_dim != cfg.state_dim or dataset.action_dim != cfg.action_dim:
        raise ValueError("expert dataset dimensions do not match Config")
    log_dir, checkpoint_dir = output_paths(cfg)
    logger = build_experiment_logger(cfg, log_dir, asdict(cfg))
    try:
        if cfg.algorithm == "bc":
            train_bc(cfg, dataset, logger, checkpoint_dir)
        else:
            train_adversarial(cfg, dataset, logger, checkpoint_dir)
    finally:
        logger.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CarlaRLLab imitation trainer")
    parser.add_argument("--algorithm", "--algo", choices=imitation_algorithms(), default="bc")
    parser.add_argument("--expert-dataset", dest="expert_dataset_path", required=True)
    parser.add_argument("--updates", dest="imitation_updates", type=int, default=Config.imitation_updates)
    parser.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    parser.add_argument("--rollout-steps", type=int, default=Config.rollout_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--checkpoint-interval", type=int, default=Config.checkpoint_interval)
    parser.add_argument("--town", default=Config.town)
    parser.add_argument("--port", type=int, default=Config.port)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--reward", dest="reward_profile", choices=list(list_reward_profiles()), default=Config.reward_profile)
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
