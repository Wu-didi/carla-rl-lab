# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import traceback
from dataclasses import  asdict
from typing import List, Dict, Any

import numpy as np

# ----- 外部依赖（与你原项目一致） -----
from carla_rl_lab.config import Config
from carla_rl_lab.envs import make_carla_env
from carla_rl_lab.algorithms import create_agent, get_algorithm, list_algorithms
from carla_rl_lab.buffers import ReplayBuffer
from carla_rl_lab.logging import ExperimentLogger, build_experiment_logger
from carla_rl_lab.observations import VectorObservationAdapter
from carla_rl_lab.rewards import list_reward_profiles
from carla_rl_lab.utils import set_seed



# ===================== 实用工具 =====================


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs_dir(cfg: Config) -> str:
    run_name = cfg.run_name or '{}_seed{}'.format(cfg.algorithm, cfg.seed)
    return os.path.join(project_root(), 'artifacts', 'runs', run_name)


def params_dir(cfg) -> str:
    return os.path.join(runs_dir(cfg), 'checkpoints')


# ===================== 日志器 =====================
class CarlaLogger:
    def __init__(self, logger: ExperimentLogger):
        self.logger = logger
        self.ep_actions: List[np.ndarray] = []

    def step(self, action: np.ndarray) -> None:
        self.ep_actions.append(action)

    def episode_done(self, gstep: int) -> None:
        if not self.ep_actions:
            return
        ep_actions = np.stack(self.ep_actions)
        self.ep_actions.clear()
        names = ["throttle", "steer", "brake"]
        metrics = {}
        for i, name in enumerate(names):
            metrics[f"action/{name}_mean"] = float(ep_actions[:, i].mean())
            metrics[f"action/{name}_std"] = float(ep_actions[:, i].std())
            metrics[f"action/{name}_min"] = float(ep_actions[:, i].min())
            metrics[f"action/{name}_max"] = float(ep_actions[:, i].max())
        self.logger.log(metrics, gstep)

def make_agent(cfg: Config):
    spec = get_algorithm(cfg.algorithm)
    if spec.runner != 'off_policy':
        raise ValueError(
            f"Algorithm '{cfg.algorithm}' requires runner='{spec.runner}', "
            "but scripts/train.py currently provides runner='off_policy'."
        )
    return create_agent(cfg.algorithm, cfg)


# ===================== 训练核心 =====================
def log_losses(logger: ExperimentLogger, loss_dict: Dict[str, Any], gstep: int, log_attention_image: bool) -> None:
    metrics = {}
    for key, value in loss_dict.items():
        if key == 'attention_img' and log_attention_image:
            logger.log_image('attention/alpha_heatmap', loss_dict['attention_img'], gstep)
        elif key != 'attention_img':
            metrics[f'train/{key}'] = float(value)
    if metrics:
        logger.log(metrics, gstep)


def save_checkpoint(cfg: Config, agent, gstep: int) -> None:
    save_dir = params_dir(cfg)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_step_path = os.path.join(save_dir, "gstep_last.txt")
    with open(ckpt_step_path, 'w') as f:
        f.write(str(gstep))
    agent.save(save_dir, step_id="last")


def load_checkpoint_if_any(cfg: Config, agent) -> None:
    if not os.path.isfile(cfg.pretrained_model_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(cfg.pretrained_model_path))
    agent.load(cfg.pretrained_model_path)
    print("[OK] Loaded model {}".format(cfg.pretrained_model_path))


# ===================== 主训练流程 =====================
def train(cfg: Config) -> None:
    set_seed(cfg.seed)

    # Experiment logger
    tb_dir = runs_dir(cfg)
    os.makedirs(tb_dir, exist_ok=True)
    config_dict = asdict(cfg)
    logger = build_experiment_logger(cfg, tb_dir, config_dict)
    print(f"Experiment logs -> {tb_dir} ({cfg.logger_backend})")
    action_logger = CarlaLogger(logger)

    # Environment, agent, observation adapter, and replay buffer
    env = make_carla_env(cfg)
    agent = make_agent(cfg)
    obs_adapter = VectorObservationAdapter(
        expected_dim=cfg.state_dim,
        risk_field_dim=cfg.risk_field_sectors,
    )

    # use pre-trained model
    if cfg.use_pretrained_model:
        load_checkpoint_if_any(cfg, agent)
    replay_buffer = ReplayBuffer(cfg.buffer_size)

    gstep = 0  # 全局步数计数（按 episode 累加也可）

    try:
        for ep in range(cfg.max_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_cost = 0.0
            ep_step = 0
            while not done:
                obs_vec = obs_adapter.encode(obs)
                action = agent.act(obs_vec)
                action_logger.step(action)

                try:
                    next_obs, reward, cost, done, info = env.step(action)
                except Exception:
                    traceback.print_exc()
                    print("[Error] Carla step failed; resetting env...")
                    obs = env.reset()
                    continue

                next_obs_vec_np = obs_adapter.encode(next_obs)

                # ===== 经验入池 =====
                replay_buffer.add(obs_vec, action, reward, next_obs_vec_np, done)

                # ===== 训练步 =====
                ready_size = max(cfg.minimal_size, cfg.batch_size)
                if replay_buffer.size() >= ready_size and cfg.train_every_step:
                    batch = replay_buffer.sample(cfg.batch_size)
                    loss_dict = agent.update(batch)
                    log_losses(logger, loss_dict, gstep, cfg.log_attention_image)

                # 汇总
                obs = next_obs
                ep_reward += reward
                ep_cost += cost
                ep_step += 1
                reward_metrics = {
                    key: float(value)
                    for key, value in info.get('reward_terms', {}).items()
                }
                if reward_metrics:
                    logger.log(reward_metrics, gstep)
                gstep += 1

            # ===== Episode 结束处理 =====
            logger.log({
                'episode/reward': float(ep_reward),
                'episode/cost': float(ep_cost),
                'episode/length': float(ep_step),
                'episode/index': float(ep),
            }, gstep)
            action_logger.episode_done(gstep)

            # 保存 checkpoint（这里按 episode 保存；可改为间隔保存）
            save_checkpoint(cfg, agent, gstep)

            print(f"[Episode {ep:03d}] Reward={ep_reward:.2f} Steps={ep_step} gstep={gstep}")

    finally:
        try:
            env.close()
            print("Cleared all Carla actors.")
        except Exception:
            traceback.print_exc()
        logger.close()


# ===================== CLI =====================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CarlaRLLab off-policy trainer")
    # 只暴露常用可变项；其余请直接改 Config 默认值或追加参数
    p.add_argument('--algorithm', '--algo', dest='algorithm', type=str,
                   default=Config.algorithm, choices=list(list_algorithms()))
    p.add_argument('--town', type=str, default=Config.town)
    p.add_argument('--port', type=int, default=Config.port)
    p.add_argument('--network', type=str, default=Config.network, choices=['SAC', 'Attention_SAC'])  # TODO: add more networks  e.g., 'ppo', 'dqn'
    p.add_argument('--max_episodes', type=int, default=Config.max_episodes)
    p.add_argument('--seed', type=int, default=Config.seed)
    p.add_argument('--reward', dest='reward_profile', type=str,
                   default=Config.reward_profile, choices=list(list_reward_profiles()))
    p.add_argument('--logger', dest='logger_backend', type=str,
                   default=Config.logger_backend, choices=['tensorboard', 'wandb', 'both', 'none'])
    p.add_argument('--run-name', type=str, default=Config.run_name)
    p.add_argument('--wandb-mode', type=str, default=Config.wandb_mode,
                   choices=['online', 'offline', 'disabled'])
    p.add_argument('--checkpoint', type=str, default=Config.pretrained_model_path,
                   help='Optional algorithm checkpoint to resume from')
    return p


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


if __name__ == '__main__':
    main()
