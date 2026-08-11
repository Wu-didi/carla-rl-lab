from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from carla_rl_lab.algorithms.base import BaseAgent
from carla_rl_lab.algorithms.common import ContinuousCritic, DeterministicActor, soft_update
from carla_rl_lab.algorithms.registry import AlgorithmSpec, register_algorithm
from carla_rl_lab.utils.checkpoint import torch_load


class Td3Agent(BaseAgent):
    """Twin Delayed DDPG for continuous CARLA control."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_bound = float(cfg.action_bound)
        self.gamma = float(cfg.gamma)
        self.tau = float(cfg.tau)
        self.exploration_noise = float(getattr(cfg, "exploration_noise", 0.1))
        self.policy_noise = float(getattr(cfg, "td3_policy_noise", 0.2))
        self.noise_clip = float(getattr(cfg, "td3_noise_clip", 0.5))
        self.policy_delay = int(getattr(cfg, "td3_policy_delay", 2))
        self.update_step = 0

        self.actor = DeterministicActor(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.action_bound).to(self.device)
        self.actor_target = DeterministicActor(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.action_bound).to(self.device)
        self.critic_1 = ContinuousCritic(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.critic_2 = ContinuousCritic(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.critic_target_1 = ContinuousCritic(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.critic_target_2 = ContinuousCritic(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=cfg.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=cfg.critic_lr)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().reshape(-1)
        if not deterministic and self.exploration_noise > 0.0:
            action = action + np.random.normal(0.0, self.exploration_noise, size=action.shape)
        return np.clip(action, -self.action_bound, self.action_bound).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.update_step += 1

        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.action_bound, self.action_bound)
            target_q1 = self.critic_target_1(next_states, next_actions)
            target_q2 = self.critic_target_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards + self.gamma * (1.0 - dones) * target_q

        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target)
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=10.0)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=10.0)
        self.critic_2_optimizer.step()

        logs = {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
        }

        if self.update_step % self.policy_delay == 0:
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic_1, self.critic_target_1, self.tau)
            soft_update(self.critic_2, self.critic_target_2, self.tau)
            logs["actor_loss"] = actor_loss.item()

        with torch.no_grad():
            logs["avg_q"] = torch.min(self.critic_1(states, actions), self.critic_2(states, actions)).mean().item()
        return logs

    def save(self, directory: str, step_id: Optional[Union[int, str]] = None) -> None:
        os.makedirs(directory, exist_ok=True)
        ckpt_id = "last" if step_id is None else step_id
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_target_1": self.critic_target_1.state_dict(),
                "critic_target_2": self.critic_target_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "update_step": self.update_step,
            },
            os.path.join(directory, "td3_ckpt_{}.pt".format(ckpt_id)),
        )

    def load(self, checkpoint_path: str) -> None:
        ckpt = torch_load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.critic_target_1.load_state_dict(ckpt["critic_target_1"])
        self.critic_target_2.load_state_dict(ckpt["critic_target_2"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
        self.update_step = int(ckpt.get("update_step", 0))


register_algorithm(
    AlgorithmSpec(
        name="td3",
        factory=Td3Agent,
        family="off_policy",
        data_source="online",
        runner="off_policy",
        action_space="continuous",
        status="implemented",
        description="TD3 baseline with twin critics and delayed policy updates.",
    )
)
