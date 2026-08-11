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


class DdpgAgent(BaseAgent):
    """Deep Deterministic Policy Gradient.

    This is intentionally compact and editable. It shares the same replay
    buffer training path as SAC and TD3.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_bound = float(cfg.action_bound)
        self.gamma = float(cfg.gamma)
        self.tau = float(cfg.tau)
        self.exploration_noise = float(getattr(cfg, "exploration_noise", 0.1))

        self.actor = DeterministicActor(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.action_bound).to(self.device)
        self.actor_target = DeterministicActor(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.action_bound).to(self.device)
        self.critic = ContinuousCritic(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.critic_target = ContinuousCritic(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().reshape(-1)
        if not deterministic and self.exploration_noise > 0.0:
            action = action + np.random.normal(0.0, self.exploration_noise, size=action.shape)
        return np.clip(action, -self.action_bound, self.action_bound).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            td_target = rewards + self.gamma * (1.0 - dones) * target_q

        critic_loss = F.mse_loss(self.critic(states, actions), td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "avg_q": self.critic(states, actions).mean().item(),
        }

    def save(self, directory: str, step_id: Optional[Union[int, str]] = None) -> None:
        os.makedirs(directory, exist_ok=True)
        ckpt_id = "last" if step_id is None else step_id
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "ddpg_ckpt_{}.pt".format(ckpt_id)),
        )

    def load(self, checkpoint_path: str) -> None:
        ckpt = torch_load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])


register_algorithm(
    AlgorithmSpec(
        name="ddpg",
        factory=DdpgAgent,
        family="off_policy",
        data_source="online",
        runner="off_policy",
        action_space="continuous",
        status="implemented",
        description="DDPG baseline for continuous vehicle control.",
    )
)
