from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from carla_rl_lab.algorithms.base import BaseAgent
from carla_rl_lab.algorithms.common import (
    ContinuousCritic,
    DeterministicActor,
    mlp,
    soft_update,
)
from carla_rl_lab.algorithms.registry import AlgorithmSpec, register_algorithm
from carla_rl_lab.utils.checkpoint import torch_load


def transition_tensors(
    batch: Dict[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("states", "actions", "rewards", "next_states", "dones")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError("offline batch is missing: {}".format(", ".join(missing)))
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    rewards = torch.as_tensor(
        batch["rewards"], dtype=torch.float32, device=device
    ).view(-1, 1)
    next_states = torch.as_tensor(
        batch["next_states"], dtype=torch.float32, device=device
    )
    dones = torch.as_tensor(
        batch["dones"], dtype=torch.float32, device=device
    ).view(-1, 1)
    return states, actions, rewards, next_states, dones


class SquashedGaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy with action likelihood evaluation."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_bound: float,
    ) -> None:
        super().__init__()
        self.body = mlp([state_dim, hidden_dim, hidden_dim], nn.ReLU, nn.ReLU)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = float(action_bound)

    def distribution(self, states: torch.Tensor) -> Normal:
        features = self.body(states)
        return Normal(
            self.mean(features),
            self.log_std(features).clamp(-5.0, 2.0).exp(),
        )

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_actions = distribution.rsample()
        unit_actions = torch.tanh(raw_actions)
        actions = unit_actions * self.action_bound
        return actions, self._log_prob(distribution, raw_actions, unit_actions)

    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(states).mean) * self.action_bound

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        unit_actions = (actions / self.action_bound).clamp(-0.999999, 0.999999)
        raw_actions = 0.5 * (
            torch.log1p(unit_actions) - torch.log1p(-unit_actions)
        )
        return self._log_prob(self.distribution(states), raw_actions, unit_actions)

    def _log_prob(
        self,
        distribution: Normal,
        raw_actions: torch.Tensor,
        unit_actions: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(
            self.action_bound * (1.0 - unit_actions.pow(2)) + 1e-6
        )
        return (distribution.log_prob(raw_actions) - correction).sum(
            dim=-1, keepdim=True
        )


class OfflineAgent(BaseAgent):
    algorithm_name = "offline"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_bound = float(cfg.action_bound)
        self.gamma = float(cfg.gamma)
        self.tau = float(cfg.tau)
        self.update_step = 0

    def _checkpoint_path(
        self, directory: str, step_id: Optional[Union[int, str]]
    ) -> str:
        os.makedirs(directory, exist_ok=True)
        checkpoint_id = "last" if step_id is None else step_id
        return os.path.join(
            directory,
            "{}_ckpt_{}.pt".format(self.algorithm_name, checkpoint_id),
        )


class Td3BcAgent(OfflineAgent):
    """TD3+BC: TD3 value learning with behavior-regularized policy updates."""

    algorithm_name = "td3_bc"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.alpha = float(getattr(cfg, "td3_bc_alpha", 2.5))
        self.policy_noise = float(getattr(cfg, "td3_policy_noise", 0.2))
        self.noise_clip = float(getattr(cfg, "td3_noise_clip", 0.5))
        self.policy_delay = int(getattr(cfg, "td3_policy_delay", 2))

        actor_args = (
            cfg.state_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.action_bound,
        )
        critic_args = (cfg.state_dim, cfg.hidden_dim, cfg.action_dim)
        self.actor = DeterministicActor(*actor_args).to(self.device)
        self.actor_target = DeterministicActor(*actor_args).to(self.device)
        self.critic_1 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_2 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_target_1 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_target_2 = ContinuousCritic(*critic_args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=cfg.critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=cfg.critic_lr
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.update_step += 1
        states, actions, rewards, next_states, dones = transition_tensors(
            batch, self.device
        )
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.action_bound, self.action_bound
            )
            next_q = torch.min(
                self.critic_target_1(next_states, next_actions),
                self.critic_target_2(next_states, next_actions),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        critic_1_loss = F.mse_loss(self.critic_1(states, actions), target_q)
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        logs = {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
        }
        if self.update_step % self.policy_delay == 0:
            policy_actions = self.actor(states)
            q_values = self.critic_1(states, policy_actions)
            scale = self.alpha / q_values.abs().mean().detach().clamp_min(1e-6)
            bc_loss = F.mse_loss(policy_actions, actions)
            actor_loss = -scale * q_values.mean() + bc_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic_1, self.critic_target_1, self.tau)
            soft_update(self.critic_2, self.critic_target_2, self.tau)
            logs.update(
                {
                    "actor_loss": actor_loss.item(),
                    "bc_loss": bc_loss.item(),
                    "td3_bc_scale": scale.item(),
                }
            )
        return logs

    def save(
        self, directory: str, step_id: Optional[Union[int, str]] = None
    ) -> None:
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
            self._checkpoint_path(directory, step_id),
        )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch_load(checkpoint_path, map_location=self.device)
        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_2",
            "critic_target_1",
            "critic_target_2",
        ):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.update_step = int(checkpoint.get("update_step", 0))


class CqlAgent(OfflineAgent):
    """Continuous CQL(H) with SAC policy learning and density correction."""

    algorithm_name = "cql"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.cql_alpha = float(getattr(cfg, "cql_alpha", 1.0))
        self.temperature = float(getattr(cfg, "cql_temperature", 1.0))
        self.num_random = int(getattr(cfg, "cql_num_random", 10))
        self.entropy_alpha = float(getattr(cfg, "offline_entropy_alpha", 0.2))
        self.action_dim = int(cfg.action_dim)
        if self.temperature <= 0.0:
            raise ValueError("cql_temperature must be positive")
        if self.num_random <= 0:
            raise ValueError("cql_num_random must be positive")

        self.actor = SquashedGaussianPolicy(
            cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.action_bound
        ).to(self.device)
        critic_args = (cfg.state_dim, cfg.hidden_dim, cfg.action_dim)
        self.critic_1 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_2 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_target_1 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_target_2 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=cfg.critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=cfg.critic_lr
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = (
                self.actor.deterministic(state)
                if deterministic
                else self.actor.sample(state)[0]
            )
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def _conservative_loss(
        self,
        critic: ContinuousCritic,
        states: torch.Tensor,
        next_states: torch.Tensor,
        data_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = states.shape[0]
        repeated_states = states[:, None, :].expand(
            batch_size, self.num_random, states.shape[-1]
        ).reshape(-1, states.shape[-1])
        repeated_next_states = next_states[:, None, :].expand(
            batch_size, self.num_random, next_states.shape[-1]
        ).reshape(-1, next_states.shape[-1])
        random_actions = torch.empty(
            batch_size * self.num_random,
            self.action_dim,
            device=self.device,
        ).uniform_(-self.action_bound, self.action_bound)
        with torch.no_grad():
            current_policy_actions, current_log_probs = self.actor.sample(
                repeated_states
            )
            next_policy_actions, next_log_probs = self.actor.sample(
                repeated_next_states
            )

        random_log_density = -self.action_dim * np.log(2.0 * self.action_bound)
        q_random = (
            critic(repeated_states, random_actions) - random_log_density
        ).reshape(batch_size, -1)
        q_current = (
            critic(repeated_states, current_policy_actions) - current_log_probs
        ).reshape(batch_size, -1)
        q_next = (
            critic(repeated_states, next_policy_actions) - next_log_probs
        ).reshape(batch_size, -1)
        candidates = torch.cat([q_random, q_current, q_next], dim=1)
        conservative_q = torch.logsumexp(
            candidates / self.temperature, dim=1
        ).mean() * self.temperature
        return conservative_q - critic(states, data_actions).mean()

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.update_step += 1
        states, actions, rewards, next_states, dones = transition_tensors(
            batch, self.device
        )
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q = torch.min(
                self.critic_target_1(next_states, next_actions),
                self.critic_target_2(next_states, next_actions),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * (
                next_q - self.entropy_alpha * next_log_probs
            )

        bellman_1 = F.mse_loss(self.critic_1(states, actions), target_q)
        bellman_2 = F.mse_loss(self.critic_2(states, actions), target_q)
        conservative_1 = self._conservative_loss(
            self.critic_1, states, next_states, actions
        )
        conservative_2 = self._conservative_loss(
            self.critic_2, states, next_states, actions
        )
        critic_1_loss = bellman_1 + self.cql_alpha * conservative_1
        critic_2_loss = bellman_2 + self.cql_alpha * conservative_2

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        policy_actions, log_probs = self.actor.sample(states)
        policy_q = torch.min(
            self.critic_1(states, policy_actions),
            self.critic_2(states, policy_actions),
        )
        actor_loss = (self.entropy_alpha * log_probs - policy_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        soft_update(self.critic_1, self.critic_target_1, self.tau)
        soft_update(self.critic_2, self.critic_target_2, self.tau)

        return {
            "actor_loss": actor_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "bellman_loss": 0.5 * (bellman_1.item() + bellman_2.item()),
            "conservative_loss": 0.5
            * (conservative_1.item() + conservative_2.item()),
            "avg_q": policy_q.mean().item(),
        }

    def save(
        self, directory: str, step_id: Optional[Union[int, str]] = None
    ) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_target_1": self.critic_target_1.state_dict(),
                "critic_target_2": self.critic_target_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "update_step": self.update_step,
            },
            self._checkpoint_path(directory, step_id),
        )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch_load(checkpoint_path, map_location=self.device)
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "critic_target_1",
            "critic_target_2",
        ):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.update_step = int(checkpoint.get("update_step", 0))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = mlp([state_dim, hidden_dim, hidden_dim, 1])

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class IqlAgent(OfflineAgent):
    """Implicit Q-Learning with expectile values and advantage-weighted BC."""

    algorithm_name = "iql"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.expectile = float(getattr(cfg, "iql_expectile", 0.7))
        self.beta = float(getattr(cfg, "iql_beta", 3.0))
        self.max_weight = float(getattr(cfg, "iql_max_weight", 100.0))
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("iql_expectile must be between 0 and 1")

        self.actor = SquashedGaussianPolicy(
            cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.action_bound
        ).to(self.device)
        critic_args = (cfg.state_dim, cfg.hidden_dim, cfg.action_dim)
        self.critic_1 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_2 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_target_1 = ContinuousCritic(*critic_args).to(self.device)
        self.critic_target_2 = ContinuousCritic(*critic_args).to(self.device)
        self.value_net = ValueNetwork(cfg.state_dim, cfg.hidden_dim).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=cfg.critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=cfg.critic_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=cfg.critic_lr
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = (
                self.actor.deterministic(state)
                if deterministic
                else self.actor.sample(state)[0]
            )
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.update_step += 1
        states, actions, rewards, next_states, dones = transition_tensors(
            batch, self.device
        )
        with torch.no_grad():
            target_q = torch.min(
                self.critic_target_1(states, actions),
                self.critic_target_2(states, actions),
            )
        values = self.value_net(states)
        difference = target_q - values
        expectile_weight = torch.where(
            difference > 0.0,
            torch.full_like(difference, self.expectile),
            torch.full_like(difference, 1.0 - self.expectile),
        )
        value_loss = (expectile_weight * difference.pow(2)).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            td_target = rewards + self.gamma * (1.0 - dones) * self.value_net(
                next_states
            )
        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target)
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        with torch.no_grad():
            advantages = target_q - self.value_net(states)
            weights = torch.exp(self.beta * advantages).clamp(max=self.max_weight)
        actor_loss = -(weights * self.actor.log_prob(states, actions)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        soft_update(self.critic_1, self.critic_target_1, self.tau)
        soft_update(self.critic_2, self.critic_target_2, self.tau)

        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "advantage": advantages.mean().item(),
            "actor_weight": weights.mean().item(),
        }

    def save(
        self, directory: str, step_id: Optional[Union[int, str]] = None
    ) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_target_1": self.critic_target_1.state_dict(),
                "critic_target_2": self.critic_target_2.state_dict(),
                "value_net": self.value_net.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "update_step": self.update_step,
            },
            self._checkpoint_path(directory, step_id),
        )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch_load(checkpoint_path, map_location=self.device)
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "critic_target_1",
            "critic_target_2",
            "value_net",
        ):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.update_step = int(checkpoint.get("update_step", 0))


register_algorithm(
    AlgorithmSpec(
        name="td3_bc",
        factory=Td3BcAgent,
        family="offline_rl",
        data_source="offline",
        runner="offline",
        action_space="continuous",
        status="implemented",
        description="TD3+BC with value-scaled policy improvement and behavior cloning.",
    )
)

register_algorithm(
    AlgorithmSpec(
        name="cql",
        factory=CqlAgent,
        family="offline_rl",
        data_source="offline",
        runner="offline",
        action_space="continuous",
        status="implemented",
        description="Continuous CQL with conservative twin critics and a Gaussian actor.",
    )
)

register_algorithm(
    AlgorithmSpec(
        name="iql",
        factory=IqlAgent,
        family="offline_rl",
        data_source="offline",
        runner="offline",
        action_space="continuous",
        status="implemented",
        description="IQL with expectile regression and advantage-weighted cloning.",
    )
)
