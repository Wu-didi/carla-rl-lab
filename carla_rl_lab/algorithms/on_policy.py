from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from carla_rl_lab.algorithms.base import BaseAgent
from carla_rl_lab.algorithms.common import mlp
from carla_rl_lab.algorithms.registry import AlgorithmSpec, register_algorithm


class ActorCritic(nn.Module):
    """Gaussian policy and value function used by PPO, A2C, GAIL, and AIRL."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_bound: float,
    ) -> None:
        super().__init__()
        self.policy_body = mlp([state_dim, hidden_dim, hidden_dim], nn.Tanh, nn.Tanh)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_net = mlp([state_dim, hidden_dim, hidden_dim, 1], nn.Tanh)
        self.action_bound = float(action_bound)

    def distribution(self, states: torch.Tensor) -> Normal:
        mean = self.policy_mean(self.policy_body(states))
        log_std = self.policy_log_std.clamp(-5.0, 2.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_net(states).squeeze(-1)

    def sample(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_actions = distribution.rsample()
        unit_actions = torch.tanh(raw_actions)
        actions = unit_actions * self.action_bound
        log_probs = self._log_prob(distribution, raw_actions, unit_actions)
        return actions, log_probs, self.value(states)

    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(states).mean) * self.action_bound

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unit_actions = (actions / self.action_bound).clamp(-0.999999, 0.999999)
        raw_actions = 0.5 * (
            torch.log1p(unit_actions) - torch.log1p(-unit_actions)
        )
        distribution = self.distribution(states)
        log_probs = self._log_prob(distribution, raw_actions, unit_actions)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, entropy, self.value(states)

    def _log_prob(
        self,
        distribution: Normal,
        raw_actions: torch.Tensor,
        unit_actions: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(
            self.action_bound * (1.0 - unit_actions.pow(2)) + 1e-6
        )
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)


class OnPolicyAgent(BaseAgent):
    algorithm_name = "on_policy"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.5))
        self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.0))
        self.value_coef = float(getattr(cfg, "value_coef", 0.5))
        self.model = ActorCritic(
            cfg.state_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.action_bound,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(getattr(cfg, "policy_lr", 3e-4))
        )
        self.update_step = 0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = (
                self.model.deterministic(state)
                if deterministic
                else self.model.sample(state)[0]
            )
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def act_with_info(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.model.sample(state)
        return (
            action.cpu().numpy().reshape(-1).astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def value(self, obs: np.ndarray) -> float:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.model.value(state)
        return float(value.item())

    def action_log_probs(self, states: Any, actions: Any) -> torch.Tensor:
        states_tensor = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions_tensor = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        )
        return self.model.evaluate_actions(states_tensor, actions_tensor)[0]

    def save(
        self, directory: str, step_id: Optional[Union[int, str]] = None
    ) -> None:
        os.makedirs(directory, exist_ok=True)
        checkpoint_id = "last" if step_id is None else step_id
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_step": self.update_step,
            },
            os.path.join(
                directory,
                "{}_ckpt_{}.pt".format(self.algorithm_name, checkpoint_id),
            ),
        )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_step = int(checkpoint.get("update_step", 0))

    def _tensors(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        names = ("states", "actions", "old_log_probs", "advantages", "returns")
        missing = [name for name in names if name not in batch]
        if missing:
            raise ValueError("on-policy batch is missing: {}".format(", ".join(missing)))
        return {
            name: torch.as_tensor(batch[name], dtype=torch.float32, device=self.device)
            for name in names
        }


class PpoAgent(OnPolicyAgent):
    """Clipped Proximal Policy Optimization with GAE targets."""

    algorithm_name = "ppo"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.clip_ratio = float(getattr(cfg, "ppo_clip", 0.2))
        self.epochs = int(getattr(cfg, "ppo_epochs", 10))
        self.minibatch_size = int(getattr(cfg, "ppo_minibatch_size", 64))

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        tensors = self._tensors(batch)
        advantages = tensors["advantages"]
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        sample_count = tensors["states"].shape[0]
        minibatch_size = min(self.minibatch_size, sample_count)
        totals = {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        optimizer_steps = 0

        for _ in range(self.epochs):
            indices = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, minibatch_size):
                selected = indices[start : start + minibatch_size]
                log_probs, entropy, values = self.model.evaluate_actions(
                    tensors["states"][selected], tensors["actions"][selected]
                )
                log_ratio = log_probs - tensors["old_log_probs"][selected]
                ratio = log_ratio.exp()
                selected_advantages = advantages[selected]
                unclipped = ratio * selected_advantages
                clipped = ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * selected_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, tensors["returns"][selected])
                loss = (
                    actor_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    totals["actor_loss"] += float(actor_loss.item())
                    totals["value_loss"] += float(value_loss.item())
                    totals["entropy"] += float(entropy.mean().item())
                    totals["approx_kl"] += float(((ratio - 1.0) - log_ratio).mean().item())
                    totals["clip_fraction"] += float(
                        ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                    )
                optimizer_steps += 1

        self.update_step += 1
        return {name: value / optimizer_steps for name, value in totals.items()}


class A2cAgent(OnPolicyAgent):
    """Synchronous Advantage Actor-Critic over a fresh rollout."""

    algorithm_name = "a2c"

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        tensors = self._tensors(batch)
        advantages = tensors["advantages"]
        log_probs, entropy, values = self.model.evaluate_actions(
            tensors["states"], tensors["actions"]
        )
        actor_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, tensors["returns"])
        loss = (
            actor_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy.mean()
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.update_step += 1
        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }


register_algorithm(
    AlgorithmSpec(
        name="ppo",
        factory=PpoAgent,
        family="on_policy",
        data_source="online",
        runner="on_policy",
        action_space="continuous",
        status="implemented",
        description="PPO with a squashed Gaussian policy, clipped objective, and GAE.",
    )
)

register_algorithm(
    AlgorithmSpec(
        name="a2c",
        factory=A2cAgent,
        family="on_policy",
        data_source="online",
        runner="on_policy",
        action_space="continuous",
        status="implemented",
        description="Synchronous advantage actor-critic for continuous control.",
    )
)
