from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from carla_rl_lab.algorithms.base import BaseAgent
from carla_rl_lab.algorithms.common import DeterministicActor, mlp
from carla_rl_lab.algorithms.on_policy import PpoAgent
from carla_rl_lab.algorithms.registry import AlgorithmSpec, register_algorithm
from carla_rl_lab.buffers import generalized_advantage_estimate
from carla_rl_lab.utils.checkpoint import torch_load


class BcAgent(BaseAgent):
    """Behavior cloning baseline for continuous expert actions."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actor = DeterministicActor(
            cfg.state_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.action_bound,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.update_step = 0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        missing = [name for name in ("states", "actions") if name not in batch]
        if missing:
            raise ValueError("BC batch is missing: {}".format(", ".join(missing)))
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        predicted_actions = self.actor(states)
        loss = F.mse_loss(predicted_actions, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_step += 1
        return {
            "bc_loss": loss.item(),
            "action_error": (predicted_actions.detach() - actions).abs().mean().item(),
        }

    def save(
        self, directory: str, step_id: Optional[Union[int, str]] = None
    ) -> None:
        os.makedirs(directory, exist_ok=True)
        checkpoint_id = "last" if step_id is None else step_id
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_step": self.update_step,
            },
            os.path.join(directory, "bc_ckpt_{}.pt".format(checkpoint_id)),
        )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch_load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_step = int(checkpoint.get("update_step", 0))


class StateActionDiscriminator(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = mlp(
            [state_dim + action_dim, hidden_dim, hidden_dim, 1],
            nn.Tanh,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)


class AirlDiscriminator(nn.Module):
    """AIRL reward/potential decomposition f=r+gamma*h(s')-h(s)."""

    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, gamma: float
    ) -> None:
        super().__init__()
        self.reward = mlp(
            [state_dim + action_dim, hidden_dim, hidden_dim, 1], nn.Tanh
        )
        self.potential = mlp([state_dim, hidden_dim, hidden_dim, 1], nn.Tanh)
        self.gamma = float(gamma)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        reward = self.reward(torch.cat([states, actions], dim=-1)).squeeze(-1)
        potential = self.potential(states).squeeze(-1)
        next_potential = self.potential(next_states).squeeze(-1)
        return reward + self.gamma * (1.0 - dones) * next_potential - potential


class AdversarialImitationAgent(BaseAgent):
    """Adversarial discriminator composed with a PPO policy."""

    def __init__(self, cfg: Any, algorithm_name: str, airl: bool) -> None:
        self.cfg = cfg
        self.algorithm_name = algorithm_name
        self.airl = bool(airl)
        self.policy = PpoAgent(cfg)
        self.device = self.policy.device
        if self.airl:
            self.discriminator = AirlDiscriminator(
                cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.gamma
            ).to(self.device)
        else:
            self.discriminator = StateActionDiscriminator(
                cfg.state_dim, cfg.hidden_dim, cfg.action_dim
            ).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(getattr(cfg, "discriminator_lr", 3e-4)),
        )
        self.discriminator_updates = int(
            getattr(cfg, "discriminator_updates", 1)
        )
        self.gamma = float(cfg.gamma)
        self.gae_lambda = float(getattr(cfg, "gae_lambda", 0.95))

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.policy.act(obs, deterministic)

    def act_with_info(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        return self.policy.act_with_info(obs)

    def value(self, obs: np.ndarray) -> float:
        return self.policy.value(obs)

    def action_log_probs(self, states: Any, actions: Any) -> torch.Tensor:
        return self.policy.action_log_probs(states, actions)

    def _logits(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: Optional[torch.Tensor],
        policy_log_probs: Optional[torch.Tensor],
        dones: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.airl:
            return self.discriminator(states, actions)
        if next_states is None or policy_log_probs is None or dones is None:
            raise ValueError(
                "AIRL requires next_states, dones, and policy log probabilities"
            )
        return (
            self.discriminator(states, actions, next_states, dones)
            - policy_log_probs
        )

    def _as_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        policy_required = (
            "states",
            "actions",
            "old_log_probs",
            "values",
            "dones",
        )
        expert_required = ("expert_states", "expert_actions")
        if self.airl:
            policy_required += ("next_states",)
            expert_required += ("expert_next_states", "expert_dones")
        missing = [
            name
            for name in policy_required + expert_required
            if name not in batch
        ]
        if missing:
            raise ValueError(
                "{} batch is missing: {}".format(
                    self.algorithm_name.upper(), ", ".join(missing)
                )
            )

        states = self._as_tensor(batch["states"])
        actions = self._as_tensor(batch["actions"])
        policy_log_probs = self._as_tensor(batch["old_log_probs"]).reshape(-1)
        policy_dones = self._as_tensor(batch["dones"]).reshape(-1)
        next_states = (
            self._as_tensor(batch["next_states"]) if self.airl else None
        )
        expert_states = self._as_tensor(batch["expert_states"])
        expert_actions = self._as_tensor(batch["expert_actions"])
        expert_next_states = (
            self._as_tensor(batch["expert_next_states"]) if self.airl else None
        )
        expert_dones = (
            self._as_tensor(batch["expert_dones"]).reshape(-1)
            if self.airl
            else None
        )

        discriminator_loss_value = 0.0
        expert_accuracy_value = 0.0
        policy_accuracy_value = 0.0
        for _ in range(self.discriminator_updates):
            with torch.no_grad():
                expert_log_probs = self.action_log_probs(
                    expert_states, expert_actions
                ).reshape(-1)
            expert_logits = self._logits(
                expert_states,
                expert_actions,
                expert_next_states,
                expert_log_probs,
                expert_dones,
            )
            policy_logits = self._logits(
                states,
                actions,
                next_states,
                policy_log_probs,
                policy_dones,
            )
            discriminator_loss = F.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            ) + F.binary_cross_entropy_with_logits(
                policy_logits, torch.zeros_like(policy_logits)
            )
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            discriminator_loss_value += float(discriminator_loss.item())
            with torch.no_grad():
                expert_accuracy_value += float((expert_logits > 0).float().mean().item())
                policy_accuracy_value += float((policy_logits < 0).float().mean().item())

        with torch.no_grad():
            current_log_probs = self.action_log_probs(states, actions).reshape(-1)
            logits = self._logits(
                states, actions, next_states, current_log_probs, policy_dones
            )
            shaped_rewards = logits if self.airl else F.softplus(logits)

        targets = generalized_advantage_estimate(
            shaped_rewards.cpu().numpy(),
            np.asarray(batch["dones"], dtype=np.float32),
            np.asarray(batch["values"], dtype=np.float32),
            float(batch.get("last_value", 0.0)),
            self.gamma,
            self.gae_lambda,
            episode_ends=np.asarray(
                batch.get("episode_ends", batch["dones"]), dtype=np.float32
            ),
            next_values=np.asarray(batch["next_values"], dtype=np.float32)
            if "next_values" in batch
            else None,
        )
        policy_batch = dict(batch)
        policy_batch["rewards"] = shaped_rewards.cpu().numpy()
        policy_batch["advantages"] = targets["advantages"]
        policy_batch["returns"] = targets["returns"]
        logs = self.policy.update(policy_batch)
        divisor = float(self.discriminator_updates)
        logs.update(
            {
                "discriminator_loss": discriminator_loss_value / divisor,
                "discriminator_expert_accuracy": expert_accuracy_value / divisor,
                "discriminator_policy_accuracy": policy_accuracy_value / divisor,
                "imitation_reward": shaped_rewards.mean().item(),
            }
        )
        return logs

    def save(
        self, directory: str, step_id: Optional[Union[int, str]] = None
    ) -> None:
        os.makedirs(directory, exist_ok=True)
        checkpoint_id = "last" if step_id is None else step_id
        torch.save(
            {
                "model": self.policy.model.state_dict(),
                "optimizer": self.policy.optimizer.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                "update_step": self.policy.update_step,
            },
            os.path.join(
                directory,
                "{}_ckpt_{}.pt".format(self.algorithm_name, checkpoint_id),
            ),
        )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch_load(checkpoint_path, map_location=self.device)
        self.policy.model.load_state_dict(checkpoint["model"])
        self.policy.optimizer.load_state_dict(checkpoint["optimizer"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer"]
        )
        self.policy.update_step = int(checkpoint.get("update_step", 0))


def make_gail_agent(cfg: Any) -> AdversarialImitationAgent:
    return AdversarialImitationAgent(cfg, algorithm_name="gail", airl=False)


def make_airl_agent(cfg: Any) -> AdversarialImitationAgent:
    return AdversarialImitationAgent(cfg, algorithm_name="airl", airl=True)


register_algorithm(
    AlgorithmSpec(
        name="bc",
        factory=BcAgent,
        family="imitation",
        data_source="expert",
        runner="imitation",
        action_space="continuous",
        status="implemented",
        description="Supervised behavior cloning from expert state-action pairs.",
    )
)

register_algorithm(
    AlgorithmSpec(
        name="gail",
        factory=make_gail_agent,
        family="imitation",
        data_source="expert_mixed",
        runner="imitation",
        action_space="continuous",
        status="implemented",
        description="GAIL with a state-action discriminator and PPO policy updates.",
    )
)

register_algorithm(
    AlgorithmSpec(
        name="airl",
        factory=make_airl_agent,
        family="imitation",
        data_source="expert_mixed",
        runner="imitation",
        action_space="continuous",
        status="implemented",
        description="AIRL with decomposed reward shaping and PPO policy updates.",
    )
)
