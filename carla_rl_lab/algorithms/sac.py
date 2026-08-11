from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from carla_rl_lab.algorithms.base import BaseAgent
from carla_rl_lab.algorithms.registry import AlgorithmSpec, register_algorithm
from carla_rl_lab.utils.checkpoint import torch_load


class SemanticAttentionEncoder(nn.Module):
    """Attention encoder for the default 299-dimensional observation vector."""

    def __init__(self, hidden_dim: int, key_dim: int = 64):
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                "ego": nn.Linear(9, key_dim),
                "lane": nn.Linear(2, key_dim),
                "risk": nn.Linear(12, key_dim),
                "lidar": nn.Linear(8, key_dim),
                "waypoint": nn.Linear(9, key_dim),
            }
        )
        self.query = nn.Linear(key_dim, key_dim)
        self.key = nn.Linear(key_dim, key_dim, bias=False)
        self.score = nn.Linear(key_dim, 1, bias=False)
        self.backbone = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-1] != 299:
            raise ValueError("Attention SAC expects a 299-dimensional observation")
        batch_size = obs.shape[0]
        tokens = torch.cat(
            [
                self.projections["ego"](obs[:, 0:9].unsqueeze(1)),
                self.projections["lane"](obs[:, 9:11].unsqueeze(1)),
                self.projections["risk"](obs[:, 11:23].unsqueeze(1)),
                self.projections["lidar"](obs[:, 23:263].reshape(batch_size, 30, 8)),
                self.projections["waypoint"](obs[:, 263:299].reshape(batch_size, 4, 9)),
            ],
            dim=1,
        )
        query = self.query(tokens.mean(dim=1)).unsqueeze(1)
        scores = self.score(torch.tanh(query + self.key(tokens))).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        encoded = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        return self.backbone(encoded), weights


class MlpEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.net(obs), None


def make_encoder(state_dim: int, hidden_dim: int, network: str) -> nn.Module:
    normalized = network.lower()
    if normalized in ("sac", "mlp"):
        return MlpEncoder(state_dim, hidden_dim)
    if normalized in ("attention_sac", "attention"):
        if state_dim != 299:
            raise ValueError("Attention SAC currently requires state_dim=299")
        return SemanticAttentionEncoder(hidden_dim)
    raise ValueError("Unknown SAC network: {}".format(network))


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: float, network: str):
        super().__init__()
        self.encoder = make_encoder(state_dim, hidden_dim, network)
        self.fc_mu = nn.Linear(self.encoder.output_dim, action_dim)
        self.fc_log_std = nn.Linear(self.encoder.output_dim, action_dim)
        self.action_bound = float(action_bound)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features, attention = self.encoder(obs)
        mean = self.fc_mu(features)
        log_std = torch.clamp(self.fc_log_std(features), -20.0, 2.0)
        distribution = Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        squashed_action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - squashed_action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        return squashed_action * self.action_bound, log_prob, attention

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        features, _ = self.encoder(obs)
        return torch.tanh(self.fc_mu(features)) * self.action_bound


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, network: str):
        super().__init__()
        self.encoder = make_encoder(state_dim, hidden_dim, network)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.output_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features, _ = self.encoder(obs)
        return self.net(torch.cat([features, action], dim=-1))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


class SacAgent(BaseAgent):
    """Readable Soft Actor-Critic implementation for continuous CARLA control."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gamma = float(cfg.gamma)
        self.tau = float(cfg.tau)
        network = getattr(cfg, "network", "SAC")

        self.actor = GaussianActor(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, cfg.action_bound, network).to(self.device)
        self.critic_1 = QNetwork(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, network).to(self.device)
        self.critic_2 = QNetwork(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, network).to(self.device)
        self.target_critic_1 = QNetwork(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, network).to(self.device)
        self.target_critic_2 = QNetwork(cfg.state_dim, cfg.hidden_dim, cfg.action_dim, network).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=cfg.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(
            np.log(0.2), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = float(cfg.target_entropy)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state)
            else:
                action = self.actor(state)[0]
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor(next_states)
            next_q = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * (
                next_q - self.log_alpha.exp() * next_log_prob
            )

        critic_1_loss = F.mse_loss(self.critic_1(states, actions), target_q)
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 10.0)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 10.0)
        self.critic_2_optimizer.step()

        new_actions, log_prob, attention = self.actor(states)
        q_value = torch.min(
            self.critic_1(states, new_actions),
            self.critic_2(states, new_actions),
        )
        actor_loss = (self.log_alpha.exp().detach() * log_prob - q_value).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.critic_1, self.target_critic_1, self.tau)
        soft_update(self.critic_2, self.target_critic_2, self.tau)

        logs = {
            "actor_loss": actor_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "avg_q": q_value.mean().item(),
            "entropy": -log_prob.mean().item(),
        }
        if attention is not None:
            mean_attention = attention.detach().mean(dim=0)
            logs["attention_img"] = mean_attention[None, None, :].cpu()
        return logs

    def save(self, directory: str, step_id: Optional[Union[int, str]] = None) -> None:
        os.makedirs(directory, exist_ok=True)
        checkpoint_id = "last" if step_id is None else step_id
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            os.path.join(directory, "sac_ckpt_{}.pt".format(checkpoint_id)),
        )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch_load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        alpha_optimizer = checkpoint.get("alpha_optimizer", checkpoint.get("log_alpha_optimizer"))
        if alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(alpha_optimizer)


register_algorithm(
    AlgorithmSpec(
        name="sac",
        factory=SacAgent,
        family="off_policy",
        data_source="online",
        runner="off_policy",
        action_space="continuous",
        status="implemented",
        description="Soft Actor-Critic with editable MLP or semantic-attention networks.",
    )
)
