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


class PixelEncoder(nn.Module):
    """Small editable encoder for packed RGB, route, speed, and steer input."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        image_size: int,
        frame_stack: int,
        num_waypoints: int,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.frame_stack = int(frame_stack)
        self.num_waypoints = int(num_waypoints)
        self.image_values = 3 * self.frame_stack * self.image_size ** 2
        expected_dim = self.image_values + 2 * self.num_waypoints + 2
        if int(state_dim) != expected_dim:
            raise ValueError(
                "Pixel_SAC state_dim mismatch: expected {}, got {}".format(
                    expected_dim, state_dim
                )
            )

        channels = 3 * self.frame_stack
        self.image_net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, self.image_size, self.image_size)
            image_features = int(self.image_net(dummy).numel())
        self.image_projection = nn.Sequential(
            nn.Linear(image_features, 128), nn.LayerNorm(128), nn.Tanh()
        )
        self.route_conv = nn.Conv1d(2, 32, kernel_size=2)
        self.route_projection = nn.Sequential(
            nn.Linear(32 * (self.num_waypoints - 1), 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        self.measurement_projection = nn.Sequential(
            nn.Linear(2, 16), nn.LayerNorm(16), nn.ReLU()
        )
        self.backbone = nn.Sequential(
            nn.Linear(176, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def _random_shift(self, image: torch.Tensor, pad: int = 4) -> torch.Tensor:
        if not self.training or pad <= 0:
            return image
        padded = F.pad(image, (pad, pad, pad, pad), mode="replicate")
        offset_y = int(torch.randint(0, 2 * pad + 1, (1,), device=image.device))
        offset_x = int(torch.randint(0, 2 * pad + 1, (1,), device=image.device))
        return padded[
            :, :, offset_y : offset_y + self.image_size, offset_x : offset_x + self.image_size
        ]

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = obs.shape[0]
        image = obs[:, : self.image_values].reshape(
            batch_size,
            3 * self.frame_stack,
            self.image_size,
            self.image_size,
        )
        image = self._random_shift(image / 255.0 - 0.5)
        image_features = self.image_projection(
            self.image_net(image).reshape(batch_size, -1)
        )

        route_end = self.image_values + 2 * self.num_waypoints
        route = obs[:, self.image_values : route_end] / 127.5 - 1.0
        route = route.reshape(batch_size, self.num_waypoints, 2).permute(0, 2, 1)
        route_features = self.route_projection(
            F.relu(self.route_conv(route)).reshape(batch_size, -1)
        )

        measurements = obs[:, route_end : route_end + 2]
        measurements = torch.stack(
            (measurements[:, 0] / 255.0, measurements[:, 1] / 127.5 - 1.0),
            dim=1,
        )
        measurement_features = self.measurement_projection(measurements)
        features = torch.cat(
            (image_features, route_features, measurement_features), dim=1
        )
        return self.backbone(features), None


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


def make_encoder(
    state_dim: int,
    hidden_dim: int,
    network: str,
    image_size: int = 84,
    frame_stack: int = 3,
    num_waypoints: int = 10,
) -> nn.Module:
    normalized = network.lower()
    if normalized in ("sac", "mlp"):
        return MlpEncoder(state_dim, hidden_dim)
    if normalized in ("pixel_sac", "pixel"):
        return PixelEncoder(
            state_dim, hidden_dim, image_size, frame_stack, num_waypoints
        )
    raise ValueError("Unknown SAC network: {}".format(network))


class GaussianActor(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.encoder = make_encoder(
            cfg.state_dim,
            cfg.hidden_dim,
            cfg.network,
            getattr(cfg, "image_size", 84),
            getattr(cfg, "frame_stack", 3),
            getattr(cfg, "max_waypoints", 10),
        )
        self.fc_mu = nn.Linear(self.encoder.output_dim, cfg.action_dim)
        self.fc_log_std = nn.Linear(self.encoder.output_dim, cfg.action_dim)
        self.action_bound = float(cfg.action_bound)

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
    def __init__(self, cfg: Any):
        super().__init__()
        self.encoder = make_encoder(
            cfg.state_dim,
            cfg.hidden_dim,
            cfg.network,
            getattr(cfg, "image_size", 84),
            getattr(cfg, "frame_stack", 3),
            getattr(cfg, "max_waypoints", 10),
        )
        self.net = nn.Sequential(
            nn.Linear(self.encoder.output_dim + cfg.action_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features, _ = self.encoder(obs)
        return self.net(torch.cat([features, action], dim=-1))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def adaptive_demonstration_weights(
    advantage: torch.Tensor,
    disagreement: torch.Tensor,
    temperature: float,
    advantage_beta: float,
    uncertainty_beta: float,
    weight_min: float,
    weight_max: float,
) -> torch.Tensor:
    """Return detached per-state BC weights with a fixed-BC neutral point."""

    advantage_gate = 2.0 * torch.sigmoid(advantage / max(temperature, 1e-6))
    uncertainty_gate = 1.0 - torch.exp(-disagreement)
    return (
        advantage_beta * advantage_gate
        + uncertainty_beta * uncertainty_gate
    ).clamp(min=weight_min, max=weight_max)


class SacAgent(BaseAgent):
    """Readable Soft Actor-Critic implementation for continuous CARLA control."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gamma = float(cfg.gamma)
        self.tau = float(cfg.tau)
        self.actor = GaussianActor(cfg).to(self.device)
        self.critic_1 = QNetwork(cfg).to(self.device)
        self.critic_2 = QNetwork(cfg).to(self.device)
        self.target_critic_1 = QNetwork(cfg).to(self.device)
        self.target_critic_2 = QNetwork(cfg).to(self.device)
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
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state)
            else:
                action = self.actor(state)[0]
        if was_training:
            self.actor.train()
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def update(
        self,
        batch: Dict[str, Any],
        expert_batch: Optional[Dict[str, Any]] = None,
        bc_coef: float = 0.0,
        bc_mode: str = "fixed",
    ) -> Dict[str, Any]:
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
        policy_q_1 = self.critic_1(states, new_actions)
        policy_q_2 = self.critic_2(states, new_actions)
        q_value = torch.min(policy_q_1, policy_q_2)
        actor_rl_loss = (
            self.log_alpha.exp().detach() * log_prob - q_value
        ).mean()
        actor_loss = actor_rl_loss
        demo_bc_loss = None
        demo_action_mae = None
        demo_logs = {}
        if expert_batch is not None:
            if bc_coef <= 0.0:
                raise ValueError("bc_coef must be positive with an expert batch")
            if bc_mode not in ("fixed", "adaptive"):
                raise ValueError("bc_mode must be 'fixed' or 'adaptive'")
            expert_states = torch.as_tensor(
                expert_batch["states"], dtype=torch.float32, device=self.device
            )
            expert_actions = torch.as_tensor(
                expert_batch["actions"], dtype=torch.float32, device=self.device
            )
            predicted_expert_actions = self.actor.deterministic(expert_states)
            per_sample_bc = F.mse_loss(
                predicted_expert_actions, expert_actions, reduction="none"
            ).mean(dim=1, keepdim=True)
            bc_weights = torch.ones_like(per_sample_bc)
            if bc_mode == "adaptive":
                critic_training = (self.critic_1.training, self.critic_2.training)
                self.critic_1.eval()
                self.critic_2.eval()
                with torch.no_grad():
                    expert_q_1 = self.critic_1(expert_states, expert_actions)
                    expert_q_2 = self.critic_2(expert_states, expert_actions)
                    policy_expert_q_1 = self.critic_1(
                        expert_states, predicted_expert_actions.detach()
                    )
                    policy_expert_q_2 = self.critic_2(
                        expert_states, predicted_expert_actions.detach()
                    )
                    expert_q = torch.min(expert_q_1, expert_q_2)
                    policy_expert_q = torch.min(
                        policy_expert_q_1, policy_expert_q_2
                    )
                    q_scale = 1.0 + 0.25 * (
                        expert_q_1.abs()
                        + expert_q_2.abs()
                        + policy_expert_q_1.abs()
                        + policy_expert_q_2.abs()
                    )
                    normalized_advantage = (
                        expert_q - policy_expert_q
                    ) / q_scale
                    normalized_disagreement = torch.maximum(
                        (expert_q_1 - expert_q_2).abs(),
                        (policy_expert_q_1 - policy_expert_q_2).abs(),
                    ) / q_scale
                    bc_weights = adaptive_demonstration_weights(
                        normalized_advantage,
                        normalized_disagreement,
                        temperature=float(
                            getattr(self.cfg, "demo_q_temperature", 0.1)
                        ),
                        advantage_beta=float(
                            getattr(self.cfg, "demo_advantage_beta", 1.0)
                        ),
                        uncertainty_beta=float(
                            getattr(self.cfg, "demo_uncertainty_beta", 1.0)
                        ),
                        weight_min=float(
                            getattr(self.cfg, "demo_bc_weight_min", 0.1)
                        ),
                        weight_max=float(
                            getattr(self.cfg, "demo_bc_weight_max", 2.0)
                        ),
                    )
                self.critic_1.train(critic_training[0])
                self.critic_2.train(critic_training[1])
                demo_logs = {
                    "demo_expert_advantage": normalized_advantage.mean().item(),
                    "demo_critic_disagreement": normalized_disagreement.mean().item(),
                    "demo_bc_weight_mean": bc_weights.mean().item(),
                    "demo_bc_weight_min": bc_weights.min().item(),
                    "demo_bc_weight_max": bc_weights.max().item(),
                    "demo_expert_preferred_rate": (
                        normalized_advantage > 0.0
                    ).float().mean().item(),
                }
            demo_bc_loss = (bc_weights * per_sample_bc).mean()
            demo_unweighted_bc_loss = per_sample_bc.mean()
            demo_action_mae = F.l1_loss(
                predicted_expert_actions, expert_actions
            )
            actor_loss = actor_loss + float(bc_coef) * demo_bc_loss
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
            "actor_rl_loss": actor_rl_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "avg_q": q_value.mean().item(),
            "critic_disagreement": (
                policy_q_1.detach() - policy_q_2.detach()
            ).abs().mean().item(),
            "entropy": -log_prob.mean().item(),
        }
        if demo_bc_loss is not None and demo_action_mae is not None:
            logs["demo_bc_loss"] = demo_bc_loss.item()
            logs["demo_unweighted_bc_loss"] = demo_unweighted_bc_loss.item()
            logs["demo_bc_action_mae"] = demo_action_mae.item()
            logs.update(demo_logs)
        if attention is not None:
            mean_attention = attention.detach().mean(dim=0)
            logs["attention_img"] = mean_attention[None, None, :].cpu()
        return logs

    def behavior_clone(
        self, batch: Dict[str, Any], coefficient: float = 1.0
    ) -> Dict[str, float]:
        """Supervise the SAC mean action without hiding the actor architecture."""

        if coefficient <= 0.0:
            raise ValueError("behavior-cloning coefficient must be positive")
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device
        )
        expert_actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        predicted_actions = self.actor.deterministic(states)
        bc_loss = F.mse_loss(predicted_actions, expert_actions)
        self.actor_optimizer.zero_grad()
        (float(coefficient) * bc_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        return {
            "bc_loss": float(bc_loss.item()),
            "bc_action_mae": float(
                F.l1_loss(predicted_actions, expert_actions).item()
            ),
        }

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
        description="Soft Actor-Critic with editable MLP or pixel encoders.",
    )
)
