from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def mlp(sizes: Sequence[int], activation: nn.Module = nn.ReLU, output_activation: nn.Module = nn.Identity) -> nn.Sequential:
    layers = []
    for index in range(len(sizes) - 1):
        act = activation if index < len(sizes) - 2 else output_activation
        layers.extend([nn.Linear(sizes[index], sizes[index + 1]), act()])
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: float):
        super().__init__()
        self.net = mlp([state_dim, hidden_dim, hidden_dim, hidden_dim, action_dim])
        self.action_bound = action_bound

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state)) * self.action_bound


class ContinuousCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = mlp([state_dim + action_dim, hidden_dim, hidden_dim, hidden_dim, 1])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
