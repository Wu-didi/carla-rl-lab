from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from carla_rl_lab.algorithms.imitation import AirlDiscriminator


class ConstantNetwork(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, inputs):
        return torch.full(
            (inputs.shape[0], 1), self.value, dtype=inputs.dtype, device=inputs.device
        )


class AlgorithmDetailsTest(unittest.TestCase):
    def test_airl_masks_terminal_next_potential(self):
        discriminator = AirlDiscriminator(2, 4, 1, gamma=0.9)
        discriminator.reward = ConstantNetwork(0.0)
        discriminator.potential = ConstantNetwork(1.0)
        states = torch.zeros((2, 2))
        actions = torch.zeros((2, 1))
        next_states = torch.zeros((2, 2))
        dones = torch.tensor([0.0, 1.0])
        output = discriminator(states, actions, next_states, dones)
        torch.testing.assert_close(output, torch.tensor([-0.1, -1.0]))


if __name__ == "__main__":
    unittest.main()
