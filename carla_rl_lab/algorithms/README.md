# Algorithms

This project is research-first: algorithm files should stay readable and easy
to edit. Avoid hiding actor/critic/loss details behind heavy wrappers.

## Taxonomy

`online/offline` and `on-policy/off-policy` describe different things:

- `data_source=online`: collect transitions from CARLA during training.
- `data_source=offline`: train from a fixed dataset without CARLA interaction.
- `family=on_policy`: update from fresh rollouts, such as PPO and A2C.
- `family=off_policy`: update from replay data, such as SAC, TD3, and DDPG.
- `family=offline_rl`: learn from a fixed dataset, such as CQL, IQL, and TD3+BC.
- `family=imitation`: learn from demonstrations, such as BC, GAIL, and AIRL.

The `runner` metadata prevents an algorithm from being connected to the wrong
training loop. The v1 trainer is an `off_policy` runner.

## v1 Support

| Algorithm | Family | Status | Notes |
| --- | --- | --- | --- |
| SAC | off-policy | implemented | Editable MLP or semantic-attention actor and critics. |
| PPO / A2C | on-policy | implemented | Dedicated rollout buffer, GAE, clipped PPO and synchronous A2C losses. |
| TD3 | off-policy | implemented | Compact baseline with twin critics, delayed actor update, target policy smoothing. |
| DDPG | off-policy | implemented | Compact deterministic actor/critic baseline. |
| BC / GAIL / AIRL | imitation | implemented | Expert-only BC and PPO-based adversarial mixed runner. |
| CQL / IQL / TD3+BC | offline RL | implemented | Validated `.npz` dataset API and fixed-dataset runner. |

## Dataset Format

Offline RL and AIRL consume `.npz` files with equally sized `states`,
`actions`, `rewards`, `next_states`, and `dones` arrays. BC and GAIL accept a
smaller expert file containing only `states` and `actions`.

## Adding an Algorithm

1. Create a module such as `carla_rl_lab/algorithms/td3.py`.
2. Implement `BaseAgent.act`, `BaseAgent.update`, `BaseAgent.save`, and `BaseAgent.load`.
3. Register it with `register_algorithm(AlgorithmSpec(...))`.
4. Import the module in `carla_rl_lab/algorithms/__init__.py`.

Keep neural network definitions next to the algorithm implementation unless
they are shared by multiple algorithms.
