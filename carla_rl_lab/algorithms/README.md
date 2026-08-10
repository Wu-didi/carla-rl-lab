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
| PPO | on-policy | planned | Add rollout buffer, clipped policy loss, GAE. |
| TD3 | off-policy | implemented | Compact baseline with twin critics, delayed actor update, target policy smoothing. |
| DDPG | off-policy | implemented | Compact deterministic actor/critic baseline. |
| GAIL | imitation | planned | A pre-refactor experiment is preserved locally under `legacy/`. |
| CQL / IQL / TD3+BC | offline RL | planned | Requires a stable dataset API and offline evaluator. |

## Adding an Algorithm

1. Create a module such as `carla_rl_lab/algorithms/td3.py`.
2. Implement `BaseAgent.act`, `BaseAgent.update`, `BaseAgent.save`, and `BaseAgent.load`.
3. Register it with `register_algorithm(AlgorithmSpec(...))`.
4. Import the module in `carla_rl_lab/algorithms/__init__.py`.

Keep neural network definitions next to the algorithm implementation unless
they are shared by multiple algorithms.
