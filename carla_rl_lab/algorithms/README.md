# Algorithms

Algorithm modules keep actor, critic, and loss equations visible. The registry
classifies methods by data source, update family, and runner so incompatible
training loops fail early.

| Data source | Family | Algorithms | Runner |
| --- | --- | --- | --- |
| online | off-policy | SAC, TD3, DDPG | `scripts/train.py` |
| online | on-policy | PPO, A2C | `scripts/train_on_policy.py` |
| offline | offline RL | TD3+BC, CQL, IQL | `scripts/train_offline.py` |
| expert/mixed | imitation | BC, GAIL, AIRL | `scripts/train_imitation.py` |

SAC currently owns the `pixel_v1` CNN/route encoder. Other algorithms retain
small MLP implementations while their shared pixel adapter is pending.

## Dataset Schema

Schema v2 stores `states`, `actions`, `rewards`, `next_states`, `terminals`,
`timeouts`, `episode_ids`, and `costs`, plus JSON metadata. Pixel states remain
`uint8` on disk and in replay. Training exposes `dones=terminals`, allowing
targets to bootstrap across time-limit truncations.

## Adding An Algorithm

1. Add one module under `carla_rl_lab/algorithms/`.
2. Implement `act`, `update`, `save`, and `load` on `BaseAgent`.
3. Register one `AlgorithmSpec` with the correct runner.
4. Import the module from `carla_rl_lab/algorithms/__init__.py`.
5. Add a finite-update and checkpoint-roundtrip test, then a named CARLA smoke.

Keep networks beside the algorithm until two real implementations share the
same behavior. Avoid inheritance chains for code reuse.
