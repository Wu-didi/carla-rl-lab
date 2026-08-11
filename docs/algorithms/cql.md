# Conservative Q-Learning (CQL(H))

**Family:** offline RL. **Paper:** [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779).
**Implementation:** [`carla_rl_lab/algorithms/offline.py`](../../carla_rl_lab/algorithms/offline.py).

## Principle

Offline Q-learning can assign unrealistically high values to actions absent
from the dataset. Continuous CQL(H) adds a conservative log-sum-exp penalty to
each critic:

```text
L_CQL(Q) = temperature * logsumexp(Q(s, sampled actions) / temperature)
           - Q(s, a_data)
L_critic = L_Bellman + cql_alpha * L_CQL
```

Candidate actions come from a uniform distribution, the current policy at the
current state, and the policy at the next state. Their known sampling
densities are subtracted before log-sum-exp. The Gaussian actor otherwise uses
the SAC objective with fixed `offline_entropy_alpha`.

## Start Training

```bash
python scripts/train_offline.py \
  --algo cql \
  --dataset artifacts/datasets/nocrash_expert_seed0_100k.npz \
  --updates 100000 --batch-size 256 --checkpoint-interval 10000 \
  --seed 0 --logger tensorboard --run-name nocrash/cql_mlp_seed0
```

The dataset is produced with the collection command in [TD3+BC](td3_bc.md).
Its metadata automatically sets state dimension, action mode, and observation
layout.

## Metrics And Results

Plot `bellman_loss`, `conservative_loss`, both total critic losses,
`actor_loss`, and `avg_q`. An excessively dominant conservative loss can
collapse all values; a small value can fail to control extrapolation. CPU
update/checkpoint tests pass. Formal CARLA-dataset training and curves are
**Pending**.
The current CQL networks are MLPs; pixel-native training remains pending.
