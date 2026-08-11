# Implicit Q-Learning (IQL)

**Family:** offline RL. **Paper:** [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).
**Implementation:** [`carla_rl_lab/algorithms/offline.py`](../../carla_rl_lab/algorithms/offline.py).

## Principle

IQL avoids evaluating unseen next actions. A value network fits an upper
expectile of the target critics on dataset actions, critics bootstrap from that
value, and the actor performs advantage-weighted behavior cloning:

```text
L_V     = mean[|expectile - I(Q-V < 0)| * (Q-V)^2]
y_Q     = r + gamma * (1-terminal) * V(s')
weight  = clip(exp(beta * (Q(s,a)-V(s))), max=iql_max_weight)
L_actor = -mean[weight * log pi(a_data|s)]
```

The actor never directly maximizes Q over arbitrary actions, reducing
out-of-distribution action queries.

## Start Training

```bash
python scripts/train_offline.py \
  --algo iql \
  --dataset artifacts/datasets/v0.1/town05_autopilot_seed0_100k.npz \
  --updates 100000 --batch-size 256 --checkpoint-interval 10000 \
  --seed 0 --logger tensorboard --run-name v0.1/iql_town05_seed0
```

## Metrics And Results

Inspect `value_loss`, both critic losses, `actor_loss`, mean `advantage`, and
`actor_weight`. Saturated weights indicate that `iql_beta` or reward scale may
be too aggressive. CPU update/checkpoint tests pass. Formal dataset training,
CARLA evaluation, and curves are **Pending**.
