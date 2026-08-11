# Generative Adversarial Imitation Learning (GAIL)

**Family:** expert data plus online on-policy interaction. **Paper:** [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476).
**Implementation:** [`carla_rl_lab/algorithms/imitation.py`](../../carla_rl_lab/algorithms/imitation.py).

## Principle

GAIL trains a discriminator to distinguish expert state-action pairs from
policy pairs. The discriminator uses binary cross-entropy, while the policy
receives the non-saturating shaped reward `softplus(discriminator_logit)` and
updates through the repository's PPO implementation. This matches occupancy
measures instead of directly regressing expert actions.

```text
L_D = BCE(D(s_E,a_E), 1) + BCE(D(s_pi,a_pi), 0)
r_imitation = softplus(D_logit(s_pi,a_pi))
```

## Start Training

```bash
python scripts/train_imitation.py \
  --benchmark nocrash_train_empty_v0 \
  --algo gail \
  --expert-dataset artifacts/datasets/nocrash_expert_seed0_100k.npz \
  --total-timesteps 100000 --rollout-steps 2048 \
  --checkpoint-interval 10000 --seed 0 --logger tensorboard \
  --run-name nocrash/gail_mlp_seed0
```

The dataset action representation is loaded automatically and must match the
online environment.

## Metrics And Results

Inspect discriminator loss, expert/policy accuracy, imitation reward, PPO
actor/value losses, entropy, KL, clip fraction, and environment benchmark
metrics. Persistent 100% discriminator accuracy usually means poor reward
signal rather than success. CPU mixed-batch update/checkpoint tests pass;
formal CARLA training and curves are **Pending**.
The current policy/discriminator are MLPs; pixel-native GAIL remains pending.
