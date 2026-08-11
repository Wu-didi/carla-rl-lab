# Adversarial Inverse Reinforcement Learning (AIRL)

**Family:** expert data plus online on-policy interaction. **Paper:** [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248).
**Implementation:** [`carla_rl_lab/algorithms/imitation.py`](../../carla_rl_lab/algorithms/imitation.py).

## Principle

AIRL decomposes the discriminator score into a learned reward and a potential
shaping term:

```text
f(s,a,s') = r_theta(s,a) + gamma*(1-terminal)*h(s') - h(s)
logit     = f(s,a,s') - log pi(a|s)
```

The terminal mask prevents potential from leaking across true episode ends.
The policy receives the AIRL logit as its shaped reward and updates with PPO.
Unlike GAIL, AIRL requires complete expert transitions, including next state
and terminal semantics.

## Start Training

```bash
python scripts/train_imitation.py \
  --benchmark nocrash_train_empty_v0 \
  --algo airl \
  --expert-dataset artifacts/datasets/nocrash_expert_seed0_100k.npz \
  --total-timesteps 100000 --rollout-steps 2048 \
  --checkpoint-interval 10000 --seed 0 --logger tensorboard \
  --run-name nocrash/airl_mlp_seed0
```

## Metrics And Results

Inspect discriminator loss/accuracies, `imitation_reward`, PPO losses,
entropy/KL, and fixed benchmark metrics. The schema-v2 dataset distinguishes
true terminals from timeouts, which is required by the potential term. CPU
transition/update/checkpoint tests pass; formal CARLA training and curves are
**Pending**.
The current policy/discriminator are MLPs; pixel-native AIRL remains pending.
