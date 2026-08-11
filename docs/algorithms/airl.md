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
  --algo airl \
  --expert-dataset artifacts/datasets/v0.1/town05_autopilot_seed0_100k.npz \
  --total-timesteps 100000 --rollout-steps 2048 \
  --checkpoint-interval 10000 --town Town05 --vehicles 50 --walkers 0 \
  --traffic off --view-mode none --max-time-episode 500 \
  --reward research_v2 --seed 0 --logger tensorboard \
  --run-name v0.1/airl_town05_seed0
```

## Metrics And Results

Inspect discriminator loss/accuracies, `imitation_reward`, PPO losses,
entropy/KL, and fixed benchmark metrics. The schema-v2 dataset distinguishes
true terminals from timeouts, which is required by the potential term. CPU
transition/update/checkpoint tests pass; formal CARLA training and curves are
**Pending**.
