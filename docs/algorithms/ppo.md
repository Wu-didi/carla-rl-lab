# Proximal Policy Optimization (PPO)

**Family:** online, on-policy. **Paper:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
**Implementation:** [`carla_rl_lab/algorithms/on_policy.py`](../../carla_rl_lab/algorithms/on_policy.py).
**Runner:** [`scripts/train_on_policy.py`](../../scripts/train_on_policy.py).

## Principle

PPO collects a fresh rollout, computes generalized advantage estimates (GAE),
then reuses that rollout for several minibatch epochs. The clipped policy loss
limits destructive updates:

```text
ratio = exp(log pi_new(a|s) - log pi_old(a|s))
L_pi  = -mean[min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)]
L     = L_pi + value_coef*L_value - entropy_coef*entropy
```

The implementation uses a tanh-squashed Gaussian actor, a value network,
timeout-aware GAE, normalized advantages, and gradient clipping.

## Start Training

```bash
python scripts/train_on_policy.py \
  --algo ppo --network Pixel_SAC \
  --benchmark nocrash_train_regular_v0 \
  --total-timesteps 20000 --rollout-steps 1024 \
  --checkpoint-interval 4000 --hidden-dim 128 \
  --ppo-epochs 5 --ppo-minibatch-size 64 \
  --view-mode none --logger tensorboard \
  --run-name pilots/rlfold_town01_regular_pixel_ppo_seed0_20k \
  --seed 0 --port 2000 --require-clean-git
```

## Metrics And Results

Inspect `actor_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`,
return/cost, and action statistics. `approx_kl` and `clip_fraction` diagnose
updates that are too large even when return is noisy.

The 2026-08-13 pixel-native PPO seed-0 pilot completed 20k CARLA 0.9.15 steps
and 20 rollout updates. Its selected 20k checkpoint scored **0% success**,
0.482 mean route completion, 60% collision rate, and 80% off-road rate on the
fixed 10-episode selector. This is a useful negative pilot rather than a
competitive baseline: it verifies the full image runner and shows that this
budget/configuration is insufficient. Curves and exact metadata are in the
[evidence bundle](../../results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/README.md).
