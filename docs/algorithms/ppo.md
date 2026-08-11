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
  --algo ppo --total-timesteps 100000 --rollout-steps 2048 \
  --checkpoint-interval 10000 --ppo-epochs 10 --ppo-minibatch-size 64 \
  --action-mode longitudinal_2d --reward research_v2 \
  --town Town05 --vehicles 50 --walkers 0 --traffic off --view-mode none \
  --max-time-episode 500 --seed 0 --logger tensorboard \
  --run-name v0.1/ppo_town05_seed0
```

## Metrics And Results

Inspect `actor_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`,
return/cost, and action statistics. `approx_kl` and `clip_fraction` diagnose
updates that are too large even when return is noisy. A 32-step CARLA 0.9.15
integration smoke completed with one rollout/update and valid checkpoint. That
is runner evidence only; a formal result and curves are **Pending**.
