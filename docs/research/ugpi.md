# UGPI Research Prototype

**Working name:** Uncertainty-Gated Policy Improvement (UGPI).
**Status:** implemented and unit-tested; CARLA seed-0 ablation pending.
**Claim boundary:** a testable prototype, not a novelty or performance claim.

## Motivation

Fixed-BC Pixel SAC reached 60% success at 8k steps on the 10-episode Empty
selector, then regressed to 0% at 20k. CADR attempted to prevent this by
increasing demonstration regularization when twin critics disagreed. Its four
evaluated checkpoints all reached only 20% selector success. During the CADR
run, mean online critic disagreement increased from 0.011 to 1.535 while the
adaptive BC weight increased from 1.008 to 1.303.

This negative result suggests that critic disagreement should control the
critic-driven policy update directly, rather than indirectly strengthening BC.

## Method

For each online state and sampled policy action, evaluate both critics with
image augmentation disabled so they receive the same observation:

```text
Q_scale = 1 + mean(|Q1(s, a_pi)|, |Q2(s, a_pi)|)
U       = |Q1(s, a_pi) - Q2(s, a_pi)| / Q_scale
c       = clip(exp(-beta * U), c_min, 1)
L_actor = mean[alpha * log pi(a_pi|s) - c * min(Q1, Q2)]
          + lambda * L_BC
```

The confidence `c` is detached. It only scales the critic-driven part of each
actor update; critic targets, critic losses, entropy regularization,
entropy-temperature learning, and fixed demonstration loss are unchanged. At
`U=0`, `c=1`, so the update exactly recovers the fixed-BC SAC objective. The
first ablation uses `beta=2`, `c_min=0.1`, and fixed `lambda=0.5`.

Enable it with:

```bash
python scripts/train.py \
  --algo sac --network Pixel_SAC \
  --benchmark nocrash_train_regular_v0 \
  --expert-dataset artifacts/datasets/rlfold_town01_regular_behavior_agent_seed0_10k.npz \
  --demo-pretrain-updates 5000 --demo-bc-coef 0.5 \
  --demo-bc-mode fixed --actor-update-mode confidence \
  --actor-uncertainty-beta 2.0 --actor-confidence-min 0.1
```

Diagnostics are logged as `train/actor_confidence_mean`,
`train/actor_confidence_min`, `train/actor_normalized_disagreement`, and both
weighted and unweighted actor-RL losses.

## Evaluation Contract

The seed-0 pilot keeps the fixed-BC run's dataset, training benchmark,
hyperparameters, 20k online-step budget, and 8k/12k/16k/20k selector grid.
Primary endpoint is selector success. Route completion, collision rate,
off-road rate, and best-to-final regression are secondary endpoints. At least
three seeds and the complete NoCrash density suite are required before making
a method claim.
