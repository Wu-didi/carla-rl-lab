# CADR Research Prototype

**Working name:** Confidence-Adaptive Demonstration Regularization (CADR).
**Status:** implemented and unit-tested; CARLA ablation pending.
**Claim boundary:** research hypothesis, not a novel-method or performance
claim until literature review and multi-seed experiments are complete.

## Problem Observed

In the first seed-0 pilot, fixed-weight demonstration-assisted Pixel SAC peaked
at step 8k with 60% selector success, then fell to 10% at 12k and 0% at 16k and
20k. Pixel TD3 showed a similar 40% to 0% collapse. Training return remained
positive while critic losses grew, so return alone did not expose the policy
regression. The frozen SAC checkpoint also fell from 46% Empty success to 24%
Regular and lower performance in Dense traffic.

This motivates two testable questions:

1. Does twin-critic disagreement predict checkpoint regression before route
   success collapses?
2. Can a state-dependent demonstration constraint preserve the early policy
   while still allowing SAC to improve beyond BehaviorAgent actions?

## Method

For a demonstration state-action pair `(s, a_E)`, let `a_pi` be the
deterministic actor action and use the conservative twin-critic estimate:

```text
Q_min(s, a) = min(Q1(s, a), Q2(s, a))
A_E         = [Q_min(s, a_E) - Q_min(s, a_pi)] / Q_scale
U           = max(|Q1(s, a_E) - Q2(s, a_E)|,
                  |Q1(s, a_pi) - Q2(s, a_pi)|) / Q_scale
Q_scale     = 1 + mean(|Q1_E|, |Q2_E|, |Q1_pi|, |Q2_pi|)
w           = clip(beta_A * 2 * sigmoid(A_E / T)
                   + beta_U * [1 - exp(-U)], w_min, w_max)
L_actor     = L_SAC + lambda * mean[w * ||a_pi - a_E||^2]
```

The advantage gate emphasizes expert actions that the critics currently rank
above the policy. The disagreement gate retains an expert anchor where that
ranking is unreliable. `Q_scale` makes both signals less sensitive to critic
magnitude drift. The factor `2` makes a neutral advantage produce weight `1`,
matching fixed BC rather than silently reducing its scale. Critic values are
detached, so CADR only changes the actor regularizer and remains a small
modification to ordinary SAC.

Defaults:

```text
lambda=0.5, T=0.1, beta_A=1.0, beta_U=1.0, w_min=0.1, w_max=2.0
```

Enable it with:

```bash
python scripts/train.py \
  --algo sac --benchmark nocrash_train_regular_v0 \
  --total-timesteps 20000 --checkpoint-interval 2000 \
  --minimal-size 1500 --batch-size 64 --buffer-size 15000 \
  --hidden-dim 128 \
  --expert-dataset artifacts/datasets/rlfold_town01_regular_behavior_agent_seed0_10k.npz \
  --demo-pretrain-updates 5000 --demo-bc-coef 0.5 \
  --demo-bc-mode adaptive --demo-q-temperature 0.1 \
  --demo-advantage-beta 1.0 --demo-uncertainty-beta 1.0 \
  --demo-bc-weight-min 0.1 --demo-bc-weight-max 2.0 \
  --view-mode none --logger tensorboard \
  --run-name ablations/cadr_seed0_20k --seed 0 --port 2000
```

## Logged Diagnostics

- `train/critic_disagreement`: disagreement on current online policy actions.
- `train/demo_expert_advantage`: normalized expert-versus-policy value gap.
- `train/demo_critic_disagreement`: uncertainty on demonstration states.
- `train/demo_expert_preferred_rate`: fraction where conservative critics rank
  the expert action higher.
- `train/demo_bc_weight_mean|min|max`: realized CADR weights.
- `train/demo_bc_loss` and `train/demo_unweighted_bc_loss`: weighted and raw BC
  error, needed to distinguish weighting from easier samples.

## Preregistered Ablation

Keep observation, reward, action, dataset, seed, training traffic, update count,
candidate steps, and selector grid fixed.

| Arm | Pretrain | Online demonstration loss | Purpose |
| --- | ---: | --- | --- |
| Plain SAC | none | none | RL baseline |
| Pretrain only | 5k | none | isolate actor initialization |
| Fixed BC | 5k | `lambda=0.5` | current best pilot |
| Advantage only | 5k | `beta_A=1, beta_U=0` | compare with Q-filter family |
| Disagreement only | 5k | `beta_A=0, beta_U=1` | isolate uncertainty anchoring |
| CADR | 5k | full adaptive weight | proposed prototype |

Primary pilot endpoint: 10-episode selector success at fixed steps 8k, 12k,
16k, and 20k. Secondary endpoints: route completion, collisions, off-road rate,
performance drop from best to final checkpoint, critic disagreement, and three
traffic-density test performance for the selected checkpoint. The full suite
is run only after selecting on the limited grid.

At least seeds 0/1/2 are required before drawing a method conclusion. Report
all arms even if CADR underperforms.

## Related Work Boundary

CADR combines known ideas in a specific CARLA online-fine-tuning setting; it
must not be described as generally unprecedented.

- [RLfOLD](https://doi.org/10.1609/aaai.v38i10.29049) uses an online privileged
  expert and separate policy standard deviations for RL and imitation.
- [AWAC](https://arxiv.org/abs/2006.09359) uses advantage-weighted maximum
  likelihood to combine prior data and online learning.
- Q-filter methods imitate an expert when a critic ranks its action above the
  policy action.
- [UWAC](https://proceedings.mlr.press/v139/wu21i.html) uses uncertainty
  weighting to stabilize offline actor-critic learning.
- [Cycle-of-Learning](https://arxiv.org/abs/1910.04281) studies fixed combined
  BC and actor-critic losses during the transition to RL.

The narrower hypothesis is whether *twin-critic disagreement plus conservative
expert advantage* can prevent late-training regression and improve
traffic-density transfer for pixel SAC in this open CARLA benchmark. A novelty
claim requires a broader systematic search and positive controlled evidence.
