# Soft Actor-Critic (SAC)

**Family:** online, off-policy.
**Paper:** [Soft Actor-Critic](https://arxiv.org/abs/1801.01290).
**Implementation:** [`carla_rl_lab/algorithms/sac.py`](../../carla_rl_lab/algorithms/sac.py).
**Runner:** [`scripts/train.py`](../../scripts/train.py).

## Principle

SAC learns a stochastic tanh-squashed Gaussian actor and two Q-functions. The
critic target is:

```text
y = r + gamma * (1 - terminal) *
    [min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a'|s')]
```

Each critic minimizes Bellman mean-squared error. The actor minimizes
`E[alpha * log pi(a|s) - min(Q1, Q2)]`. Entropy temperature `alpha` is learned
against `target_entropy`, and target critics use Polyak updates.

`PixelEncoder` decodes the packed RGB/route/measurement state. Four image
convolutions, a route convolution, and a measurement MLP are fused before the
actor or critic head. Random-shift augmentation is active only during training.
The implementation deliberately keeps separate editable encoders for actor and
critics instead of hiding them in a framework policy class.

## Start

Plain pixel SAC:

```bash
python scripts/train.py \
  --algo sac --network Pixel_SAC \
  --benchmark nocrash_train_regular_v0 \
  --total-timesteps 20000 --checkpoint-interval 2000 \
  --minimal-size 1500 --batch-size 64 --buffer-size 15000 \
  --hidden-dim 128 --view-mode none --logger tensorboard \
  --run-name pilots/rlfold_town01_regular_pixel_sac_seed0_20k \
  --seed 0 --port 2000 --require-clean-git
```

The runner can also initialize the actor from demonstrations and retain an
editable BC term during online SAC updates:

```bash
python scripts/train.py \
  --algo sac --network Pixel_SAC \
  --benchmark nocrash_train_regular_v0 \
  --total-timesteps 20000 --checkpoint-interval 2000 \
  --minimal-size 1500 --batch-size 64 --buffer-size 15000 \
  --hidden-dim 128 \
  --expert-dataset artifacts/datasets/rlfold_town01_regular_behavior_agent_seed0_10k.npz \
  --demo-pretrain-updates 5000 --demo-bc-coef 0.5 \
  --view-mode none --logger tensorboard \
  --run-name pilots/rlfold_town01_regular_pixel_sac_demo_seed0_20k \
  --seed 0 --port 2000 --require-clean-git
```

This uses `L_actor = L_SAC + 0.5 * L_BC`; it is demonstration-assisted SAC,
not plain SAC and not a new algorithm name. Resume with `--checkpoint` and a
larger absolute `--total-timesteps`.

An optional [CADR research prototype](../research/cadr.md) replaces the fixed
per-sample BC weight with a conservative expert-advantage and twin-critic
disagreement gate. It is an ablation target, not a validated improvement.

## Inspect

Plot `train/critic_1_loss`, `train/critic_2_loss`, `train/actor_loss`,
`train/alpha_loss`, `train/alpha`, `train/entropy`, and `train/avg_q` together
with `episode/reward`, `episode/cost`, episode length, action distributions, and
every `reward/*` term. Finite losses do not prove that the vehicle can finish a
route.

Quick evaluation:

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/sac_ckpt_last.pt \
  --benchmark nocrash_empty_v0 --routes 1 --weathers 1
```

Formal evaluation replaces the limited target with `--suite
rlfold_nocrash_0915_v0 --output-tag selected`. Add `--resume` after an
interrupted evaluation; completed episodes are validated and reused.

## Current Result

The 2026-08-13 CARLA 0.9.15 seed-0 pilot trained both variants for 20k online
steps in Town01 Regular traffic. On the common 10-episode Town02 Empty selector,
plain SAC selected step 8k with 0% success and 0.360 mean route completion.
Demonstration-assisted SAC selected step 8k with **60% success**, 0.786 mean
completion, 20% collision rate, and 20% off-road rate.

The frozen assisted checkpoint then scored **46% (23/50)** on the full Empty
split, **24% (12/50)** on Regular, and **4% (2/50)** on Dense. Exact
per-episode reports are published in the [seed-0 evidence bundle](../../results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/README.md).
These are pilot results, not a three-seed baseline or a reproduction claim for
the original RLfOLD paper. The old Town05 vector/risk-field experiment is not
part of the current `pixel_v1` contract.
