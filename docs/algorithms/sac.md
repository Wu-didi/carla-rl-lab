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

```bash
python scripts/train.py \
  --benchmark nocrash_train_empty_v0 \
  --algo sac --network Pixel_SAC \
  --total-timesteps 100000 \
  --minimal-size 1500 --batch-size 128 --buffer-size 30000 \
  --hidden-dim 256 --checkpoint-interval 10000 \
  --logger tensorboard \
  --run-name nocrash/pixel_sac_empty_seed0 --seed 0
```

After the curriculum, train the primary regular-traffic configuration by using
`--benchmark nocrash_train_v0`. Resume with `--checkpoint` and a larger absolute
`--total-timesteps`.

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
nocrash_0915_v0`.

## Current Result

**Evidence: CARLA integration smoke, not a baseline.** On 2026-08-12 the pixel
path completed 64 real CARLA 0.9.15 steps and 57 gradient updates, writing
TensorBoard scalars and checkpoints at steps 32 and 64. A one-route,
one-weather Town02 evaluation terminated in a layout collision after 70 steps,
with about 9.1% route completion. This negative result is expected at 64 steps
and establishes only that training and evaluation use the same sensor/action
contract.

The earlier 10k Town05 vector/risk-field pilot is not a current input baseline
and must not be compared with `pixel_v1`. A publishable SAC row still requires
three converged seeds and the complete 150-episode suite.
