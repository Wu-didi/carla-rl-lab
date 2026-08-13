# Experiment Protocol

This document defines what may be called a CarlaRLLab result. A successful
smoke proves integration only; it does not prove driving quality.

## Evidence Levels

| Level | Required evidence |
| --- | --- |
| CPU test | One finite update plus checkpoint round trip |
| CARLA smoke | Real reset/step, sensor data, update, log, and checkpoint with a small fixed budget |
| Pilot | One training seed and explicitly limited fixed-route evaluation |
| Baseline | Three training seeds and the complete 150-episode NoCrash 0.9.15 suite |

The evidence label must appear next to every result table and curve.

## Primary Baseline Contract

| Field | Fixed value |
| --- | --- |
| CARLA | 0.9.15 client and server |
| Observation | `pixel_v1`: two `84x84` RGB frames, 10 route points, speed, steer |
| Action | `target_speed_2d` |
| Reward | `nocrash_v0` |
| Train map | Town01 |
| Primary train benchmark | `nocrash_train_v0` |
| Curriculum | `nocrash_train_empty_v0`, labeled separately |
| Train seeds | 0, 1, 2 |
| Evaluation | `rlfold_nocrash_0915_v0`, all 25 routes x 2 weathers x 3 densities |
| Logging | TensorBoard; W&B optional; full config always saved |

No two algorithm rows may use different observations, route files, weather,
traffic, reward, action representation, or checkpoint-selection rules unless
the table labels the difference as an ablation.

## Required Outputs

For every training seed retain:

- episode return, cost, length, and every reward term;
- actor/critic/value/discriminator losses as applicable;
- entropy, Q estimates, alpha, and action distributions as applicable;
- wall-clock duration, device, source commit, CARLA versions, and full config;
- immutable step checkpoints and `*_ckpt_last.pt`.

For every evaluation episode retain route, weather, traffic density, success,
route completion, distance, speed, collision category, red-light event,
blockage, requested and actually spawned actor counts, termination reason, and
checkpoint SHA-256. Publish per-seed values, then mean and standard deviation.
Checkpoint selection must not use the final test suite unless that selection
procedure is disclosed.

## Current Seed-0 Pilot

On 2026-08-13, five pixel-native methods were trained with seed 0 on
`nocrash_train_regular_v0` (fixed Town01 20/50 traffic). Checkpoints were
selected on the same limited 10-episode Town02 Empty grid.

| Method | Budget | Selected checkpoint | Selection success |
| --- | ---: | ---: | ---: |
| SAC | 20k environment steps | 8k | 0% |
| TD3 | 20k environment steps | 8k | 40% |
| BC | 10k updates / 10k demonstrations | 10k | 20% |
| PPO | 20k environment steps | 20k | 0% |
| SAC + demonstrations | 5k BC pretrain + 20k environment steps | 8k | 60% |

The frozen SAC + demonstrations checkpoint achieved 46% success on all 50
Empty episodes, 24% on all 50 Regular episodes, and 4% on all 50 Dense
episodes. See the
[tracked evidence bundle](../results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/)
for curves, raw scalar exports, per-episode reports, commands, and hashes.

This remains a **pilot** because it has one seed, a 20k budget, and a fixed
20/50 curriculum rather than the primary sampled 0-150/0-300 training
distribution. Test-suite scores must not be used for further checkpoint tuning.

## Verified Smoke

On 2026-08-12, the current `pixel_v1` SAC path was executed against a real
CARLA 0.9.15 server:

| Item | Result |
| --- | --- |
| Camera/route smoke | 10 steps, nonblank RGB, packed state 42,358 |
| Online training smoke | 64 CARLA steps, 57 SAC updates |
| Artifacts | TensorBoard events plus checkpoints at 32 and 64 steps |
| Limited evaluation | One Town02 route and one weather; failed after collision |

The poor limited evaluation is expected for a 64-step policy and remains a
negative integration record, not a performance result.

## Integration Matrix

With a CARLA server already running:

```bash
CARLA_PORT=2000 scripts/run_research_smoke.sh
```

Defaults are 64 expert transitions, 64 online environment steps, and 8 offline
updates. This checks collection, SAC, PPO, BC, TD3+BC, TensorBoard, and
checkpoint wiring on the named Town01 curriculum. SAC, TD3, PPO, and BC have
pixel-native policies; offline and adversarial-imitation tracks remain MLP
interface tests.

## Publishing

Use `scripts/export_curves.py` to create PNG/CSV/JSON summaries from TensorBoard
and an evaluation report. Large checkpoints and datasets belong in GitHub
Releases or external storage, not Git history. A release table should link the
exact checkpoint, config, raw scalars, evaluation JSON, and source commit.
