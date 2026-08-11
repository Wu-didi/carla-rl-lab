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
| Observation | `pixel_v1`: three `84x84` RGB frames, 10 route points, speed, steer |
| Action | `target_speed_2d` |
| Reward | `nocrash_v0` |
| Train map | Town01 |
| Primary train benchmark | `nocrash_train_v0` |
| Curriculum | `nocrash_train_empty_v0`, labeled separately |
| Train seeds | 0, 1, 2 |
| Evaluation | `nocrash_0915_v0`, all 25 routes x 2 weathers x 3 densities |
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
blockage, termination reason, and checkpoint SHA-256. Publish per-seed values,
then mean and standard deviation. Checkpoint selection must not use the final
test suite unless that selection procedure is disclosed.

## Current Verified Smoke

On 2026-08-12, the current `pixel_v1` SAC path was executed against a real
CARLA 0.9.15 server:

| Item | Result |
| --- | --- |
| Camera/route smoke | 10 steps, nonblank RGB, packed state 63,526 |
| Online training smoke | 64 CARLA steps, 57 SAC updates |
| Artifacts | TensorBoard events plus checkpoints at 32 and 64 steps |
| Limited evaluation | One Town02 route and one weather; failed after collision |

The poor limited evaluation is expected for a 64-step policy and is recorded
as a negative smoke, not a baseline. Longer pilots and full multi-seed results
must replace this section before a release claims learned driving performance.

## Integration Matrix

With a CARLA server already running:

```bash
CARLA_PORT=2000 scripts/run_research_smoke.sh
```

Defaults are 64 expert transitions, 64 online environment steps, and 8 offline
updates. This checks collection, SAC, PPO, BC, TD3+BC, TensorBoard, and
checkpoint wiring on the named Town01 curriculum. Only SAC currently uses the
pixel-native encoder; the other tracks are interface smoke tests.

## Publishing

Use `scripts/export_curves.py` to create PNG/CSV/JSON summaries from TensorBoard
and an evaluation report. Large checkpoints and datasets belong in GitHub
Releases or external storage, not Git history. A release table should link the
exact checkpoint, config, raw scalars, evaluation JSON, and source commit.
