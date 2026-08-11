# Algorithm Guides

Each guide explains the implemented objective, its direct source mapping, the
exact launch command, the metrics that should be inspected, and the current
CarlaRLLab result. A method is not called a baseline until it has completed the
fixed multi-seed protocol in [`../experiments.md`](../experiments.md).

## Evidence Levels

| Level | Meaning |
| --- | --- |
| CPU test | One update and checkpoint round trip without CARLA |
| Integration smoke | A bounded runner/data/log/checkpoint run on CARLA 0.9.15 |
| Pilot | A real training run and fixed benchmark evaluation, usually one training seed |
| Baseline | Three training seeds and all five benchmark seeds with published checkpoints |

## Current Matrix

| Algorithm | Family | Current evidence | Guide |
| --- | --- | --- | --- |
| SAC | Online off-policy | 10k pilot; stationary-policy failure documented | [SAC](sac.md) |
| TD3 | Online off-policy | CPU test | [TD3](td3.md) |
| DDPG | Online off-policy | CPU test | [DDPG](ddpg.md) |
| PPO | Online on-policy | CARLA integration smoke | [PPO](ppo.md) |
| A2C | Online on-policy | CPU test | [A2C](a2c.md) |
| TD3+BC | Offline RL | CARLA-dataset integration smoke | [TD3+BC](td3_bc.md) |
| CQL(H) | Offline RL | CPU test | [CQL](cql.md) |
| IQL | Offline RL | CPU test | [IQL](iql.md) |
| BC | Imitation | CARLA-dataset integration smoke | [BC](bc.md) |
| GAIL | Imitation + online | CPU test | [GAIL](gail.md) |
| AIRL | Imitation + online | CPU test | [AIRL](airl.md) |

## Start CARLA

Use one of the remembered launch commands before online training or evaluation:

```bash
cd "$CARLA_ROOT"

# Headless server
./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia

# Windowed server
./CarlaUE4.sh -quality_level=Low-prefernvidia
```

The current research protocol uses CARLA 0.9.15, Town05, 50 background
vehicles, no walkers, frozen green traffic lights, `longitudinal_2d` actions,
and the versioned reward named in each result record.

## Evaluate And Export

Every algorithm checkpoint uses the same internal benchmark command:

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/<algorithm>_ckpt_last.pt \
  --benchmark lane_following_v0
```

Export publication-ready curves and machine-readable data directly from the
TensorBoard event directory:

```bash
python scripts/export_curves.py \
  --run-dir artifacts/runs/<run-name> \
  --output-dir docs/results/<result-name> \
  --benchmark-report artifacts/evaluations/lane_following_v0/<algorithm>/report.json \
  --title "<algorithm and protocol>"
```

The exporter writes `episode_reward.png`, `training_losses.png`, a downsampled
`scalars.csv`, and `result.json`. The JSON retains the full run config,
checkpoint SHA-256, CARLA versions, per-seed benchmark rows, and scalar
statistics.
