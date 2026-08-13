# Experiment Log

All timestamps are UTC. This is the operator record for the seed-0 pilot; the
JSON files under `runs/` and `evaluations/` are the machine-readable source of
truth.

## Environment

| Field | Value |
| --- | --- |
| Host | Linux 6.8.0-107-generic x86_64 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| Python | 3.7.16 |
| PyTorch | 1.13.1+cu117 |
| NumPy | 1.21.6 |
| CARLA | client 0.9.15, server 0.9.15 |
| Server command | `./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia` |

## Chronology

| Start | Finish | Job | Budget | Source commit | Status |
| --- | --- | --- | ---: | --- | --- |
| 2026-08-12 17:58 | 18:42 | Plain Pixel SAC | 20k env steps | `05efb2d` | completed |
| 2026-08-12 19:05 | 19:47 | Pixel TD3 | 20k env steps | `16b7e44` | completed |
| 2026-08-12 20:36 | 20:38 | Pixel BC | 10k updates | `5ffa947` | completed |
| 2026-08-12 20:55 | 21:39 | Demo-assisted Pixel SAC | 5k BC + 20k env steps | `5ffa947` | completed |
| 2026-08-12 22:01 | 22:49 | Pixel PPO | 20k env steps | `2a84019` | completed |
| 2026-08-12 23:00 | 23:46 | Selected SAC, Empty test | 50 episodes | evaluator `1e0c665` | completed |
| 2026-08-13 | 2026-08-13 | Selected SAC, Regular test | 50 episodes | evaluator `1e0c665` | completed after clean retry |
| 2026-08-13 00:43 | interrupted | Selected SAC, Dense attempt 1 | reached 45/50 | evaluator `1e0c665` | interrupted; no partial report |
| 2026-08-13 16:56 | 17:59 | Selected SAC, Dense resumable run | 50 episodes | evaluator `fc61d43` | completed |

The first Dense attempt predates per-episode atomic progress and is retained
locally as an interruption record, not merged with the new run. The second run
started from episode 0 and writes `progress.json` after every episode.

## Demonstrations

BehaviorAgent (`normal`) collected 10,000 transitions in Town01 with fixed
20 vehicles / 50 walkers. The policy observation and action contracts match
online training.

```text
file:   rlfold_town01_regular_behavior_agent_seed0_10k.npz
states: (10000, 42358) uint8
actions: (10000, 2) float32
sha256: aa39eb1f06341574c6c7dc693cfb8265014db21fb1da4b7f5e9b604c71ace9de
```

The dataset is 245 MB and is intentionally outside Git history. Its exact
collection command is in the main evidence README.

## Checkpoint Selection

Every candidate below was evaluated on the same Town02 Empty grid: routes 0-4,
both held-out weathers, 10 episodes. Selection occurred before the 50-episode
test runs.

| Method | Step | Success | Completion | Collision | Off-road |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC | 2k | 10% | 0.529 | 50% | 90% |
| BC | 6k | 20% | 0.295 | 40% | 40% |
| BC | **10k** | **20%** | **0.530** | **40%** | **80%** |
| PPO | 4,096 | 0% | 0.024 | 10% | 100% |
| PPO | 12,288 | 0% | 0.056 | 50% | 60% |
| PPO | **20k** | **0%** | **0.482** | **60%** | **80%** |
| TD3 | **8k** | **40%** | **0.999** | **40%** | **20%** |
| TD3 | 12k | 20% | 0.420 | 20% | 80% |
| TD3 | 16k | 0% | 0.323 | 40% | 60% |
| TD3 | 20k | 0% | 0.125 | 0% | 100% |
| Assisted SAC | **8k** | **60%** | **0.786** | **20%** | **20%** |
| Assisted SAC | 12k | 10% | 0.320 | 40% | 50% |
| Assisted SAC | 16k | 0% | 0.347 | 40% | 100% |
| Assisted SAC | 20k | 0% | 0.253 | 50% | 60% |

Plain SAC candidates at 8k/12k/16k/20k scored 0% success; step 8k had the
highest completion at 0.360. Its original per-episode 8k report was overwritten
before evaluation output scoping was fixed, so no raw selection JSON is claimed
for that row.

## Frozen Test Checkpoint

```text
method: demonstration-assisted Pixel SAC
step: 8000
sha256: 9ff30f291781814d33f6ee56005eb78d878de9340065c78ca41a4f3124349c2a
training source: 5ffa947808071ed87194a6480cec6c4c3dd66171
```

| Split | Episodes | Success | Completion | Collision | Off-road | Collisions/km |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empty 0/0 | 50 | 46% | 0.753 | 18% | 34% | 0.759 |
| Regular 15/50 | 50 | 24% | 0.674 | 52% | 36% | 4.771 |
| Dense 70/150 requested | 50 | 4% | 0.469 | 82% | 24% | 15.432 |

## Observations

- Demonstration assistance produced the strongest checkpoint under this small
  budget, but this comparison has one seed and is not a significance claim.
- TD3 learned a competitive early checkpoint and then regressed.
- Assisted SAC also regressed after step 8k; late critic losses rose while the
  selector score fell. Training return alone was not a reliable selector.
- Vehicle collisions dominate Regular failures. Dense currently exposes severe
  blockage and collision sensitivity.
- A formal baseline still requires seeds 0/1/2, the sampled Town01 0-150 / 0-300
  training distribution, a preregistered selection rule, and the full suite.
