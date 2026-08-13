# CADR Ablation Log

All timestamps are UTC. Runs are compared only when they start from a clean,
committed source tree and complete the preregistered protocol.

| Start | Run | Source | Online steps | Status | Reason |
| --- | --- | --- | ---: | --- | --- |
| 2026-08-14 | `rlfold_town01_regular_pixel_sac_cadr_seed0_20k_20260814` | `03aa1f2` | 0 | invalid | Operator stopped after 5k BC pretraining: the initial gate used `sigmoid(A/T)`, giving neutral samples weight 0.5 and confounding adaptive weighting with a lower global BC scale. The interrupted process emitted a segmentation fault while inside CARLA walker spawning during the first reset. |
| 2026-08-14 | `rlfold_town01_regular_pixel_sac_cadr_seed0_20k_retry1_20260814` | `6f725c3` | 20,000 | completed | Valid corrected CADR run. All four selector candidates reached 20% success; current full CADR configuration rejected. |

The corrected gate is `2 * sigmoid(A/T)`, so zero expert advantage and zero
critic disagreement recover weight 1. The invalid run is retained locally
under `artifacts/runs/ablations/` but must not appear in result tables.

## Valid Seed-0 Result

Dataset SHA-256:
`aa39eb1f06341574c6c7dc693cfb8265014db21fb1da4b7f5e9b604c71ace9de`.
CARLA client/server: 0.9.15. Training: Town01 Regular traffic. Selection:
Town02 Empty, routes 0-4, `SoftRainSunset` and `WetSunset`, one episode per
route-weather pair.

| Step | Checkpoint SHA-256 | Success | Completion | Collision | Off-road |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8k | `72ca2c60cb4f...` | 20% | 32.5% | 40% | 50% |
| 12k | `844682ecd18e...` | 20% | 54.6% | 40% | 60% |
| 16k | `82f4d9f2b492...` | 20% | 33.1% | 40% | 40% |
| 20k | `71d91c42ebc7...` | 20% | 50.6% | 40% | 60% |

The primary endpoint is below fixed-BC SAC's 60% at 8k. This is a negative
result, not evidence of improvement. Scalar exports and raw evaluation reports
are stored beside this log so the conclusion can be independently checked.
