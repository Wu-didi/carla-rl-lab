# CADR Ablation Log

All timestamps are UTC. Runs are compared only when they start from a clean,
committed source tree and complete the preregistered protocol.

| Start | Run | Source | Online steps | Status | Reason |
| --- | --- | --- | ---: | --- | --- |
| 2026-08-14 | `rlfold_town01_regular_pixel_sac_cadr_seed0_20k_20260814` | `03aa1f2` | 0 | invalid | Operator stopped after 5k BC pretraining: the initial gate used `sigmoid(A/T)`, giving neutral samples weight 0.5 and confounding adaptive weighting with a lower global BC scale. The interrupted process emitted a segmentation fault while inside CARLA walker spawning during the first reset. |

The corrected gate is `2 * sigmoid(A/T)`, so zero expert advantage and zero
critic disagreement recover weight 1. The invalid run is retained locally
under `artifacts/runs/ablations/` but must not appear in result tables.
