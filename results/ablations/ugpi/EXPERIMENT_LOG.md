# UGPI Ablation Log

All timestamps are UTC. This ledger is committed before the first CARLA run.

## Preregistered Seed-0 Pilot

| Field | Value |
| --- | --- |
| Method | Fixed-BC Pixel SAC + UGPI |
| Dataset SHA-256 | `aa39eb1f06341574c6c7dc693cfb8265014db21fb1da4b7f5e9b604c71ace9de` |
| Training | `nocrash_train_regular_v0`, Town01, fixed 20 vehicles / 50 walkers |
| Budget | 5k BC pretrain updates + 20k online environment steps |
| Seed | 0 |
| UGPI | `beta=2.0`, `confidence_min=0.1` |
| Candidate steps | 8k, 12k, 16k, 20k |
| Selector | `nocrash_empty_v0`, Town02 routes 0-4, both test weathers, 10 episodes |
| Selection | success descending, route completion descending, collision ascending, off-road ascending, earlier step ascending |
| Baseline | Fixed-BC SAC: 60%, 10%, 0%, 0% success at 8k, 12k, 16k, 20k |

Primary endpoint: selector success. Secondary endpoints: route completion,
collision rate, off-road rate, and best-to-final success regression. A result
is promising only if it matches or exceeds the fixed-BC 60% peak or materially
reduces its late regression without degrading the peak. This single seed cannot
support a general method claim.

```bash
python scripts/train.py \
  --algo sac --network Pixel_SAC \
  --benchmark nocrash_train_regular_v0 \
  --total-timesteps 20000 --checkpoint-interval 2000 --checkpoint-keep 10 \
  --minimal-size 1500 --batch-size 64 --buffer-size 15000 \
  --hidden-dim 128 \
  --expert-dataset artifacts/datasets/rlfold_town01_regular_behavior_agent_seed0_10k.npz \
  --demo-pretrain-updates 5000 --demo-bc-coef 0.5 \
  --demo-bc-mode fixed --actor-update-mode confidence \
  --actor-uncertainty-beta 2.0 --actor-confidence-min 0.1 \
  --view-mode none --logger tensorboard \
  --run-name ablations/rlfold_town01_regular_pixel_sac_ugpi_seed0_20k_20260814 \
  --seed 0 --port 2000 --require-clean-git
```

## Runs

| Start | Run | Source | Online steps | Status | Note |
| --- | --- | --- | ---: | --- | --- |
| 2026-08-14 | `rlfold_town01_regular_pixel_sac_ugpi_seed0_20k_20260814` | `3943c36` | 0 | invalid | CARLA 0.9.15 PythonAPI segmentation fault in `WalkerAIController.go_to_location` during the first reset after BC pretraining. No online method update occurred. |
| 2026-08-14 | `rlfold_town01_regular_pixel_sac_ugpi_seed0_20k_retry1_20260814` | `3943c36` | 20,000 | incomplete | Training completed, but the default bounded checkpoint retention deleted 8k before selection. Diagnostic selector results were 60% at 12k, 30% at 16k, and 40% at 20k. These cannot replace the preregistered four-checkpoint run. |
| 2026-08-14 | `rlfold_town01_regular_pixel_sac_ugpi_seed0_20k_retry2_20260814` | `7283d92` | 0 | invalid | Repeated the CARLA 0.9.15 segmentation fault at `WalkerAIController.go_to_location`. The environment issued controller commands before a world tick, unlike CARLA's official traffic example. No online method update occurred. |

The completed incomplete run is promising but not the primary result. Relative
to fixed BC, success changed from 10%/0%/0% to 60%/30%/40% at 12k/16k/20k;
the 20k collision rate changed from 50% to 0%, while off-road rate remained
60%. A clean rerun retains all ten 2k checkpoints via `--checkpoint-keep 10`.
