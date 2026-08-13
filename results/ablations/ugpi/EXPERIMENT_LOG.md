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
  --total-timesteps 20000 --checkpoint-interval 2000 \
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
