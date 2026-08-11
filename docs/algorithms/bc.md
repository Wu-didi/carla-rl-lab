# Behavior Cloning (BC)

**Family:** imitation learning from a fixed expert dataset.
**Implementation:** [`carla_rl_lab/algorithms/imitation.py`](../../carla_rl_lab/algorithms/imitation.py).
**Runner:** [`scripts/train_imitation.py`](../../scripts/train_imitation.py).

## Principle

BC treats control as supervised regression. The deterministic actor predicts
the expert action and minimizes:

```text
L_BC = mean[(mu(s_expert) - a_expert)^2]
```

It is simple and stable but does not correct compounding errors after the
policy visits states absent from the demonstrations. Dataset coverage and
closed-loop CARLA evaluation therefore matter more than training loss alone.

## Prepare Data And Train

```bash
python scripts/collect_dataset.py \
  --policy autopilot --transitions 100000 \
  --output artifacts/datasets/v0.1/town05_autopilot_seed0_100k.npz \
  --town Town05 --vehicles 50 --walkers 0 --traffic off --view-mode none \
  --action-mode longitudinal_2d --reward research_v2 --seed 0

python scripts/train_imitation.py \
  --algo bc \
  --expert-dataset artifacts/datasets/v0.1/town05_autopilot_seed0_100k.npz \
  --updates 100000 --batch-size 256 --checkpoint-interval 10000 \
  --seed 0 --logger tensorboard --run-name v0.1/bc_town05_seed0
```

## Metrics And Results

Plot `bc_loss` and mean absolute `action_error`, then evaluate return, success,
distance, stationary rate, and action distributions in CARLA. A four-update
CARLA 0.9.15 dataset integration smoke passed and produced a checkpoint. It is
not a driving result; formal training and curves are **Pending**.
