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
  --benchmark nocrash_train_regular_v0 \
  --policy autopilot --transitions 10000 \
  --action-mode target_speed_2d --reward nocrash_v0 \
  --view-mode none --seed 0 --port 2000 \
  --output artifacts/datasets/rlfold_town01_regular_behavior_agent_seed0_10k.npz

python scripts/train_imitation.py \
  --algo bc \
  --expert-dataset artifacts/datasets/rlfold_town01_regular_behavior_agent_seed0_10k.npz \
  --updates 10000 --batch-size 64 --hidden-dim 128 \
  --checkpoint-interval 2000 --logger tensorboard \
  --run-name pilots/rlfold_town01_regular_pixel_bc_seed0_10k \
  --seed 0 --require-clean-git
```

`autopilot` uses CARLA BehaviorAgent in `normal` mode. The stored action is the
same normalized target-speed/steering command consumed by the RL policy. The
dataset embeds its observation/action schema, CARLA versions, collector config,
source commit, and SHA-256 identity.

## Metrics And Results

Plot `bc_loss` and mean absolute `action_error`, then evaluate return, success,
distance, stationary rate, and action distributions in CARLA.

The 2026-08-13 pilot trained the pixel actor for 10k updates on 10k Town01
Regular demonstrations. The selected 10k checkpoint scored **20% success**,
0.530 mean route completion, 40% collision rate, and 80% off-road rate on the
fixed 10-episode selector. Closed-loop performance remains far below the
expert, illustrating BC's distribution-shift limitation. The dataset hash,
loss curve, scalar CSV, run metadata, and checkpoint manifest are in the
[evidence bundle](../../results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/README.md).
