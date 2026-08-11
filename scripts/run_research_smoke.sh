#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON_BIN:-python}"
carla_port="${CARLA_PORT:-2000}"
transitions="${SMOKE_TRANSITIONS:-64}"
timesteps="${SMOKE_TIMESTEPS:-64}"
updates="${SMOKE_UPDATES:-8}"
output_root="${SMOKE_OUTPUT:-${project_root}/artifacts/research-smoke}"
dataset_path="${output_root}/town05_autopilot.npz"

mkdir -p "${output_root}"

common_env=(
  --town Town05
  --port "${carla_port}"
  --vehicles 0
  --walkers 0
  --traffic off
  --view-mode none
  --max-time-episode "${timesteps}"
)

echo "[1/5] Collecting ${transitions} CARLA transitions"
"${python_bin}" scripts/collect_dataset.py \
  --policy autopilot \
  --transitions "${transitions}" \
  --output "${dataset_path}" \
  --town Town05 \
  --port "${carla_port}" \
  --vehicles 0 \
  --walkers 0 \
  --traffic off \
  --view-mode none \
  --action-mode longitudinal_2d \
  --reward research_v1

echo "[2/5] Running SAC online smoke"
"${python_bin}" scripts/train.py \
  --algo sac \
  --total-timesteps "${timesteps}" \
  --minimal-size 8 \
  --batch-size 8 \
  --buffer-size 1000 \
  --hidden-dim 32 \
  --checkpoint-interval 32 \
  --action-mode longitudinal_2d \
  --reward research_v1 \
  --logger tensorboard \
  --run-name "${output_root}/sac" \
  "${common_env[@]}"

echo "[3/5] Running PPO online smoke"
"${python_bin}" scripts/train_on_policy.py \
  --algo ppo \
  --total-timesteps "${timesteps}" \
  --rollout-steps 32 \
  --hidden-dim 32 \
  --ppo-epochs 2 \
  --ppo-minibatch-size 8 \
  --checkpoint-interval 32 \
  --action-mode longitudinal_2d \
  --reward research_v1 \
  --logger tensorboard \
  --run-name "${output_root}/ppo" \
  "${common_env[@]}"

echo "[4/5] Running behavior cloning smoke"
"${python_bin}" scripts/train_imitation.py \
  --algo bc \
  --expert-dataset "${dataset_path}" \
  --updates "${updates}" \
  --batch-size 8 \
  --hidden-dim 32 \
  --checkpoint-interval 4 \
  --logger tensorboard \
  --run-name "${output_root}/bc"

echo "[5/5] Running TD3+BC offline smoke"
"${python_bin}" scripts/train_offline.py \
  --algo td3_bc \
  --dataset "${dataset_path}" \
  --updates "${updates}" \
  --batch-size 8 \
  --hidden-dim 32 \
  --checkpoint-interval 4 \
  --logger tensorboard \
  --run-name "${output_root}/td3_bc"

echo "Research smoke passed. Outputs -> ${output_root}"
