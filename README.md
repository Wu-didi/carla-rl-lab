<div align="center">

# CarlaRLLab

**An editable reinforcement-learning research stack for vision-based urban driving in CARLA.**

[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-e11d48?style=flat-square)](https://github.com/carla-simulator/carla/releases/tag/0.9.15)
[![Python](https://img.shields.io/badge/Python-3.7-3776ab?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-ee4c2c?style=flat-square)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](LICENSE)

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

CarlaRLLab is built for researchers who need to change the policy network,
observation, reward, or training equation without fighting a deeply wrapped RL
framework. The first release stays deliberately small: one primary pixel SAC
baseline, one versioned NoCrash adaptation, transparent PyTorch code, and
reproducible experiment artifacts.

> **Research status:** the CARLA 0.9.15 connection, pixel observation, route
> execution, replay updates, checkpoints, TensorBoard logging, and a real
> end-to-end training smoke have been verified. A smoke run is not a converged
> benchmark result. Multi-seed baselines are still in progress.

## Why CarlaRLLab

- **Editable by design.** Algorithm equations live in ordinary PyTorch files;
  reward terms and observation packing are short functions.
- **Protocol before score.** Training and evaluation select named benchmark
  configurations instead of relying on undocumented command-line settings.
- **Comparable inputs.** The current policy sees only front RGB, route
  waypoints, speed, and previous steering. Simulator state remains evaluation
  telemetry, not a hidden policy input.
- **Research logging.** TensorBoard works out of the box; Weights & Biases is an
  optional backend. Configs, checkpoint metadata, reward terms, losses, actions,
  and benchmark metrics are recorded.
- **Shallow structure.** Agents compose small networks and buffers. There is no
  hierarchy of environment wrappers or nested algorithm classes.

## Current Scope

| Family | Algorithms | Implementation status |
| --- | --- | --- |
| Online, off-policy | SAC, TD3, DDPG | SAC has the primary pixel encoder; MLP cores are tested for all three |
| Online, on-policy | PPO, A2C | Update equations and runner implemented; pixel encoder integration pending |
| Offline RL | TD3+BC, CQL(H), IQL | Dataset runners and MLP cores implemented; pixel baselines pending |
| Imitation | BC, GAIL, AIRL | Runners implemented; pixel baselines pending |

The registry prevents an algorithm from being launched with an incompatible
runner. “Implemented” does not mean that a paper-quality CARLA checkpoint has
already been produced; published baselines require the fixed protocol below.

## Observation And Control

The `pixel_v1` observation follows the policy-input pattern used by
[RLAD](https://arxiv.org/abs/2305.18510) and
[RLfOLD](https://ojs.aaai.org/index.php/AAAI/article/view/29049):

```text
front RGB:            2 frames x 3 x 84 x 84, uint8 (RLfOLD profile)
route:                10 ego-frame (x, y) waypoints
vehicle measurements: normalized speed + previous steering
packed replay state:  42,358 uint8 values
action:               target speed + steering, both in [-1, 1]
low-level control:    target speed -> throttle/brake PID
```

Global pose, lane measurements, nearby actor state, LiDAR, and risk fields are
not policy inputs. They may be used by the environment to calculate reward,
termination, or evaluation metrics. The RLfOLD benchmark profile uses one
front RGB sensor at the official `(x=1.5 m, z=2.4 m)` position and 90-degree
field of view. CARLA renders `256x256`; the lightweight baseline resizes it to
`84x84` before the policy.
RLAD used `256x256`, CARLA 0.9.10.1, and a different training budget, so results
from this CARLA 0.9.15 adaptation must not be presented as protocol-equivalent
RLAD numbers. See [Observation Contract](docs/observations.md).

## NoCrash 0.9.15

The primary `rlfold_nocrash_0915_v0` suite ports RLfOLD's NoCrash task protocol
to the CARLA 0.9.15 API. `nocrash_0915_v0` remains a compatibility alias.

| Split | Town | Vehicles / walkers | Weather | Episodes |
| --- | --- | ---: | --- | ---: |
| Train | Town01 | sampled 0-150 / 0-300 | 4 training weathers | endless route sampling |
| Empty | Town02 | 0 / 0 | 2 held-out weathers | 25 routes x 2 = 50 |
| Regular | Town02 | 15 / 50 | 2 held-out weathers | 50 |
| Dense | Town02 | 70 / 150 | 2 held-out weathers | 50 |

Success means route completion without collision. Lane departure, wrong-way
driving, blockage, and timeout terminate unsuccessfully; red lights are counted
without ending a fixed evaluation route. Reports include route completion, return,
speed, collision categories, red lights, blockage, and infractions per km. The
red-light detector is a documented CARLA 0.9.15 approximation; this suite is
therefore named as an adaptation, not as the legacy CARLA 0.8 NoCrash runner.

## Repository Layout

```text
carla_rl_lab/
  algorithms/       editable PyTorch algorithms and registry
  benchmarks/       named protocols, NoCrash routes, weather groups
  buffers/          replay, rollout, and offline datasets
  envs/             one CARLA environment and control conversion
  evaluation/       route-level metrics and suite aggregation
  logging/          TensorBoard and optional W&B backend
  observations/     policy observation packing
  rewards/          versioned reward functions
scripts/             train, collect, evaluate, smoke, export curves
docs/                protocols, architecture, experiments, algorithm guides
tests/               CPU unit and integration tests
```

## Installation

The reference setup is Linux x86_64, an NVIDIA GPU, CARLA 0.9.15, and Python
3.7. CARLA itself is a separate simulator process and is not installed by
`pip install -e .`.

### 1. Choose installation directories

```bash
export CARLA_ROOT="$HOME/simulators/CARLA_0.9.15"
export CARLA_RL_LAB_ROOT="$HOME/code/carla-rl-lab"
mkdir -p "$CARLA_ROOT" "$(dirname "$CARLA_RL_LAB_ROOT")"
```

Add these exports to your shell profile if you want them to persist.

### 2. Download and extract CARLA 0.9.15

Download the official prebuilt Linux release directly into `CARLA_ROOT`, then
extract it there. The archive already contains `CarlaUE4.sh`; do not add another
nested `CARLA_0.9.15` directory.

```bash
cd "$CARLA_ROOT"
wget -c \
  https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz \
  -O CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
```

The important paths should now look like this:

```text
$CARLA_ROOT/
  CarlaUE4.sh
  CarlaUE4/
  PythonAPI/
    carla/
      agents/
      dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

Verify the package before continuing:

```bash
test -x "$CARLA_ROOT/CarlaUE4.sh" && echo "CARLA server: OK"
ls "$CARLA_ROOT"/PythonAPI/carla/dist/*0.9.15*py3.7*
```

Town01 and Town02 are included in the base package; no additional-map download
is required for the NoCrash suite.

### 3. Clone the project and create the environment

```bash
git clone git@github.com:Wu-didi/carla-rl-lab.git "$CARLA_RL_LAB_ROOT"
cd "$CARLA_RL_LAB_ROOT"

conda create -n carla-rl-lab python=3.7 -y
conda activate carla-rl-lab
python -m pip install --upgrade "pip<24.1"
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 4. Expose the matching CARLA Python API

CARLA 0.9.15 ships a Python 3.7 egg. Add both the API root (for navigation
agents) and the egg (for `import carla`) to `PYTHONPATH`:

```bash
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"

python -c "import carla; from agents.navigation.behavior_agent import BehaviorAgent; print(carla.__file__)"
```

Do not install a different `carla` package from PyPI into this environment.
Client/server version mismatches often fail only after the simulator connects.

### 5. Start CARLA

Open terminal A and use one of the project’s two reference commands exactly as
shown.

Off-screen training:

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia
```

Windowed debugging:

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -quality_level=Low-prefernvidia
```

The repository wrapper runs the same commands:

```bash
CARLA_ROOT="$CARLA_ROOT" scripts/launch_carla.sh offscreen
```

### 6. Verify the complete stack

Keep CARLA running. In terminal B:

```bash
conda activate carla-rl-lab
cd "$CARLA_RL_LAB_ROOT"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"

python -m unittest discover -s tests -v
python scripts/smoke_carla.py \
  --benchmark nocrash_empty_v0 --steps 10 \
  --frame-output artifacts/smoke/nocrash_town02_rgb.png
```

The smoke command must print matching CARLA client/server versions, a
`state_dim` of `42358`, `num_cameras` of `1`, non-zero front-image statistics,
and ten successful steps.

## Train Pixel SAC

Start with the empty Town01 curriculum before adding regular traffic:

```bash
python scripts/train.py \
  --benchmark nocrash_train_empty_v0 \
  --algo sac --network Pixel_SAC \
  --total-timesteps 100000 \
  --minimal-size 1500 --batch-size 128 --buffer-size 30000 \
  --checkpoint-interval 10000 \
  --logger tensorboard \
  --run-name nocrash/pixel_sac_empty_seed0 \
  --seed 0
```

Use `--benchmark nocrash_train_v0` for RLfOLD's Town01 traffic distribution
(uniformly sampled 0-150 vehicles and 0-300 walkers per episode). The fixed
20/50 curriculum is available as `nocrash_train_regular_v0`.
Resume by passing the last checkpoint and a larger absolute step budget:

```bash
python scripts/train.py \
  --checkpoint artifacts/runs/nocrash/pixel_sac_empty_seed0/checkpoints/sac_ckpt_last.pt \
  --total-timesteps 200000
```

Each packed transition contains two `42,358`-byte observations. A 30,000-entry
replay buffer therefore needs roughly 2.5 GB for observations; size it for your
machine.

## Evaluate On The Same Protocol

Run one route and one weather first:

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/sac_ckpt_last.pt \
  --benchmark nocrash_empty_v0 \
  --routes 1 --weathers 1 --logger tensorboard
```

Then run all 150 Town02 episodes under the three traffic densities:

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/sac_ckpt_last.pt \
  --suite rlfold_nocrash_0915_v0 \
  --logger tensorboard
```

Use the same checkpoint, route files, weather grid, traffic counts, image
contract, and seed policy for every algorithm row. Results are written under
`artifacts/evaluations/` with checkpoint SHA-256 and full metadata.

## Logging And Curves

TensorBoard is the default and has no extra dependency:

```bash
tensorboard --logdir artifacts/runs --port 6006
```

For Weights & Biases:

```bash
python -m pip install -r requirements-wandb.txt
wandb login
python scripts/train.py \
  --benchmark nocrash_train_empty_v0 --algo sac \
  --logger both --wandb-mode online --run-name nocrash/pixel_sac_seed0
```

Export publication-ready PNG/CSV summaries from a completed run:

```bash
python scripts/export_curves.py \
  --run-dir artifacts/runs/nocrash/pixel_sac_empty_seed0 \
  --benchmark-report artifacts/evaluations/nocrash_empty_v0/sac/report.json
```

## Modify The Research Code

| Research change | Primary file |
| --- | --- |
| Pixel encoder or SAC equations | [`carla_rl_lab/algorithms/sac.py`](carla_rl_lab/algorithms/sac.py) |
| Observation fields and packing | [`carla_rl_lab/observations/vector.py`](carla_rl_lab/observations/vector.py) |
| Reward terms | [`carla_rl_lab/rewards/profiles.py`](carla_rl_lab/rewards/profiles.py) |
| Camera, route, PID, termination | [`carla_rl_lab/envs/carla_env.py`](carla_rl_lab/envs/carla_env.py) |
| Benchmark traffic/routes/weather | [`carla_rl_lab/benchmarks/specs.py`](carla_rl_lab/benchmarks/specs.py) |
| Metrics and success rules | [`carla_rl_lab/evaluation/evaluator.py`](carla_rl_lab/evaluation/evaluator.py) |

New reward functions should return both a scalar and named terms. New
observation versions should receive a new protocol name and state shape instead
of silently changing `pixel_v1`.

## Additional Training Paths

```bash
# Expert dataset from the same Town01 protocol
python scripts/collect_dataset.py \
  --benchmark nocrash_train_empty_v0 --policy autopilot \
  --transitions 100000 --output artifacts/datasets/nocrash_expert.npz

# Offline and imitation runners (MLP baseline today; pixel adapter is pending)
python scripts/train_offline.py --algo td3_bc --dataset artifacts/datasets/nocrash_expert.npz
python scripts/train_imitation.py --algo bc --expert-dataset artifacts/datasets/nocrash_expert.npz

# End-to-end 64-step integration matrix, not a performance benchmark
scripts/run_research_smoke.sh
```

Algorithm principles, commands, expected logs, and result status are indexed in
the [algorithm guide](docs/algorithms/README.md). Benchmark definitions are in
[docs/benchmarks.md](docs/benchmarks.md), and the reporting contract is in
[docs/experiments.md](docs/experiments.md).

## Roadmap

- [ ] Validate CARLA 0.9.15 with three full training seeds and publish
  checkpoints plus raw TensorBoard/W&B exports.
- [ ] Add pixel-native encoders for TD3, DDPG, PPO, A2C, offline RL, and
  imitation learning; train every algorithm on the same NoCrash suite.
- [ ] Report loss, return, route completion, success, collision categories,
  red-light infractions, and blockage for every released checkpoint.
- [ ] Version and ablate camera history, route representation, speed, steering,
  and future sensor-fusion observations without using simulator-only policy
  inputs.
- [ ] Publish a detailed tutorial for every algorithm, including equations,
  data flow, launch, debugging, evaluation, and result interpretation.
- [ ] Provide one-command installation, environment diagnostics, and a pinned
  Docker image for CARLA 0.9.15.
- [ ] Add CI for CPU tests and a separately scheduled CARLA integration test.

## License And Attribution

CarlaRLLab is released under the [MIT License](LICENSE). Bundled NoCrash route
files are adapted from the official RLfOLD repository; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). CARLA and referenced papers
retain their own licenses and citation requirements.
