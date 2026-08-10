# CarlaRLLab

[简体中文](README_zh-CN.md)

CarlaRLLab is a research-first reinforcement learning lab for CARLA. It keeps
algorithms, neural networks, reward terms, logging, and benchmark protocols
explicit so researchers can modify experiments without working through a
large third-party RL abstraction.

## v1 Scope

- Online off-policy algorithms: SAC, TD3, DDPG
- Editable MLP and semantic-attention SAC networks
- Composable reward terms with a legacy-compatible default
- TensorBoard and optional Weights & Biases logging
- A fixed `lane_following_v0` benchmark protocol
- Continuous control: throttle, steering, and brake

PPO and other on-policy algorithms will use a separate rollout runner. Offline
RL algorithms such as CQL, IQL, and TD3+BC will use a dataset runner rather
than being forced into the online replay-buffer loop.

## Layout

```text
carla_rl_lab/
  algorithms/       # Editable SAC, TD3, DDPG and algorithm registry
  benchmarks/       # Fixed experimental protocols
  buffers/          # Replay and future rollout buffers
  envs/             # CARLA environment and factory
  evaluation/       # Benchmark evaluator and reports
  logging/          # TensorBoard and W&B backends
  observations/     # Observation adapters and validation
  rewards/          # Reward terms and profiles
scripts/
  train.py
  evaluate.py
  launch_carla.sh
tests/
docs/
artifacts/           # Ignored logs, checkpoints, and reports
legacy/              # Ignored local archive of pre-refactor experiments
```

## Installation

The current environment targets CARLA 0.9.13 and Python 3.7.

```bash
conda create -n carla37 python=3.7
conda activate carla37
pip install -r requirements.txt
```

Install the matching CARLA Python API separately from the CARLA distribution.
For W&B support, install `requirements-wandb.txt`.

## Start CARLA

From the CARLA installation directory, use either project command:

```bash
./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia
```

```bash
./CarlaUE4.sh -quality_level=Low-prefernvidia
```

The included wrapper runs the same commands:

```bash
CARLA_ROOT=/path/to/CARLA scripts/launch_carla.sh offscreen
CARLA_ROOT=/path/to/CARLA scripts/launch_carla.sh window
```

## Train

```bash
python scripts/train.py --algo sac --reward legacy --logger tensorboard
python scripts/train.py --algo td3 --reward research_v1 --logger both --wandb-mode offline
python scripts/train.py --algo ddpg --checkpoint /path/to/ddpg_ckpt.pt
```

Outputs are written under `artifacts/runs/`. Metrics include algorithm losses,
entropy and Q statistics, episode reward and safety cost, action distributions,
and per-term reward decomposition.

## Benchmark

Run deterministic evaluation against the fixed `lane_following_v0` protocol:

```bash
python scripts/evaluate.py \
  --algo sac \
  --checkpoint /path/to/sac_ckpt.pt \
  --benchmark lane_following_v0
```

The evaluator reports mean return, mean cost, mean speed, collision rate,
off-road rate, and success rate. JSON reports are stored under
`artifacts/evaluations/`.

## Research Extension Points

- Implement `BaseAgent` and register an `AlgorithmSpec` to add an algorithm.
- Change model structure directly in the algorithm module.
- Add reward terms in `carla_rl_lab/rewards/terms.py` and compose a profile.
- Register fixed benchmark specs without changing training code.

CarlaRLLab is an independent research project built on CARLA and is not an
official CARLA project.
