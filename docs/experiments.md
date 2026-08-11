# Experiment Protocol

This document separates cheap integration checks from results that may be
reported as CarlaRLLab baselines. A successful smoke run proves that data,
training, logging, and checkpoint interfaces compose; it does not establish
driving performance.

## Representative V0.1 Tracks

| Track | Algorithm | Data source | Purpose |
| --- | --- | --- | --- |
| Online off-policy | SAC | Fresh CARLA transitions | Primary editable actor-critic baseline |
| Online on-policy | PPO | Fresh CARLA rollouts | On-policy reference |
| Imitation | BC | Fixed autopilot dataset | Expert-only reference |
| Offline RL | TD3+BC | Same fixed dataset | Offline policy improvement reference |

Run these four tracks before expanding the baseline matrix to TD3, DDPG, A2C,
CQL(H), IQL, GAIL, and AIRL.

## Integration Smoke

Start a CARLA 0.9.15 server, activate the Python 3.7 environment, and run:

```bash
CARLA_PORT=2000 scripts/run_research_smoke.sh
```

Default budgets are 64 collected transitions, 64 online environment steps,
and 8 offline updates. The script must produce a versioned dataset,
`run_config.json`, TensorBoard events, checkpoint manifests, immutable step
checkpoints, and `*_ckpt_last.pt` for all four algorithms.

The workload can be changed without editing source:

```bash
SMOKE_TRANSITIONS=128 \
SMOKE_TIMESTEPS=128 \
SMOKE_UPDATES=16 \
SMOKE_OUTPUT=/tmp/carla-rl-lab-smoke \
CARLA_PORT=3000 \
PYTHON_BIN=/path/to/python \
scripts/run_research_smoke.sh
```

## Baseline Contract

The first publishable baseline should hold these choices fixed:

| Field | Value |
| --- | --- |
| CARLA | 0.9.15 |
| Training town | Town05 |
| Action representation | `longitudinal_2d` |
| Reward | `research_v1` |
| Training seeds | 0, 1, 2 |
| Evaluation protocol | `lane_following_v0` |
| Evaluation seeds | 0, 1, 2, 3, 4 |
| Logging | TensorBoard plus saved `run_config.json` |

Every reported row must include the Git commit, checkpoint SHA-256, CARLA
client/server versions, full config, dataset metadata when applicable, hardware,
wall-clock training time, and all random seeds. Do not compare algorithms using
different observations, action modes, reward profiles, traffic settings, or
evaluation horizons without labeling the comparison as an ablation.

## Reporting

For each training seed, retain episode return, safety cost, success rate,
collision rate, route completion or distance, lane offset, episode length, and
wall-clock time. Publish per-seed values as compact JSON or CSV, then report
mean and standard deviation across seeds. Checkpoints belong in GitHub Releases;
datasets and model binaries do not belong in Git history.
