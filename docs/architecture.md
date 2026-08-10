# Architecture

CarlaRLLab separates research concerns that change at different rates.

```text
Benchmark spec
    -> environment configuration
    -> observation adapter
    -> algorithm and network
    -> runner
    -> logger and evaluator
```

## Algorithm Taxonomy

| Data source | Update family | Examples | Runner |
| --- | --- | --- | --- |
| online | off-policy | SAC, TD3, DDPG | replay buffer |
| online | on-policy | PPO, A2C | rollout buffer |
| offline | offline RL | CQL, IQL, TD3+BC | dataset |
| expert data or mixed | imitation | BC, GAIL, AIRL | dataset or mixed |

Registry metadata records `data_source`, `family`, and the required `runner`.
A trainer rejects algorithms requiring a different runner instead of silently
using an invalid update schedule.

## Reward Boundary

The environment owns simulation state and termination. Reward terms consume
observations and a small event context. The default `legacy` profile preserves
the pre-refactor behavior; `research_v1` demonstrates weighted terms and
per-term logging.

## Benchmark Boundary

A benchmark fixes environment overrides, seeds, and metrics. Reward is logged,
but collision, off-road, success, speed, and cost metrics remain the primary
way to compare algorithms across reward designs.
