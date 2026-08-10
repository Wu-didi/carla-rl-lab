# Architecture

CarlaRLLab keeps one visible training path and separates only the research
concerns that change independently.

```text
Benchmark spec
    -> environment configuration
    -> observation function
    -> algorithm and network
    -> replay-buffer loop
    -> logger and benchmark function
```

## Simplicity Rules

1. Use a function for stateless observation, reward, metric, and evaluation logic.
2. Use a class only when an object owns meaningful state or follows a required
   PyTorch/Gym interface.
3. Keep algorithm update equations inside the algorithm module.
4. Add a new runner only when the data flow is genuinely different.

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

The environment owns simulation state and termination. Reward functions consume
observations and a small event context. The default `legacy` profile preserves
the pre-refactor behavior; `research_v1` keeps all weights in one editable
function and returns per-term logs.

## Benchmark Boundary

A benchmark fixes environment overrides, seeds, and metrics. Reward is logged,
but collision, off-road, success, speed, and cost metrics remain the primary
way to compare algorithms across reward designs.
