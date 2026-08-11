# Architecture

CarlaRLLab keeps the research path explicit:

```text
benchmark spec -> CarlaEnv -> pixel_v1 packing -> agent -> buffer/runner
                                                   |
                                                   +-> TensorBoard / W&B
                                                   +-> checkpoint metadata
checkpoint + same benchmark -> evaluator -> route report
```

## Simplicity Rules

1. Use functions for stateless observation, reward, metric, and configuration
   transforms.
2. Use classes only for stateful CARLA resources, PyTorch modules, agents,
   buffers, and logging backends.
3. Keep each algorithm's update equations in its algorithm module.
4. Add a runner only when data flow changes: replay, rollout, fixed dataset, or
   mixed expert/online data.
5. Do not hide policy inputs inside environment wrappers.

## Algorithm Taxonomy

`online/offline` describes the source of training data;
`on-policy/off-policy` describes how that data may be reused.

| Data source | Update family | Algorithms | Runner |
| --- | --- | --- | --- |
| Online | Off-policy | SAC, TD3, DDPG | Replay buffer |
| Online | On-policy | PPO, A2C | Fresh rollout buffer |
| Offline | Offline RL | TD3+BC, CQL, IQL | Versioned dataset |
| Expert or mixed | Imitation | BC, GAIL, AIRL | Dataset or expert + rollout |

Registry metadata binds each algorithm to one runner. GAIL and AIRL compose a
PPO policy rather than inheriting through another agent class.

## Observation Boundary

The CARLA environment returns policy fields plus telemetry. Only
`image`, `waypoints`, and `vehicle_measurements` enter
`encode_observation`. `ego_state` and `lane_info` are reserved for reward,
termination, and evaluation. See `docs/observations.md` for shapes and units.

## Reward Boundary

The environment owns simulator state and terminal events. A reward function
receives the observation and a small context, then returns:

```python
reward, named_terms
```

`nocrash_v0` is the primary pixel reward. It uses privileged simulator truth
only to derive a safe desired speed and event penalties; that truth does not
enter the policy. Reward terms are logged individually so a high return can be
audited rather than accepted as a driving result.

## Artifact Boundary

Datasets store observation/action semantics, terminal versus timeout flags,
source config, CARLA versions, and collection provenance. Checkpoints store the
model, optimizers, global step, source commit, RNG state, software/hardware
metadata, and optionally replay state. CARLA world state is not serializable;
resume begins from a fresh seeded episode.

## Benchmark Boundary

Named benchmark specs own map, traffic, weather, route, reward, action, and
observation overrides. Evaluation then applies the same spec to a checkpoint
and emits native route metrics. External Leaderboard tools stay behind a
separate launcher and are never conflated with local NoCrash-adaptation scores.
