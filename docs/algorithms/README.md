# Algorithm Guides

Each guide states the objective, runner, launch command, metrics, and current
evidence. “Implemented” refers to code and tests; “baseline” requires the full
contract in [`../experiments.md`](../experiments.md).

## Matrix

| Algorithm | Family | Pixel-native | Current evidence | Guide |
| --- | --- | --- | --- | --- |
| SAC | Online off-policy | Yes | CARLA 0.9.15 train/eval smoke | [SAC](sac.md) |
| TD3 | Online off-policy | No | CPU update/checkpoint test | [TD3](td3.md) |
| DDPG | Online off-policy | No | CPU update/checkpoint test | [DDPG](ddpg.md) |
| PPO | Online on-policy | No | Runner integration smoke | [PPO](ppo.md) |
| A2C | Online on-policy | No | CPU update/checkpoint test | [A2C](a2c.md) |
| TD3+BC | Offline RL | No | Dataset runner integration smoke | [TD3+BC](td3_bc.md) |
| CQL(H) | Offline RL | No | CPU update/checkpoint test | [CQL](cql.md) |
| IQL | Offline RL | No | CPU update/checkpoint test | [IQL](iql.md) |
| BC | Imitation | No | Dataset runner integration smoke | [BC](bc.md) |
| GAIL | Imitation + online | No | CPU update/checkpoint test | [GAIL](gail.md) |
| AIRL | Imitation + online | No | CPU update/checkpoint test | [AIRL](airl.md) |

The immediate baseline target is Pixel SAC. Pixel adapters for the remaining
algorithms are roadmap work; running their current MLP directly on packed image
bytes is useful only as an interface smoke and should not be reported as a
vision baseline.

## Shared Protocol

Start CARLA with one of the fixed commands:

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia
# or
./CarlaUE4.sh -quality_level=Low-prefernvidia
```

Formal comparisons train in Town01 and evaluate the unchanged checkpoint with:

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --suite nocrash_0915_v0
```

See [`../observations.md`](../observations.md) for inputs and
[`../experiments.md`](../experiments.md) for reporting rules.
