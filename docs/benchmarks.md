# Benchmark Protocols

CarlaRLLab separates locally runnable RL protocols from external CARLA
Leaderboard protocols. A local route-completion result is never renamed as a
Leaderboard driving score.

## Primary: `rlfold_nocrash_0915_v0`

This suite adapts the NoCrash route grid used by RLAD/RLfOLD to CARLA 0.9.15.
The original endpoint pairs are bundled under
`carla_rl_lab/benchmarks/assets/nocrash/` with third-party attribution.

### Training

| Name | Town | Traffic | Weather | Route mode |
| --- | --- | --- | --- | --- |
| `nocrash_train_empty_v0` | Town01 | 0 vehicles, 0 walkers | 4 train weathers | endless sampling |
| `nocrash_train_regular_v0` | Town01 | 20 vehicles, 50 walkers | 4 train weathers | endless sampling |
| `nocrash_train_v0` | Town01 | uniform 0-150 vehicles, 0-300 walkers per episode | 4 train weathers | endless sampling |

The empty and fixed regular settings are curriculum and integration targets.
`nocrash_train_v0` is the RLfOLD training distribution. It samples distant
Town01 spawn points continuously and uses `ClearNoon`, `WetNoon`,
`HardRainNoon`, and `ClearSunset`.

### Evaluation

| Name | Town | Vehicles | Walkers | Episodes |
| --- | --- | ---: | ---: | ---: |
| `nocrash_empty_v0` | Town02 | 0 | 0 | 25 routes x 2 weather = 50 |
| `nocrash_regular_v0` | Town02 | 15 | 50 | 50 |
| `nocrash_dense_v0` | Town02 | 70 | 150 | 50 |

All three use held-out `SoftRainSunset` and `WetSunset`. The full suite is 150
episodes:

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/sac_ckpt_last.pt \
  --suite rlfold_nocrash_0915_v0
```

Use `--routes 1 --weathers 1` only for a quick integration check. Results from
a limited grid must be labeled as smoke or pilot results.

Long evaluations atomically write `progress.json` after every episode. If the
evaluator is interrupted, repeat the identical command with `--resume`.
Checkpoint SHA-256, algorithm, benchmark, and evaluation scope must all match
before completed episodes are reused.

### Success And Metrics

An episode succeeds only when its destination is reached with zero collisions.
Lane departure, wrong-way driving, blockage, and timeout terminate as failures.
Red-light infractions are recorded without terminating the fixed route. The
report stores per-episode route ID, weather, return, cost, route
completion, distance, speed, lane offset, collision category, termination
reason, and success.

Aggregate outputs include mean/std return and distance, success rate, route
completion, stationary rate, requested and actually spawned traffic counts,
and pedestrian/vehicle/layout collision, red-light, blockage, and off-road
events per km.

CARLA 0.9.15 does not expose the old NoCrash evaluator unchanged. Route
execution, traffic spawning, and red-light detection are local adaptations.
The exact CARLA build, source commit, checkpoint hash, route assets, and config
must accompany a reported result. These numbers are not directly comparable to
the original CARLA 0.8 NoCrash table or to RLAD's CARLA 0.9.10.1 table.

The current seed-0 pilot and complete per-episode reports are published in the
[results directory](../results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/).
Its fixed Town01 20/50 curriculum is labeled separately from the primary
RLfOLD-style sampled training distribution.

## Lightweight Compatibility Suite

`carla_lightweight_v0` retains earlier Town03/Town05 fixed-horizon checks for
regression testing. It is not the primary pixel research protocol and does not
measure route-completion generalization. New algorithm comparisons should use
`rlfold_nocrash_0915_v0` (`nocrash_0915_v0` is a compatibility alias).

## External Paper Evaluators

`scripts/evaluate_paper.py` validates assets and builds commands for external
evaluators. Those tasks require their own CARLA, Leaderboard, ScenarioRunner,
and agent contracts.

| Name | Typical CARLA | Local role |
| --- | --- | --- |
| `corl2017` | 0.8.2 | Protocol registry only |
| `nocrash` | 0.8.4 | Legacy protocol registry only |
| `town05_short`, `town05_long` | 0.9.10 | Leaderboard 1 launcher |
| `longest6` | 0.9.10 | Leaderboard 1 launcher |
| `longest6_v2` | 0.9.15 | Leaderboard 2 adaptation launcher |
| `bench2drive220` | 0.9.15 | Bench2Drive launcher |

List and preflight them with:

```bash
python scripts/evaluate_paper.py --list
python scripts/evaluate_paper.py \
  --benchmark bench2drive220 \
  --carla-root /path/to/CARLA_0.9.15 \
  --agent /path/to/leaderboard_agent.py \
  --agent-config /path/to/checkpoint
```

The launcher refuses missing assets and records route XML hashes. It does not
make CarlaRLLab's local policy satisfy the external `AutonomousAgent` sensor
contract.

Sources: [NoCrash](https://openaccess.thecvf.com/content_ICCV_2019/html/Codevilla_Exploring_the_Limitations_of_Behavior_Cloning_for_Autonomous_Driving_ICCV_2019_paper.html),
[RLAD](https://arxiv.org/abs/2305.18510),
[RLfOLD](https://ojs.aaai.org/index.php/AAAI/article/view/29049),
[CARLA Leaderboard](https://github.com/carla-simulator/leaderboard), and
[Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive).
