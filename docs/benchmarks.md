# CARLA Benchmark Protocols

CarlaRLLab separates paper-standard route benchmarks from its small internal
control checks. This distinction matters: surviving a fixed-horizon random
spawn episode is not CARLA route completion and must not be reported as a
Leaderboard result.

## Paper-standard benchmarks

`scripts/evaluate_paper.py` validates the official route/scenario assets and
builds the command for the matching external evaluator.

| Name | Typical CARLA | Protocol | Local support |
| --- | --- | --- | --- |
| `corl2017` | 0.8.2 | Straight, One Turn, Navigation, Navigation + dynamic obstacles | Registered, legacy runner required |
| `nocrash` | 0.8.4 | Town01/Town02, empty/regular/dense traffic, train/test weather | Registered, legacy runner required |
| `town05_short` | 0.9.10 | 32 short routes with Leaderboard scenarios | Runnable when LB1 assets are installed |
| `town05_long` | 0.9.10 | 10 long routes with Leaderboard scenarios | Runnable when LB1 assets are installed |
| `longest6` | 0.9.10 | 36 long routes across six towns | Runner ready; pass the paper's route XML |
| `longest6_v2` | 0.9.15 | Longest6 adapted to Leaderboard 2.x scenario logic | Runner ready; pass the v2 route XML |
| `carla_leaderboard1` | 0.9.10 | Official public Leaderboard 1.x routes | Runnable when LB1 assets are installed |
| `bench2drive220` | 0.9.15 | 220 routes, 44 scenario types, 23 weathers, 12 towns | Runnable when Bench2Drive is installed |

The launcher records the route XML SHA-256, route count, towns, embedded
scenario types, and weather-entry count during preflight. It also refuses to
run when required files are missing and warns when the detected CARLA version
differs from the version normally used for the protocol. An evaluator import
smoke catches missing Python dependencies before a long simulator run. Install
the matching Leaderboard and ScenarioRunner `requirements.txt` files in a
dedicated environment when it reports a missing module.

List the protocols:

```bash
python scripts/evaluate_paper.py --list
```

Check a one-route Bench2Drive smoke command without starting it:

```bash
python scripts/evaluate_paper.py \
  --benchmark bench2drive220 \
  --carla-root /path/to/CARLA_0.9.15 \
  --agent /path/to/leaderboard_agent.py \
  --agent-config /path/to/agent_config_or_checkpoint \
  --route-subset 0
```

Add `--check-server` to require an already-running CARLA server during the
check. Add `--run` to execute the generated command; `--run` always checks the
server first. The launcher does not start or kill CARLA processes.

Town05 uses the Leaderboard 1.x evaluator and separate scenario annotations:

```bash
python scripts/evaluate_paper.py \
  --benchmark town05_long \
  --carla-root /path/to/CARLA_0.9.10.1 \
  --leaderboard-root /path/to/leaderboard \
  --scenario-runner-root /path/to/scenario_runner \
  --agent /path/to/leaderboard_agent.py \
  --agent-config /path/to/agent_config \
  --run
```

For Longest6, also pass `--routes /path/to/longest6.xml`. Use `longest6` for
the Leaderboard 1.0/CARLA 0.9.10 protocol and `longest6_v2` for its CARLA
0.9.15 adaptation. Their results are not directly comparable. Different
codebases publish different route revisions, so the launcher deliberately does
not silently substitute another XML.

The output JSON is produced by the official evaluator and contains its native
driving score, route completion, infraction score, and per-route records.
Bench2Drive additionally exposes scenario success statistics through its own
evaluation tooling.

### Agent compatibility

`--agent` must point to a CARLA Leaderboard `AutonomousAgent` implementation.
The vector policies trained by this repository's current `CarlaEnv` are local
lane-following policies: they use simulator actor state and do not consume a
destination route command. They therefore cannot be relabelled as official
Town05, Longest6, or Bench2Drive agents. A future adapter must define the
Leaderboard sensor contract and train the policy with route-conditioned
observations before those checkpoints can produce meaningful paper numbers.

CoRL2017 and NoCrash depend on the legacy CARLA 0.8.x driving-benchmarks API.
Their task definitions are kept in the registry for experiment planning, but
the launcher fails clearly instead of pretending that a CARLA 0.9.x random
spawn episode is equivalent.

Protocol sources: [original CARLA paper](https://arxiv.org/abs/1711.03938),
[NoCrash paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Codevilla_Exploring_the_Limitations_of_Behavior_Cloning_for_Autonomous_Driving_ICCV_2019_paper.html),
[CARLA Leaderboard](https://github.com/carla-simulator/leaderboard),
[TransFuser](https://github.com/autonomousvision/transfuser), and
[Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive).

## Lightweight internal suite

`carla_lightweight_v0` runs directly through this repository's vector RL
environment. It evaluates five fixed configurations with seeds `0, 1, 2, 3,
4`. The old name `carla_common_v0` remains as a compatibility alias only.

| Protocol | Purpose | Horizon |
| --- | --- | ---: |
| `lane_following_v0` | Moderate-traffic control baseline | 500 |
| `urban_traffic_v0` | Active signals and mixed Town03 traffic | 750 |
| `dense_traffic_v0` | Dense Town05 traffic stress test | 750 |
| `adverse_weather_v0` | Hard-rain configuration check | 750 |
| `town02_generalization_v0` | Held-out-map generalization | 500 |

`lane_following_empty_v0` is a separate no-traffic sanity check.

```bash
python scripts/evaluate.py \
  --algo sac \
  --checkpoint /path/to/sac_ckpt.pt \
  --suite carla_lightweight_v0
```

These reports contain horizon survival, distance, return/cost, speed, lane
offset, stationary and overspeed ratios, collision/off-road rates, and event
counts per kilometre. `horizon_fraction` is not route completion, and the
internal success rule is not an official Leaderboard metric.
