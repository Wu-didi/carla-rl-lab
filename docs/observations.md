# Observation Contract

`pixel_v1` is the policy-facing contract for the first CarlaRLLab vision
baseline. It is inspired by RLAD and RLfOLD, but its CARLA version and image
resolution are explicitly different.

## Policy Inputs

| Field | Shape before packing | Range | Meaning |
| --- | --- | --- | --- |
| `image` | `(6, 84, 84)` | uint8 `[0, 255]` | Two chronological front RGB frames, channel-first |
| `waypoints` | `(20,)` | float32 `[-1, 1]` | Ten `(forward, lateral)` route points in the ego frame, divided by 25 m |
| `vehicle_measurements` | `(2,)` | `[0, 1]`, `[-1, 1]` | Speed divided by desired speed and previous steering |

The RLfOLD profile renders a single front camera at `256x256`, mounts it at
`(x=1.5 m, z=2.4 m)`, and uses a 90-degree field of view. Each frame is resized
to `84x84` before entering the lightweight policy. CARLA runs at 10 Hz
(`dt=0.1`) and the sensor produces one frame per
world tick. At reset, the first frame is repeated until the temporal stack is
full.

`encode_observation` keeps image bytes unchanged and quantizes route,
normalized speed, and steering to bytes. The resulting replay state has:

```text
2 * 3 * 84 * 84 + 10 * 2 + 2 = 42,358 uint8 values
```

The SAC encoder decodes each segment, applies random-shift image augmentation
during training, processes the route with a small 1-D convolution, and fuses
image, route, speed, and steering features. This code is intentionally local to
the algorithm and can be modified without an external RL framework.

## Explicitly Excluded

The policy does not receive:

- global position, yaw, acceleration, or angular velocity;
- lane width, lane offset, or simulator waypoint identity;
- nearby vehicle or pedestrian state;
- LiDAR, semantic segmentation, or privileged maps;
- any handcrafted risk field.

`ego_state` and `lane_info` still exist in the environment observation
dictionary as telemetry. Reward, termination, and the evaluator consume them,
but `encode_observation` excludes them from the policy tensor. Tests lock this
boundary.

## Reward Privilege

`nocrash_v0` uses a simulator-derived safe desired speed to reward slowing for
actors and traffic lights. This signal is available only to the reward
function, following the asymmetric training pattern used by the referenced
work; it is not a policy observation. Every future privileged signal must be
documented at this boundary.

## Action Contract

`target_speed_2d` contains two bounded values:

```text
action[0] in [-1, 1] -> target speed in [0, desired_speed]
action[1] in [-1, 1] -> steering
```

The environment converts target speed to throttle/brake with a visible PID in
`carla_rl_lab/envs/carla_env.py`. This avoids asking the policy to learn engine
dynamics and mirrors the action abstraction used by RLAD/RLfOLD.

## Paper Relationship

| Field | RLAD | RLfOLD | CarlaRLLab `pixel_v1` |
| --- | --- | --- | --- |
| Camera frames | 3 | 2 | 2 |
| Image size | `256x256` | paper implementation setting | `84x84` practical default |
| Route points | 10 | 10 | 10 |
| Measurements | speed + steer | speed + steer | speed + steer |
| Action | target speed + steer | target speed + steer | target speed + steer |
| CARLA | 0.9.10.1 | 0.9.10.1-era stack | 0.9.15 |

Increasing `image_size` to 256 produces a packed state of 393,238 bytes and
substantially increases replay memory and compute. Such a run must use a new
named experiment protocol and cannot be mixed with the default results.

Sources: [RLAD paper](https://arxiv.org/abs/2305.18510),
[RLfOLD paper](https://ojs.aaai.org/index.php/AAAI/article/view/29049), and
[official RLfOLD code](https://github.com/DanielCoelho112/rlfold).
