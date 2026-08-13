# Twin Delayed DDPG (TD3)

**Family:** online, off-policy. **Paper:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477).
**Implementation:** [`carla_rl_lab/algorithms/td3.py`](../../carla_rl_lab/algorithms/td3.py).

## Principle

TD3 extends deterministic policy gradients with three controls against Q-value
overestimation: two critics and their minimum target, noise added to target
actions, and a delayed actor update. Its target is:

```text
a' = clip(mu'(s') + clip(epsilon, -c, c), action bounds)
y  = r + gamma * (1 - terminal) * min(Q1'(s', a'), Q2'(s', a'))
```

The actor maximizes `Q1(s, mu(s))`; actor and target networks update only every
`td3_policy_delay` critic steps. Online exploration adds Gaussian noise to the
deterministic action.

## Start Training

```bash
python scripts/train.py \
  --algo td3 --network Pixel_SAC \
  --benchmark nocrash_train_regular_v0 \
  --total-timesteps 20000 --checkpoint-interval 2000 \
  --minimal-size 1500 --batch-size 64 --buffer-size 15000 \
  --hidden-dim 128 --view-mode none --logger tensorboard \
  --run-name pilots/rlfold_town01_regular_pixel_td3_seed0_20k \
  --seed 0 --port 2000 --require-clean-git
```

Resume by adding `--checkpoint /path/to/td3_ckpt_last.pt` and setting the new
absolute `--total-timesteps` target.

## Metrics And Results

Plot both critic losses, delayed `actor_loss`, `avg_q`, episode return/cost,
action distributions, termination reasons, and reward terms.

The 2026-08-13 seed-0 pilot completed 20k real CARLA 0.9.15 steps with the
pixel encoder. Step 8k was best on the fixed 10-episode selector: **40%
success**, 0.999 mean route completion, 40% collision rate, and 20% off-road
rate. Later checkpoints regressed, which is why checkpoint selection is kept
separate from the full test suite. Curves, scalar CSV files, run metadata, and
the checkpoint manifest are in the [evidence bundle](../../results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/README.md).
This is a small single-seed pilot; TD3 has not yet received a 150-episode full
suite evaluation.
