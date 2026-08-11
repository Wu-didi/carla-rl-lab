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
  --algo td3 --total-timesteps 100000 --checkpoint-interval 10000 \
  --action-mode longitudinal_2d --reward research_v2 \
  --town Town05 --vehicles 50 --walkers 0 --traffic off --view-mode none \
  --max-time-episode 500 --seed 0 --logger tensorboard \
  --checkpoint-replay-buffer --run-name v0.1/td3_town05_seed0
```

Resume by adding `--checkpoint /path/to/td3_ckpt_last.pt` and setting the new
absolute `--total-timesteps` target.

## Metrics And Results

Plot both critic losses, delayed `actor_loss`, `avg_q`, episode return/cost,
action distributions, termination reasons, and reward terms. CPU update and
checkpoint round-trip tests pass. A formal CARLA training run and performance
curve have not been completed; status is **Pending**, not zero performance.
