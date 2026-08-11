# Deep Deterministic Policy Gradient (DDPG)

**Family:** online, off-policy. **Paper:** [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971).
**Implementation:** [`carla_rl_lab/algorithms/ddpg.py`](../../carla_rl_lab/algorithms/ddpg.py).

## Principle

DDPG learns a deterministic actor `mu(s)` and one critic `Q(s,a)` from replay.
The critic regresses to a slowly moving target and the actor follows the
deterministic policy gradient:

```text
y       = r + gamma * (1 - terminal) * Q'(s', mu'(s'))
L_Q     = mean[(Q(s,a) - y)^2]
L_actor = -mean[Q(s, mu(s))]
```

Target actor and critic parameters use Polyak updates. Gaussian action noise is
used only during online data collection.

## Start Training

```bash
python scripts/train.py \
  --algo ddpg --total-timesteps 100000 --checkpoint-interval 10000 \
  --action-mode longitudinal_2d --reward research_v2 \
  --town Town05 --vehicles 50 --walkers 0 --traffic off --view-mode none \
  --max-time-episode 500 --seed 0 --logger tensorboard \
  --checkpoint-replay-buffer --run-name v0.1/ddpg_town05_seed0
```

## Metrics And Results

Track `critic_loss`, `actor_loss`, `avg_q`, return, cost, action saturation, and
termination reasons. DDPG is intentionally retained as a simple reference but
is more sensitive to critic error than TD3. CPU update/checkpoint tests pass;
formal CARLA training and curves are **Pending**.
