# Advantage Actor-Critic (A2C)

**Family:** online, on-policy. **Reference:** synchronous counterpart of [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).
**Implementation:** [`carla_rl_lab/algorithms/on_policy.py`](../../carla_rl_lab/algorithms/on_policy.py).

## Principle

A2C performs one synchronous update from each fresh rollout. GAE supplies the
advantage and return targets:

```text
L_actor = -mean[log pi(a|s) * stop_gradient(A)]
L_value = mean[(V(s) - return)^2]
L       = L_actor + value_coef*L_value - entropy_coef*entropy
```

Unlike PPO, A2C does not clip ratios or run repeated minibatch epochs. It is a
useful lower-complexity on-policy reference but can be more sensitive to
rollout variance and learning rate.

## Start Training

```bash
python scripts/train_on_policy.py \
  --benchmark nocrash_train_empty_v0 \
  --algo a2c --total-timesteps 100000 --rollout-steps 2048 \
  --checkpoint-interval 10000 --seed 0 --logger tensorboard \
  --run-name nocrash/a2c_mlp_smoke_seed0
```

## Metrics And Results

Plot `actor_loss`, `value_loss`, `entropy`, return/cost, and action statistics.
CPU update/checkpoint tests pass. A formal CARLA run and curves are **Pending**.
The current runner uses an MLP; pixel-native A2C remains pending.
