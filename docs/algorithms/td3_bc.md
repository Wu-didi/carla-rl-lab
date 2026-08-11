# TD3+Behavior Cloning (TD3+BC)

**Family:** offline RL. **Paper:** [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860).
**Implementation:** [`carla_rl_lab/algorithms/offline.py`](../../carla_rl_lab/algorithms/offline.py).
**Runner:** [`scripts/train_offline.py`](../../scripts/train_offline.py).

## Principle

TD3+BC keeps TD3's twin-critic target but trains only on a fixed dataset. Its
actor balances Q maximization with supervised behavior cloning:

```text
lambda  = alpha / mean(|Q(s, mu(s))|)
L_actor = -lambda * mean[Q(s, mu(s))] + mean[(mu(s) - a_data)^2]
```

The value-dependent scale makes `alpha` easier to reuse across reward scales.
Target policy smoothing, the minimum twin target, delayed actor updates, and
soft target updates follow TD3. No new CARLA data is collected during updates.

## Prepare Data And Train

```bash
python scripts/collect_dataset.py \
  --benchmark nocrash_train_empty_v0 \
  --policy autopilot --transitions 100000 \
  --output artifacts/datasets/nocrash_expert_seed0_100k.npz --seed 0

python scripts/train_offline.py \
  --algo td3_bc \
  --dataset artifacts/datasets/nocrash_expert_seed0_100k.npz \
  --updates 100000 --batch-size 256 --checkpoint-interval 10000 \
  --seed 0 --logger tensorboard --run-name nocrash/td3_bc_mlp_seed0
```

## Metrics And Results

Inspect twin critic losses, delayed `actor_loss`, `bc_loss`, and
`td3_bc_scale`. Evaluation return and success are essential because a falling
BC term does not imply improved driving. A four-update CARLA-dataset integration
smoke completed with valid TensorBoard logs and checkpoint. Formal dataset
training, evaluation, and curves are **Pending**.
The current actor/critics are MLPs; a pixel-native offline baseline is pending.
