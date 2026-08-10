# CarlaRLLab

[English](README.md)

CarlaRLLab 是一个面向 CARLA 的科研型强化学习平台。项目将算法、网络结构、奖励函数、
实验日志和 benchmark 协议明确拆分，便于研究者直接修改模型、loss 和 reward，而不需要
穿过高度封装的第三方 RL 框架。

## v1 范围

- 在线 off-policy 算法：SAC、TD3、DDPG
- 可直接修改的 MLP 与语义注意力 SAC 网络
- 可组合奖励项，同时保留旧奖励作为默认配置
- TensorBoard 与可选的 Weights & Biases 日志
- 固定的 `lane_following_v0` benchmark 协议
- 连续控制：油门、方向盘、刹车

PPO 等 on-policy 算法会使用独立的 rollout runner。CQL、IQL、TD3+BC 等离线 RL
算法会使用 dataset runner，不会被强行塞进在线 replay-buffer 训练循环。

## 目录结构

```text
carla_rl_lab/
  algorithms/       # SAC、TD3、DDPG 与算法注册表
  benchmarks/       # 固定实验协议
  buffers/          # Replay buffer，后续加入 rollout buffer
  envs/             # CARLA 环境与环境工厂
  evaluation/       # Benchmark 评测与报告
  logging/          # TensorBoard 与 W&B 后端
  observations/     # 观测适配与维度校验
  rewards/          # 奖励项与奖励配置
scripts/
  train.py
  evaluate.py
  launch_carla.sh
tests/
docs/
artifacts/           # 日志、checkpoint、评测报告，不提交到 Git
legacy/              # 重构前实验的本地归档，不提交到 Git
```

## 安装

当前环境以 CARLA 0.9.13 和 Python 3.7 为基准：

```bash
conda create -n carla37 python=3.7
conda activate carla37
pip install -r requirements.txt
```

CARLA Python API 需要从对应的 CARLA 发行包单独安装。使用 W&B 时安装
`requirements-wandb.txt`。

## 启动 CARLA

在 CARLA 安装目录下使用以下任一命令：

```bash
./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia
```

```bash
./CarlaUE4.sh -quality_level=Low-prefernvidia
```

仓库内的启动脚本执行相同命令：

```bash
CARLA_ROOT=/path/to/CARLA scripts/launch_carla.sh offscreen
CARLA_ROOT=/path/to/CARLA scripts/launch_carla.sh window
```

## 训练

```bash
python scripts/train.py --algo sac --reward legacy --logger tensorboard
python scripts/train.py --algo td3 --reward research_v1 --logger both --wandb-mode offline
python scripts/train.py --algo ddpg --checkpoint /path/to/ddpg_ckpt.pt
```

实验输出写入 `artifacts/runs/`。日志包括算法 loss、entropy、Q 值、episode reward、
safety cost、动作分布以及每个 reward term 的分解结果。

## Benchmark

使用固定的 `lane_following_v0` 协议进行确定性评测：

```bash
python scripts/evaluate.py \
  --algo sac \
  --checkpoint /path/to/sac_ckpt.pt \
  --benchmark lane_following_v0
```

评测输出平均 return、平均 cost、平均速度、碰撞率、驶出道路率和成功率，JSON 报告位于
`artifacts/evaluations/`。

## 科研扩展入口

- 实现 `BaseAgent` 并注册 `AlgorithmSpec` 即可新增算法。
- 直接在算法模块中修改网络结构。
- 在 `carla_rl_lab/rewards/terms.py` 添加奖励项并组合成 reward profile。
- 注册固定 benchmark spec，无需修改训练代码。

CarlaRLLab 是基于 CARLA 构建的独立科研项目，不是 CARLA 官方项目。
