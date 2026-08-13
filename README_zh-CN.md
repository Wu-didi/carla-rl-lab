<div align="center">

# CarlaRLLab

**面向 CARLA 城市自动驾驶的、易修改的视觉强化学习研究框架。**

[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-e11d48?style=flat-square)](https://github.com/carla-simulator/carla/releases/tag/0.9.15)
[![Python](https://img.shields.io/badge/Python-3.7-3776ab?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-ee4c2c?style=flat-square)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](LICENSE)

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

CarlaRLLab 服务于需要频繁修改策略网络、观测、奖励函数和训练公式的研究工作，
避免把关键逻辑藏在复杂 RL 框架与多层 wrapper 中。第一版有意控制范围：先做好一个
Pixel SAC 主基线、一个有版本的 NoCrash 适配、透明的 PyTorch 实现和可复现实验产物。

> **当前研究状态：**已经在 CARLA 0.9.15 中真实完成 20k step 的 Pixel SAC、
> TD3、PPO、BC 与示范辅助 SAC pilot。选中的 seed-0 checkpoint 在 Town02 完整
> 50 回合 Empty 测试中成功率为 46%，Regular 为 24%，Dense 为 4%。这是公开
> 透明的小预算 pilot，不是三随机种子论文级 baseline。

## 为什么做 CarlaRLLab

- **方便科研修改。**算法公式使用普通 PyTorch 编写，奖励项与观测打包是短函数。
- **先固定协议，再比较分数。**训练和验证选择命名 benchmark，不依赖未记录的命令行
  参数。
- **统一策略输入。**当前策略只接收前视 RGB、路线点、速度和上一时刻转角。模拟器
  真值只用于奖励、终止和评估，不会暗中进入策略。
- **完整科研日志。**默认支持 TensorBoard，可选 W&B；记录配置、checkpoint 元数据、
  reward 分项、loss、动作统计和 benchmark 指标。
- **结构简单。**Agent 由小型网络和 buffer 组合，不建立多层环境 wrapper 或复杂类继承。

## 当前范围

| 分类 | 算法 | 实现状态 |
| --- | --- | --- |
| 在线、off-policy | SAC、TD3、DDPG | Pixel SAC、TD3 已实车训练；DDPG MLP 核心已测试 |
| 在线、on-policy | PPO、A2C | Pixel PPO 已实车训练；A2C 核心已测试 |
| 离线 RL | TD3+BC、CQL(H)、IQL | Dataset runner 与 MLP 核心已实现；像素基线待完成 |
| 模仿学习 | BC、GAIL、AIRL | Pixel BC 已实车训练；对抗模仿核心已测试 |

算法 registry 会阻止用错误 runner 启动算法。这里的“已实现”不等于已经有论文级
CARLA checkpoint；只有在下面固定协议中完成训练与验证后，才会发布正式 baseline。

## 观测与控制协议

`pixel_v1` 参考
[RLAD](https://arxiv.org/abs/2305.18510) 与
[RLfOLD](https://ojs.aaai.org/index.php/AAAI/article/view/29049) 的策略输入形式：

```text
前视 RGB：           2 帧 x 3 x 84 x 84，uint8（RLfOLD profile）
路线：               10 个 ego 坐标系下的 (x, y) 路线点
车辆测量：           归一化速度 + 上一时刻转角
Replay 打包状态：    42,358 个 uint8
策略动作：           目标速度 + 转角，均在 [-1, 1]
底层控制：           目标速度经 PID 转换为油门/刹车
```

全局位姿、车道测量、周围 actor 状态、激光雷达和 risk field 都不是策略输入；这些
信息只允许用于环境奖励、终止或评估。RLfOLD benchmark profile 只用一路前视
RGB：原始渲染 `256x256`、FOV 90°、安装位置 `(x=1.5 m, z=2.4 m)`，进入轻量
baseline 前缩放到 `84x84`。RLAD 使用
`256x256`、CARLA 0.9.10.1 和不同训练预算，因此本项目 CARLA 0.9.15 适配结果不能
直接写成与 RLAD 完全同协议的成绩。详细边界见[观测协议](docs/observations.md)。

## NoCrash 0.9.15

主 suite `rlfold_nocrash_0915_v0` 将 RLfOLD 的 NoCrash 任务协议适配到 CARLA
0.9.15 API；`nocrash_0915_v0` 保留为兼容别名：

| 划分 | 地图 | 车辆 / 行人 | 天气 | Episode 数 |
| --- | --- | ---: | --- | ---: |
| 训练 | Town01 | 每回合采样 0-150 / 0-300 | 4 种训练天气 | 持续抽取路线 |
| Empty | Town02 | 0 / 0 | 2 种未见天气 | 25 路线 x 2 = 50 |
| Regular | Town02 | 15 / 50 | 2 种未见天气 | 50 |
| Dense | Town02 | 70 / 150 | 2 种未见天气 | 50 |

成功表示完成路线且没有碰撞。驶出车道、逆行、堵塞和超时会失败终止；固定路线评测中
红灯只计数，不提前结束。报告包含成功率、路线完成度、return、速度、分类碰撞、红灯、
堵塞和每公里违规数。红灯检测是明确记录
的 CARLA 0.9.15 近似实现，因此该 suite 称为 0.9.15 adaptation，而不是旧版
CARLA 0.8 NoCrash 原生 runner。

## 可复现 Pilot 结果

第一批受 Git 跟踪的证据位于
[`results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/`](results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/)。
所有算法使用相同的前视 RGB 输入、Town01 固定 20 车/50 行人训练课程、seed 0，
并在 Town02 Empty 的同一 10 回合网格上选择 checkpoint。

| 方法 | 预算 | 选择 step | 选模成功率 |
| --- | ---: | ---: | ---: |
| Pixel SAC | 20k 环境 step | 8k | 0% |
| Pixel TD3 | 20k 环境 step | 8k | 40% |
| Pixel BC | 10k update / 10k 示范 | 10k | 20% |
| Pixel PPO | 20k 环境 step | 20k | 0% |
| Pixel SAC + 示范 | 5k BC 预训练 + 20k 环境 step | **8k** | **60%** |

冻结后的示范辅助 SAC checkpoint 在完整 50 回合条件下取得 **46% Empty**、
**24% Regular** 与 **4% Dense** 成功率。Dense 使用未修改的请求交通流 70 车/
150 行人。证据目录包含逐回合 JSON、scalar CSV、曲线、命令、哈希与局限说明。
选中的
[`sac_demo_seed0_step8000.pt`](results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/checkpoints/sac_demo_seed0_step8000.pt)
检查点直接纳入仓库，并使用 SHA-256 校验。

![示范辅助 SAC episode return](results/rlfold_nocrash_0915_v0/pilot_seed0_2026-08-13/curves/pixel_sac_demo_episode_reward.png)

## 目录结构

```text
carla_rl_lab/
  algorithms/       易修改的 PyTorch 算法与 registry
  benchmarks/       命名协议、NoCrash 路线与天气组
  buffers/          replay、rollout 和离线数据集
  envs/             单一 CARLA 环境与控制转换
  evaluation/       路线指标和 suite 汇总
  logging/          TensorBoard 与可选 W&B 后端
  observations/     策略观测打包
  rewards/          有版本的奖励函数
scripts/             训练、采集、验证、smoke、曲线导出
docs/                协议、架构、实验与算法教程
tests/               CPU 单元测试和集成测试
```

## 安装

参考环境为 Linux x86_64、NVIDIA GPU、CARLA 0.9.15 和 Python 3.7。CARLA 是
独立模拟器进程，不会随着 `pip install -e .` 安装。

### 1. 确定安装目录

```bash
export CARLA_ROOT="$HOME/simulators/CARLA_0.9.15"
export CARLA_RL_LAB_ROOT="$HOME/code/carla-rl-lab"
mkdir -p "$CARLA_ROOT" "$(dirname "$CARLA_RL_LAB_ROOT")"
```

需要长期生效时，将这两个环境变量加入 shell 配置。

### 2. 下载并解压 CARLA 0.9.15

把官方 Linux 预编译包直接下载、解压到 `CARLA_ROOT`。压缩包本身已经包含
`CarlaUE4.sh`，不要在里面再套一层 `CARLA_0.9.15` 目录。

```bash
cd "$CARLA_ROOT"
wget -c \
  https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz \
  -O CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
```

关键目录应为：

```text
$CARLA_ROOT/
  CarlaUE4.sh
  CarlaUE4/
  PythonAPI/
    carla/
      agents/
      dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

继续之前先检查：

```bash
test -x "$CARLA_ROOT/CarlaUE4.sh" && echo "CARLA server: OK"
ls "$CARLA_ROOT"/PythonAPI/carla/dist/*0.9.15*py3.7*
```

Town01 和 Town02 已包含在基础包中，运行 NoCrash 无需下载 Additional Maps。

### 3. 克隆项目并创建 Python 环境

```bash
git clone git@github.com:Wu-didi/carla-rl-lab.git "$CARLA_RL_LAB_ROOT"
cd "$CARLA_RL_LAB_ROOT"

conda create -n carla-rl-lab python=3.7 -y
conda activate carla-rl-lab
python -m pip install --upgrade "pip<24.1"
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 4. 加载完全匹配的 CARLA Python API

CARLA 0.9.15 自带 Python 3.7 egg。`PYTHONPATH` 必须同时包含 API 根目录
（提供 navigation agents）和 egg（提供 `import carla`）：

```bash
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"

python -c "import carla; from agents.navigation.behavior_agent import BehaviorAgent; print(carla.__file__)"
```

不要在该环境中从 PyPI 安装其他版本的 `carla`。Client/server 版本不匹配有时要到
连接模拟器后才会报错。

### 5. 启动 CARLA

打开终端 A，严格使用下面两条项目约定命令之一。

无界面训练：

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia
```

窗口调试：

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -quality_level=Low-prefernvidia
```

仓库脚本执行的是同样命令：

```bash
CARLA_ROOT="$CARLA_ROOT" scripts/launch_carla.sh offscreen
```

### 6. 验证完整链路

保持 CARLA 运行，在终端 B 执行：

```bash
conda activate carla-rl-lab
cd "$CARLA_RL_LAB_ROOT"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"

python -m unittest discover -s tests -v
python scripts/smoke_carla.py \
  --benchmark nocrash_empty_v0 --steps 10 \
  --frame-output artifacts/smoke/nocrash_town02_rgb.png
```

Smoke 应打印一致的 CARLA client/server 版本、`42358` 的 `state_dim`、值为 `1`
的 `num_cameras`、非零前视图像统计量，并完成 10 个 step。

## 训练 Pixel SAC

先在 Town01 空场景课程上训练，再增加常规交通：

```bash
python scripts/train.py \
  --benchmark nocrash_train_empty_v0 \
  --algo sac --network Pixel_SAC \
  --total-timesteps 100000 \
  --minimal-size 1500 --batch-size 128 --buffer-size 30000 \
  --checkpoint-interval 10000 \
  --logger tensorboard \
  --run-name nocrash/pixel_sac_empty_seed0 \
  --seed 0
```

RLfOLD 的 Town01 交通分布使用 `--benchmark nocrash_train_v0`，每回合均匀采样
0-150 辆车与 0-300 个行人；固定 20/50 课程使用
`nocrash_train_regular_v0`。恢复训练时传入最后一个
checkpoint，并把 `total-timesteps` 设置为更大的绝对步数：

```bash
python scripts/train.py \
  --checkpoint artifacts/runs/nocrash/pixel_sac_empty_seed0/checkpoints/sac_ckpt_last.pt \
  --total-timesteps 200000
```

每条 transition 包含两个 `42,358` 字节的观测；30,000 条 replay 的观测约占
2.5 GB。应根据机器内存设置 buffer。

### 使用在线示范训练

先在同一课程采集 BehaviorAgent 示范，再把 actor 预训练和显式 BC 项加入 SAC：

```bash
python scripts/collect_dataset.py \
  --benchmark nocrash_train_regular_v0 --policy autopilot \
  --transitions 10000 \
  --output artifacts/datasets/nocrash_regular_demo_seed0_10k.npz \
  --view-mode none --seed 0

python scripts/train.py \
  --benchmark nocrash_train_regular_v0 --algo sac \
  --expert-dataset artifacts/datasets/nocrash_regular_demo_seed0_10k.npz \
  --demo-pretrain-updates 5000 --demo-bc-coef 0.5 \
  --total-timesteps 20000 --minimal-size 1500 \
  --batch-size 64 --buffer-size 15000 --hidden-dim 128 \
  --checkpoint-interval 2000 --logger tensorboard \
  --run-name nocrash/pixel_sac_demo_seed0 --seed 0
```

这是受在线示范研究启发、保持代码透明的本地适配，不会宣称为 RLfOLD 完整方法的
精确复现。

## 在相同协议验证

先跑一条路线、一个天气：

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/sac_ckpt_last.pt \
  --benchmark nocrash_empty_v0 \
  --routes 1 --weathers 1 --logger tensorboard
```

确认链路后，在三种交通密度下执行全部 150 个 Town02 episode：

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/sac_ckpt_last.pt \
  --suite rlfold_nocrash_0915_v0 \
  --output-tag release --logger tensorboard
```

评估每完成一个 episode 都会写入 `progress.json`。中断后用相同命令加
`--resume`；程序会先校验 benchmark、scope 和 checkpoint 哈希。

所有算法必须使用同一 checkpoint 选择规则、路线、天气、交通量、图像协议和随机
种子。报告保存在 `artifacts/evaluations/`，并带 checkpoint SHA-256 与完整元数据。

## 日志与曲线

TensorBoard 默认可用：

```bash
tensorboard --logdir artifacts/runs --port 6006
```

启用 Weights & Biases：

```bash
python -m pip install -r requirements-wandb.txt
wandb login
python scripts/train.py \
  --benchmark nocrash_train_empty_v0 --algo sac \
  --logger both --wandb-mode online --run-name nocrash/pixel_sac_seed0
```

从已完成实验导出 PNG/CSV：

```bash
python scripts/export_curves.py \
  --run-dir artifacts/runs/nocrash/pixel_sac_empty_seed0 \
  --benchmark-report artifacts/evaluations/nocrash_empty_v0/sac/report.json
```

## 修改科研代码

| 修改内容 | 主要文件 |
| --- | --- |
| Pixel encoder 或 SAC 公式 | [`carla_rl_lab/algorithms/sac.py`](carla_rl_lab/algorithms/sac.py) |
| 观测字段与打包 | [`carla_rl_lab/observations/vector.py`](carla_rl_lab/observations/vector.py) |
| Reward 各项 | [`carla_rl_lab/rewards/profiles.py`](carla_rl_lab/rewards/profiles.py) |
| 相机、路线、PID、终止 | [`carla_rl_lab/envs/carla_env.py`](carla_rl_lab/envs/carla_env.py) |
| Benchmark 交通、路线、天气 | [`carla_rl_lab/benchmarks/specs.py`](carla_rl_lab/benchmarks/specs.py) |
| 指标和成功规则 | [`carla_rl_lab/evaluation/evaluator.py`](carla_rl_lab/evaluation/evaluator.py) |

第一批 pilot 暴露出一个具体科研问题：SAC 和 TD3 都在早期达到峰值后退化，但训练
return 仍为正。[CADR 研究原型](docs/research/cadr.md) 已完成 seed-0 实验，四个
候选 checkpoint 均只有 20% 成功率，低于固定 BC 基线的 60% 峰值。该负结果的
逐回合报告、scalar 与曲线公开在
[`results/ablations/cadr/`](results/ablations/cadr/)。后续的
[UGPI 原型](docs/research/ugpi.md) 在两个无图像增强的 critic 判断不一致时，直接
抑制 critic 驱动的 actor 更新；其方法和评测约束已在 CARLA 实验前完成预注册。

新增 reward 应同时返回标量与命名分项。新增观测应建立新的协议名和 shape，不能悄悄
改变 `pixel_v1` 的含义。

## 其他训练入口

```bash
# 在同一个 Town01 协议中采集专家数据
python scripts/collect_dataset.py \
  --benchmark nocrash_train_empty_v0 --policy autopilot \
  --transitions 100000 --output artifacts/datasets/nocrash_expert.npz

# 离线 runner（像素 baseline 仍待完成）
python scripts/train_offline.py --algo td3_bc --dataset artifacts/datasets/nocrash_expert.npz

# Pixel 行为克隆
python scripts/train_imitation.py --algo bc --expert-dataset artifacts/datasets/nocrash_expert.npz

# 64 step 端到端集成矩阵，不是性能 benchmark
scripts/run_research_smoke.sh
```

每种算法的原理、启动方法、日志与结果状态见[算法索引](docs/algorithms/README.md)，
benchmark 定义见 [docs/benchmarks.md](docs/benchmarks.md)，结果报告约束见
[docs/experiments.md](docs/experiments.md)。

## TODO

- [ ] 使用三个完整训练随机种子验证 CARLA 0.9.15，并发布 checkpoint 与原始
  TensorBoard/W&B 导出。
- [x] 在同一 NoCrash 适配上训练 seed-0 Pixel SAC、TD3、PPO、BC 与示范辅助
  SAC pilot，并发布曲线。
- [ ] 补齐 Pixel DDPG、A2C、离线 RL、GAIL 与 AIRL baseline。
- [ ] 每个 checkpoint 报告 loss、return、路线完成度、成功率、分类碰撞、红灯与
  堵塞指标。
- [ ] 对相机历史、路线表示、速度、转角及未来传感器融合观测做有版本的消融，不把
  模拟器专有真值输入策略。
- [ ] 为每种 RL 算法编写包含公式、数据流、启动、排错、验证和结果解读的详细教程。
- [ ] 提供一键安装、环境诊断和锁定版本的 CARLA 0.9.15 Docker 镜像。
- [ ] 建立 CPU CI，并单独调度真实 CARLA 集成测试。

## License 与引用

CarlaRLLab 使用 [MIT License](LICENSE)。内置 NoCrash 路线来自 RLfOLD 官方仓库的
适配，详情见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。CARLA 与引用
论文仍遵循其各自 license 和引用要求。
