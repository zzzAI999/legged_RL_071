# Dogleg 轮腿机器人强化学习

**运行平台：原生 Linux，推荐 Ubuntu 22.04 + NVIDIA GPU。** 以下命令均在这台 Linux 机器的终端执行。Windows 本机可阅读源码、运行 CPU 检查；本仓库的 GPU 监督程序使用 Linux `/proc` 和信号，不能直接在 Windows PowerShell 中运行仿真。不要把 Windows 路径直接粘贴到 Linux。


## 1. 准备机器

推荐使用独立、空闲的 NVIDIA RTX GPU。原训练使用 4096 个并行环境；历史显存占用约 8.4 GiB，但驱动、图形上下文和缓存会改变实际需求，建议准备 16–24 GiB 显存、64 GiB 内存和至少 100 GiB 可用磁盘。小规模 smoke 测试使用 32 个环境。

先运行：

```bash
nvidia-smi
```

如果命令能列出 GPU 和驱动，就继续下一步。没有驱动的新装 Ubuntu 可使用发行版提供的驱动安装方式；以下命令会安装驱动并重启机器：

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

重启后再次运行 `nvidia-smi`。驱动还必须满足 NVIDIA 对 Isaac Sim 4.5 的兼容要求；如果官方兼容检查失败，先处理驱动问题。PyTorch 的 CUDA 11.8 wheel 自带所需 CUDA 用户态库，一般不需要另装一套系统 CUDA Toolkit。

安装系统基础工具：

```bash
sudo apt update
sudo apt install -y git python3.10 python3.10-venv python3-pip build-essential cmake \
  libgl1 libglib2.0-0 libx11-6 libxext6 libxi6 libxrandr2 libxrender1 libxtst6 libvulkan1 vulkan-tools
```

## 2. 获取项目

克隆本仓库并进入目录：

```bash
git clone https://github.com/zzzAI999/legged_RL_071.git dogleg-rl
cd dogleg-rl
```

资产和最新检查点都已随仓库提供，无需再下载旧工程、Navila、机器人其它型号或历史训练输出。

## 3. 从零安装环境

固定的兼容组合如下。**不要直接升级到 Isaac Sim 5.x、Isaac Lab 2.1+ 或最新版 RSL-RL。**

| 组件 | 版本／来源 |
| --- | --- |
| Python | 3.10 |
| Isaac Sim | 4.5.0（Python 包元数据为 4.5.0.0） |
| Isaac Lab | v2.0.2（包内部版本 0.34.9） |
| PyTorch / torchvision | 2.5.1+cu118 / 0.20.1+cu118 |
| NumPy | 1.26.4 |
| PPO / ActorCritic | 随仓库保留的 TRON1 RSL-RL 2.0.2 必要源码 |

一条命令执行安装，**安装脚本只装软件，不启动仿真或训练**：

```bash
bash scripts/install_ubuntu.sh
source .venv/bin/activate
```

首次下载 Isaac Sim 体积较大，需要能访问 PyPI、NVIDIA 软件源、PyTorch 软件源和 GitHub。完整展开的手动安装命令见 [INSTALL.md](docs/INSTALL.md)。NVIDIA EULA 会在第一次启动时请求确认，请阅读后自行选择；安装器不会替你接受许可。

## 4. 先检查，再启动

在仓库根目录、已激活 `.venv` 的终端执行：

```bash
dogleg-rl check
python -m unittest discover -s tests -v
```

`check` 核对固定版本、三个资产 SHA256 和 CPU 策略前向计算，不启动仿真或 CUDA 上下文。输出中应有 `assets_verified: true`、`policy_CPU_forward: true`。

首次启动 Isaac Sim 可先用官方空场景验证安装；这会启动所选 GPU：

```bash
python _deps/IsaacLab/scripts/tutorials/00_sim/create_empty.py --headless --device cuda:0
```

确认场景启动后按 `Ctrl+C` 退出。若有多块 GPU，把 `cuda:0` 改为你选择的空闲 GPU。

然后运行项目 smoke 测试：

```bash
# 单 GPU 机器使用 --gpu 0；原双 GPU 服务器使用 --gpu 1
dogleg-rl smoke --gpu 0
```

`--gpu` 使用 `nvidia-smi` 显示的物理编号。若终端已设置 `CUDA_VISIBLE_DEVICES`，先执行 `unset CUDA_VISIBLE_DEVICES`，避免 CUDA 逻辑编号与 Isaac Sim 物理编号不一致；监督程序会拒绝有这项映射的启动。

它检查 32 个环境中的有限观测／奖励、闭链间隙和超时前的 critic 观测，不进行优化。看到 `SMOKE_COMPLETE` 才继续。第一次启动可能下载并编译缓存，耗时明显长于后续运行。纯 SSH 服务器使用默认 headless；有桌面时可加 `--gui`。

## 5. 使用当前策略评估

```bash
dogleg-rl evaluate --gpu 0 --checkpoint checkpoints/guard17.pt
```

默认使用当前 4096 环境布局中的固定前 512 个环境，分为 Soft／Protected 各 256 个，运行 35.02 仿真秒。输出保存在新的 `runs/<UTC时间>/evaluation/evaluation.json`，其中有两侧通过数、每项失败计数、模型身份和安全状态。该命令是一次新的评估，不会覆盖仓库内的历史结果，也不会训练。

## 6. 开始训练

从当前完整检查点继续，最多新增 100 次 PPO 更新／复位后 60 分钟：

```bash
dogleg-rl train --gpu 0 --checkpoint checkpoints/guard17.pt --updates 100 --minutes 60
```

如果明确要从随机策略开始，而非继续当前模型：

```bash
dogleg-rl train --gpu 0 --fresh --updates 100 --minutes 60
```

保持 4096 并行环境、每环境 rollout 64 个决策、PPO 4 epochs × 4 minibatches。每更新为 262144 个决策和 16 次 Adam step。达到本段 local100 时才执行一次计划内课程评估；少于 100 更新的早停段不会补跑评估。每段完成后停止，不自动接下一段。

输出包括 `checkpoint_initial.pt`、滚动覆盖的 `checkpoint_latest.pt`、完整 `checkpoint_final.pt`、`metrics.jsonl`、`receipt.json`、热监控和子进程退出凭证。保存完整模型、Adam、Python／NumPy／CPU／CUDA RNG 和课程状态，继续训练时重新进行物理复位，不恢复仿真轨迹／求解器 warmstart。

普通停止：在监督终端按 `Ctrl+C`，或在另一终端创建当前输出目录的停止文件：

```bash
touch runs/<当前运行目录>/STOP_AFTER_UPDATE
```

程序完成已经开始的 update／评估，再保存退出。热保护仍为 85°C 警告、86°C 持续 60 秒或 active thermal flag 请求完整保存停止、88°C／硬件 thermal flag 精确暂停该子进程。紧急暂停不代表完成保存，且不会自动恢复；查看 `supervisor_receipt.json`。外层 4500 秒请求保存、5100 秒 TERM、5400 秒 KILL，仅针对本次创建且身份匹配的子进程。

## 7. 当前机制与目录

- `src/dogleg_rl/core/`：当前环境、奖励、控制和 PPO 的实际依赖闭包。一些模块仍有 `v6/v7/v8` 等名称，因为它们是现版本仍在继承的基础实现；旧实验入口和淘汰运行器未收录。
- `src/dogleg_rl/assets/`：实际训练的 5.5 kg USD、完整物理参数、合法复位姿态表；已固定 SHA256。
- `src/dogleg_rl/_vendor/`：必要的算法源码及其原始版权声明；不安装另一份冲突的 `rsl_rl`。
- `checkpoints/guard17.pt`：最新完整检查点。
- `results/guard17_summary.json`：已完成的历史短评估结果。
- `docs/`：完整安装命令、奖励／资格说明和来源清单。

当前 base gyro 惩罚为 `-0.02 × 2.0 × min(g²,25)`，其余 21 项保持原实现。求解器 TGS32/8，物理 1000 Hz、PD 500 Hz、状态／ACK 100 Hz、策略 50 Hz，腿位置 PD 和轮直接力矩路径保持。细节见 [当前机制](docs/MECHANISM.md)。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `No module named isaaclab` | 激活本项目 `.venv`，重新执行安装脚本；不要使用系统 Python 或另一个 Conda 环境。 |
| PPO 接口或 `logstd` 不匹配 | 使用本仓库 vendored 分支；不要替换为 pip 最新 RSL-RL。 |
| `Selected GPU has active processes` | 程序拒绝与已有进程抢占；选一块空闲 GPU，勿停止他人的进程。 |
| CUDA OOM | 先退出失败进程检查 `nvidia-smi`，确保 GPU 独占且显存足够；先完成 smoke。正式 4096 规模不要在同一显存不足配置上反复重试。 |
| Vulkan／显示错误 | 原生 Linux 检查 NVIDIA 驱动和 Vulkan；纯 SSH 保持 headless，GUI 需要可用显示服务。 |
| 没有 `checkpoint_final.pt` | 检查 `receipt.json` 和 `supervisor_receipt.json`；进程暂停／崩溃与“完成保存”不同。 |
| 本轮未晋级 | 课程晋级需要连续两次两侧各 ≥244/256；训练时的高稳定比例不能替代独立评估。 |

## 已验证范围

历史 Guard17 的实际 GPU 训练及短评估已完成审查。**本仓库新增了便携入口和目录整理，发布前验证包括资产／检查点哈希、数值实现来源、CLI 和 CPU 策略／PPO 集成；没有为发布而另开 GPU 训练，也未在一台全新 Ubuntu 主机重装全部环境。** 按上述 check → 官方空场景 → smoke 的顺序验证你的机器，再进入评估或训练。

最终资格保持：课程 4、5 秒内稳定并持续至 120 秒、两侧各 ≥244/256，所有原阈值含 `abs(yaw_rate) ≤ 0.3`。当前 checkpoint 尚未达到这一资格；本项目不包含自动烧录、串口控制或实机部署流程。
