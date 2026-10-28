# Dogleg 轮腿机器人强化学习

**运行平台：原生 Linux，推荐 Ubuntu 22.04 + NVIDIA GPU。** 以下命令均在这台 Linux 机器的终端执行。Windows 本机可阅读源码、运行 CPU 检查；本仓库的 GPU 监督程序使用 Linux `/proc` 和信号，不能直接在 Windows PowerShell 中运行仿真。不要把 Windows 路径直接粘贴到 Linux。

**RTX 5080／RTX 50 系列请先阅读 [RTX 5080 安装说明](#rtx-5080)。** 默认安装脚本复现旧训练平台的 CUDA 11.8 环境；5080 使用 Blackwell 架构，需要单独准备较新的基础环境。本项目到该新环境的迁移尚未完成验证。

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

运行资产、源URDF、完整网格和一个可复现的完整基线检查点都已随仓库提供，无需再下载旧工程、Navila、机器人其它型号或历史训练输出。源模型入口见 [机器人模型文件](robots/dogleg/README.md)。`checkpoints/guard17.pt` 是保留的已审基线，不代表租用服务器此刻的最新策略；服务器后续训练的大检查点与全量输入记录没有随源码发布。

仓库同时提供两层机器人描述：

| 文件／目录 | 用途 |
| --- | --- |
| [标准源URDF](robots/dogleg/urdf/dogleg_demo_finished_motor_belt_tree.urdf) | 修正同步带关系后的树结构，19个link、18个joint |
| [闭链扩展URDF](robots/dogleg/urdf/dogleg_demo_finished_motor_belt_closed_isaac.urdf) | 包含4个非标准`loop_joint`扩展，需导入器明确支持 |
| [STL网格](robots/dogleg/meshes/) | URDF引用的完整19个网格，保持相对路径 |
| [机构配置](robots/dogleg/mechanism/motor_config.json) | 同步带比例、方向和闭链连接点 |
| [5.5kg来源场景](robots/dogleg/mujoco/dogleg_ground.xml) | 已采用物理模型的来源XML，附齐其19个网格依赖 |
| [训练运行USD](src/dogleg_rl/assets/dogleg_lab.usda) | Isaac Sim实际使用的模型，含内嵌网格 |
| [物理参数](src/dogleg_rl/assets/physical_payload.json)／[复位姿态](src/dogleg_rl/assets/reset_atlas.json) | 与运行USD配套的固定参数和合法复位数据 |

源URDF总质量约4.808793095kg；训练USD采用后续构建的5.5kg模型。训练入口直接加载固定USD，不会重新导入URDF。源URDF的历史电机测试限值也不能替代正式训练的控制限值。完整来源、差异和查看方法见模型目录说明。

## 3. 从零安装环境

原训练平台固定的兼容组合如下。**复现该平台时不要直接升级到 Isaac Sim 5.x、Isaac Lab 2.1+ 或最新版 RSL-RL。** RTX 5080 的独立基础环境说明在本节后半部分。

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

### RTX 5080

RTX 5080 是 **Blackwell、计算能力 12.0（`sm_120`）**。PyTorch 2.7 官方开始提供 Blackwell 支持及 CUDA 12.8 wheel；因此本仓库默认的 `torch 2.5.1+cu118` 安装路径不适合作为 5080 的直接安装方案。只更新显卡驱动或另外安装 CUDA Toolkit，也不会把现有 torch wheel 变成支持新架构的构建。

以下是一组来自 **Isaac Lab v2.2.0 官方安装文档**的基础环境组合，用于准备并验证 5080 上的 PyTorch／Isaac 环境：

| 组件 | 5080 独立基础环境 |
| --- | --- |
| 操作系统 | 原生 Ubuntu 22.04／24.04，Linux x86-64 |
| NVIDIA 驱动 | 支持 RTX 50 系列、满足所选 Isaac Sim 版本要求的驱动；按 Ubuntu 推荐及 NVIDIA 兼容表选择 |
| Python | 3.11 |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| Isaac Sim / Isaac Lab | 5.0.0 / v2.2.0 |

**这些命令安装的是新显卡的基础环境，当前仓库还没有把该组合声明为可直接运行 dogleg 训练的受支持配置。** `pyproject.toml`、`constraints.txt` 和 `dogleg-rl check` 目前固定 Python3.10／torch2.5.1／IsaacSim4.5。完整运行项目还需要适配依赖声明与版本检查、确认 Isaac Lab API、完成本项目 smoke 和物理对照；不能把“CUDA 张量能算”当作迁移通过。

先按第1节安装系统依赖并确认 `nvidia-smi` 能识别 RTX 5080。`nvidia-smi` 显示的 `CUDA Version` 是驱动可支持的 CUDA 上限，PyTorch 实际使用的版本查看 `torch.version.cuda`。

在已克隆的仓库根目录执行，单独使用 `dogleg5080` 环境，保留原 `.venv`。若机器尚无 Conda，可先安装 Miniforge；已有 Conda／Miniforge 的机器跳过下面这段：

```bash
sudo apt install -y curl
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

创建新环境，安装支持 Blackwell 的 PyTorch 和 Isaac：

```bash
conda create -n dogleg5080 python=3.11 -y
conda activate dogleg5080
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.0.0' \
  --extra-index-url https://pypi.nvidia.com

mkdir -p _deps
git clone --depth 1 --branch v2.2.0 https://github.com/isaac-sim/IsaacLab.git _deps/IsaacLab5080
python -m pip install -e _deps/IsaacLab5080/source/isaaclab
python -m pip check
```

此流程使用新的 `IsaacLab5080` 目录；不调用旧环境安装脚本，也不传入旧 `constraints.txt`。当前阶段到官方基础环境验证为止，不执行本项目的 `pip install -e .` 或 `dogleg-rl train`。

验证 PyTorch 确实能在5080执行 GPU 运算。单显卡使用 `cuda:0`；多显卡时选择对应5080的编号：

```bash
python - <<'PY'
import torch
device = torch.device('cuda:0')
assert torch.cuda.is_available(), 'CUDA unavailable: check driver and torch wheel'
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(device))
print('capability:', torch.cuda.get_device_capability(device))
x = torch.randn(1024, 1024, device=device)
y = x @ x
torch.cuda.synchronize(device)
assert torch.isfinite(y).all().item()
print('CUDA matrix multiplication: PASS')
PY
```

如果选择的确实是5080，计算能力应为 `(12, 0)`。然后验证官方空场景；首次启动时自行阅读并确认 NVIDIA EULA：

```bash
python _deps/IsaacLab5080/scripts/tutorials/00_sim/create_empty.py --headless --device cuda:0
```

确认场景正常启动后按 `Ctrl+C` 退出。上述两项只确认基础软件环境，未验证本项目的控制、闭链、奖励或训练结果。5080的16GiB显存也不保证新版本下4096环境一定能运行，仍需实际测量。

参考：[NVIDIA GPU计算能力表](https://developer.nvidia.com/cuda-gpus)、[PyTorch2.7的Blackwell支持](https://pytorch.org/blog/pytorch-2-7/)、[Isaac Lab v2.2.0官方安装流程](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/pip_installation.html)。该节提供版本化的基础环境准备路径，尚未在5080硬件上由本项目实测。

## 4. 先检查，再启动

以下项目命令对应已验证来源的原训练环境，不接续上面的5080基础环境准备流程。

在仓库根目录、已激活 `.venv` 的终端执行：

```bash
dogleg-rl check
python scripts/check_robot_assets.py
python -m unittest discover -s tests -v
```

`check` 核对固定版本、三个资产 SHA256 和 CPU 策略前向计算，不启动仿真或 CUDA 上下文。输出中应有 `assets_verified: true`、`policy_CPU_forward: true`。

`check_robot_assets.py`只用Python标准库检查全部已登记模型文件的SHA、两个URDF的树结构和惯量、19个STL的完整性、来源XML的全部网格路径，以及XML到训练物理参数的来源哈希；不启动Isaac或GPU。安装Python后即可单独运行。

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

**这条命令会启动 Isaac Sim，但默认使用无窗口模式（headless），不会显示训练画面。** Isaac Sim 负责物理仿真，PPO同时收集数据并更新策略。

并行环境数可以通过 `--num-envs` 调整。默认训练使用4096环境；例如，在显存和主机内存充足的机器上使用8192环境：

```bash
dogleg-rl train --gpu 0 --checkpoint checkpoints/guard17.pt \
  --num-envs 8192 --updates 100 --minutes 60
```

先测量8192的实际吞吐量，再根据资源余量尝试 `--num-envs 16384` 或 `--num-envs 32768`；当前支持的上限为32768。`metrics.jsonl` 会记录 `decisions_per_second`、采样与优化耗时，以及 PyTorch 分配／保留显存的峰值；完整GPU占用和主机内存余量记录在 `thermal.jsonl`。PyTorch峰值不包含全部PhysX和图形运行库的占用，应结合GPU总占用判断容量。

每次更新收集 `num_envs × 64` 条决策：8192环境对应524,288条，16384环境对应1,048,576条，32768环境对应2,097,152条。PPO仍执行4轮、每轮4个minibatch，即每次更新16个Adam步骤，因此增大环境数也会改变训练的batch大小。环境数在场景创建时固定，修改正在运行的训练需要先完整保存，再从该检查点启动新的运行目录。

2026-10-27在单张RTX4090 24GB、16核CPU配额、120GiB主机内存配额的租用服务器上，32768环境已实际训练验证：约30,000条决策/秒、GPU利用率约91%、总显存约17.6GiB，采样最低可用主机内存约35.4GiB。原4096环境约16,653条决策/秒。以上是该机器的运行观测；环境数需要结合GPU和主机内存实测选择，更高数据吞吐不直接代表更快通过站立资格。

扩容版本会在GPU可用显存低于2GiB，或可读取的Linux cgroup主机内存余量低于8GiB时请求完成当前更新并保存停止。仍应预留优化和评估时的峰值空间，按实际每秒采样量选择环境数。

如果明确要从随机策略开始，而非继续当前模型：

```bash
dogleg-rl train --gpu 0 --fresh --updates 100 --minutes 60
```

### 可视化训练

在训练命令末尾增加 `--gui`，会打开 Isaac Sim 窗口，显示训练中的仿真场景。先用32个环境确认窗口和渲染正常；这个 smoke 测试不进行优化：

```bash
dogleg-rl smoke --gpu 0 --gui
```

看到 `SMOKE_COMPLETE` 并正常退出后，从当前检查点开始可视化训练：

```bash
dogleg-rl train \
  --gpu 0 \
  --checkpoint checkpoints/guard17.pt \
  --updates 100 \
  --minutes 60 \
  --gui
```

明确从随机策略开始时，同样可以使用可视化：

```bash
dogleg-rl train --gpu 0 --fresh --updates 100 --minutes 60 --gui
```

运行条件与资源要求：

- 需要可用的 Linux 桌面或远程桌面会话。纯 SSH 终端不会自动把 Isaac Sim 画面传到本机；没有显示服务时使用默认 headless。
- 单显卡通常用 `--gpu 0`；原双显卡服务器使用 GPU1 时把命令改为 `--gpu 1`。所选GPU仍需通过空闲检查，已有桌面或其它进程占用同一GPU时，监督程序也会拒绝启动；`--gui` 不放宽这个检查。
- 正式训练仍为4096并行环境，开启窗口会增加渲染负担和显存占用。先完成32环境的GUI smoke，再确认正式规模的资源是否足够；GUI不会自动降低环境数量，也不会自动录制视频。
- RTX 5080 仍需先处理前面说明的版本迁移与运行验证。增加 `--gui` 只控制窗口显示，不解决 Python／PyTorch／Isaac 的版本兼容问题。

### 训练预算、输出与停止

默认4096并行环境、每环境rollout64个决策、PPO4 epochs×4 minibatches，每更新262144个决策和16次Adam step。使用`--num-envs`时，决策数随环境数变化，Adam仍为每更新16步。默认在本段local100执行唯一计划内课程评估；可通过`--evaluation-update`指定该段唯一评估所在的更新，不因早停补评。每段完成后停止，不自动接下一段。

例如，复用租用服务器已使用的32768环境规模，在最多50次更新／60分钟内，于第3次更新后执行一次课程评估：

```bash
dogleg-rl train --gpu 0 --checkpoint checkpoints/guard17.pt \
  --num-envs 32768 --updates 50 --minutes 60 --evaluation-update 3
```

这条命令从仓库内的基线开始。若需接续你自己的训练，将`--checkpoint`换成上一次实际完整保存的检查点；它会恢复模型、Adam、RNG和课程，而不是把基线文件视为服务器最新模型。运行前须满足GPU空闲、内存和记录磁盘余量。

输出包括 `checkpoint_initial.pt`、滚动覆盖的 `checkpoint_latest.pt`、完整 `checkpoint_final.pt`、`metrics.jsonl`、`receipt.json`、热监控和子进程退出凭证。保存完整模型、Adam、Python／NumPy／CPU／CUDA RNG 和课程状态，继续训练时重新进行物理复位，不恢复仿真轨迹／求解器 warmstart。

普通停止：在监督终端按 `Ctrl+C`，或在另一终端创建当前输出目录的停止文件：

```bash
touch runs/<当前运行目录>/STOP_AFTER_UPDATE
```

程序完成已经开始的 update／评估，再保存退出。热保护仍为 85°C 警告、86°C 持续 60 秒或 active thermal flag 请求完整保存停止、88°C／硬件 thermal flag 精确暂停该子进程。紧急暂停不代表完成保存，且不会自动恢复；查看 `supervisor_receipt.json`。外层 4500 秒请求保存、5100 秒 TERM、5400 秒 KILL，仅针对本次创建且身份匹配的子进程。

## 7. 当前机制与目录

训练和独立评估现支持逐次记录实际送入策略的29维float32输入，连同对应模型、动作、环境／回合编号、时间和成功标签保存到 `output/observation29/`。训练记录全部并行环境，评估记录评分的512个环境；不抽行。以后可提取已通过独立评估的完整输入轨迹，与同一合同、正确对齐的实车输入离线计算逐维误差。字段顺序、存储需求和提取／对比命令见 [29维输入记录与实车对比](docs/OBSERVATION29_RECORDING.md)。既往未记录的输入不能从模型恢复。

- `src/dogleg_rl/core/`：当前环境、奖励、控制和 PPO 的实际依赖闭包。一些模块仍有 `v6/v7/v8` 等名称，因为它们是现版本仍在继承的基础实现；旧实验入口和淘汰运行器未收录。
- `src/dogleg_rl/assets/`：实际训练的 5.5 kg USD、完整物理参数、合法复位姿态表；已固定 SHA256。
- `robots/dogleg/`：完整修正源URDF、STL、机构配置、5.5kg来源XML及其网格和来源记录。
- `src/dogleg_rl/_vendor/`：必要的算法源码及其原始版权声明；不安装另一份冲突的 `rsl_rl`。
- `checkpoints/guard17.pt`：已审完整基线检查点；服务器后续策略另行保存。
- `results/guard17_summary.json`：已完成的历史短评估结果。
- `docs/`：完整安装命令、奖励／资格说明和来源清单。
- `docs/robot_assets_manifest.json`：全部模型来源文件与运行资产的路径、字节数和SHA256。

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
