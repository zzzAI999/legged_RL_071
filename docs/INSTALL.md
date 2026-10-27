# 从零安装的完整命令

推荐原生 Ubuntu 22.04。先按根 README 检查驱动、安装系统依赖并克隆本仓库。下面从仓库根目录开始，逐条执行；不要激活原工程的环境。

**RTX 5080／RTX 50 系列请先看 [README 的专门安装说明](../README.md#rtx-5080)。** 本页命令复现 Python3.10／CUDA11.8 的原平台。5080需要新架构基础环境，且本项目在新版本上的完整迁移尚未验证。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 与原项目一致的 CUDA 11.8 PyTorch 构建。
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4

# Isaac Sim 含扩展缓存，下载量很大；不是项目本身的代码体积。
python -m pip install 'isaacsim[all,extscache]==4.5.0' \
  --extra-index-url https://pypi.nvidia.com -c constraints.txt

# 固定 Isaac Lab 源码版本。仅安装本任务需要的核心包；不装其它学习框架。
mkdir -p _deps
git clone --depth 1 --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git _deps/IsaacLab
cat _deps/IsaacLab/VERSION
python -m pip install -e _deps/IsaacLab/source/isaaclab -c constraints.txt

# 安装本项目；其内部使用保留的算法分支。
python -m pip install -e . -c constraints.txt
python -m pip check
dogleg-rl check
```

`cat .../VERSION` 应输出 `2.0.2`。首次启动 Isaac Sim 时请自行阅读并确认 NVIDIA EULA；可先运行：

```bash
python _deps/IsaacLab/scripts/tutorials/00_sim/create_empty.py --headless --device cuda:0
```

确认启动后 `Ctrl+C` 退出，再回根 README 执行项目 smoke、评估和训练。

重新打开终端只需要：

```bash
cd <你克隆仓库的位置>/dogleg-rl
source .venv/bin/activate
dogleg-rl check
```

不建议在这份环境中安装 wheeled-legged_RL V14_2 所需的 Isaac Lab2.1／RSL2.3，也不要全局升级 torch。两个项目的机器人与依赖基线不同。

## 官方资料与核对来源

- [Isaac Lab 2.0 官方 pip 安装流程](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/setup/installation/pip_installation.html)：Python3.10、torch2.5.1/cu118、IsaacSim4.5.0 的安装命令；该版本化页面在整理文档时已读取。
- [Isaac Lab v2.0.2 VERSION](https://github.com/isaac-sim/IsaacLab/blob/v2.0.2/VERSION)：确认实际固定标签为2.0.2。
- [Isaac Lab v2.0.2 setup.py](https://github.com/isaac-sim/IsaacLab/blob/v2.0.2/source/isaaclab/setup.py)：核心包要求 torch2.5.1、numpy<2，以及与IsaacSim4.5.0一致的依赖。
- [NVIDIA Isaac Sim 文档入口](https://docs.isaacsim.omniverse.nvidia.com/)：核对4.5版本的系统、驱动兼容及许可要求。

自动安装与上面的手动步骤内容相同，入口为 `scripts/install_ubuntu.sh`。下载软件需要网络；本仓库不包含 Isaac Sim 安装包或预装环境。
