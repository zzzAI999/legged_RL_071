#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Use native Ubuntu 22.04 for the documented GPU workflow.' >&2
  exit 1
fi
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'numpy==1.26.4'
python -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com -c constraints.txt
mkdir -p _deps
if [[ ! -d _deps/IsaacLab ]]; then
  git clone --depth 1 --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git _deps/IsaacLab
fi
if [[ "$(tr -d '\r\n' < _deps/IsaacLab/VERSION)" != 2.0.2 ]]; then
  echo '_deps/IsaacLab must be version 2.0.2; preserve and inspect the existing directory.' >&2
  exit 1
fi
python -m pip install -e _deps/IsaacLab/source/isaaclab -c constraints.txt
python -m pip install -e . -c constraints.txt
python -m pip check
echo 'Installation finished. Activate .venv, review the NVIDIA EULA on first launch, then follow README checks.'
