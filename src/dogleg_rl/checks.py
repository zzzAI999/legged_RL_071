"""Verify the installation without starting a simulator or a CUDA context."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform


def check_installation():
    from .cli import asset_check
    required = {'torch': '2.5.1', 'numpy': '1.26.4', 'isaacsim': '4.5.0.0', 'isaaclab': '0.34.9'}
    versions = {}
    problems = []
    for package, expected in required.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
        versions[package] = actual
        if actual is None or actual.split('+')[0] != expected:
            problems.append(f'{package}: expected {expected}, found {actual}')
    if platform.python_version_tuple()[:2] != ('3', '10'):
        problems.append('Python 3.10 is required')
    asset_check()
    if problems:
        raise RuntimeError('\n'.join(problems) + '\nRun the installation steps in README.md')
    import torch
    from .core.policy_architecture import actor_critic
    actor = actor_critic(device='cpu', bounded=True, hidden_dims=[256, 256])
    with torch.no_grad():
        action = actor.act_inference(torch.zeros(2, 29))
    assert action.shape == (2, 6) and torch.isfinite(action).all()
    return dict(python=platform.python_version(), versions=versions, assets_verified=True,
                policy_CPU_forward=True, simulator_started=False, CUDA_initialized=torch.cuda.is_initialized())
