"""Checkpoint identity and optimizer/RNG continuation, without simulator state."""
import hashlib
import random
import numpy as np
import torch

def model_identity(actor):
    digest = hashlib.sha256()
    for (name, tensor) in sorted(actor.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('ascii'))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def capture_training_state(algorithm, curriculum, device):
    return dict(optimizer_state_dict=algorithm.optimizer.state_dict(), curriculum=dict(level=curriculum.level, consecutive_passes=curriculum.consecutive_passes, history=curriculum.history), rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(), torch_device=torch.cuda.get_rng_state(device) if str(device) != 'cpu' else None), simulator_state_saved=False, resume_semantics='optimizer/curriculum/RNG restored; begin a fresh physical rollout')

def restore_training_state(state, algorithm, curriculum, device):
    algorithm.optimizer.load_state_dict(state['optimizer_state_dict'])
    algorithm.learning_rate = algorithm.optimizer.param_groups[0]['lr']
    course = state['curriculum']
    curriculum.level = int(course['level'])
    curriculum.consecutive_passes = int(course['consecutive_passes'])
    curriculum.history = list(course['history'])
    rng = state['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'].cpu())
    if rng['torch_device'] is not None:
        torch.cuda.set_rng_state(rng['torch_device'].cpu(), device)
