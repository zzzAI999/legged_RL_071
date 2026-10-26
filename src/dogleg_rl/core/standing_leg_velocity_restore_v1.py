"""Prove complete restoration before the changed-reward physical reset."""
import numpy as np
import torch
from .standing_training_state import capture_training_state, restore_training_state

def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all((equal_tree(left[k], right[k]) for k in left))
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all((equal_tree(a, b) for (a, b) in zip(left, right)))
    return left == right

def restore_velocity_state(checkpoint, algorithm, curriculum, device, actor):
    restore_training_state(checkpoint['training_state'], algorithm, curriculum, device)
    restored = capture_training_state(algorithm, curriculum, device)
    assert equal_tree(checkpoint['model_state_dict'], actor.state_dict()), 'Model restoration mismatch'
    assert equal_tree(checkpoint['training_state'], restored), 'Complete Adam/course/RNG restoration mismatch'
    steps = sorted({int(v['step']) for v in restored['optimizer_state_dict']['state'].values()})
    return dict(version='v10_complete_state_before_physical_reset_v1', source_update=checkpoint['iteration'], source_transitions=checkpoint['cumulative_transitions'], model_actor_critic_exploration_exact=True, complete_Adam_exact=True, curriculum_exact_before_reward_gate_reset=True, python_numpy_cpu_cuda_RNG_exact=True, adam_steps=steps, course_level=curriculum.level, history_length=len(curriculum.history), simulator_state_restored=False, next_action='explicit reward gate reset then fresh physical reset')
