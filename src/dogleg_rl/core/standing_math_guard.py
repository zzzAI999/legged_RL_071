"""Keep safe v8 shaping; an unsafe decision earns no positive bonuses."""
import torch
from .standing_math_v8 import balance_terms as terms_v8
from .standing_math_v8 import SuccessCurriculum, hardware_flags, stable_criteria, sustained_pass
REWARD_VERSION = 'standing_v8_guard_aligned_v1'
GUARD_PENALTY = 5.0

def balance_terms(m, q, speed, applied, previous, weight, violations, stopped, terminated):
    terms = terms_v8(m, q, speed, applied, previous, weight, violations)
    unsafe = violations.any(-1) | stopped
    terms.pop('violations')
    terms = {key: torch.where(unsafe, value.clamp(max=0.0), value) for (key, value) in terms.items()}
    terms['guard_terminal'] = -GUARD_PENALTY * unsafe.float()
    terms['fall_terminal'] = -5.0 * (terminated & ~unsafe).float()
    return terms

def record_interval_flags(interval, speeds, travel, command_age):
    interval |= hardware_flags(speeds, travel, command_age)
