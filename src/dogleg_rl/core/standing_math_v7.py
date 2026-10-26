"""Zero-command standing must track yaw as well as translation and tilt."""
import torch
from .standing_math import SuccessCurriculum, hardware_flags, sustained_pass
from .standing_math_v6 import balance_terms as terms_v6
from .standing_math import stable_criteria as criteria_v5
REWARD_VERSION = 'standing_v7_zero_command_yaw'

def balance_terms(m, q, speed, applied, previous, weight, violations):
    terms = terms_v6(m, q, speed, applied, previous, weight, violations)
    tilt = torch.deg2rad(m['tilt_deg'])
    support = (m['wheel_mean'].sum(-1) / weight).clamp(0.0, 1.0)
    both = (m['wheel_mean'].amin(-1) / 5.0).clamp(0.0, 1.0)
    gate = torch.exp(-tilt.square() / 0.02) * support * both
    terms['quiet_yaw'] = 0.02 * 2.0 * gate * torch.exp(-m['yaw_rate'].square() / 0.09)
    terms['yaw_rate'] = -0.02 * m['yaw_rate'].square().clamp(max=25.0)
    return terms

def balance_reward(m, q, speed, applied, previous, weight, violations):
    return sum(balance_terms(m, q, speed, applied, previous, weight, violations).values())

def stable_criteria(m, weight, stopped, instantaneous=False):
    tests = criteria_v5(m, weight, stopped, instantaneous=instantaneous)
    tests['yaw'] = m['yaw_rate'].abs() <= 0.3
    return tests
