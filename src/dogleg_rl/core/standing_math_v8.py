"""Standing bonuses require usable height, clearance and wheel support."""
import torch
from .standing_math_v7 import SuccessCurriculum, hardware_flags, stable_criteria, sustained_pass
from .standing_math_v7 import balance_terms as terms_v7
REWARD_VERSION = 'standing_v8_pose_quality_gate'

def balance_terms(m, q, speed, applied, previous, weight, violations):
    terms = terms_v7(m, q, speed, applied, previous, weight, violations)
    upright = torch.exp(-torch.deg2rad(m['tilt_deg']).square() / 0.02)
    support = (m['wheel_mean'].sum(-1) / weight).clamp(0.0, 1.0)
    both = (m['wheel_mean'].amin(-1) / 5.0).clamp(0.0, 1.0)
    height_gate = ((m['root_height'] - 0.095) / 0.01).clamp(0.0, 1.0)
    clearance_gate = (m['base_clearance'] / 0.015).clamp(0.0, 1.0)
    gate = upright * support * both * height_gate * clearance_gate
    terms['quiet_gyro'] = 0.02 * 2.0 * gate * torch.exp(-m['gyro'].square() / 0.09)
    terms['quiet_speed'] = 0.02 * 2.0 * gate * torch.exp(-m['speed'].square() / 0.0225)
    terms['quiet_yaw'] = 0.02 * 2.0 * gate * torch.exp(-m['yaw_rate'].square() / 0.09)
    terms['height'] = 0.02 * 3.0 * torch.exp(-((m['root_height'] - 0.124) / 0.015).square())
    deficit = ((0.105 - m['root_height']) / 0.02).clamp(0.0, 2.0)
    terms['low_height'] = -0.02 * 6.0 * deficit.square()
    return terms

def balance_reward(m, q, speed, applied, previous, weight, violations):
    return sum(balance_terms(m, q, speed, applied, previous, weight, violations).values())
