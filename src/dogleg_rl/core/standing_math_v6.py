"""Upright-gated balance shaping; qualification remains identical to v5."""
import math
import torch
from .standing_math import SuccessCurriculum, hardware_flags, stable_criteria, sustained_pass
REWARD_VERSION = 'standing_v6_upright_gate_wheel2e5'

def balance_terms(m, q, speed, applied, previous, weight, violations):
    tilt = m['tilt_deg'] * math.pi / 180.0
    upright = torch.exp(-tilt.square() / 0.02)
    support = (m['wheel_mean'].sum(-1) / weight).clamp(0.0, 1.0)
    both = (m['wheel_mean'].amin(-1) / 5.0).clamp(0.0, 1.0)
    quiet_gate = upright * support * both
    terms = dict(upright=6.0 * upright, quiet_gyro=2.0 * quiet_gate * torch.exp(-m['gyro'].square() / 0.09), quiet_speed=2.0 * quiet_gate * torch.exp(-m['speed'].square() / 0.0225), clearance=(m['base_clearance'] / 0.015).clamp(0.0, 1.0), height=torch.exp(-((m['root_height'] - 0.124) / 0.04).square()), wheel_support=support, both_wheels=both, base_speed=-3.0 * m['speed'].square().clamp(max=4.0), base_gyro=-0.5 * m['gyro'].square().clamp(max=25.0), nonwheel=-2.0 * (m['nonwheel_mean'] / weight).clamp(0.0, 2.0), leg_pose=-2.0 * q.square().sum(-1).clamp(max=8.0), action_rate=-0.05 * (applied - previous).square().sum(-1), action_effort=-0.02 * applied.square().sum(-1), wheel_speed=-2e-05 * speed[:, 4:].square().sum(-1).clamp(max=500.0), drift=-2.0 * m['drift'].square().clamp(max=4.0), violations=-0.3 * violations.float().sum(-1))
    return {key: 0.02 * value for (key, value) in terms.items()}

def balance_reward(m, q, speed, applied, previous, weight, violations):
    return sum(balance_terms(m, q, speed, applied, previous, weight, violations).values())
