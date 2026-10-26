"""Continuous balance objective and measured-success curriculum."""
import math
import torch

def hardware_flags(speeds, travel, age_ms):
    rotor_rpm = speeds[:, 4:].abs().amax(-1) * 19.0 * 60.0 / (2.0 * math.pi)
    return torch.stack((speeds[:, :4].abs().amax(-1) > 6.0, rotor_rpm > 3000.0, travel.abs().amax(-1) > 1.48, age_ms > 40), -1)

def balance_reward(m, q, speed, applied, previous, weight, violations):
    tilt = m['tilt_deg'] * math.pi / 180.0
    wheel_support = (m['wheel_mean'].sum(-1) / weight).clamp(0.0, 1.0)
    both_wheels = (m['wheel_mean'].amin(-1) / 5.0).clamp(0.0, 1.0)
    clearance = (m['base_clearance'] / 0.015).clamp(0.0, 1.0)
    height = torch.exp(-((m['root_height'] - 0.124) / 0.04).square())
    reward = 3.0 * torch.exp(-tilt.square() / 0.02)
    reward += 2.0 * torch.exp(-m['gyro'].square() / 0.09)
    reward += 2.0 * torch.exp(-m['speed'].square() / 0.0225)
    reward += clearance + height + wheel_support + both_wheels
    reward -= 3.0 * m['speed'].square().clamp(max=4.0)
    reward -= 0.5 * m['gyro'].square().clamp(max=25.0)
    reward -= 2.0 * (m['nonwheel_mean'] / weight).clamp(0.0, 2.0)
    reward -= 2.0 * q.square().sum(-1).clamp(max=8.0)
    reward -= 0.05 * (applied - previous).square().sum(-1)
    reward -= 0.02 * applied.square().sum(-1)
    reward -= 0.02 * speed[:, 4:].square().sum(-1).clamp(max=500.0)
    reward -= 2.0 * m['drift'].square().clamp(max=4.0)
    reward -= 0.3 * violations.float().sum(-1)
    return 0.02 * reward

def stable_criteria(m, weight, stopped, instantaneous=False):
    wheels = m['wheel_fz'] if instantaneous else m['wheel_mean']
    other = m['nonwheel_fz'] if instantaneous else m['nonwheel_mean']
    return dict(tilt=m['tilt_deg'] <= 8.0, speed=m['speed'] <= 0.15, gyro=m['gyro'] <= 0.3, height=m['root_height'] >= 0.105, clearance=m['base_clearance'] >= 0.015, each_wheel=wheels.amin(-1) >= 2.0, support=wheels.sum(-1) >= 0.8 * weight, nonwheel=other <= 1.0, hardware=~stopped)

def sustained_pass(hold, elapsed_steps, stopped, dt=0.02):
    return (hold >= round(30.0 / dt)) & ((elapsed_steps - hold) * dt <= 5.0) & ~stopped

class SuccessCurriculum:

    def __init__(self):
        self.level = 0
        self.consecutive_passes = 0
        self.history = []

    def observe(self, passed, cases, protected_passed):
        good = passed / cases >= 0.95 and protected_passed / cases >= 0.95
        self.consecutive_passes = self.consecutive_passes + 1 if good else 0
        self.history.append(dict(level=self.level, passed=passed, cases=cases, protected_passed=protected_passed, passed_gate=good))
        promoted = self.consecutive_passes >= 2 and self.level < 4
        if promoted:
            self.level += 1
            self.consecutive_passes = 0
        return promoted
