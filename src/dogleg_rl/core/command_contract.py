"""Shared observation layout and six-motor target conversion, without simulator dependencies."""
import torch
OBSERVATIONS = 29
ACTIONS = 6

def observations(gyro, gravity, commands, leg_deviations, motor_speeds, previous_actions, jump_height, jump_elapsed, jump_active):
    device = gyro.device
    leg_sign = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device)
    speed_scale = torch.tensor([0.05] * 4 + [0.03] * 2, device=device)
    command_scale = torch.tensor([1.0, 0.2, 1.0], device=device)
    active = jump_active.to(dtype=gyro.dtype)
    phase = jump_elapsed * 4.1887902047863905
    extra = torch.stack((jump_height / 0.1, torch.sin(phase) * active, torch.cos(phase) * active, active), dim=-1)
    return torch.cat((gyro * 0.2, gravity, commands * command_scale, leg_deviations * leg_sign / 0.5, motor_speeds * speed_scale, previous_actions, extra), dim=-1)

def motor_targets(actions, leg_scale, wheel_sign=(1.0, -1.0)):
    bounded = actions.clamp(-1.0, 1.0)
    leg_sign = torch.tensor([1.0, 1.0, -1.0, -1.0], device=bounded.device)
    wheel_sign = torch.tensor(wheel_sign, device=bounded.device)
    return (bounded[:, :4] * leg_sign * leg_scale, bounded[:, 4:] * wheel_sign * 0.8)
