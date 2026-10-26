"""Additional dense objectives for command accuracy and measured jump clearance."""
import torch

def tracking_bonus(linear, angular, commands):
    vx_error = linear[:, 0] - commands[:, 0]
    yaw_error = angular[:, 2] - commands[:, 1]
    stopped = commands[:, :2].abs().amax(dim=-1) < 0.025
    return 1.5 * torch.exp(-vx_error.square() / 0.01) + 1.5 * torch.exp(-yaw_error.square() / 0.06) - 2.0 * vx_error.square().clamp(max=9.0) - yaw_error.square().clamp(max=16.0) - stopped * (45.0 * linear[:, 0].square().clamp(max=4.0) + 2.0 * angular[:, 2].square().clamp(max=4.0))

def jump_bonus(linear, upright, height, elapsed, airborne, clearance, contact):
    target = height.clamp(min=0.005)
    takeoff = (elapsed >= 0.06) & (elapsed < 0.65)
    landing = (elapsed >= 0.65) & (elapsed < 1.5)
    progress = (clearance / target).clamp(0.0, 1.0)
    height_match = torch.exp(-((clearance - target) / 0.01).square())
    launch_speed = torch.sqrt(2.0 * 9.8 * target)
    upward_match = torch.exp(-((linear[:, 2] - launch_speed) / 0.4).square())
    return (6.0 * progress + 8.0 * height_match) * airborne * takeoff + 2.0 * upward_match * (linear[:, 2] > 0) * takeoff + 4.0 * torch.exp(-linear[:, 0].square() / 0.01) * contact.all(dim=-1) * (upright > 0.97) * landing
