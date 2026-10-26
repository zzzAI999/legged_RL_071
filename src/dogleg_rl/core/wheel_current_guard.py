"""Shared float32 nominal C620 command units, before unknown plant strength."""
import math
import torch
from .recovery_contract import WHEEL_RATIO

def guarded_wheel_current(requested_torque, measured_model_speed, enabled=None):
    rpm = measured_model_speed.abs() * WHEEL_RATIO * 60 / (2 * math.pi)
    scale = ((2500.0 - rpm) / 500.0).clamp(0.0, 1.0)
    nominal = torch.where(requested_torque * measured_model_speed > 0.0, requested_torque * scale, requested_torque)
    if enabled is not None:
        nominal = torch.where(enabled[:, None], nominal, requested_torque)
    counts_float = -nominal / 0.3 * 819.2
    counts = (torch.sign(counts_float) * torch.floor(counts_float.abs() + 0.5)).clamp(-2184.0, 2184.0)
    nominal = -counts / 819.2 * 0.3
    return (counts, nominal)
