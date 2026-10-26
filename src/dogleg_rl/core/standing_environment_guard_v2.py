"""Finalize endpoint protection before DirectRLEnv builds its reset mask."""
import torch
from .standing_environment_guard import StandingCfg, StandingEnv as GuardEnvV1
from .standing_math_guard import hardware_flags
TRAINING_GUARD_VERSION = 'latched_1ms_endpoint_before_done_v2'

class StandingEnv(GuardEnvV1):

    def _get_dones(self):
        speeds = self.robot.data.joint_vel[:, self.motor_ids]
        travel = self.robot.data.joint_pos[:, self.motor_ids[:4]] - self.begin_q
        flags = hardware_flags(speeds, travel, self.command_age)
        self.interval_flags |= flags
        first = flags.any(-1) & ~self.would_stop
        self.first_violation_ms[first] = self.episode_length_buf[first] * 20
        self.violation_flags[first] = flags[first]
        self.would_stop |= flags.any(-1)
        self.speed_peak = torch.maximum(self.speed_peak, speeds.abs())
        self.safety_latched |= self.interval_flags.any(-1) & self.hardware_gate_mask
        self.held_torque[self.safety_latched] = 0.0
        self.applied[self.safety_latched] = 0.0
        self.actions.copy_(self.applied)
        self.robot.set_joint_effort_target(self.held_torque)
        return super()._get_dones()
