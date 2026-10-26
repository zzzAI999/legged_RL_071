"""Supplementary execution model matching the reviewed C adapter's ACK units."""
import torch
from .recovery_environment import RecoveryEnv
from .recovery_contract import LEG_SCALE, WHEEL_SCALE

class DeploymentRecoveryEnv(RecoveryEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.ack_history = torch.zeros((51, self.num_envs, 6), device=self.device)
        self.initial_admission_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if ids is None:
            ids = self.robot._ALL_INDICES
        measured = self.reset_reference_q[ids] + self.q_bias[ids]
        admitted = measured.abs().amax(-1) <= LEG_SCALE
        self.slew_target[ids] = measured
        self.applied[ids, :4] = measured / (self.action_sign[:4] * LEG_SCALE)
        self.applied[ids[~admitted]] = 0.0
        self.initial_admission_failed[ids] = ~admitted
        self.safety_latched[ids] |= ~admitted
        self.actions[ids] = self.applied[ids]
        self.previous_actions[ids] = self.applied[ids]
        self.action_history[:, ids] = self.applied[ids]
        self.ack_history[:, ids] = self.applied[ids]

    def _apply_action(self):
        tick = self.control_tick
        super()._apply_action()
        if tick % 2 == 0:
            ids = torch.arange(self.num_envs, device=self.device)
            delayed = self.action_history[(self.history_cursor - self.latency_ms) % 51, ids]
            native_A = -self.action_sign[4:] * WHEEL_SCALE * delayed[:, 4:] / 0.3
            counts_float = native_A * 819.2
            counts = torch.sign(counts_float) * torch.floor(counts_float.abs() + 0.5)
            counts = counts.clamp(-2184.0, 2184.0)
            nominal = -(counts / 819.2) * 0.3
            physical = (nominal * self.strengths[:, 4:]).clamp(-WHEEL_SCALE, WHEEL_SCALE)
            speeds = self.robot.data.joint_vel[:, self.motor_ids[4:]]
            self.held_torque[:, self.motor_ids[4:]] = physical - speeds * self.wheel_drag
            self.held_torque[self.safety_latched] = 0.0
            self.applied[:, 4:] = nominal / (self.action_sign[4:] * WHEEL_SCALE)
            self.applied[self.safety_latched] = 0.0
            self.actions.copy_(self.applied)
            self.robot.set_joint_effort_target(self.held_torque)
        if tick % 10 == 0:
            self.ack_history[self.sensor_cursor] = self.applied

    def _get_observations(self):
        result = super()._get_observations()
        ids = torch.arange(self.num_envs, device=self.device)
        previous = self.ack_history[(self.sensor_cursor - (self.sensor_latency_ms + 9) // 10) % 51, ids]
        result['policy'][:, 19:25] = previous
        result['critic'][:, :29] = result['policy']
        return result
