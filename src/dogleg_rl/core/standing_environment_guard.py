"""Train with the existing protected actuator path and preserve substep events."""
import torch
from .standing_environment_v8 import StandingCfg, StandingEnv as StandingEnvV8
from .standing_math_guard import balance_terms, record_interval_flags, sustained_pass

class StandingEnv(StandingEnvV8):

    def __init__(self, cfg, **kwargs):
        cfg.hardware_gates = True
        super().__init__(cfg, **kwargs)
        self.interval_flags = torch.zeros_like(self.violation_flags)
        self.guard_step_stats = {}

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if ids is None:
            ids = self.robot._ALL_INDICES
        self.interval_flags[ids] = False

    def _pre_physics_step(self, actions):
        self.interval_flags.zero_()
        if self.training_mode and (not self.cfg.evaluation):
            assert self.hardware_gate_mask.all(), 'Training must use the protected execution path'
        super()._pre_physics_step(actions)

    def _record_flags(self):
        record_interval_flags(self.interval_flags, self.robot.data.joint_vel[:, self.motor_ids], self.robot.data.joint_pos[:, self.motor_ids[:4]] - self.begin_q, self.command_age)

    def _apply_action(self):
        self._record_flags()
        super()._apply_action()

    def _get_rewards(self):
        self._record_flags()
        m = self.measures()
        q = self.robot.data.joint_pos[:, self.motor_ids[:4]]
        speeds = self.robot.data.joint_vel[:, self.motor_ids]
        unsafe = self.interval_flags.any(-1) | self.safety_latched
        self.reward_terms = balance_terms(m, q, speeds, self.applied, self.previous_actions, self.mass_scale * 5.5 * 9.8, self.interval_flags, self.safety_latched, self.reset_terminated)
        reward = sum(self.reward_terms.values())
        self.episode_returns += reward
        ended = self.reset_terminated | self.reset_time_outs
        passed = sustained_pass(self.hold_steps, self.episode_length_buf, self.safety_latched)
        self.guard_step_stats = dict(guard_event_decisions=self.interval_flags.any(-1).sum(), guard_flag_events=self.interval_flags.sum(0), guard_terminated=(self.reset_terminated & self.safety_latched).sum(), completed_training_episodes=ended.sum(), completed_training_sustained_passes=(passed & self.reset_time_outs & ~self.reset_terminated).sum(), completed_training_sustained_failures=(ended & ~passed).sum(), unsafe_positive_reward_decisions=(unsafe & (reward > 0)).sum())
        if self.reset_time_outs.any():
            self.extras['terminal_critic'] = super()._get_observations()['critic'].detach().clone()
        else:
            self.extras.pop('terminal_critic', None)
        assert torch.isfinite(reward).all(), 'Nonfinite reward'
        return reward
