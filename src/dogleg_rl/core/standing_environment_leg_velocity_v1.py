"""Integrate actual leg speeds without changing the v9 actuator/guard path."""
import torch
from .standing_environment_margin_v1 import StandingCfg, StandingEnv as MarginEnv
from .standing_leg_velocity_reward_v1 import DECISION_SAMPLES, PHYSICS_DT, leg_velocity_cost

class StandingEnv(MarginEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        assert cfg.decimation == DECISION_SAMPLES and abs(self.physics_dt - PHYSICS_DT) < 1e-12
        self.decision_leg_velocity_history = torch.zeros((DECISION_SAMPLES, self.num_envs, 4), device=self.device)
        self.decision_leg_velocity_square_integral = torch.zeros((self.num_envs, 4), device=self.device)
        self.decision_leg_velocity_samples = 0
        self.decision_leg_velocity_start_tick = self.control_tick
        self.decision_leg_velocity_last_tick = self.control_tick

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if hasattr(self, 'decision_leg_velocity_history'):
            ids = self.robot._ALL_INDICES if ids is None else ids
            self.decision_leg_velocity_history[:, ids] = 0.0
            self.decision_leg_velocity_square_integral[ids] = 0.0

    def _pre_physics_step(self, actions):
        self.decision_leg_velocity_history.zero_()
        self.decision_leg_velocity_square_integral.zero_()
        self.decision_leg_velocity_samples = 0
        self.decision_leg_velocity_start_tick = self.control_tick
        self.decision_leg_velocity_last_tick = self.control_tick
        super()._pre_physics_step(actions)

    def _record_flags(self):
        super()._record_flags()
        if not hasattr(self, 'decision_leg_velocity_last_tick'):
            return
        tick = self.control_tick
        if tick == self.decision_leg_velocity_last_tick:
            return
        assert tick == self.decision_leg_velocity_last_tick + 1
        elapsed = tick - self.decision_leg_velocity_start_tick
        assert 1 <= elapsed <= DECISION_SAMPLES
        speed = self.robot.data.joint_vel[:, self.motor_ids[:4]]
        self.decision_leg_velocity_history[elapsed - 1].copy_(speed)
        self.decision_leg_velocity_square_integral.add_(speed.square(), alpha=PHYSICS_DT)
        self.decision_leg_velocity_samples += 1
        self.decision_leg_velocity_last_tick = tick

    def _get_dones(self):
        self._record_flags()
        return super()._get_dones()

    def _get_rewards(self):
        reward = super()._get_rewards()
        assert self.decision_leg_velocity_samples == DECISION_SAMPLES
        cost = leg_velocity_cost(self.decision_leg_velocity_square_integral)
        self.reward_terms['leg_velocity_l2_1ms'] = cost
        self.episode_returns += cost
        self.guard_step_stats['leg_velocity_integral_sum'] = self.decision_leg_velocity_square_integral.sum()
        self.guard_step_stats['leg_velocity_nonzero_decisions'] = (cost < 0).sum()
        return reward + cost
