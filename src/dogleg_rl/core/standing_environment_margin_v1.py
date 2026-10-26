"""Track leg speed over each decision; retain the verified endpoint guard."""
import torch
from .standing_environment_guard_v2 import StandingCfg, StandingEnv as EndpointEnv
from .standing_margin_reward import SOFT_LEG_SPEED, leg_speed_margin

class StandingEnv(EndpointEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.decision_leg_speed_peak = torch.zeros((self.num_envs, 4), device=self.device)

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if hasattr(self, 'decision_leg_speed_peak'):
            self.decision_leg_speed_peak[ids if ids is not None else self.robot._ALL_INDICES] = 0.0

    def _pre_physics_step(self, actions):
        self.decision_leg_speed_peak.copy_(self.robot.data.joint_vel[:, self.motor_ids[:4]].abs())
        super()._pre_physics_step(actions)

    def _record_flags(self):
        self.decision_leg_speed_peak.copy_(torch.maximum(self.decision_leg_speed_peak, self.robot.data.joint_vel[:, self.motor_ids[:4]].abs()))
        super()._record_flags()

    def _get_rewards(self):
        reward = super()._get_rewards()
        margin = leg_speed_margin(self.decision_leg_speed_peak)
        self.reward_terms['leg_speed_margin'] = margin
        self.episode_returns += margin
        self.guard_step_stats['leg_margin_decisions'] = (self.decision_leg_speed_peak > SOFT_LEG_SPEED).any(-1).sum()
        return reward + margin
