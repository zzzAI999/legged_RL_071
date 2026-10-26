"""Preserve the v10 physics path and replace only its base_gyro term."""
import torch
from .standing_environment_leg_velocity_v1 import StandingCfg, StandingEnv as VelocityEnv
from .standing_gyro_penalty_reward_v1 import base_gyro_cost

class StandingEnv(VelocityEnv):

    def _get_rewards(self):
        reward = super()._get_rewards()
        replacement = base_gyro_cost(self.measures()['gyro'])
        delta = replacement - self.reward_terms['base_gyro']
        self.reward_terms['base_gyro'] = replacement
        self.episode_returns += delta
        result = reward + delta
        assert torch.isfinite(result).all(), 'Nonfinite gyro-weight reward'
        return result
