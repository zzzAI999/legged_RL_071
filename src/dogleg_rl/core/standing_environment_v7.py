"""Keep the v6 physical plant; add measured yaw to zero-command standing."""
import torch
from .standing_environment_v6 import StandingCfg, StandingEnv as StandingEnvV6
from .standing_math_v7 import balance_terms, hardware_flags, stable_criteria

class StandingEnv(StandingEnvV6):

    def measures(self):
        m = super().measures()
        m['yaw_rate'] = self.motion()[1][:, 2]
        return m

    def standing(self, m):
        tests = stable_criteria(m, self.mass_scale * 5.5 * 9.8, self.safety_latched)
        return torch.stack(list(tests.values()), -1).all(-1)

    def _get_rewards(self):
        m = self.measures()
        q = self.robot.data.joint_pos[:, self.motor_ids[:4]]
        speeds = self.robot.data.joint_vel[:, self.motor_ids]
        flags = hardware_flags(speeds, q - self.begin_q, self.command_age)
        self.reward_terms = balance_terms(m, q, speeds, self.applied, self.previous_actions, self.mass_scale * 5.5 * 9.8, flags)
        self.reward_terms['fall_terminal'] = -5.0 * self.reset_terminated.float()
        reward = sum(self.reward_terms.values())
        self.episode_returns += reward
        if self.reset_time_outs.any():
            self.extras['terminal_critic'] = super()._get_observations()['critic'].detach().clone()
        else:
            self.extras.pop('terminal_critic', None)
        assert torch.isfinite(reward).all(), 'Nonfinite reward'
        return reward
