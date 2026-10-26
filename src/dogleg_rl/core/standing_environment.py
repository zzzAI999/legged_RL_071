"""Learn balance before broad recovery; retain a separate hardware-gated view."""
import math
import torch
from isaaclab.utils import configclass
from .deployment_environment import DeploymentRecoveryEnv
from .recovery_environment import RecoveryCfg
from .recovery_contract import LEG_SCALE, WHEEL_SCALE
from .standing_math import balance_reward, hardware_flags, stable_criteria
from .wheel_current_guard import guarded_wheel_current

@configclass
class StandingCfg(RecoveryCfg):
    episode_length_s = 40.2
    curriculum_level = 0
    hardware_gates = False
    eval_seconds = 35.02

class StandingEnv(DeploymentRecoveryEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.level = cfg.curriculum_level
        self.nominal_row = self.atlas_label.index('nominal')
        self.pose_magnitude = self.atlas_q[:, self.motor_ids[:4]].abs().amax(-1)
        self.hardware_gate_mask = torch.full((self.num_envs,), cfg.hardware_gates, dtype=torch.bool, device=self.device)
        self.would_stop = torch.zeros_like(self.hardware_gate_mask)
        self.first_violation_ms = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.violation_flags = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.begin_q = torch.zeros((self.num_envs, 4), device=self.device)
        self.reset_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        self.contact_history = torch.zeros((5, self.num_envs, self.robot.num_bodies, 3), device=self.device)
        self.contact_cursor = 0
        self.contact_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.contact_sum = torch.zeros_like(self.contact_history[0])
        self.contact_samples = 0
        self.fall_ticks = torch.zeros_like(self.contact_counts)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.completed_returns = []
        self.training_reset_serial = 0
        self.measure_cache_tick = -1
        self.measure_cache = None

    def pose_pool(self, level=None):
        level = self.level if level is None else level
        if level == 0:
            return torch.tensor([self.nominal_row], device=self.device)
        pool = self.full_pool
        bound = 0.15 if level in (1, 2) else 0.35 if level == 3 else 1.15
        eligible = pool[self.pose_magnitude[pool] <= bound]
        assert len(eligible), ('No legal curriculum poses', level)
        return eligible

    def _randomize_domain(self, ids):
        original = self.cfg.profile
        if self.training_mode:
            self.cfg.profile = 'nominal' if self.level < 2 else 'mixed'
        super()._randomize_domain(ids)
        self.cfg.profile = original

    def _reset_idx(self, ids):
        if ids is None:
            ids = self.robot._ALL_INDICES
        original = self.override_rows
        if original is None:
            pool = self.pose_pool()
            self.override_rows = self.selected_rows.clone()
            chosen = pool[torch.randint(len(pool), (len(ids),), device=self.device)]
            if self.training_mode and self.level == 4:
                historical = torch.rand(len(ids), device=self.device) < 0.1
                history = self.historical_pool[torch.randint(len(self.historical_pool), (len(ids),), device=self.device)]
                chosen = torch.where(historical, history, chosen)
            self.override_rows[ids] = chosen
        if self.training_mode and (self.episode_length_buf[ids] > 0).any():
            self.completed_returns.extend(self.episode_returns[ids].detach().cpu().tolist())
            self.completed_returns = self.completed_returns[-2048:]
        super()._reset_idx(ids)
        self.override_rows = original
        self.begin_q[ids] = self.reset_reference_q[ids]
        self.safety_latched[ids] &= self.hardware_gate_mask[ids]
        self.would_stop[ids] = self.initial_admission_failed[ids]
        self.first_violation_ms[ids] = torch.where(self.initial_admission_failed[ids], 0, -1)
        self.violation_flags[ids] = False
        self.contact_history[:, ids] = 0.0
        self.contact_counts[ids] = 0
        self.fall_ticks[ids] = 0
        self.episode_returns[ids] = 0.0
        velocity = torch.rand((len(ids), 6), device=self.device) * 2.0 - 1.0
        velocity[:, :3] *= 0.01
        velocity[:, 2] = 0.0
        velocity[:, 3:] *= 0.05
        self.reset_velocity[ids] = velocity
        self.robot.write_root_velocity_to_sim(velocity, ids)
        self.measure_cache_tick = -1

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.contact_sum.zero_()
        self.contact_samples = 0
        if self.level < 2:
            self.drop_commands[:] = False
        begin = self.begin_q + self.q_bias
        target = self.request[:, :4] * self.action_sign[:4] * LEG_SCALE
        clamped = torch.maximum(torch.minimum(target, begin + 1.45), begin - 1.45)
        target = torch.where(self.hardware_gate_mask[:, None], clamped, target)
        self.request[:, :4] = target / (self.action_sign[:4] * LEG_SCALE)

    def _apply_action(self):
        qd = self.robot.data.joint_vel
        q = self.robot.data.joint_pos[:, self.motor_ids[:4]]
        speeds = qd[:, self.motor_ids]
        flags = hardware_flags(speeds, q - self.begin_q, self.command_age)
        first = flags.any(-1) & ~self.would_stop
        self.first_violation_ms[first] = self.episode_length_buf[first] * 20 + self.control_tick % 20
        self.violation_flags[first] = flags[first]
        self.would_stop |= flags.any(-1)
        self.safety_latched |= flags.any(-1) & self.hardware_gate_mask
        self.speed_peak = torch.maximum(self.speed_peak, speeds.abs())
        if self.control_tick % 10 == 0:
            self.desired = torch.where(self.drop_commands[:, None], self.desired, self.request)
        self.history_cursor = (self.history_cursor + 1) % 51
        self.action_history[self.history_cursor] = self.desired
        ids = torch.arange(self.num_envs, device=self.device)
        delayed = self.action_history[(self.history_cursor - self.latency_ms) % 51, ids]
        self.command_age += 1
        if self.control_tick % 10 == 0:
            self.command_age = torch.where(self.drop_commands, self.command_age, 0)
        if self.control_tick % 2 == 0:
            target = delayed[:, :4] * self.action_sign[:4] * LEG_SCALE
            begin = self.begin_q + self.q_bias
            clamped = torch.maximum(torch.minimum(target, begin + 1.45), begin - 1.45)
            target = torch.where(self.hardware_gate_mask[:, None], clamped, target)
            self.slew_target += (target - self.slew_target).clamp(-0.003, 0.003)
            force = -qd * self.damping
            leg = 30.0 * self.gains * (self.slew_target - q - self.q_bias) - 0.8 * self.kd_gains * speeds[:, :4]
            leg = (leg * self.strengths[:, :4]).clamp(-3.0, 3.0)
            requested = delayed[:, 4:] * self.action_sign[4:] * WHEEL_SCALE
            (_, nominal) = guarded_wheel_current(requested, speeds[:, 4:] * self.sensor_scale[:, 4:])
            wheel = (nominal * self.strengths[:, 4:]).clamp(-WHEEL_SCALE, WHEEL_SCALE)
            force[:, self.motor_ids[:4]] = leg
            force[:, self.motor_ids[4:]] = wheel - speeds[:, 4:] * self.wheel_drag
            self.held_torque = force
            self.applied[:, :4] = self.slew_target / (self.action_sign[:4] * LEG_SCALE)
            self.applied[:, 4:] = nominal / (self.action_sign[4:] * WHEEL_SCALE)
            self.torque_peak = torch.maximum(self.torque_peak, torch.cat((leg.abs(), wheel.abs()), -1))
        self.held_torque[self.safety_latched] = 0.0
        self.applied[self.safety_latched] = 0.0
        self.actions.copy_(self.applied)
        self.robot.set_joint_effort_target(self.held_torque)
        self.contact_sum += self.all_contacts.data.net_forces_w
        self.contact_samples += 1
        if self.control_tick % 10 == 0:
            (_, gyro) = self.motion()
            self.sensor_cursor = (self.sensor_cursor + 1) % 51
            self.sensor_history[self.sensor_cursor] = torch.cat((gyro + self.gyro_bias, self.gravity(), q + self.q_bias, speeds * self.sensor_scale), -1)
            self.ack_history[self.sensor_cursor] = self.applied
        self.control_tick += 1

    def measures(self):
        if self.measure_cache_tick == self.control_tick:
            return self.measure_cache
        m = super().measures()
        mean = self.contact_history.sum(0) / self.contact_counts.clamp(min=1)[:, None, None]
        mean = mean[:, self.contacts_ids]
        m['wheel_mean'] = mean[:, self.wheel_body_ids, 2].clamp(min=0.0)
        mask = [i for i in range(self.robot.num_bodies) if i not in self.wheel_body_ids]
        m['nonwheel_mean'] = mean[:, mask, 2].clamp(min=0.0).sum(-1)
        self.measure_cache_tick = self.control_tick
        self.measure_cache = m
        return m

    def standing(self, m):
        tests = stable_criteria(m, self.mass_scale * 5.5 * 9.8, self.safety_latched)
        return torch.stack(list(tests.values()), -1).all(-1)

    def _get_dones(self):
        self.contact_history[self.contact_cursor] = self.contact_sum / max(1, self.contact_samples)
        self.contact_cursor = (self.contact_cursor + 1) % 5
        self.contact_counts = (self.contact_counts + 1).clamp(max=5)
        self.measure_cache_tick = -1
        m = self.measures()
        stable = self.standing(m)
        self.hold_steps = torch.where(stable, self.hold_steps + 1, 0)
        self.has_recovered |= stable
        self.fall_ticks = torch.where(m['tilt_deg'] > 75.0, self.fall_ticks + 1, 0)
        invalid = ~torch.isfinite(self.robot.data.joint_pos).all(-1)
        invalid |= ~torch.isfinite(self.robot.data.joint_vel).all(-1)
        done = (self.fall_ticks >= 5) | (m['drift'] > 3.0) | (m['root_height'] < -0.02) | invalid
        done |= self.safety_latched
        if self.cfg.evaluation:
            done[:] = False
        return (done, self.episode_length_buf >= self.max_episode_length - 1)

    def _get_rewards(self):
        m = self.measures()
        q = self.robot.data.joint_pos[:, self.motor_ids[:4]]
        speeds = self.robot.data.joint_vel[:, self.motor_ids]
        violations = hardware_flags(speeds, q - self.begin_q, self.command_age)
        reward = balance_reward(m, q, speeds, self.applied, self.previous_actions, self.mass_scale * 5.5 * 9.8, violations)
        reward -= 5.0 * self.reset_terminated.float()
        self.episode_returns += reward
        if self.reset_time_outs.any():
            self.extras['terminal_critic'] = super()._get_observations()['critic'].detach().clone()
        else:
            self.extras.pop('terminal_critic', None)
        return reward
