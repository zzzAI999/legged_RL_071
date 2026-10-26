"""Feasible randomized four-motor resets and a measured-timing recovery task."""
import json
import math
from pathlib import Path
import sys
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
ROOT = Path(__file__).resolve().parents[1] / 'assets'
from .skills_environment import SkillsCfg, SkillsEnv, PAYLOAD
from .command_contract import observations
from .inertial_randomization import sample_inertials
from .inertia_contract import write_body_inertias
from .recovery_contract import MOTORS, LEG_SCALE, WHEEL_SCALE, WHEEL_RATIO, LIMITS

@configclass
class RecoveryCfg(SkillsCfg):
    stage = 'robust'
    domain_randomization = False
    terrain = False
    episode_length_s = 35.2
    observation_noise = False
    profile = 'mixed'
    audit_directory = ''
    evaluation = False

class RecoveryEnv(SkillsEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        (self.motor_ids, _) = self.robot.find_joints(MOTORS, preserve_order=True)
        (self.wheel_body_ids, _) = self.robot.find_bodies(['right_rear4_link', 'left_rear4_link'], preserve_order=True)
        self.contacts_ids = [self.all_contacts.body_names.index(n) for n in self.robot.body_names]
        self.mass_reference = self.robot.root_physx_view.get_masses().clone()
        self.inertia_reference = self.robot.root_physx_view.get_inertias().clone()
        self.com_reference = self.robot.root_physx_view.get_coms().clone()
        self.material_reference = self.robot.root_physx_view.get_material_properties().clone()
        self.audit_dir = Path(cfg.audit_directory)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        atlas = json.loads((ROOT / 'reset_atlas.json').read_text())
        self.atlas = atlas
        self.rows = atlas['rows']
        order = [atlas['joint_names'].index(n) for n in self.robot.joint_names]
        self.atlas_q = torch.tensor([r['joint_delta_rad'] for r in self.rows], device=self.device)[:, order]
        self.atlas_root = torch.tensor([[0.0, 0.0, r['root_height_m'], *r['quaternion_wxyz']] for r in self.rows], device=self.device)
        self.atlas_label = [r['label'] for r in self.rows]
        indices = torch.arange(len(self.rows), device=self.device)
        near = torch.tensor([l in ('near', 'nominal') for l in self.atlas_label], device=self.device)
        historical = torch.tensor([l.startswith('v') for l in self.atlas_label], device=self.device)
        split = indices % 3 == 0 if cfg.evaluation else indices % 3 != 0
        self.near_pool = indices[near & split]
        self.full_pool = indices[~historical & split]
        self.historical_pool = indices[historical]
        self.coverage = 0.15
        self.reset_serial = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.selected_rows = torch.zeros_like(self.reset_serial)
        self.total_resets = 0
        self.audit_count = 0
        self.override_rows = None
        self.slew_target = torch.zeros((self.num_envs, 4), device=self.device)
        self.desired = torch.zeros((self.num_envs, 6), device=self.device)
        self.request = torch.zeros_like(self.desired)
        self.applied = torch.zeros_like(self.desired)
        self.held_torque = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.action_history = torch.zeros((51, self.num_envs, 6), device=self.device)
        self.sensor_history = torch.zeros((51, self.num_envs, 16), device=self.device)
        self.history_cursor = 0
        self.sensor_cursor = 0
        self.control_tick = 0
        self.command_age = torch.zeros_like(self.reset_serial)
        self.latency_ms = torch.zeros_like(self.reset_serial)
        self.sensor_latency_ms = torch.zeros_like(self.reset_serial)
        self.drop_commands = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gains = torch.ones((self.num_envs, 4), device=self.device)
        self.kd_gains = torch.ones_like(self.gains)
        self.strengths = torch.ones((self.num_envs, 6), device=self.device)
        self.q_bias = torch.zeros_like(self.gains)
        self.gyro_bias = torch.zeros((self.num_envs, 3), device=self.device)
        self.sensor_scale = torch.ones((self.num_envs, 6), device=self.device)
        self.wheel_drag = torch.ones((self.num_envs, 2), device=self.device) * 0.005
        self.safety_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.speed_peak = torch.zeros((self.num_envs, 6), device=self.device)
        self.torque_peak = torch.zeros_like(self.speed_peak)
        self.reset_reference_q = torch.zeros((self.num_envs, 4), device=self.device)
        self.reset_reference_root = torch.zeros((self.num_envs, 7), device=self.device)
        self.start_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.has_recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.first_recover_time = torch.full((self.num_envs,), -1.0, device=self.device)
        self.hold_steps = torch.zeros_like(self.reset_serial)
        self.pushes_enabled = False
        self.training_mode = not cfg.evaluation
        self.action_sign = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, 1.0], device=self.device)
        self._colliders = []
        for g in PAYLOAD['geoms']:
            if not g['collision']:
                continue
            body = self.robot.body_names.index(g['body'])
            self._colliders.append((g, body, torch.tensor(g['pos'], device=self.device), torch.tensor(g['quat'], device=self.device)))

    def _randomize_domain(self, ids):
        cpu = ids.cpu().to(torch.int)
        count = len(ids)
        nominal = torch.ones(count, dtype=torch.bool) if self.cfg.profile == 'nominal' else torch.zeros(count, dtype=torch.bool)
        corners = torch.ones(count, dtype=torch.bool) if self.cfg.profile == 'corners' else torch.zeros(count, dtype=torch.bool)
        if self.cfg.profile == 'mixed':
            nominal = torch.rand(count) < 0.25
            corners = (torch.rand(count) < 0.2) & ~nominal
        (m, i, c, audit) = sample_inertials(self.mass_reference[cpu], self.inertia_reference[cpu], self.com_reference[cpu], self.robot.body_names, self.cfg.seed + int(self.total_resets) + 713, nominal, corners)
        view = self.robot.root_physx_view
        mass = view.get_masses().clone()
        inertia = view.get_inertias().clone()
        com = view.get_coms().clone()
        (mass[cpu], inertia[cpu], com[cpu]) = (m, i, c)
        view.set_masses(mass, cpu)
        view.set_coms(com, cpu)
        write_body_inertias(view, inertia, cpu)
        material = view.get_material_properties().clone()
        friction = torch.empty(count).uniform_(0.55, 1.1)
        friction[corners] = torch.where(torch.arange(count)[corners] % 2 == 0, 0.55, 1.1)
        friction[nominal] = 0.9
        material[cpu, :, 0] = friction[:, None]
        material[cpu, :, 1] = friction[:, None] * 0.95
        material[cpu, :, 2] = 0.0
        view.set_material_properties(material, cpu)
        torch.testing.assert_close(view.get_masses()[cpu], m, atol=1e-06, rtol=1e-05)
        torch.testing.assert_close(view.get_coms()[cpu], c, atol=1e-06, rtol=1e-05)
        torch.testing.assert_close(view.get_material_properties()[cpu], material[cpu], atol=1e-06, rtol=1e-05)
        self.mass_scale[ids] = (m.sum(-1) / 5.5).to(self.device)
        self.friction[ids] = friction.to(self.device)
        self.gains[ids] = torch.empty((count, 4), device=self.device).uniform_(0.85, 1.15)
        self.kd_gains[ids] = torch.empty((count, 4), device=self.device).uniform_(0.85, 1.15)
        self.strengths[ids] = torch.empty((count, 6), device=self.device).uniform_(0.85, 1.1)
        self.latency_ms[ids] = torch.randint(0, 16, (count,), device=self.device)
        self.sensor_latency_ms[ids] = torch.randint(0, 16, (count,), device=self.device)
        self.q_bias[ids] = torch.empty((count, 4), device=self.device).uniform_(-0.025, 0.025)
        self.gyro_bias[ids] = torch.empty((count, 3), device=self.device).uniform_(-0.015, 0.015)
        self.sensor_scale[ids] = torch.empty((count, 6), device=self.device).uniform_(0.97, 1.03)
        self.wheel_drag[ids] = torch.empty((count, 2), device=self.device).uniform_(0.003, 0.012)
        corner_ids = ids[corners.to(self.device)]
        self.gains[corner_ids] = 0.85
        self.kd_gains[corner_ids] = 0.85
        self.strengths[corner_ids] = 0.85
        self.latency_ms[corner_ids] = 15
        self.sensor_latency_ms[corner_ids] = 15
        self.q_bias[corner_ids] = torch.tensor([0.025, -0.025, -0.025, 0.025], device=self.device)
        nominal_ids = ids[nominal.to(self.device)]
        self.gains[nominal_ids] = 1.0
        self.kd_gains[nominal_ids] = 1.0
        self.strengths[nominal_ids] = 1.0
        self.latency_ms[nominal_ids] = 0
        self.sensor_latency_ms[nominal_ids] = 0
        self.q_bias[nominal_ids] = 0.0
        self.gyro_bias[nominal_ids] = 0.0
        self.sensor_scale[nominal_ids] = 1.0
        self.wheel_drag[nominal_ids] = 0.005
        self.material_audit = dict(audit)
        self.material_audit.update(initialized_once=False, resampled_each_episode=True, friction_range=[0.55, 1.1], kp_gain_range=[0.85, 1.15], strength_range=[0.85, 1.1], sensor_action_latency_range_ms=[0, 15], sensor_zero_bias_range_rad=[-0.025, 0.025])

    def _reset_idx(self, ids):
        if ids is None:
            ids = self.robot._ALL_INDICES
        DirectRLEnv._reset_idx(self, ids)
        self.robot.reset(ids)
        count = len(ids)
        self._randomize_domain(ids)
        near = self.near_pool[torch.randint(len(self.near_pool), (count,), device=self.device)]
        full = self.full_pool[torch.randint(len(self.full_pool), (count,), device=self.device)]
        selected = torch.where(torch.rand(count, device=self.device) < self.coverage, full, near)
        historical = torch.rand(count, device=self.device) < self.coverage * 0.15
        if historical.any():
            selected[historical] = self.historical_pool[torch.randint(3, (int(historical.sum()),), device=self.device)]
        if self.override_rows is not None:
            selected = self.override_rows[ids]
        self.selected_rows[ids] = selected
        q = self.atlas_q[selected]
        state = self.atlas_root[selected].clone()
        state[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim(state, ids)
        self.robot.write_root_velocity_to_sim(torch.zeros((count, 6), device=self.device), ids)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.reset_reference_q[ids] = q[:, self.motor_ids[:4]]
        self.reset_reference_root[ids] = state
        self.start_xy[ids] = state[:, :2]
        self.slew_target[ids] = q[:, self.motor_ids[:4]]
        self.request[ids] = 0.0
        self.desired[ids] = 0.0
        self.held_torque[ids] = 0.0
        self.applied[ids] = 0.0
        self.applied[ids, :4] = self.slew_target[ids] / (self.action_sign[:4] * LEG_SCALE)
        self.actions[ids] = self.applied[ids]
        self.previous_actions[ids] = self.applied[ids]
        self.action_history[:, ids] = self.applied[ids]
        self.sensor_history[:, ids] = 0.0
        self.command_age[ids] = 0
        self.safety_latched[ids] = False
        self.speed_peak[ids] = 0.0
        self.torque_peak[ids] = 0.0
        self.first_recover_time[ids] = -1.0
        self.has_recovered[ids] = False
        self.hold_steps[ids] = 0
        self.commands[ids] = 0.0
        self.targets[ids] = 0.0
        self.reset_serial[ids] += 1
        self.total_resets += count

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.applied)
        self.request.copy_(actions.clamp(-1.0, 1.0))
        self.drop_commands = (torch.rand(self.num_envs, device=self.device) < 0.001) & (self.cfg.profile != 'nominal')

    def _apply_action(self):
        q = self.robot.data.joint_pos[:, self.motor_ids[:4]]
        qd_all = self.robot.data.joint_vel
        speeds = qd_all[:, self.motor_ids]
        self.speed_peak = torch.maximum(self.speed_peak, speeds.abs())
        rpm = speeds[:, 4:].abs() * WHEEL_RATIO * 60 / (2 * math.pi)
        self.safety_latched |= (speeds[:, :4].abs().amax(-1) > 6.0) | (rpm.amax(-1) > 3000.0)
        if self.control_tick % 10 == 0:
            self.desired = torch.where(self.drop_commands[:, None], self.desired, self.request)
        self.history_cursor = (self.history_cursor + 1) % 51
        self.action_history[self.history_cursor] = self.desired
        ids = torch.arange(self.num_envs, device=self.device)
        delayed = self.action_history[(self.history_cursor - self.latency_ms) % 51, ids]
        self.command_age += 1
        if self.control_tick % 10 == 0:
            self.command_age = torch.where(self.drop_commands, self.command_age, 0)
        self.safety_latched |= self.command_age > 40
        if self.control_tick % 2 == 0:
            target = delayed[:, :4] * self.action_sign[:4] * LEG_SCALE
            self.slew_target += (target - self.slew_target).clamp(-0.003, 0.003)
            torque = -qd_all * self.damping
            leg = 30 * self.gains * (self.slew_target - q - self.q_bias) - 0.8 * self.kd_gains * speeds[:, :4]
            leg = (leg * self.strengths[:, :4]).clamp(-3.0, 3.0)
            wheel = (delayed[:, 4:] * self.action_sign[4:] * WHEEL_SCALE * self.strengths[:, 4:]).clamp(-0.8, 0.8)
            torque[:, self.motor_ids[:4]] = leg
            torque[:, self.motor_ids[4:]] = wheel - speeds[:, 4:] * self.wheel_drag
            self.held_torque = torque
            self.applied[:, :4] = self.slew_target / (self.action_sign[:4] * LEG_SCALE)
            self.applied[:, 4:] = wheel / (self.action_sign[4:] * WHEEL_SCALE)
            self.actions.copy_(self.applied)
            self.torque_peak = torch.maximum(self.torque_peak, torch.cat((leg.abs(), wheel.abs()), -1))
        self.held_torque[self.safety_latched] = 0.0
        self.applied[self.safety_latched] = 0.0
        self.robot.set_joint_effort_target(self.held_torque)
        if self.control_tick % 10 == 0:
            (_, gyro) = self.motion()
            self.sensor_cursor = (self.sensor_cursor + 1) % 51
            self.sensor_history[self.sensor_cursor] = torch.cat((gyro + self.gyro_bias, self.gravity(), q + self.q_bias, speeds * self.sensor_scale), -1)
        self.control_tick += 1

    def measures(self):
        from isaaclab.utils.math import quat_mul
        (linear, angular) = self.motion()
        forces = self.all_contacts.data.net_forces_w[:, self.contacts_ids]
        wheel_fz = forces[:, self.wheel_body_ids, 2].clamp(min=0.0)
        mask = [i for i in range(len(self.robot.body_names)) if i not in self.wheel_body_ids]
        nonwheel_fz = forces[:, mask, 2].clamp(min=0.0).sum(-1)
        base_clearance = None
        for (g, body, pos, rot) in self._colliders:
            if g['body'] != 'base_link':
                continue
            quat = self.robot.data.body_quat_w[:, body]
            center = self.robot.data.body_pos_w[:, body] + quat_apply(quat, pos.expand(self.num_envs, 3))
            geom_quat = quat_mul(quat, rot.expand(self.num_envs, 4))
            axes = torch.eye(3, device=self.device).expand(self.num_envs, 3, 3)
            q_axes = quat_apply(geom_quat[:, None, :].expand(-1, 3, -1).reshape(-1, 4), axes.reshape(-1, 3)).reshape(self.num_envs, 3, 3)
            extent = q_axes[:, :, 2].abs() @ torch.tensor(g['size'], device=self.device)
            base_clearance = center[:, 2] - extent
        return dict(tilt_deg=torch.acos((-self.gravity()[:, 2]).clamp(-1.0, 1.0)) * 180 / math.pi, speed=torch.linalg.vector_norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1), gyro=torch.linalg.vector_norm(angular[:, :2], dim=-1), root_height=self.robot.data.root_pos_w[:, 2], base_clearance=base_clearance, wheel_fz=wheel_fz, nonwheel_fz=nonwheel_fz, drift=torch.linalg.vector_norm(self.robot.data.root_pos_w[:, :2] - self.start_xy, dim=-1))

    def standing(self, m):
        load = self.mass_scale * 5.5 * 9.8
        return (m['tilt_deg'] <= LIMITS['tilt_deg']) & (m['speed'] <= LIMITS['horizontal_speed_m_s']) & (m['gyro'] <= LIMITS['roll_pitch_gyro_rad_s']) & (m['base_clearance'] >= LIMITS['body_clearance_m']) & (m['root_height'] >= LIMITS['root_height_m']) & (m['wheel_fz'].amin(-1) >= LIMITS['each_wheel_load_min_N']) & (m['wheel_fz'].sum(-1) >= load * LIMITS['wheel_load_fraction_min']) & (m['nonwheel_fz'] <= LIMITS['nonwheel_load_max_N']) & ~self.safety_latched

    def _get_dones(self):
        m = self.measures()
        stable = self.standing(m)
        first = stable & ~self.has_recovered
        self.first_recover_time[first] = self.episode_length_buf[first] * 0.02
        self.has_recovered |= stable
        self.hold_steps = torch.where(stable, self.hold_steps + 1, 0)
        done = self.safety_latched | (m['tilt_deg'] > 75.0) | (m['drift'] > 1.5)
        done |= (m['root_height'] < -0.02) | ~torch.isfinite(self.robot.data.joint_pos).all(-1)
        done |= (self.episode_length_buf > 250) & ~self.has_recovered
        if self.cfg.evaluation:
            done[:] = False
        return (done, self.episode_length_buf >= self.max_episode_length - 1)

    def _get_rewards(self):
        m = self.measures()
        height = ((m['root_height'] - 0.01) / 0.12).clamp(0.0, 1.0)
        clearance = (m['base_clearance'] / 0.015).clamp(0.0, 1.0)
        upright = torch.exp(-m['tilt_deg'].square() / 100.0)
        speed_reward = torch.exp(-m['speed'].square() / 0.01)
        load = (m['wheel_fz'].amin(-1) / 8.0).clamp(0.0, 1.0)
        unsupported = clearance * load
        reward = 1.0 * height + 1.0 * clearance + 1.5 * upright + 2.0 * unsupported * speed_reward + 2.0 * self.standing(m).float()
        reward -= 0.8 * m['speed'].square().clamp(max=4.0) + 0.3 * m['gyro'].square().clamp(max=9.0)
        reward -= 1.5 * m['nonwheel_fz'].clamp(max=60.0) / 54.0
        reward -= 0.1 * (self.applied - self.previous_actions).square().sum(-1)
        reward -= 0.015 * self.applied.square().sum(-1) + 0.3 * (self.applied.abs() > 0.97).float().sum(-1)
        reward -= (1.0 - unsupported) * self.robot.data.joint_vel[:, self.motor_ids[4:]].square().sum(-1) * 0.002
        reward -= 0.5 * m['drift'].square().clamp(max=4.0) + 5.0 * self.safety_latched.float() + 4.0 * self.reset_terminated.float()
        return reward * 0.02

    def _get_observations(self):
        ids = torch.arange(self.num_envs, device=self.device)
        sensor = self.sensor_history[(self.sensor_cursor - (self.sensor_latency_ms + 9) // 10) % 51, ids].clone()
        fresh = self.episode_length_buf == 0
        if fresh.any():
            (_, gyro) = self.motion()
            sensor[fresh] = torch.cat((gyro[fresh] + self.gyro_bias[fresh], self.gravity()[fresh], self.robot.data.joint_pos[fresh][:, self.motor_ids[:4]] + self.q_bias[fresh], self.robot.data.joint_vel[fresh][:, self.motor_ids] * self.sensor_scale[fresh]), -1)
            self.sensor_history[:, fresh] = sensor[fresh]
        if self.cfg.profile != 'nominal':
            sensor[:, :3] += torch.randn_like(sensor[:, :3]) * 0.006
            sensor[:, 3:6] += torch.randn_like(sensor[:, 3:6]) * 0.003
            sensor[:, 3:6] /= torch.linalg.vector_norm(sensor[:, 3:6], dim=-1, keepdim=True).clamp(min=0.01)
            sensor[:, 6:10] += torch.randn_like(sensor[:, 6:10]) * 0.002
            sensor[:, 10:] += torch.randn_like(sensor[:, 10:]) * 0.025
        zeros = torch.zeros(self.num_envs, device=self.device)
        obs = observations(sensor[:, :3], sensor[:, 3:6], self.commands, sensor[:, 6:10], sensor[:, 10:], self.applied, zeros, zeros, zeros.bool())
        m = self.measures()
        privileged = torch.cat((self.motion()[0], m['root_height'][:, None], (m['wheel_fz'] > 2.0).float(), m['base_clearance'][:, None], m['nonwheel_fz'][:, None] / 54.0, self.mass_scale[:, None], self.friction[:, None]), -1)
        self._audit_reset_readback()
        return {'policy': obs.clamp(-100.0, 100.0), 'critic': torch.cat((obs, privileged), -1)}

    def _audit_reset_readback(self):
        new = (self.episode_length_buf <= 1).nonzero().flatten()
        if not len(new) or self.audit_count >= 8192:
            return
        actual = self.robot.data.joint_pos[new][:, self.motor_ids[:4]]
        gaps = self.pin_gaps()[new].amax(-1)
        belts = self.belt_errors()[new].amax(-1)
        m = self.measures()
        with (self.audit_dir / 'resets.jsonl').open('a') as f:
            for (j, env_id) in enumerate(new.tolist()):
                row_id = int(self.selected_rows[env_id])
                f.write(json.dumps(dict(env_id=env_id, serial=int(self.reset_serial[env_id]), step=int(self.episode_length_buf[env_id]), atlas_row=row_id, label=self.atlas_label[row_id], requested_leg_delta_rad=self.rows[row_id]['requested_leg_delta_rad'], actual_leg_delta_rad=actual[j].tolist(), pin_gap_m=float(gaps[j]), belt_error_rad=float(belts[j]), requested_root_height_m=float(self.reset_reference_root[env_id, 2]), actual_root_height_m=float(m['root_height'][env_id]), base_clearance_m=float(m['base_clearance'][env_id]), wheel_load_N=m['wheel_fz'][env_id].tolist(), nonwheel_load_N=float(m['nonwheel_fz'][env_id]))) + '\n')
        self.audit_count += len(new)
