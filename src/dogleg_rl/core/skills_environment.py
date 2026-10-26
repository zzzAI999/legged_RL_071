"""Sensor-based actor observations, curriculum and physical domain randomization."""
import json
import math
from pathlib import Path
import sys
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_euler_xyz, quat_mul
ROOT = Path(__file__).resolve().parents[1] / 'assets'
from .lab_environment import BASE, PAYLOAD, DoglegCfg, DoglegLabEnv
from .command_contract import observations, motor_targets
from .inertia_contract import write_body_inertias
BASE_CFG = DoglegCfg()

@configclass
class SkillsCfg(DoglegCfg):
    observation_space = 29
    state_space = 39
    episode_length_s = 20.0
    stage = 'steer'
    domain_randomization = True
    terrain = False
    terrain_wavelength = 0.6
    terrain_extent = 6.0
    terrain_offset = 0.12
    command_vx_max = 0.5
    command_yaw_max = 2.0
    preserve_low_jump_probability = 0.0
    jump_leg_scale = 0.3
    jump_probability = 0.55
    jump_height_max = 0.025
    randomize_jump = False
    randomization_profile = 'standard'
    inertial_randomization = False
    maximum_push_N = 0.0
    reward_scale = 0.02
    observation_noise = True
    repair_curriculum = False
    scene = BASE_CFG.scene.replace(num_envs=256, env_spacing=14.0)
    robot_cfg = BASE_CFG.robot_cfg.replace(spawn=BASE_CFG.robot_cfg.spawn.replace(activate_contact_sensors=True))

class SkillsEnv(DoglegLabEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.stage = cfg.stage
        self.sensor_noise_enabled = cfg.observation_noise
        self.leg_scale = torch.full((self.num_envs, 1), 0.18, device=self.device)
        self.travel_limit = 6.0
        self.command_fixed = False
        self.push_limit = {'steer': 5.0, 'jump': 2.0, 'robust': 20.0}[self.stage]
        if cfg.maximum_push_N > 0.0:
            self.push_limit = cfg.maximum_push_N
        self.push_duration = 5
        (self.wheel_body_ids, _) = self.robot.find_bodies(['left_rear4_link', 'right_rear4_link'], preserve_order=True)
        self.skill = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.jump_height = torch.zeros(self.num_envs, device=self.device)
        self.jump_start = torch.full_like(self.skill, 150)
        self.jump_time = torch.zeros(self.num_envs, device=self.device)
        self.jump_peak = torch.zeros(self.num_envs, device=self.device)
        self.air_steps = torch.zeros_like(self.skill)
        self.jump_successes = torch.zeros((), dtype=torch.long, device=self.device)
        self.delay = torch.zeros_like(self.skill)
        self.action_history = torch.zeros((3, self.num_envs, 6), device=self.device)
        self.gain = torch.ones((self.num_envs, 1), device=self.device)
        self.strength = torch.ones_like(self.gain)
        self.mass_scale = torch.ones(self.num_envs, device=self.device)
        self.friction = torch.full_like(self.mass_scale, 0.9)
        self.material_audit = {}
        self._randomize_physics(cfg.domain_randomization)
        self.actor_noise = torch.tensor([0.006] * 3 + [0.004] * 3 + [0.0] * 3 + [0.004] * 4 + [0.002] * 4 + [0.006] * 2 + [0.0] * 10, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        floor = sim_utils.CuboidCfg(size=(1024.0, 1024.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9, restitution=0.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.42, 0.38)))
        floor.func('/World/ground', floor, translation=(0.0, 0.0, -0.05))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot'] = self.robot
        light = sim_utils.DomeLightCfg(intensity=900.0)
        light.func('/World/Light', light)
        self.contacts = ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*_rear4_link', update_period=0.0, history_length=3))
        self.scene.sensors['wheel_contact'] = self.contacts
        self.all_contacts = ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*', update_period=0.0, history_length=1))
        self.scene.sensors['whole_robot_contact'] = self.all_contacts
        self.terrain_parameters = None
        if self.cfg.terrain:
            self._create_terrain()
        global_colliders = ['/World/ground'] + (['/World/terrain'] if self.cfg.terrain else [])
        self.scene.filter_collisions(global_prim_paths=global_colliders)

    def _create_terrain(self):
        import numpy as np
        from pxr import Gf, UsdGeom, UsdPhysics
        rng = np.random.default_rng(self.cfg.seed)
        parameters = np.column_stack((rng.uniform(-0.008, 0.008, (self.num_envs, 2)), rng.uniform(0.0, 0.002, self.num_envs), rng.uniform(0.0, 2.0 * math.pi, (self.num_envs, 2))))
        self.terrain_parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device)
        coordinates = np.linspace(-self.cfg.terrain_extent, self.cfg.terrain_extent, 49)
        (xx, yy) = np.meshgrid(coordinates, coordinates, indexing='ij')
        (vertices, faces) = ([], [])
        origins = self.scene.env_origins.cpu().numpy()
        for (index, (sx, sy, amplitude, px, py)) in enumerate(parameters):
            height = self.cfg.terrain_offset + sx * xx + sy * yy + amplitude * np.sin(xx * 2.0 * math.pi / self.cfg.terrain_wavelength + px) * np.sin(yy * 2.0 * math.pi / self.cfg.terrain_wavelength + py)
            points = np.stack((xx + origins[index, 0], yy + origins[index, 1], height), axis=-1).reshape(-1, 3)
            offset = len(vertices)
            vertices.extend(map(tuple, points))
            for x in range(48):
                for y in range(48):
                    a = offset + x * 49 + y
                    faces.extend((a, a + 49, a + 50, a, a + 50, a + 1))
        mesh = UsdGeom.Mesh.Define(self.sim.stage, '/World/terrain')
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.42, 0.38)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    def ground_height(self, position):
        if self.terrain_parameters is None:
            return torch.zeros(position.shape[0], device=self.device)
        local = position - self.scene.env_origins
        (sx, sy, amplitude, px, py) = self.terrain_parameters.unbind(dim=-1)
        scale = 2.0 * math.pi / self.cfg.terrain_wavelength
        (x, y) = (local[:, 0], local[:, 1])
        extent = self.cfg.terrain_extent
        spacing = 2.0 * extent / 48
        (x0, y0) = (torch.floor((x + extent) / spacing) * spacing - extent, torch.floor((y + extent) / spacing) * spacing - extent)
        (u, v) = ((x - x0) / spacing, (y - y0) / spacing)

        def node(xx, yy):
            return self.cfg.terrain_offset + sx * xx + sy * yy + amplitude * torch.sin(xx * scale + px) * torch.sin(yy * scale + py)
        (h00, h10, h11, h01) = (node(x0, y0), node(x0 + spacing, y0), node(x0 + spacing, y0 + spacing), node(x0, y0 + spacing))
        height = torch.where(u >= v, h00 + u * (h10 - h00) + v * (h11 - h10), h00 + v * (h01 - h00) + u * (h11 - h01))
        return torch.where((x.abs() <= extent) & (y.abs() <= extent), height, 0.0)

    def _get_dones(self):
        (done, timeout) = super()._get_dones()
        done |= self.robot.data.root_pos_w[:, 2] - self.ground_height(self.robot.data.root_pos_w) < 0.06
        return (done, timeout)

    def _randomize_physics(self, enabled):
        if not enabled:
            return
        view = self.robot.root_physx_view
        reference = (view.get_masses().clone(), view.get_inertias().clone(), view.get_coms().clone()) if self.cfg.inertial_randomization else None
        ids = torch.arange(self.num_envs, dtype=torch.int, device='cpu')
        extended = self.cfg.randomization_profile == 'extended'
        scale = torch.empty((self.num_envs, 1), device='cpu').uniform_(*((0.8, 1.25) if extended else (0.9, 1.15)))
        mass = view.get_masses().clone()
        inertia = view.get_inertias().clone()
        view.set_masses(mass * scale, ids)
        inertia_error = write_body_inertias(view, inertia * scale.unsqueeze(-1), ids)
        material = view.get_material_properties().clone()
        friction = torch.empty((self.num_envs, 1), device='cpu').uniform_(*((0.45, 1.2) if extended else (0.6, 1.1)))
        material[:, :, 0] = friction
        material[:, :, 1] = friction * 0.95
        material[:, :, 2] = 0.0
        view.set_material_properties(material, ids)
        com = view.get_coms().clone()
        offsets = torch.empty((self.num_envs, 3), device='cpu').uniform_(-1.0, 1.0)
        offsets *= torch.tensor([0.005, 0.008, 0.005])
        com[:, self.base_ids[0], :3] += offsets
        view.set_coms(com, ids)
        self.mass_scale.copy_(scale[:, 0].to(self.device))
        self.friction.copy_(friction[:, 0].to(self.device))
        self.material_audit = dict(profile=self.cfg.randomization_profile, body_frame_inertia_max_abs_error=inertia_error, mass_scale_min=float(scale.min()), mass_scale_max=float(scale.max()), friction_min=float(friction.min()), friction_max=float(friction.max()), com_offset_abs_max_m=offsets.abs().max(dim=0).values.tolist())
        assert torch.allclose(view.get_masses(), mass * scale, atol=1e-05)
        assert torch.allclose(view.get_coms(), com, atol=1e-06)
        assert torch.allclose(view.get_material_properties(), material, atol=1e-05)
        if reference is not None:
            from .inertial_randomization import apply_inertials
            apply_inertials(self, view, *reference)

    def wheel_contact(self):
        return torch.linalg.vector_norm(self.contacts.data.net_forces_w, dim=-1) > 1.0

    def fully_airborne(self):
        return torch.linalg.vector_norm(self.all_contacts.data.net_forces_w, dim=-1).sum(dim=-1) < 1.0

    def wheel_clearance(self):
        values = []
        for (index, name) in enumerate(('left_tire', 'right_tire')):
            geom = next((g for g in PAYLOAD['geoms'] if g['name'] == name))
            body = self.wheel_body_ids[index]
            quaternion = self.robot.data.body_quat_w[:, body]
            local = torch.tensor(geom['pos'], device=self.device).expand(self.num_envs, 3)
            center = self.robot.data.body_pos_w[:, body] + quat_apply(quaternion, local)
            axis = quat_apply(quaternion, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
            extent = geom['size'][0] * torch.sqrt((1.0 - axis[:, 2].square()).clamp(min=0.0))
            extent += geom['size'][1] * axis[:, 2].abs()
            values.append(center[:, 2] - extent - self.ground_height(center))
        return torch.stack(values, dim=-1)

    def _get_observations(self):
        t = (self.episode_length_buf - self.jump_start) * 0.02
        active = (self.skill == 5) & (t >= 0) & (t < 1.5)
        (_, gyro) = self.motion()
        policy = observations(gyro, self.gravity(), self.commands, self.robot.data.joint_pos[:, self.motor_ids[:4]], self.robot.data.joint_vel[:, self.motor_ids], self.actions, self.jump_height, t, active)
        if self.training_mode or self.sensor_noise_enabled:
            policy += torch.randn_like(policy) * self.actor_noise
        (linear, _) = self.motion()
        privileged = torch.cat((linear, self.robot.data.root_pos_w[:, 2:3], self.wheel_contact().float(), self.last_force[:, 0, [0, 2]] * 0.05, self.mass_scale[:, None], self.friction[:, None]), dim=-1)
        return {'policy': policy.clamp(-100, 100), 'critic': torch.cat((policy, privileged), dim=-1)}

    def _sample_commands(self, ids):
        count = len(ids)
        skill = torch.randint(0, 6 if self.stage != 'steer' else 5, (count,), device=self.device)
        if self.stage != 'steer':
            skill = torch.where(torch.rand(count, device=self.device) < self.cfg.jump_probability, 5, skill.clamp(max=4))
        self.skill[ids] = skill
        v = torch.empty(count, device=self.device).uniform_(-self.cfg.command_vx_max, self.cfg.command_vx_max)
        yaw = torch.empty(count, device=self.device).uniform_(-self.cfg.command_yaw_max, self.cfg.command_yaw_max)
        commands = torch.zeros((count, 3), device=self.device)
        commands[:, 0] = torch.where((skill == 1) | (skill == 4), v, 0.0)
        commands[:, 1] = torch.where((skill == 2) | (skill == 4), yaw * 0.5, 0.0)
        commands[:, 1] = torch.where(skill == 3, torch.sign(yaw) * torch.empty_like(yaw).uniform_(1.0, self.cfg.command_yaw_max), commands[:, 1])
        self.targets[ids] = commands
        self.jump_height[ids] = torch.where(skill == 5, torch.empty_like(v).uniform_(0.02, self.cfg.jump_height_max), 0.0)
        if self.cfg.preserve_low_jump_probability:
            low = (skill == 5) & (torch.rand(count, device=self.device) < self.cfg.preserve_low_jump_probability)
            self.jump_height[ids] = torch.where(low, 0.025, self.jump_height[ids])
        if self.cfg.repair_curriculum and self.stage == 'jump':
            choices = torch.rand(count, device=self.device)
            heights = torch.where(choices < 0.25, 0.025, torch.where(choices < 0.5, 0.035, 0.05))
            self.jump_height[ids] = torch.where(skill == 5, heights.clamp(max=self.cfg.jump_height_max), 0.0)
        self.jump_start[ids] = self.episode_length_buf[ids] + torch.randint(100, 151, (count,), device=self.device)
        self.command_at[ids] = self.episode_length_buf[ids] + torch.randint(220, 401, (count,), device=self.device)
        if self.cfg.repair_curriculum and self.stage == 'jump':
            self.command_at[ids] = torch.where(skill == 5, self.jump_start[ids] + 90, self.command_at[ids])

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.action_history[2] = self.action_history[1].clone()
        self.action_history[1] = self.action_history[0].clone()
        self.action_history[0] = actions.clamp(-1.0, 1.0)
        self.actions = self.action_history[self.delay, torch.arange(self.num_envs, device=self.device)].clone()
        self.action_change = self.actions - self.previous_actions
        if self.training_mode and (not self.command_fixed):
            ids = (self.episode_length_buf >= self.command_at).nonzero().flatten()
            self._sample_commands(ids)
        limits = torch.tensor([0.8, 3.0, 0.0], device=self.device) * 0.02
        self.commands += (self.targets - self.commands).clamp(-limits, limits)
        self.jump_time = (self.episode_length_buf - self.jump_start) * 0.02
        jumping = (self.skill == 5) & (self.jump_time >= 0) & (self.jump_time < 1.5)
        scale = torch.where(jumping[:, None], torch.full_like(self.leg_scale, self.cfg.jump_leg_scale), self.leg_scale)
        (self.leg_targets, self.wheel_targets) = motor_targets(self.actions, scale)
        active = (self.episode_length_buf >= self.push_at) & (self.episode_length_buf < self.push_at + self.push_duration)
        if not self.pushes_enabled:
            active[:] = False
        self.last_force = (self.push_vectors * active[:, None]).unsqueeze(1)
        offset = torch.tensor([0.0, 0.08, 0.0], device=self.device).expand_as(self.last_force)
        self.robot.set_external_force_and_torque(self.last_force, torch.cross(offset, self.last_force, dim=-1), body_ids=self.base_ids)
        self.push_total += active.sum()
        if self.training_mode:
            finished = self.episode_length_buf == self.push_at + self.push_duration
            ids = finished.nonzero().flatten()
            self.push_at[ids] += torch.randint(150, 351, (len(ids),), device=self.device)
            angle = torch.rand(len(ids), device=self.device) * 6.2831853
            magnitude = torch.rand(len(ids), device=self.device) * self.push_limit
            self.push_vectors[ids] = torch.stack((magnitude * torch.cos(angle), torch.zeros_like(angle), -magnitude * torch.sin(angle)), dim=-1)

    def _apply_action(self):
        (q, qd) = (self.robot.data.joint_pos, self.robot.data.joint_vel)
        force = -qd * self.damping
        force[:, self.motor_ids[:4]] = (30 * self.gain * (self.leg_targets - q[:, self.motor_ids[:4]]) - 0.8 * self.gain * qd[:, self.motor_ids[:4]]).clamp(-3.0, 3.0) * self.strength
        force[:, self.motor_ids[4:]] = self.wheel_targets * self.strength
        self.robot.set_joint_effort_target(force)

    def _get_rewards(self):
        (linear, angular) = self.motion()
        vx = linear[:, 0] - self.commands[:, 0]
        yaw = angular[:, 2] - self.commands[:, 1]
        upright = -self.gravity()[:, 2]
        jumping = (self.skill == 5) & (self.jump_time >= 0) & (self.jump_time < 1.5)
        contact = self.wheel_contact()
        airborne = self.fully_airborne()
        rise = (self.robot.data.root_pos_w[:, 2] - self.ground_height(self.robot.data.root_pos_w) - float(BASE['pos'][2])).clamp(min=0)
        self.air_steps = torch.where(jumping & airborne, self.air_steps + 1, 0)
        self.jump_peak = torch.where(jumping, torch.maximum(self.jump_peak, rise), torch.zeros_like(rise))
        takeoff = (self.jump_time >= 0.15) & (self.jump_time < 0.55)
        landing = (self.jump_time >= 0.8) & (self.jump_time < 1.5)
        clearance = self.wheel_clearance().min(dim=-1).values.clamp(min=0.0)
        jump_reward = 3.0 * airborne.float() * takeoff + 3.0 * (clearance / self.jump_height.clamp(min=0.005)).clamp(max=1.5) * takeoff + 2.0 * contact.all(dim=-1).float() * landing - 0.15 * linear[:, 2].square().clamp(max=25.0) * landing
        reward = 1.0 + 2.0 * torch.exp(-20.0 * (1.0 - upright)) + 3.0 * torch.exp(-vx.square() / 0.025) + 4.0 * torch.exp(-yaw.square() / 0.16) - 1.0 * vx.square().clamp(max=9.0) - 0.7 * yaw.square().clamp(max=16.0) - 0.4 * angular[:, :2].square().sum(dim=-1).clamp(max=16.0) - 0.25 * linear[:, 1].square().clamp(max=9.0) - 0.15 * linear[:, 2].square().clamp(max=16.0) * ~jumping - 0.005 * self.actions.square().sum(dim=-1) - 0.015 * self.action_change.square().sum(dim=-1) + jump_reward * jumping - 15.0 * self.reset_terminated.float()
        if self.cfg.repair_curriculum:
            from .repair_rewards import tracking_bonus, jump_bonus
            reward += tracking_bonus(linear, angular, self.commands)
            reward += jumping * jump_bonus(linear, upright, self.jump_height, self.jump_time, airborne, clearance, contact)
        return reward * self.cfg.reward_scale

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if not hasattr(self, 'skill'):
            return
        if self.terrain_parameters is not None:
            state = self.robot.data.default_root_state[env_ids].clone()
            state[:, :3] += self.scene.env_origins[env_ids]
            state[:, 2] += self.ground_height(self.scene.env_origins)[env_ids] + 0.001
            if self.randomize_reset:
                pitch = torch.empty(len(env_ids), device=self.device).uniform_(-0.02, 0.02)
                zeros = torch.zeros_like(pitch)
                state[:, 3:7] = quat_mul(quat_from_euler_xyz(zeros, pitch, zeros), state[:, 3:7])
            self.robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self.action_history[:, env_ids] = 0.0
        self.jump_time[env_ids] = -3.0
        self.jump_peak[env_ids] = 0.0
        self.air_steps[env_ids] = 0
        self.skill[env_ids] = 0
        self.jump_height[env_ids] = 0.0
        self.command_at[env_ids] = 25
        if self.training_mode:
            randomize = self.cfg.domain_randomization
            delay_count = 3 if self.cfg.randomization_profile == 'extended' else 2 if self.cfg.randomize_jump else 3 if self.stage == 'robust' else 1
            self.delay[env_ids] = torch.randint(0, delay_count if randomize else 1, (len(env_ids),), device=self.device)
            if randomize:
                extended = self.cfg.randomization_profile == 'extended'
                self.gain[env_ids] = torch.empty((len(env_ids), 1), device=self.device).uniform_(*((0.8, 1.2) if extended else (0.85, 1.15)))
                self.strength[env_ids] = torch.empty((len(env_ids), 1), device=self.device).uniform_(*((0.85, 1.1) if extended else (0.9, 1.1)))
        self.push_vectors[env_ids] *= self.push_limit / 10.0
