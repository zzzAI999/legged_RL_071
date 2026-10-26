"""Isaac Lab command task for the accepted six-motor closed-chain mechanism."""
import json
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_euler_xyz, quat_mul
ROOT = Path(__file__).resolve().parents[1] / 'assets'
PAYLOAD = json.loads((ROOT / 'physical_payload.json').read_text())
BASE = next((b for b in PAYLOAD['bodies'] if b['name'] == 'base_link'))
MOTORS = ['left_front1_joint', 'left_rear1_joint', 'right_front1_joint', 'right_rear1_joint', 'left_rear4_joint', 'right_rear4_joint']

@configclass
class DoglegCfg(DirectRLEnvCfg):
    decimation = 20
    episode_length_s = 12.0
    action_space = 6
    observation_space = 25
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=0.001, render_interval=20, device='cuda:1', gravity=(0.0, 0.0, -9.8), use_fabric=True, physx=sim_utils.PhysxCfg(solver_type=1, min_position_iteration_count=32, max_position_iteration_count=32, min_velocity_iteration_count=8, max_velocity_iteration_count=8))
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=False)
    robot_cfg = ArticulationCfg(prim_path='/World/envs/env_.*/Robot', spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT / 'dogleg_lab.usda')), init_state=ArticulationCfg.InitialStateCfg(pos=tuple(BASE['pos']), rot=tuple(BASE['quat']), joint_pos={'.*': 0.0}, joint_vel={'.*': 0.0}), actuators={'effort': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0.0, damping=0.0, effort_limit_sim=1000.0, velocity_limit_sim=150.0)})

class DoglegLabEnv(DirectRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        (self.motor_ids, _) = self.robot.find_joints(MOTORS, preserve_order=True)
        (self.base_ids, _) = self.robot.find_bodies('base_link')
        self.legsign = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device)
        self.wheelsign = torch.tensor([1.0, -1.0], device=self.device)
        self.speedscale = torch.tensor([0.05] * 4 + [0.03] * 2, device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.targets = torch.zeros_like(self.commands)
        self.actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.action_change = torch.zeros_like(self.actions)
        self.damping = torch.tensor([next((j['damping'] for j in PAYLOAD['joints'] if j['name'] == n)) for n in self.robot.joint_names], device=self.device)
        self.push_vectors = torch.zeros((self.num_envs, 3), device=self.device)
        self.push_at = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_at = torch.zeros_like(self.push_at)
        self.push_total = torch.zeros((), device=self.device, dtype=torch.long)
        self.training_mode = True
        self.randomize_reset = True
        self.pushes_enabled = True
        self.travel_limit = 0.8
        self.last_force = torch.zeros((self.num_envs, 1, 3), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        floor = sim_utils.CuboidCfg(size=(200.0, 200.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9, restitution=0.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.69, 0.67)))
        floor.func('/World/ground', floor, translation=(0.0, 0.0, -0.05))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        self.scene.articulations['robot'] = self.robot
        sim_utils.DomeLightCfg(intensity=1800.0).func('/World/Light', sim_utils.DomeLightCfg(intensity=1800.0))

    def robot_local(self, vector):
        cad = quat_apply(quat_conjugate(self.robot.data.root_quat_w), vector)
        return torch.stack((cad[:, 0], -cad[:, 2], cad[:, 1]), dim=-1)

    def motion(self):
        return (self.robot_local(self.robot.data.root_lin_vel_w), self.robot_local(self.robot.data.root_ang_vel_w))

    def gravity(self):
        return self.robot_local(torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3))

    def pin_gaps(self):
        values = []
        for loop in PAYLOAD['loops']:
            points = []
            for k in (0, 1):
                idx = self.robot.body_names.index(loop['body' + str(k)])
                v = torch.tensor(loop['local_pos' + str(k) + '_m'], device=self.device).expand(self.num_envs, 3)
                points.append(self.robot.data.body_pos_w[:, idx] + quat_apply(self.robot.data.body_quat_w[:, idx], v))
            values.append(torch.linalg.vector_norm(points[0] - points[1], dim=-1))
        return torch.stack(values, dim=-1)

    def belt_errors(self):
        q = self.robot.data.joint_pos
        return torch.stack([(q[:, self.robot.joint_names.index(b['output_joint'])] - b['multiplier'] * q[:, self.robot.joint_names.index(b['motor_joint'])]).abs() for b in PAYLOAD['belts']], dim=-1)

    def _get_observations(self):
        (_, gyro) = self.motion()
        (q, qd) = (self.robot.data.joint_pos, self.robot.data.joint_vel)
        obs = torch.cat((gyro * 0.2, self.gravity(), self.commands * torch.tensor([1.0, 0.2, 1.0], device=self.device), q[:, self.motor_ids[:4]] * self.legsign / 0.5, qd[:, self.motor_ids] * self.speedscale, self.actions), dim=-1)
        if self.training_mode:
            scale = torch.tensor([0.002] * 6 + [0.0] * 3 + [0.002] * 4 + [0.001] * 4 + [0.003] * 2 + [0.0] * 6, device=self.device)
            obs = obs + torch.randn_like(obs) * scale
        return {'policy': obs.clamp(-100, 100)}

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clamp(-1, 1).clone()
        self.action_change = self.actions - self.previous_actions
        if self.training_mode:
            change = (self.episode_length_buf >= self.command_at).nonzero().flatten()
            self.targets[change, :2] = (torch.rand((len(change), 2), device=self.device) * 2 - 1) * torch.tensor([0.3, 0.8], device=self.device)
            self.command_at[change] = self.episode_length_buf[change] + torch.randint(100, 201, (len(change),), device=self.device)
        limits = torch.tensor([0.6, 1.5, 0.0], device=self.device) * 0.02
        self.commands += (self.targets - self.commands).clamp(-limits, limits)
        active = (self.episode_length_buf >= self.push_at) & (self.episode_length_buf < self.push_at + 5)
        if not self.pushes_enabled:
            active[:] = False
        self.last_force = (self.push_vectors * active[:, None]).unsqueeze(1)
        offset = torch.tensor([0.0, 0.08, 0.0], device=self.device).expand_as(self.last_force)
        torque = torch.cross(offset, self.last_force, dim=-1)
        self.robot.set_external_force_and_torque(self.last_force, torque, body_ids=self.base_ids)
        self.push_total += active.sum()

    def _apply_action(self):
        (q, qd) = (self.robot.data.joint_pos, self.robot.data.joint_vel)
        force = -qd * self.damping
        force[:, self.motor_ids[:4]] = (30 * (0.18 * self.actions[:, :4] * self.legsign - q[:, self.motor_ids[:4]]) - 0.8 * qd[:, self.motor_ids[:4]]).clamp(-3, 3)
        force[:, self.motor_ids[4:]] = 0.8 * self.actions[:, 4:] * self.wheelsign
        self.robot.set_joint_effort_target(force)

    def _get_dones(self):
        upright = -self.gravity()[:, 2]
        relative = self.robot.data.root_pos_w - self.scene.env_origins
        done = (upright < 0.65) | (relative[:, 2] < 0.06) | (relative[:, :2].abs().max(dim=-1).values > self.travel_limit)
        done |= ~torch.isfinite(self.robot.data.joint_vel).all(dim=-1)
        return (done, self.episode_length_buf >= self.max_episode_length - 1)

    def _get_rewards(self):
        (linear, angular) = self.motion()
        (vx, yaw) = (linear[:, 0] - self.commands[:, 0], angular[:, 2] - self.commands[:, 1])
        upright = -self.gravity()[:, 2]
        reward = 1 + torch.exp(-10 * (1 - upright)) + 2 * torch.exp(-vx.square() / 0.04) + 1.5 * torch.exp(-yaw.square() / 0.25) - 0.5 * vx.square().clamp(max=16) - 0.2 * yaw.square().clamp(max=16) - 0.3 * angular[:, :2].square().sum(dim=-1).clamp(max=16) - 0.1 * linear[:, 1:].square().sum(dim=-1).clamp(max=16) - 0.002 * self.actions.square().sum(dim=-1) - 0.005 * self.action_change.square().sum(dim=-1) - 8 * self.reset_terminated.float()
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.robot.reset(env_ids)
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self.scene.env_origins[env_ids]
        state[:, 2] += 0.001
        if self.randomize_reset:
            pitch = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * 0.02
            zeros = torch.zeros_like(pitch)
            state[:, 3:7] = quat_mul(quat_from_euler_xyz(zeros, pitch, zeros), state[:, 3:7])
        self.robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids].clone(), self.robot.data.default_joint_vel[env_ids].clone(), env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        self.targets[env_ids] = 0.0
        self.command_at[env_ids] = 50
        self.push_at[env_ids] = torch.randint(100, 201, (len(env_ids),), device=self.device)
        angle = torch.rand(len(env_ids), device=self.device) * 6.283185307
        magnitude = torch.rand(len(env_ids), device=self.device) * 7.5 + 2.5
        self.push_vectors[env_ids] = torch.stack((magnitude * torch.cos(angle), torch.zeros_like(angle), -magnitude * torch.sin(angle)), dim=-1)
