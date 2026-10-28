"""Lossless decision-input capture; no actor forward, RNG draw, or simulator advance."""
import hashlib
import io
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import zipfile

import numpy as np
import torch


CRITERIA = ('tilt', 'speed', 'gyro', 'height', 'clearance', 'each_wheel', 'support', 'nonwheel', 'hardware', 'yaw')
MOTORS = ('left_front1_joint', 'left_rear1_joint', 'right_front1_joint', 'right_rear1_joint', 'right_rear4_joint', 'left_rear4_joint')


def observation_contract():
    features = []
    def add(name, unit, scale, sign=1):
        features.append(dict(index=len(features), name=name, physical_unit=unit,
                             input_equals_physical_times=scale*sign))
    for axis in 'xyz': add('body_gyro_'+axis, 'rad/s', .2)
    for axis in 'xyz': add('body_unit_gravity_'+axis, 'dimensionless', 1.)
    for name, unit, scale in [('command_vx', 'm/s', 1.), ('command_yaw', 'rad/s', .2), ('command_reserved', 'dimensionless', 1.)]:
        add(name, unit, scale)
    for name, sign in zip(MOTORS[:4], [1, 1, -1, -1]): add('leg_position_'+name, 'model-offset rad', 2., sign)
    for name, scale in zip(MOTORS, [.05]*4+[.03]*2): add('motor_speed_'+name, 'joint rad/s', scale)
    for name in MOTORS: add('sampled_completed_action_'+name, 'normalized action', 1.)
    for name in ['jump_height_div_0p1', 'jump_phase_sin_active', 'jump_phase_cos_active', 'jump_active']:
        add(name, 'dimensionless; zero for this standing task', 1.)
    assert len(features) == 29
    return dict(schema='dogleg_exact_actor_observation29_v1', dtype='float32', features=features,
        policy_hz=50, physics_hz=1000, final_input_clip=[-100., 100.],
        coordinate_frame='BODY; gyro/gravity use the existing INS-to-BODY adapter contract',
        motor_order=list(MOTORS), wheel_speed_is_joint_not_motor_rotor=True,
        previous_action_source='existing delayed sampled ack_history, final policy columns19:25; phase/units unchanged',
        original_sensor_latency_and_noise_preserved=True,
        source_sensor_not_reconstructed_from_policy=True,
        first_leg_positions_are_model_offsets_not_absolute_encoder_angles=True,
        original_encoder_zero_scale_verified=False, ready_for_hardware=False,
        scope='each successfully returned 50Hz control actor call; PPO minibatch reuse is not a new sensor sample',
        criteria_bit_order=list(CRITERIA),
        training_success='episode timeout, no termination/safety stop, full criteria continuous>=30s entered within5s',
        evaluation_success='actual independent evaluation.case_passed; labels committed after the complete evaluation',
        labels_are_not_final_hardware_qualification=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class Observation29Recorder:
    def __init__(self, out, num_envs, source_updates, source_sha, chunk_frames=8, queue_chunks=2, disk_reserve_bytes=8*2**30):
        self.root = Path(out)/'observation29'
        self.root.mkdir(exist_ok=False)
        (self.root/'models').mkdir()
        self.num_envs = num_envs
        self.chunk_frames = chunk_frames
        self.disk_reserve_bytes = disk_reserve_bytes
        self.contract = observation_contract()
        raw = json.dumps(self.contract, sort_keys=True, separators=(',', ':')).encode()
        self.contract_sha = hashlib.sha256(raw).hexdigest()
        atomic_json(self.root/'contract.json', dict(self.contract, canonical_SHA256=self.contract_sha))
        atomic_json(self.root/'coverage_start.json', dict(source_cumulative_updates=source_updates,
            source_checkpoint_SHA256=source_sha, past_unrecorded_inputs_reconstructed=False,
            capture_starts_only_with_this_run=True, every_training_environment=True,
            every_scored_evaluation_environment=True, row_subsampling=False))
        self.work = queue.Queue(maxsize=queue_chunks)
        self.error = None
        self.buffer = []
        self.pending = None
        self.context = None
        self.chunk_serial = 0
        self.frame_serial = 0
        self.committed_frames = 0
        self.committed_rows = 0
        self.closed = False
        self.hold = np.zeros(num_envs, dtype=np.int64)
        self.previous_serial = np.full(num_envs, -1, dtype=np.int64)
        self.worker = threading.Thread(target=self._writer, name='observation29-disk', daemon=False)
        self.worker.start()

    def _check(self):
        if self.error is not None: raise RuntimeError('Observation29 storage failure; stop without dropping inputs') from self.error
        if self.closed: raise RuntimeError('Observation29 recorder already closed')

    def _writer(self):
        while True:
            item = self.work.get()
            try:
                if item is None: return
                if self.error is not None: continue
                number, context, frames = item
                arrays = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
                arrays['env_ids'] = np.asarray(context['env_ids'], dtype=np.int32)
                arrays['metadata_json'] = np.asarray(json.dumps(context, sort_keys=True))
                estimate = sum(value.nbytes for value in arrays.values())+2**20
                if shutil.disk_usage(self.root).free < self.disk_reserve_bytes+estimate:
                    raise OSError('Observation29 disk reserve below8GiB plus the pending complete chunk')
                name = 'chunk_%06d.npz' % number
                target = self.root/name
                temporary = target.with_suffix('.tmp')
                with temporary.open('xb') as destination:
                    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                        for key, array in arrays.items():
                            with z.open(key+'.npy', 'w', force_zip64=True) as f:
                                np.lib.format.write_array(f, array, allow_pickle=False)
                    destination.flush()
                temporary.replace(target)
                entry = dict(file=name, sha256=sha(target), bytes=target.stat().st_size,
                    first_frame=int(frames[0]['frame_serial']), last_frame=int(frames[-1]['frame_serial']),
                    frames=len(frames), rows=len(frames)*len(context['env_ids']), mode=context['mode'],
                    model_sha256=context['model_sha256'], cumulative_policy_updates=context['cumulative_policy_updates'])
                with (self.root/'chunks.jsonl').open('a', encoding='utf-8') as manifest:
                    manifest.write(json.dumps(entry)+'\n')
                    manifest.flush()
                self.committed_frames += entry['frames']
                self.committed_rows += entry['rows']
                atomic_json(self.root/'progress.json', dict(committed_frames=self.committed_frames,
                    committed_rows=self.committed_rows, latest_chunk=entry, queue_is_bounded=True,
                    dropped_rows=0, full_run_complete=False))
            except BaseException as error:
                self.error = error
            finally:
                self.work.task_done()

    def _flush(self):
        self._check()
        if self.buffer:
            self.chunk_serial += 1
            self.work.put((self.chunk_serial, dict(self.context), self.buffer))
            self.buffer = []
        self._check()

    def sync(self):
        self._flush()
        self.work.join()
        self._check()

    def begin(self, mode, actor, cumulative_policy_updates, model_checkpoint, *, evaluation=None):
        from .core.standing_training_state import model_identity
        self.sync()
        self.context = dict(schema='dogleg_exact_actor_observation29_chunk_v1', mode=mode,
            observation_contract_SHA256=self.contract_sha, model_sha256=model_identity(actor),
            model_checkpoint=str(model_checkpoint), model_checkpoint_SHA256=sha(model_checkpoint),
            cumulative_policy_updates=int(cumulative_policy_updates),
            env_ids=list(range(self.num_envs if mode == 'training' else 512)),
            action_mean_is_deterministic_network_output=True, sampled_action_is_training_exploration=(mode=='training'),
            evaluation=evaluation, observation_sample_hz=50, rows_are_time_then_env=True)
        atomic_json(self.root/'current_context.json', self.context)

    @staticmethod
    def _cpu(value, dtype=None):
        result = value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()
        return result.astype(dtype, copy=False) if dtype is not None else result

    def accepted(self, env, observation, action, action_mean):
        self._check()
        assert self.context is not None and self.pending is None
        n = len(self.context['env_ids'])
        obs = self._cpu(observation[:n], np.float32)
        sampled = self._cpu(action[:n], np.float32)
        mean = self._cpu(action_mean[:n], np.float32)
        assert obs.shape == (n, 29) and sampled.shape == mean.shape == (n, 6)
        assert np.isfinite(obs).all() and np.isfinite(sampled).all() and np.isfinite(mean).all()
        serial = self._cpu(env.reset_serial[:n], np.int64)
        changed = self.previous_serial[:n] != serial
        self.hold[:n][changed] = 0
        self.previous_serial[:n] = serial
        at_limits = np.abs(obs) >= 100.
        self.frame_serial += 1
        self.pending = dict(frame_serial=np.int64(self.frame_serial), observation=obs,
            action_sampled=sampled, action_mean=mean,
            episode_serial=serial, episode_step_before=self._cpu(env.episode_length_buf[:n], np.int32),
            control_tick_before=np.int64(env.control_tick), actor_result_host_UTC_ns=np.int64(time.time_ns()),
            actor_result_host_monotonic_ns=np.int64(time.monotonic_ns()),
            input_at_clip_limit_bits=(at_limits.astype(np.uint32)*(np.uint32(1)<<np.arange(29, dtype=np.uint32))).sum(-1, dtype=np.uint32),
            sampled_sensor_latency_ms=self._cpu(env.sensor_latency_ms[:n], np.uint8),
            sensor_history_index=((int(env.sensor_cursor)-(self._cpu(env.sensor_latency_ms[:n], np.int64)+9)//10)%51).astype(np.uint8))

    def attach(self, env):
        from .core.standing_math_guard import stable_criteria, sustained_pass
        original = env._get_rewards
        assert not hasattr(env, '_observation29_recorder')
        env._observation29_recorder = self
        def rewards_with_record():
            reward = original()
            if self.pending is not None:
                n = len(self.context['env_ids'])
                tests = stable_criteria(env.measures(), env.mass_scale*5.5*9.8, env.safety_latched)
                assert tuple(tests) == CRITERIA
                matrix = self._cpu(torch.stack([tests[key][:n] for key in CRITERIA], -1), np.bool_)
                bits = (matrix.astype(np.uint16)*(np.uint16(1)<<np.arange(len(CRITERIA), dtype=np.uint16))).sum(-1, dtype=np.uint16)
                good = matrix.all(-1)
                self.hold[:n] = np.where(good, self.hold[:n]+1, 0)
                after = self._cpu(env.episode_length_buf[:n], np.int32)
                stopped = self._cpu(env.safety_latched[:n], np.bool_)
                terminated = self._cpu(env.reset_terminated[:n], np.bool_)
                timeout = self._cpu(env.reset_time_outs[:n], np.bool_)
                full_pass = (self.hold[:n] >= 1500) & ((after-self.hold[:n])*.02 <= 5.) & ~stopped
                native_pass = self._cpu(sustained_pass(env.hold_steps[:n], env.episode_length_buf[:n], env.safety_latched[:n]), np.bool_)
                frame = self.pending
                assert int(env.control_tick)-int(frame['control_tick_before']) == 20
                frame.update(control_tick_after=np.int64(env.control_tick), episode_step_after=after,
                    applied_action_after=self._cpu(env.applied[:n], np.float32), reward=self._cpu(reward[:n], np.float32),
                    post_criteria_bits=bits, hold_steps_all_criteria=self.hold[:n].astype(np.int32).copy(),
                    native_hold_steps=self._cpu(env.hold_steps[:n], np.int32),
                    episode_terminated=terminated, episode_timeout=timeout, safety_latched=stopped,
                    would_stop=self._cpu(env.would_stop[:n], np.bool_),
                    native_training_episode_success=native_pass & timeout & ~terminated,
                    all_criteria_training_episode_success=full_pass & timeout & ~terminated)
                self.pending = None
                self.buffer.append(frame)
                if len(self.buffer) >= self.chunk_frames: self._flush()
            return reward
        env._get_rewards = rewards_with_record

    def evaluation_labels(self, result):
        self.sync()
        assert self.context['mode'] == 'evaluation'
        assert result['model_sha256'] == self.context['model_sha256']
        assert len(result['case_passed']) == 512
        target = self.root/('evaluation_labels_%06d.json' % self.context['cumulative_policy_updates'])
        assert not target.exists()
        atomic_json(target, dict(result, observation_contract_SHA256=self.contract_sha,
            model_checkpoint=self.context['model_checkpoint'], capture_uses_actual_case_labels=True,
            inputs_can_be_selected_by_env_id_without_row_subsampling=True))

    def close(self, reason):
        if self.closed: return
        issue = None
        try:
            self.sync()
            if self.pending is not None: issue = 'Actor call accepted but its transition did not complete; pending input not labeled a completed transition'
        except BaseException as error:
            issue = repr(error)
        finally:
            self.work.put(None)
            self.work.join()
            self.worker.join()
            self.closed = True
            atomic_json(self.root/'recording_receipt.json', dict(reason=reason,
                accepted_actor_calls=self.frame_serial, committed_frames=self.committed_frames,
                committed_rows=self.committed_rows, issue=issue, dropped_rows=0 if issue is None else None,
                full_capture_of_completed_transitions=(issue is None), ready_for_hardware=False))
        if issue is not None: raise RuntimeError('Observation29 recording incomplete: '+issue)
