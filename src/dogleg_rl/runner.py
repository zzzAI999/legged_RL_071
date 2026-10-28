"""Portable execution of the current protected environment and bounded PPO."""
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import traceback


def run(args):
    # Isaac Sim must start before importing Isaac Lab or pxr-backed environments.
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=not args.gui, active_gpu=args.gpu, physics_gpu=args.gpu,
                             multi_gpu=False, create_new_stage=False))
    env = None
    observer = None
    out = args.out
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def stopping():
        return stop_requested or (out/'STOP_AFTER_UPDATE').exists()

    def status(phase, **values):
        record = dict(values)
        record.update(phase=phase, at=time.time())
        temporary = out/'receipt.tmp'
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(out/'receipt.json')
        print(json.dumps(record), flush=True)

    try:
        from isaacsim.core.utils.stage import create_new_stage
        create_new_stage()
        import numpy as np
        import torch
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(4)
        from pxr import UsdPhysics
        from .core.standing_environment_gyro_penalty_v1 import StandingCfg, StandingEnv
        from .core.policy_architecture import actor_critic, checkpoint_architecture
        from .core.standing_gyro_penalty_reward_v1 import REWARD_VERSION, gyro_contract
        from .core.standing_leg_velocity_reward_v1 import velocity_contract
        from .core.standing_training_state import capture_training_state, model_identity
        from .core.standing_leg_velocity_restore_v1 import restore_velocity_state
        from .core.standing_math_guard import SuccessCurriculum
        from .core.bounded_adaptive_ppo import BoundedAdaptivePPO, OPTIMIZATION_VERSION, LR_MIN, LR_MAX
        from .core.standing_guard_protocol import GUARD_V2, RETURN_ACCOUNTING_V2
        from .core.recovery_contract_v4 import contract_v4
        from .evaluation import evaluate
        from .observation29_recording_v1 import Observation29Recorder
        random.seed(125021)
        np.random.seed(125021)
        torch.manual_seed(125021)
        cfg = StandingCfg()
        cfg.sim.device = 'cuda:' + str(args.gpu)
        cfg.scene.num_envs = args.num_envs
        cfg.scene.env_spacing = 2.
        cfg.seed = 125021
        cfg.profile = 'nominal'
        cfg.evaluation = args.mode == 'evaluate'
        cfg.audit_directory = str(out)
        env = StandingEnv(cfg)
        scenes = [p for p in env.sim.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        assert len(scenes) == 1
        assert cfg.sim.physx.solver_type == 1 and scenes[0].GetAttribute('physxScene:solverType').Get() == 'TGS'
        assert cfg.sim.physx.min_position_iteration_count == cfg.sim.physx.max_position_iteration_count == 32
        assert cfg.sim.physx.min_velocity_iteration_count == cfg.sim.physx.max_velocity_iteration_count == 8
        obs, _ = env.reset()
        assert obs['policy'].shape == (cfg.scene.num_envs, 29)
        assert obs['critic'].shape == (cfg.scene.num_envs, 39)
        checkpoint = None if args.fresh or args.mode == 'smoke' else torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        if checkpoint is not None:
            assert checkpoint['reward_version'] == REWARD_VERSION
            assert checkpoint['optimization_version'] == OPTIMIZATION_VERSION
            assert checkpoint['training_guard_version'] == GUARD_V2
            assert checkpoint['return_accounting_version'] == RETURN_ACCOUNTING_V2
        actor = actor_critic(checkpoint, device=env.device, bounded=True, hidden_dims=[256, 256])
        source_updates = int(checkpoint['iteration']) if checkpoint else 0
        source_transitions = int(checkpoint['cumulative_transitions']) if checkpoint else 0
        env.level = int(checkpoint['curriculum_level']) if checkpoint else 0
        curriculum = SuccessCurriculum()
        curriculum.level = env.level
        algorithm = None
        if args.mode == 'train':
            class EmptyEncoder:
                num_output_dim = 0

                def encode(self, history):
                    return history
            algorithm = BoundedAdaptivePPO(cfg.scene.num_envs, EmptyEncoder(), actor,
                num_learning_epochs=4, num_mini_batches=4, learning_rate=3e-4, schedule='adaptive',
                desired_kl=.01, early_stop=False, entropy_coef=.001, gamma=.995, lam=.95,
                use_clipped_value_loss=False, device=env.device)
            algorithm.init_storage(cfg.scene.num_envs, 64, [29], [39], [0], [0], [6])
            if checkpoint:
                assert checkpoint['resumable'] and checkpoint['training_state']
                proof = restore_velocity_state(checkpoint, algorithm, curriculum, env.device, actor)
                (out/'restore_proof.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
            else:
                head = [m for m in actor.actor if isinstance(m, torch.nn.Linear)][-1]
                torch.nn.init.orthogonal_(head.weight, gain=.01)
                head.bias.data.zero_()
                actor.logstd.data[:4].fill_(math.log(.03))
                actor.logstd.data[4:].fill_(math.log(.20))
        contract = contract_v4()
        contract.update(schema='dogleg_standing_portable_guard17', reward=REWARD_VERSION,
            training_guard_version=GUARD_V2, return_accounting_version=RETURN_ACCOUNTING_V2,
            body_gyro_penalty=gyro_contract(), leg_velocity_l2_1ms=velocity_contract(),
            physics_dt=.001, decimation=20, solver='TGS32/8', training_envs=cfg.scene.num_envs,
            rollout_steps=64, epochs=4, minibatches=4, ppo_gamma=.995, ppo_lam=.95,
            source_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if checkpoint else None,
            source_updates=source_updates, source_decisions=source_transitions,
            maximum_updates=args.updates, maximum_minutes=args.minutes,
            scheduled_local_evaluation_update=args.evaluation_update,
            source_training_envs=(checkpoint.get('control_contract', {}).get('training_envs') if checkpoint else None),
            minibatch_samples=cfg.scene.num_envs*64//4,
            batch_geometry_may_differ_from_source=True,
            qualified_deployment=False, ready_for_hardware=False,
            portable_wrapper_is_new=True, simulator_state_restored=False,
            observation29_recording=dict(schema='dogleg_exact_actor_observation29_v1',
                all_training_envs=True, all_scored_evaluation_cases=True, policy_hz=50,
                lossless_float32=True, row_subsampling=False, disk_reserve_bytes=8*2**30))
        (out/'control_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
        architecture = checkpoint_architecture(dict(model_state_dict=actor.state_dict(), actor_output_transform='tanh'))

        def save(name, updates):
            payload = dict(model_state_dict=actor.state_dict(), iteration=source_updates+updates,
                run_updates=updates, run_transitions=updates*cfg.scene.num_envs*64,
                cumulative_transitions=source_transitions+updates*cfg.scene.num_envs*64,
                observations=29, actions=6, actor_output_transform='tanh', controller_version='standing_v8',
                network_architecture=architecture, curriculum_level=env.level, control_contract=contract,
                reward_version=REWARD_VERSION, optimization_version=OPTIMIZATION_VERSION,
                model_sha256=model_identity(actor), training_guard_version=GUARD_V2,
                return_accounting_version=RETURN_ACCOUNTING_V2,
                training_state=capture_training_state(algorithm, curriculum, env.device) if algorithm else None,
                resumable=algorithm is not None)
            target = out/name
            temporary = target.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(target)
            return target

        if args.mode in ('train', 'evaluate'):
            observer = Observation29Recorder(out, cfg.scene.num_envs, source_updates, contract['source_checkpoint_sha256'])
            observer.attach(env)
        if args.mode == 'smoke':
            zero = torch.zeros((cfg.scene.num_envs, 6), device=env.device)
            gaps = []
            for _ in range(20):
                obs, reward, _, _, _ = env.step(zero)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                gaps.append(float(env.pin_gaps().max()))
            assert max(gaps) < .003
            env.episode_length_buf[:] = env.max_episode_length-2
            obs, reward, _, timeout, info = env.step(zero)
            assert timeout.all() and info['terminal_critic'].shape == (cfg.scene.num_envs, 39)
            status('SMOKE_COMPLETE', optimizer_updates=0, max_pin_gap_m=max(gaps),
                   terminal_bootstrap_observation_checked=True)
            return
        if args.mode == 'evaluate':
            if args.seconds >= 120 and not (env.level == 4 and checkpoint['training_state']['curriculum']['consecutive_passes'] >= 2):
                raise RuntimeError('Final 120 second qualification requires course4 and two consecutive course gates')
            result, _ = evaluate(env, actor, out/'evaluation', args.seed, args.seconds,
                observer=observer, model_checkpoint=args.checkpoint, cumulative_updates=source_updates)
            observer.close('evaluation_complete')
            status('EVALUATION_COMPLETE', **{k: result[k] for k in ('soft','protected','seconds','level','final_qualified')})
            return
        save('checkpoint_initial.pt', 0)
        obs, _ = env.reset()
        empty = torch.empty((cfg.scene.num_envs, 0), device=env.device)
        updates = 0
        started = time.monotonic()
        with (out/'metrics.jsonl').open('x', encoding='utf-8', buffering=1) as log:
            while updates < args.updates and time.monotonic()-started < args.minutes*60 and not stopping():
                observer.begin('training', actor, source_updates+updates,
                    save('observation29/models/policy_%06d.pt' % (source_updates+updates), updates))
                update_started = time.monotonic()
                torch.cuda.reset_peak_memory_stats(env.device)
                reward_sum = bootstrap_sum = 0.
                reward_terms = {}
                before = max((int(v['step'].item()) for v in algorithm.optimizer.state.values()), default=0)
                with torch.no_grad():
                    for _ in range(64):
                        action = algorithm.act(obs['policy'], empty, empty, obs['critic'])
                        observer.accepted(env, obs['policy'], action, actor.action_mean)
                        obs, reward, done, timeout, info = env.step(action)
                        reward_sum += float(reward.mean())
                        if timeout.any():
                            value = actor.evaluate(info['terminal_critic']).squeeze(-1)
                            bootstrap = algorithm.gamma*value*timeout.float()*(~done).float()
                            bootstrap_sum += float(bootstrap.mean())
                            reward = reward+bootstrap
                        for key, value in env.reward_terms.items():
                            reward_terms[key] = reward_terms.get(key, 0.) + float(value.detach().mean())
                        assert int(env.guard_step_stats['unsafe_positive_reward_decisions']) == 0
                        algorithm.process_env_step(reward, (done|timeout).long(), {}, obs['policy'])
                    observer.sync()
                    algorithm.compute_returns(obs['critic'])
                rollout_finished = time.monotonic()
                losses = algorithm.update()
                optimizer_finished = time.monotonic()
                after = max((int(v['step'].item()) for v in algorithm.optimizer.state.values()), default=0)
                assert after-before == 16 and all(math.isfinite(float(v)) for v in losses)
                assert LR_MIN <= algorithm.learning_rate <= LR_MAX
                assert all(LR_MIN <= group['lr'] <= LR_MAX for group in algorithm.optimizer.param_groups)
                actor.logstd.data[:4].clamp_(math.log(.015), math.log(.08))
                actor.logstd.data[4:].clamp_(math.log(.06), math.log(.30))
                updates += 1
                row = dict(at=time.time(), update=updates, cumulative_updates=source_updates+updates,
                    cumulative_transitions=source_transitions+updates*cfg.scene.num_envs*64,
                    num_envs=cfg.scene.num_envs, update_transitions=cfg.scene.num_envs*64,
                    rollout_wall_s=rollout_finished-update_started,
                    optimizer_wall_s=optimizer_finished-rollout_finished,
                    update_wall_s=optimizer_finished-update_started,
                    decisions_per_second=cfg.scene.num_envs*64/(optimizer_finished-update_started),
                    CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(env.device),
                    CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(env.device),
                    mean_reward=reward_sum/64, timeout_bootstrap_mean=bootstrap_sum/64,
                    reward_terms={key: value/64 for key, value in reward_terms.items()},
                    losses=list(map(float, losses)), adam_steps=after, ppo_minibatches_this_update=after-before,
                    learning_rate=algorithm.learning_rate, level=env.level, wall_s=time.monotonic()-started)
                log.write(json.dumps(row)+'\n')
                status('TRAINING', **row)
                if updates % 10 == 0:
                    save('checkpoint_latest.pt', updates)
                if stopping():
                    break
                if updates == args.evaluation_update:
                    eval_checkpoint = save('checkpoint_pre_evaluation.pt', updates)
                    result, obs = evaluate(env, actor, out/f'eval_{updates:04d}', 125071+source_updates+updates,
                        observer=observer, model_checkpoint=eval_checkpoint, cumulative_updates=source_updates+updates)
                    promoted = curriculum.observe(result['soft']['passed'], 256, result['protected']['passed'])
                    if promoted:
                        env.level = curriculum.level
                        obs, _ = env.reset()
                    save('checkpoint_post_evaluation.pt', updates)
            final = save('checkpoint_final.pt', updates)
            observer.close('training_complete')
        (out/'curriculum_history.json').write_text(json.dumps(curriculum.history, indent=2), encoding='utf-8')
        status('STOPPED' if stopping() else 'TRAINING_COMPLETE', updates=updates,
            cumulative_updates=source_updates+updates, checkpoint=str(final),
            checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
            level=env.level, final_qualification_performed=False, ready_for_hardware=False)
    except BaseException:
        error_text = traceback.format_exc()
        error_checkpoint = None
        error_save_issue = None
        # Preserve a completed update if later reporting fails. A partial Adam update
        # is never relabeled as a complete checkpoint.
        if args.mode == 'train' and 'save' in locals() and 'updates' in locals() and algorithm is not None:
            try:
                steps = {int(value['step'].item()) for value in algorithm.optimizer.state.values()}
                if steps == {(source_updates+updates)*16}:
                    error_checkpoint = str(save('checkpoint_error.pt', updates))
            except BaseException:
                error_save_issue = traceback.format_exc()[-2000:]
        status('FAIL', error=error_text, complete_update_error_checkpoint=error_checkpoint,
               error_checkpoint_save_issue=error_save_issue)
        raise
    finally:
        if observer is not None and not observer.closed:
            try:
                observer.close('exception_or_shutdown')
            except BaseException as recording_error:
                print('Observation29 recording close issue: '+repr(recording_error), flush=True)
        try:
            if env is not None:
                env.close()
        finally:
            app.close()
