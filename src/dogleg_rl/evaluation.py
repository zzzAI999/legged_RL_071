"""Deterministic paired standing evaluation with the unchanged acceptance criteria."""
import json
from pathlib import Path
import time


def evaluate(env, actor, out, seed, seconds=35.02):
    import torch
    from .core.standing_math_guard import stable_criteria, sustained_pass
    from .core.standing_training_state import model_identity
    from .core.standing_guard_protocol import reset_after_evaluation

    out = Path(out)
    out.mkdir(exist_ok=False)
    original = (env.cfg.evaluation, env.cfg.profile, env.training_mode,
                env.level, env.hardware_gate_mask.clone(), env.cfg.episode_length_s)
    completed_returns = tuple(env.completed_returns)
    env.cfg.evaluation = True
    env.cfg.profile = 'nominal' if env.level < 2 else 'mixed'
    env.training_mode = False
    env.cfg.episode_length_s = seconds+.2
    cases, half = 512, 256
    assert env.num_envs >= cases and env.num_envs % cases == 0
    env.hardware_gate_mask[:] = False
    env.hardware_gate_mask[half:] = True
    torch.manual_seed(seed)
    pool = env.pose_pool(env.level)
    if env.level:
        candidates = torch.arange(len(env.atlas_label), device=env.device)
        bound = .15 if env.level < 3 else .35 if env.level == 3 else 1.15
        candidates = candidates[(candidates % 3 == 0) & (env.pose_magnitude <= bound)]
        candidates = candidates[torch.tensor([not env.atlas_label[i].startswith('v') for i in candidates.tolist()], device=env.device)]
        if len(candidates):
            pool = candidates
    rows = pool[torch.randint(len(pool), (half,), device=env.device)]
    env.override_rows = torch.cat((rows, rows)).repeat(env.num_envs//cases)
    obs, _ = env.reset(seed=seed)
    env.override_rows = None
    env.reset_velocity[half:cases] = env.reset_velocity[:half]
    env.robot.write_root_velocity_to_sim(env.reset_velocity[half:cases], torch.arange(half, cases, device=env.device))
    env.measure_cache_tick = -1
    obs = env._get_observations()
    steps = round(seconds/.02)
    hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    longest = torch.zeros_like(hold)
    bad = {key: torch.zeros_like(hold) for key in stable_criteria(env.measures(), env.mass_scale*5.5*9.8, env.safety_latched)}
    identity = model_identity(actor)
    was_training = actor.training
    actor.eval()
    started = time.monotonic()
    with torch.no_grad():
        for step in range(steps):
            action = actor.act_inference(obs['policy']).clamp(-1., 1.)
            obs, _, _, timeout, _ = env.step(action)
            assert not timeout.any(), 'Evaluation reset is prohibited'
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(env.robot.data.joint_vel).all()
            tests = stable_criteria(env.measures(), env.mass_scale*5.5*9.8, env.safety_latched)
            stable = torch.stack(list(tests.values()), -1).all(-1)
            hold = torch.where(stable, hold+1, 0)
            longest = torch.maximum(longest, hold)
            if step >= 250:
                for key in bad:
                    bad[key] += (~tests[key]).long()
    passed = sustained_pass(hold, steps, env.safety_latched)[:cases]
    result = dict(level=env.level, seed=seed, seconds=steps*.02, elapsed_s=time.monotonic()-started,
        model_sha256=identity, soft=dict(passed=int(passed[:half].sum()), cases=half),
        protected=dict(passed=int(passed[half:].sum()), cases=half),
        case_passed=passed.cpu().tolist(),
        bad_counts_after5s={key: value[:cases].cpu().tolist() for key, value in bad.items()},
        safety_stop=env.safety_latched[:cases].cpu().tolist(), would_stop=env.would_stop[:cases].cpu().tolist(),
        final_hold_s=(hold[:cases].float()*.02).cpu().tolist(),
        longest_hold_s=(longest[:cases].float()*.02).cpu().tolist(),
        evaluation_env_ids=list(range(cases)), training_grid_envs=env.num_envs,
        paired_poses=True, exact_random_noise_pair=False, ready_for_hardware=False)
    result['both_arms_pass'] = result['soft']['passed'] >= 244 and result['protected']['passed'] >= 244
    result['final_qualified'] = env.level == 4 and seconds >= 120 and result['both_arms_pass']
    (out/'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    env.cfg.evaluation, env.cfg.profile, env.training_mode, env.level, gates, env.cfg.episode_length_s = original
    env.hardware_gate_mask.copy_(gates)
    actor.train(was_training)
    obs, _ = reset_after_evaluation(env, completed_returns)
    return result, obs
