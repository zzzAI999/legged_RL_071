"""Continuous measured-leg-speed cost added to the preserved v9 objective."""
import math
import torch
from .standing_guard_protocol import GUARD_V2, RETURN_ACCOUNTING_V2, guard_version
from .standing_margin_reward import REWARD_VERSION as BASE_REWARD_VERSION
REWARD_VERSION = 'standing_v10_leg_velocity_l2_1ms_v1'
LEG_VELOCITY_WEIGHT = 0.01
PHYSICS_DT = 0.001
DECISION_SAMPLES = 20
SAMPLING_VERSION = 'right_endpoint_20_unique_1ms_leg_speed_squares_v1'

def leg_velocity_cost(square_integral):
    assert square_integral.shape[-1] == 4
    assert torch.isfinite(square_integral).all() and (square_integral >= 0).all()
    return -LEG_VELOCITY_WEIGHT * square_integral.sum(-1)

def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all((finite_tree(v) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return all((finite_tree(v) for v in value))
    if isinstance(value, float):
        return math.isfinite(value)
    return True

def validate_velocity_restore(checkpoint, target_reward=REWARD_VERSION, migrate=False):
    assert target_reward == REWARD_VERSION
    assert checkpoint.get('controller_version') == 'standing_v8'
    assert checkpoint.get('optimization_version') == 'standing_v8_bounded_adaptive_ppo'
    assert guard_version(checkpoint) == GUARD_V2
    assert checkpoint.get('return_accounting_version') == RETURN_ACCOUNTING_V2
    assert checkpoint.get('resumable') and checkpoint.get('training_state')
    assert checkpoint.get('model_state_dict') and finite_tree(checkpoint['model_state_dict'])
    state = checkpoint['training_state']
    assert state.get('simulator_state_saved') is False
    optimizer = state['optimizer_state_dict']
    assert optimizer['state'] and optimizer['param_groups'] and finite_tree(optimizer)
    assert all((1e-05 <= group['lr'] <= 0.0003 for group in optimizer['param_groups']))
    assert all(({'step', 'exp_avg', 'exp_avg_sq'} <= set(item) for item in optimizer['state'].values()))
    course = state['curriculum']
    assert course['level'] == checkpoint['curriculum_level']
    assert isinstance(course['history'], list) and course['consecutive_passes'] >= 0
    rng = state['rng']
    assert all((rng.get(k) is not None for k in ('python', 'numpy', 'torch_cpu', 'torch_device')))
    assert finite_tree(rng)
    source_reward = checkpoint.get('reward_version')
    if not migrate:
        assert source_reward == target_reward, 'Changed reward requires explicit v9-to-v10 migration'
        return None
    assert source_reward == BASE_REWARD_VERSION, 'Only the preserved v9 reward may migrate here'
    return dict(version='explicit_v9_to_v10_leg_velocity_reward_migration_v1', source_reward=source_reward, target_reward=target_reward, model_policy='retain_actor_critic_and_exploration_exactly', optimizer_policy='restore_complete_Adam_as_warm_start_for_changed_reward', critic_policy='retain_source_value_and_Adam_then_fit_new_reward_targets', curriculum_policy='retain_level_and_history_reset_consecutive_gate_count', rng_policy='restore_all_source_RNG_then_fresh_physical_reset', physical_rollout_restored=False, homogeneous_reward_continuation=False, guard_protocol_migrated=False)

def velocity_contract():
    return dict(reward_version=REWARD_VERSION, base_reward_version=BASE_REWARD_VERSION, sampling_version=SAMPLING_VERSION, weight_before_dt=LEG_VELOCITY_WEIGHT, joint_scope='four measured actuated leg speeds; exclude wheel and passive joints', formula='-0.01 * sum_legs(sum_k=1..20(0.001 * measured_qdot[k]^2))', quadrature='right endpoints at 1..20 ms; initial 0 ms excluded; endpoint counted once', penalty_applies_to_unsafe_and_terminal_decisions=True, existing_peak_margin_and_all_hard_guards_retained=True, physical_model_and_action_units_changed=False, new_acceleration_or_action_penalty=False)
