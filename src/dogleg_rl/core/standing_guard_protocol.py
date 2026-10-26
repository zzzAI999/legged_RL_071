"""Explicit guard-state migration and evaluation return accounting."""
GUARD_V1 = 'latched_1ms_interval_or_v1'
GUARD_V2 = 'latched_1ms_endpoint_before_done_v2'
RETURN_ACCOUNTING_V2 = 'preserve_completed_training_returns_v2'

def guard_version(checkpoint):
    return checkpoint.get('training_guard_version', checkpoint.get('control_contract', {}).get('training_guard_version', 'ungated_v8'))

def validate_guard_restore(checkpoint, target_version, migrate_v1=False):
    source_version = guard_version(checkpoint)
    if migrate_v1:
        assert source_version == GUARD_V1 and target_version == GUARD_V2
        assert checkpoint.get('controller_version') == 'standing_v8'
        assert checkpoint.get('optimization_version') == 'standing_v8_bounded_adaptive_ppo'
        assert checkpoint.get('reward_version') == 'standing_v8_guard_aligned_v1'
        assert checkpoint.get('resumable') and checkpoint.get('training_state')
        assert checkpoint['training_state'].get('simulator_state_saved') is False
        course = checkpoint['training_state']['curriculum']
        assert course['level'] == 0 and course['consecutive_passes'] == 0, 'Do not transfer successful gates across guard protocols'
        return 'v8_guard_v1_to_endpoint_v2_explicit_state_migration'
    assert source_version == target_version, 'Guard changed; require explicit validated migration'
    return 'v8_bounded_adaptive_training_state'

def reset_after_evaluation(env, completed_training_returns):
    result = env.reset()
    env.completed_returns = list(completed_training_returns)
    return result
