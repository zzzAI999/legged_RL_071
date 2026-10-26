"""Versioned prelimit leg-speed shaping with unchanged physical hard guards."""
import torch
from .standing_guard_protocol import GUARD_V2, guard_version
REWARD_VERSION = 'standing_v9_leg_speed_margin_v1'
BASE_REWARD_VERSION = 'standing_v8_guard_aligned_v1'
SOFT_LEG_SPEED = 4.5
HARD_LEG_SPEED = 6.0
MARGIN_WEIGHT = 0.5
DT = 0.02

def leg_speed_margin(peak_speeds):
    excess = ((peak_speeds.abs() - SOFT_LEG_SPEED) / (HARD_LEG_SPEED - SOFT_LEG_SPEED)).clamp(0.0, 1.0)
    return -DT * MARGIN_WEIGHT * excess.square().sum(-1)

def validate_reward_restore(checkpoint, target_reward, migrate=False):
    source_reward = checkpoint.get('reward_version')
    if not migrate:
        assert source_reward == target_reward, 'Reward changed; require an explicit reward migration'
        return None
    assert source_reward == BASE_REWARD_VERSION and target_reward == REWARD_VERSION
    assert checkpoint.get('controller_version') == 'standing_v8'
    assert checkpoint.get('optimization_version') == 'standing_v8_bounded_adaptive_ppo'
    assert guard_version(checkpoint) == GUARD_V2
    assert checkpoint.get('resumable') and checkpoint.get('training_state')
    assert checkpoint['training_state'].get('simulator_state_saved') is False
    return dict(version='explicit_leg_speed_margin_reward_migration_v1', source_reward=source_reward, target_reward=target_reward, model_policy='retain_actor_critic_and_exploration_exactly', optimizer_policy='restore_complete_Adam_as_warm_start_for_changed_reward', critic_policy='warm_start_old_value_estimates_then_fit_changed_reward_targets', curriculum_policy='retain_level_and_history_reset_consecutive_gate_count', rng_policy='restore_source_RNG_then_fresh_physical_reset', homogeneous_reward_continuation=False)
