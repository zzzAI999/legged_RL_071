"""Isolated v10-to-v11 change of roll/pitch angular velocity cost."""
import torch
from .standing_leg_velocity_reward_v1 import REWARD_VERSION as SOURCE_REWARD_VERSION, validate_velocity_restore
REWARD_VERSION = 'standing_v11_body_gyro_weight2_v1'
GYRO_WEIGHT = 2.0
OLD_GYRO_WEIGHT = 0.5
DECISION_DT = 0.02
GYRO_SQUARE_CAP = 25.0

def base_gyro_cost(gyro):
    return -DECISION_DT * GYRO_WEIGHT * gyro.square().clamp(max=GYRO_SQUARE_CAP)

def validate_gyro_restore(checkpoint, target_reward=REWARD_VERSION, migrate=False):
    assert target_reward == REWARD_VERSION and migrate
    validate_velocity_restore(checkpoint, SOURCE_REWARD_VERSION, migrate=False)
    return dict(version='explicit_v10_to_v11_body_gyro_weight2_v1', source_reward=SOURCE_REWARD_VERSION, target_reward=REWARD_VERSION, changed_term='base_gyro', old_weight_before_dt=OLD_GYRO_WEIGHT, new_weight_before_dt=GYRO_WEIGHT, model_policy='retain_actor_critic_and_exploration_exactly', optimizer_policy='restore_complete_Adam_as_warm_start_for_changed_reward', critic_policy='retain_source_value_and_Adam_then_fit_new_reward_targets', curriculum_policy='retain_level_and_history_reset_consecutive_gate_count', rng_policy='restore_all_source_RNG_then_fresh_physical_reset', physical_rollout_restored=False, homogeneous_reward_continuation=False, guard_protocol_migrated=False)

def gyro_contract():
    return dict(reward_version=REWARD_VERSION, source_reward_version=SOURCE_REWARD_VERSION, changed_term='base_gyro', weight_before_dt=GYRO_WEIGHT, previous_weight_before_dt=OLD_GYRO_WEIGHT, decision_dt=DECISION_DT, formula='-0.02 * 2.0 * min(roll_pitch_gyro_norm^2, 25)', scope='same measured roll/pitch norm and 20ms endpoint as v10', applies_to_unsafe_and_terminal=True, other_21_terms_preserved=True, quiet_gyro_width_squared=0.09, quiet_gyro_weight=2.0, qualification_gyro_rad_s=0.3, qualification_changed=False, physical_model_and_action_units_changed=False, hypothesis='stronger direct cost may reduce angular motion; no success guarantee')
