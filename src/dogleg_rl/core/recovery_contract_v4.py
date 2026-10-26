"""Candidate v4 contract; the independent skill thresholds remain frozen."""
from .recovery_contract_v3 import contract_v3

def contract_v4():
    value = contract_v3()
    value.update(schema='dogleg_random_recovery_controller_v4', leg_begin_relative_hard_guard_rad=1.48, leg_requested_target_begin_relative_limit_rad=1.45, leg_target_margin_is_design_assumption=True, begin_angle=dict(source='C-board can_recovery_begin frozen measured model-angle snapshot', simulation='position_start+q_bias; no per-sample angle noise in frozen begin', freeze_timestamp_and_valid_required=True, host_must_not_substitute_last_uart_angle=True, new_state_protocol_required=True), target_clamp_order='actor 50Hz request absolute model target -> measured begin q +/-1.45rad -> UART delay -> 500Hz 1.5rad/s slew -> external PD', training_changes=dict(potential='2*upright+clipped_height+clipped_base_clearance', shaping='6*(.999*next_potential-previous_potential); terminal potential=0', dense_tilt_and_outward_rotation_penalty=True, reset_cohort_percent=dict(near=37.5, hard=37.5, historical=25.0), rollout_steps=64, gamma=0.999, lam=0.98, finite_episode_timeout_bootstrap=False, critic_and_optimizer_reset_on_v3_transfer=True, wheel_exploration_std=0.25), physics_asset_changed=False, acceptance_thresholds_unchanged=True, qualified_deployment=False, ready_for_hardware=False)
    value['deployment_changes'] += ['apply the same 50Hz begin-relative target clamp before UART delay; retain existing actual position hard stop at 1.48rad']
    return value
