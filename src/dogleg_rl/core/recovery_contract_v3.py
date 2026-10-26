"""Candidate executable recovery contract; unchanged independent acceptance."""
from .deployment_contract import deployment_contract

def contract_v3():
    value = deployment_contract()
    value.update(schema='dogleg_random_recovery_controller_v3', actor_output_transform='tanh before Gaussian exploration; clamp sampled actions to [-1,1]', wheel_speed_current_guard=dict(onset_rotor_rpm=2000.0, cutoff_rotor_rpm=2500.0, formula='scale outward nominal torque by clamp((2500-abs(measured_rotor_rpm))/500,0,1); retain inward braking torque', update_hz=500, units='RPM uses calibrated model-axis speed and retained ratio 19'), wheel_guard_before_quantization_and_plant_strength=True, previous_action='same completed nominal quantized ACK as deployment_v2, after wheel speed-current guard', firmware_change_required=True, existing_firmware_compatible=False, armature_changed=False, uncalibrated_armature_probe_not_adopted=True, acceptance_thresholds_unchanged=True, ready_for_hardware=False, qualified_deployment=False)
    value['deployment_changes'] += ['implement same 500Hz directional wheel-current guard before integer C620 quantization and ACK']
    return value
