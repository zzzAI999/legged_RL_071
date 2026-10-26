"""Unchanged established thermal parser and policy; UUID supplied by the supervisor."""
import re
import xml.etree.ElementTree as ET
GPU_UUID = None

def number(text):
    if not text or text.strip() in ('N/A', '[N/A]', 'Not Supported'):
        return None
    match = re.match('^\\s*(-?\\d+(?:\\.\\d+)?)', text)
    return float(match.group(1)) if match else None

def parse_gpu(xml):
    gpu = ET.fromstring(xml).find('gpu')
    assert gpu is not None and gpu.findtext('uuid') == GPU_UUID, 'GPU identity mismatch'

    def get(path):
        return number(gpu.findtext(path))

    def section(tag):
        element = gpu.find(tag)
        return {node.tag: node.text for node in element} if element is not None else {}
    return dict(gpu_uuid=GPU_UUID, name=gpu.findtext('product_name'), core_c=get('temperature/gpu_temp'), memory_c=get('temperature/memory_temp'), hotspot_c=None, hotspot_unavailable='No documented hotspot field in installed nvidia-smi XML', target_c=get('temperature/gpu_target_temperature'), max_operating_c=get('temperature/gpu_temp_max_gpu_threshold'), slowdown_c=get('temperature/gpu_temp_slow_threshold'), shutdown_c=get('temperature/gpu_temp_max_threshold'), memory_max_operating_c=get('temperature/gpu_temp_max_mem_threshold'), fan_reported_percent=get('fan_speed'), gpu_busy_percent=get('utilization/gpu_util'), memory_activity_percent=get('utilization/memory_util'), memory_used_mib=get('fb_memory_usage/used'), memory_total_mib=get('fb_memory_usage/total'), power_average_w=get('gpu_power_readings/average_power_draw'), power_instant_w=get('gpu_power_readings/instant_power_draw'), power_limit_w=get('gpu_power_readings/current_power_limit'), sm_mhz=get('clocks/sm_clock'), memory_mhz=get('clocks/mem_clock'), clocks_events=section('clocks_event_reasons'), event_counters_us={key: number(value) for (key, value) in section('clocks_event_reasons_counters').items()}, volatile_ecc={node.tag: number(node.text) for node in gpu.findall('ecc_errors/volatile/*')}, fan_percent_is_reported_speed_not_verified_airflow=True)

class Policy:

    def __init__(self):
        self.hot_since = None
        self.previous_counters = None
        self.previous_ecc = None

    def assess(self, sample, now):
        gpu = sample['gpu']
        temp = gpu['core_c']
        (warn, stop, pause) = ([], [], [])
        maximum = gpu.get('max_operating_c')
        save_at = min(86.0, maximum - 7.0) if maximum is not None else 86.0
        urgent_at = min(88.0, maximum - 5.0) if maximum is not None else 88.0
        if temp is None:
            stop.append('core temperature sensor unavailable')
        else:
            if temp >= 85.0:
                warn.append('core >=85C')
            self.hot_since = (now if self.hot_since is None else self.hot_since) if temp >= save_at else None
            if self.hot_since is not None and now - self.hot_since >= 60.0:
                stop.append('core above conservative save threshold for 60s')
            if temp >= urgent_at:
                stop.append('core at conservative urgent threshold')
                pause.append('core at conservative urgent threshold; do not wait through evaluation')
        events = gpu.get('clocks_events', {})
        for (key, value) in events.items():
            if value == 'Active' and ('sw_thermal' in key or 'hw_thermal' in key):
                stop.append('thermal clock event: ' + key)
                if 'hw_thermal' in key:
                    pause.append('hardware thermal slowdown active')
        counters = gpu.get('event_counters_us', {})
        delta = {}
        if self.previous_counters is not None:
            for (key, value) in counters.items():
                old = self.previous_counters.get(key)
                if value is not None and old is not None:
                    delta[key] = value - old
                    if 'therm' in key and value > old:
                        warn.append('uncorroborated thermal counter change: ' + key)
        self.previous_counters = counters.copy()
        current_ecc = gpu.get('volatile_ecc', {})
        if self.previous_ecc is not None:
            for (key, value) in current_ecc.items():
                old = self.previous_ecc.get(key)
                if value is not None and old is not None and (value > old):
                    (stop if 'uncorrectable' in key else warn).append('new ECC count: ' + key)
        self.previous_ecc = current_ecc.copy()
        (memory, limit) = (gpu.get('memory_c'), gpu.get('memory_max_operating_c'))
        if memory is not None and limit is not None and (memory >= limit - 5.0):
            stop.append('memory temperature within 5C of reported operating limit')
        return dict(warnings=warn, stop_reasons=stop, emergency_pause_reasons=pause, thermal_counter_delta_us=delta, policy=dict(warn_core_c=85, sustained_save_core_c=save_at, sustained_seconds=60, urgent_pause_core_c=urgent_at))
