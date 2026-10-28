"""Own one Linux simulator child; preserve the established thermal and outer guards."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def generation(pid):
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
        return int(text.rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def exact_child(child, start_ticks, command):
    if child.poll() is not None or generation(child.pid) != start_ticks:
        return False
    try:
        current = Path('/proc', str(child.pid), 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    except (OSError, ProcessLookupError):
        return False
    return [os.fsdecode(v) for v in current] == command


def json_file(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def selected_gpu(index):
    xml = subprocess.check_output(['nvidia-smi', '-i', str(index), '-q', '-x'], text=True, timeout=15)
    gpu = ET.fromstring(xml).find('gpu')
    if gpu is None:
        raise RuntimeError('No selected GPU')
    apps = gpu.findall('processes/process_info')
    if apps:
        raise RuntimeError('Selected GPU has active processes; choose an idle GPU')
    return gpu.findtext('uuid'), xml


def supervise(args):
    from .core import thermal_policy as thermal
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('Unset CUDA_VISIBLE_DEVICES; --gpu selects the physical GPU index for both Isaac Sim and CUDA')
    uuid, xml = selected_gpu(args.gpu)
    thermal.GPU_UUID = uuid
    policy = thermal.Policy()
    initial = policy.assess({'gpu': thermal.parse_gpu(xml)}, time.monotonic())
    if initial['stop_reasons'] or initial['emergency_pause_reasons']:
        raise RuntimeError('GPU preflight rejected: ' + json.dumps(initial))
    out = (args.out or Path('runs')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-u', '-m', 'dogleg_rl.cli', args.mode, '--child',
               '--gpu', str(args.gpu), '--out', str(out), '--num-envs', str(args.num_envs),
               '--evaluation-update', str(args.evaluation_update), '--updates', str(args.updates),
               '--minutes', str(args.minutes), '--seconds', str(args.seconds), '--seed', str(args.seed),
               '--checkpoint', str(args.checkpoint.resolve())]
    command += ['--fresh'] if args.fresh else []
    command += ['--gui'] if args.gui else []
    json_file(out/'supervision.json', dict(command=command, gpu_uuid=uuid, automatic_resume=False,
        outer_save_s=4500, outer_TERM_s=5100, outer_KILL_s=5400,
        warn_c=85, save_c=86, sustained_s=60, urgent_pause_c=88,
        resource_save_margin_GPU_mib=2048, resource_save_margin_host_mib=8192,
        resource_sample_interval_s=1 if args.num_envs >= 8192 else 10))
    print('Output: ' + str(out), flush=True)
    started = time.monotonic()
    paused = stop_sent = term_sent = False
    errors = 0
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    ticks = generation(child.pid)
    if ticks is None:
        code = child.wait()
        json_file(out/'supervisor_receipt.json', dict(child_exit_code=code, actual_reaped=True))
        return code
    json_file(out/'child_identity.json', dict(pid=child.pid, start_ticks=ticks, command=command))

    def stop(reason):
        nonlocal stop_sent
        if not stop_sent:
            (out/'STOP_AFTER_UPDATE').write_text(reason, encoding='utf-8')
            stop_sent = True
            print('Complete-save stop requested: ' + reason, flush=True)

    def interrupt(signum, frame):
        stop('user interrupt')

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    last_sample = -10.
    with (out/'thermal.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        while child.poll() is None:
            elapsed = time.monotonic()-started
            if not exact_child(child, ticks, command):
                # /proc/cmdline may empty before the exiting child becomes waitable.
                # Wait only on our Popen child; never signal a mismatched identity.
                try:
                    terminal = {}
                    try:
                        terminal = json.loads((out/'receipt.json').read_text())
                    except (OSError, ValueError):
                        pass
                    terminal_phases = {'STOPPED', 'SMOKE_COMPLETE', 'FAIL', 'TRAINING_COMPLETE', 'EVALUATION_COMPLETE'}
                    exiting = generation(child.pid) == ticks and terminal.get('phase') in terminal_phases
                    child.wait(timeout=60. if exiting else 2.)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    break
                json_file(out/'supervisor_receipt.json', dict(child_pid=child.pid,
                    start_ticks=ticks, actual_reaped=False,
                    requires_user_action=True, error='child identity no longer matches; no signal sent'))
                return 2
            if elapsed-last_sample >= (1. if args.num_envs >= 8192 else 10.):
                last_sample = elapsed
                try:
                    xml = subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-x'], text=True, timeout=12)
                    sample = dict(at=time.time(), gpu=thermal.parse_gpu(xml))
                    assessment = policy.assess(sample, time.monotonic())
                    gpu_info = sample['gpu']
                    free_mib = gpu_info.get('memory_total_mib', 0)-gpu_info.get('memory_used_mib', 0)
                    sample['GPU_free_mib'] = free_mib
                    if free_mib < 2048:
                        assessment['stop_reasons'].append('GPU memory reserve below 2048 MiB')
                    host_max = Path('/sys/fs/cgroup/memory.max')
                    host_current = Path('/sys/fs/cgroup/memory.current')
                    if host_max.is_file() and host_current.is_file():
                        limit_text = host_max.read_text().strip()
                        if limit_text.isdigit():
                            host_free_mib = (int(limit_text)-int(host_current.read_text()))/2**20
                            sample['host_memory_free_mib'] = host_free_mib
                            if host_free_mib < 8192:
                                assessment['stop_reasons'].append('host memory reserve below 8192 MiB')
                    sample['assessment'] = assessment
                    errors = 0
                except (subprocess.SubprocessError, OSError, ET.ParseError, ValueError) as error:
                    errors += 1
                    sample = dict(at=time.time(), error=str(error))
                    assessment = dict(stop_reasons=['three sensor errors'] if errors >= 3 else [], emergency_pause_reasons=[])
                log.write(json.dumps(sample)+'\n')
                if assessment['stop_reasons']:
                    stop(json.dumps(assessment))
                if assessment['emergency_pause_reasons']:
                    if exact_child(child, ticks, command):
                        os.kill(child.pid, signal.SIGSTOP)
                        paused = True
                    break
            if elapsed >= 4500:
                stop('outer 4500 second budget')
            if elapsed >= 5400 and exact_child(child, ticks, command):
                os.kill(child.pid, signal.SIGKILL)
            elif elapsed >= 5100 and not term_sent and exact_child(child, ticks, command):
                os.kill(child.pid, signal.SIGTERM)
                term_sent = True
            time.sleep(.5)
    if paused:
        json_file(out/'supervisor_receipt.json', dict(actual_reaped=False, child_pid=child.pid,
            start_ticks=ticks, emergency_paused=True, automatic_resume=False,
            complete_save_confirmed=False, requires_user_action=True))
        print('Emergency pause: child remains paused. Inspect supervisor_receipt.json; there is no automatic resume.', flush=True)
        return 2
    code = child.wait()
    json_file(out/'supervisor_receipt.json', dict(child_pid=child.pid, start_ticks=ticks,
        child_exit_code=code, actual_reaped=True, emergency_paused=False,
        complete_final_checkpoint=(out/'checkpoint_final.pt').is_file(),
        selected_GPU_process_release_independently_verified=False))
    return code
