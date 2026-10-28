"""Export a successful simulated evaluation input trace and compare an aligned real CSV offline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def successful_reference(root, env_id):
    root = Path(root)
    labels = [json.loads(p.read_text()) for p in root.glob('evaluation_labels_*.json')]
    labels = [v for v in labels if v['case_passed'][env_id]]
    if len(labels) != 1:
        raise ValueError('Require exactly one completed independent evaluation where this environment passed')
    label = labels[0]
    frames, steps = [], []
    for line in (root/'chunks.jsonl').read_text().splitlines():
        entry = json.loads(line)
        if entry['mode'] != 'evaluation' or entry['model_sha256'] != label['model_sha256']: continue
        p = root/entry['file']
        if digest(p) != entry['sha256']: raise ValueError('Reference chunk SHA mismatch: '+str(p))
        with np.load(p, allow_pickle=False) as chunk:
            ids = chunk['env_ids']
            positions = np.flatnonzero(ids == env_id)
            if len(positions) != 1: raise ValueError('Environment identity missing or duplicated')
            i = int(positions[0])
            frames.append(chunk['observation'][:, i])
            steps.append(chunk['episode_step_before'][:, i])
    values, step = np.concatenate(frames), np.concatenate(steps)
    expected = round(label['seconds']/.02)
    if values.shape != (expected, 29) or not np.array_equal(step, np.arange(expected)):
        raise ValueError('Successful reference has missing/duplicated/out-of-order50Hz inputs')
    return step, values, label


def compare(reference, actual, contract):
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] != 29:
        raise ValueError('Need equally aligned N by29 arrays')
    rows = []
    for feature in contract['features']:
        i = feature['index']
        valid = np.isfinite(reference[:, i]) & np.isfinite(actual[:, i])
        difference = actual[valid, i].astype(np.float64)-reference[valid, i].astype(np.float64)
        unsaturated = valid & (np.abs(reference[:, i]) < 100) & (np.abs(actual[:, i]) < 100)
        physical = (actual[unsaturated, i].astype(np.float64)-reference[unsaturated, i].astype(np.float64))/feature['input_equals_physical_times']
        rows.append(dict(feature, finite_pairs=int(valid.sum()), nonfinite_pairs=int((~valid).sum()),
            normalized_bias=float(difference.mean()) if len(difference) else None,
            normalized_RMSE=float(np.sqrt(np.mean(difference**2))) if len(difference) else None,
            normalized_abs_p95=float(np.quantile(np.abs(difference), .95)) if len(difference) else None,
            normalized_abs_max=float(np.abs(difference).max()) if len(difference) else None,
            inverse_scaled_RMSE_excluding_clipped_values=float(np.sqrt(np.mean(physical**2))) if len(physical) else None))
    return dict(aligned_rows=len(reference), exact_float32_equal=bool(np.array_equal(reference.view(np.uint32), actual.view(np.uint32))),
        features=rows, automatic_pass_fail=False,
        interpretation='Descriptive input comparison. Matching model/coordinate/units/initial state/time alignment must be established separately; inverse scaling is not a raw sensor reconstruction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True, help='The observation29 directory')
    parser.add_argument('--env-id', type=int, required=True, choices=range(512))
    parser.add_argument('--export-csv', type=Path)
    parser.add_argument('--real-csv', type=Path, help='Requires episode_step_before and obs_00 through obs_28 columns')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--real-contract-sha256')
    args = parser.parse_args()
    if not args.export_csv and not args.real_csv: parser.error('Choose export or comparison')
    if args.real_csv and not args.report: parser.error('Comparison requires --report')
    contract = json.loads((args.recording/'contract.json').read_text())
    step, reference, label = successful_reference(args.recording, args.env_id)
    if args.export_csv:
        with args.export_csv.open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_step_before']+['obs_%02d' % i for i in range(29)])
            for t, row in zip(step, reference): writer.writerow([int(t)]+[format(float(v), '.9g') for v in row])
    if args.real_csv:
        with args.real_csv.open(newline='', encoding='utf-8-sig') as f: real = list(csv.DictReader(f))
        by_step = {}
        for row in real:
            key = int(row['episode_step_before'])
            if key in by_step: raise ValueError('Real CSV has duplicated decision step identities')
            by_step[key] = [float(row['obs_%02d' % i]) for i in range(29)]
        mask = np.asarray([int(t) in by_step for t in step])
        if not mask.any(): raise ValueError('No shared decision steps; explicit time alignment is required')
        actual = np.asarray([by_step[int(t)] for t in step[mask]], dtype=np.float32)
        result = compare(reference[mask], actual, contract)
        result.update(reference_case_id=args.env_id, reference_evaluation_model_SHA256=label['model_sha256'],
            reference_observation_contract_SHA256=contract['canonical_SHA256'],
            real_contract_identity_verified=args.real_contract_sha256 == contract['canonical_SHA256'],
            reference_rows_not_matched=int((~mask).sum()), reference_success_is_independent_evaluation=True)
        with args.report.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2)
        print(json.dumps(dict(report=str(args.report), aligned_rows=result['aligned_rows'], exact_equal=result['exact_float32_equal'])))


if __name__ == '__main__': main()
