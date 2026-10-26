"""Run environment checks, a bounded smoke test, evaluation or explicit training."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ASSETS = Path(__file__).resolve().parent / 'assets'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT / 'checkpoints/guard17.pt'
ASSET_HASHES = {
    'dogleg_lab.usda': 'fdf4c05cb5acbcf547aa8eee458707ba41efd643ea038e077d0341273087b06c',
    'physical_payload.json': 'a982791ce998da6d9472d48ff18f179c90288743e686cc6115be6c5fae2c47df',
    'reset_atlas.json': 'c5a58dcbad1792727cb64e5d9c9733fc53e487b51be9d32975bdd96918693af5',
}


def asset_check():
    for name, expected in ASSET_HASHES.items():
        path = ASSETS / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Asset hash mismatch: ' + str(path))
    return {'files': len(ASSET_HASHES), 'sha256_passed': True}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode', choices=('check', 'smoke', 'evaluate', 'train'))
    result.add_argument('--gpu', type=int, default=1, help='Physical GPU index; use 0 on a single-GPU machine')
    result.add_argument('--out', type=Path, default=None, help='New output directory; existing paths are refused')
    result.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument('--fresh', action='store_true', help='Explicitly train a new policy instead of restoring a checkpoint')
    result.add_argument('--updates', type=int, default=100, help='Maximum new PPO updates, 1..100')
    result.add_argument('--minutes', type=float, default=60., help='Maximum post-reset training minutes, at most 60')
    result.add_argument('--seconds', type=float, default=35.02, help='Evaluation duration; 35.02 by default')
    result.add_argument('--seed', type=int, default=126105)
    result.add_argument('--gui', action='store_true', help='Display the Isaac Sim window; requires a desktop')
    result.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    return result


def validate(args):
    if args.gpu < 0 or not 1 <= args.updates <= 100 or not 0 < args.minutes <= 60:
        raise ValueError('Require GPU index >=0, updates 1..100, and minutes in (0,60]')
    if not 35.02 <= args.seconds <= 120. or abs(args.seconds/.02 - round(args.seconds/.02)) > 1e-6:
        raise ValueError('Evaluation seconds must be 35.02..120 in 20 ms increments')
    if args.fresh and args.mode != 'train':
        raise ValueError('--fresh is only valid for train')


def main():
    p = parser()
    args = p.parse_args()
    try:
        validate(args)
        asset_check()
        if args.mode == 'check':
            from .checks import check_installation
            print(json.dumps(check_installation(), indent=2))
            return
        if sys.platform != 'linux':
            raise RuntimeError('GPU commands require native Linux; see README for the supported setup')
        if not args.fresh and args.mode != 'smoke' and not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        if args.child:
            if args.out is None or not (args.out/'supervision.json').is_file():
                raise RuntimeError('Simulator child requires the owning supervisor contract')
            from .runner import run
            run(args)
        else:
            from .monitor import supervise
            raise SystemExit(supervise(args))
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        p.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
