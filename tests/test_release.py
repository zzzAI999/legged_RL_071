"""CPU integration checks for the packaged entry points and pinned algorithm."""
import contextlib
import io
import json
from pathlib import Path
import unittest

import torch

from dogleg_rl.cli import ASSETS, asset_check, parser, validate
from dogleg_rl.core.bounded_adaptive_ppo import BoundedAdaptivePPO, LR_MIN, LR_MAX
from dogleg_rl.core.policy_architecture import actor_critic
from dogleg_rl.core.standing_gyro_penalty_reward_v1 import base_gyro_cost
from dogleg_rl.core import thermal_policy as thermal


class EmptyEncoder:
    num_output_dim = 0

    def encode(self, history):
        return history


class ReleaseTests(unittest.TestCase):
    def test_asset_integrity_and_mechanism(self):
        self.assertTrue(asset_check()['sha256_passed'])
        payload = json.loads((ASSETS/'physical_payload.json').read_text())
        self.assertAlmostEqual(payload['mass'], 5.5)
        self.assertEqual(len(payload['joints']), 18)
        self.assertEqual(len(payload['loops']), 4)
        self.assertEqual(len(payload['belts']), 4)

    def test_cli_enforces_bounded_budget(self):
        for options in (['train','--updates','101'], ['train','--minutes','61'], ['train','--gpu','-1'], ['evaluate','--fresh']):
            with self.assertRaises(ValueError):
                validate(parser().parse_args(options))
        validate(parser().parse_args(['train','--updates','100','--minutes','60']))

    def test_help_has_no_simulator_dependency(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable,'-B','-m','dogleg_rl.cli','--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('smoke', result.stdout)

    def test_checkpoint_policy_matches_packaged_network(self):
        import hashlib
        from dogleg_rl.cli import DEFAULT_CHECKPOINT
        self.assertEqual(hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest(),
            '65bf85f0fc9fb5ee9bcdff97cda9e96527c71c6c27c974c334edf602878c974f')
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', weights_only=False)
        with contextlib.redirect_stdout(io.StringIO()):
            actor = actor_critic(checkpoint, device='cpu', hidden_dims=[256,256])
        with torch.no_grad():
            action = actor.act_inference(torch.zeros(3,29))
        self.assertEqual(action.shape, (3,6))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue((action.abs() <= 1).all())

    def test_current_gyro_formula(self):
        g = torch.tensor([0., .3, 6.])
        torch.testing.assert_close(base_gyro_cost(g), torch.tensor([0., -.0036, -1.]))

    def test_vendored_ppo_completes_sixteen_minibatches(self):
        torch.manual_seed(21)
        with contextlib.redirect_stdout(io.StringIO()):
            actor = actor_critic(device='cpu', bounded=True, hidden_dims=[256,256])
        algorithm = BoundedAdaptivePPO(8, EmptyEncoder(), actor, num_learning_epochs=4,
            num_mini_batches=4, learning_rate=3e-4, schedule='adaptive', desired_kl=.01,
            early_stop=False, entropy_coef=.001, gamma=.995, lam=.95,
            use_clipped_value_loss=False, device='cpu')
        algorithm.init_storage(8, 4, [29], [39], [0], [0], [6])
        obs = torch.randn(8,29)
        critic = torch.randn(8,39)
        empty = torch.empty(8,0)
        initial = next(actor.parameters()).detach().clone()
        with torch.no_grad():
            for _ in range(4):
                algorithm.act(obs, empty, empty, critic)
                algorithm.process_env_step(torch.randn(8), torch.zeros(8,dtype=torch.long), {}, obs)
            algorithm.compute_returns(critic)
        losses = algorithm.update()
        steps = {int(v['step']) for v in algorithm.optimizer.state.values()}
        self.assertEqual(steps, {16})
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertTrue(LR_MIN <= algorithm.learning_rate <= LR_MAX)
        self.assertFalse(torch.equal(initial, next(actor.parameters()).detach()))

    def test_thermal_save_and_emergency_pause_are_distinct(self):
        def sample(temp, events=None):
            return {'gpu': dict(core_c=temp, clocks_events=events or {}, event_counters_us={}, volatile_ecc={})}
        policy = thermal.Policy()
        self.assertFalse(policy.assess(sample(86), 0)['stop_reasons'])
        self.assertTrue(policy.assess(sample(86), 60)['stop_reasons'])
        self.assertFalse(policy.assess(sample(86), 60)['emergency_pause_reasons'])
        self.assertTrue(policy.assess(sample(88), 61)['emergency_pause_reasons'])
        self.assertTrue(thermal.Policy().assess(sample(70, {'hw_thermal_slowdown':'Active'}), 0)['emergency_pause_reasons'])


if __name__ == '__main__':
    unittest.main()
