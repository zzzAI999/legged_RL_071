"""Use upstream adaptive PPO with an explicit conservative learning-rate cap."""
import math
from dogleg_rl._vendor.rsl_rl.algorithm import PPO
LR_MIN = 1e-05
LR_MAX = 0.0003
OPTIMIZATION_VERSION = 'standing_v8_bounded_adaptive_ppo'

class BoundedAdaptivePPO(PPO):

    @property
    def learning_rate(self):
        return self._bounded_learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        assert math.isfinite(float(value)) and float(value) > 0.0
        self._bounded_learning_rate = max(LR_MIN, min(LR_MAX, float(value)))

    def __init__(self, *args, **kwargs):
        assert kwargs.get('schedule') == 'adaptive'
        assert kwargs.get('early_stop') is False
        assert kwargs.get('anneal_lr', False) is False
        assert LR_MIN <= kwargs['learning_rate'] <= LR_MAX
        super().__init__(*args, **kwargs)
