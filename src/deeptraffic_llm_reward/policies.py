"""Non-learning controls for the JS observation encoding."""
import numpy as np
from .env import Action

class RandomPolicy:
    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)
    def act(self, observation):
        return int(self._rng.integers(5))

class RuleBasedPolicy:
    """Simple observation-only control; no claim of reproducing a submission."""
    def __init__(self, lanes_side=3, patches_ahead=40, patches_behind=5):
        self.side, self.ahead, self.behind = lanes_side, patches_ahead, patches_behind
    def act(self, observation):
        grid = np.asarray(observation).reshape(self.ahead+self.behind, 2*self.side+1)
        def gap(col):
            if col < 0 or col >= grid.shape[1]:
                return -1
            # Zero may be a stopped car or off-road: both conservatively blocked.
            occupied = np.flatnonzero(grid[:self.ahead, col][::-1] < 0.5)
            return int(occupied[0]) if len(occupied) else self.ahead
        own, left, right = gap(self.side), gap(self.side-1), gap(self.side+1)
        if own < 12:
            if left > own+3 and left >= right:
                return int(Action.CHANGE_LEFT)
            if right > own+3:
                return int(Action.CHANGE_RIGHT)
        return int(Action.ACCELERATE)
