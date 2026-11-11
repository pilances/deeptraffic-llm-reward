"""Small policies used to smoke-test the shared environment contract."""

from typing import Optional

import numpy as np

from .env import Action


class RandomPolicy:
    """Select actions uniformly using its own reproducible random generator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        del observation
        return int(self._rng.integers(0, len(Action)))


class RuleBasedPolicy:
    """Smoke heuristic for paper observations (zero means empty).

    Stopped cars and off-road cells are also zero and cannot be distinguished
    from empty space here. Environment safety retains the full geometry.
    """

    def act(self, observation: np.ndarray) -> int:
        if observation.ndim != 2 or observation.shape[0] != 3:
            raise ValueError("RuleBasedPolicy expects a three-lane observation")

        center = observation[1]
        if center[0] == 0.0 or center[0] >= 75.0:
            return int(Action.ACCELERATE)

        clear_distances = np.array([self._clear_distance(row) for row in observation])
        best_row = int(np.argmax(clear_distances))
        if clear_distances[best_row] > clear_distances[1]:
            if best_row == 0:
                return int(Action.CHANGE_LEFT)
            if best_row == 2:
                return int(Action.CHANGE_RIGHT)
        if center[0] < 55.0:
            return int(Action.DECELERATE)
        return int(Action.MAINTAIN)

    @staticmethod
    def _clear_distance(lane: np.ndarray) -> int:
        occupied = np.flatnonzero(lane > 0.0)
        return int(occupied[0]) if occupied.size else int(lane.size)

