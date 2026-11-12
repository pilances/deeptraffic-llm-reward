"""Explicit runner for reproducing MIT's historical single-ego starter training.

This is deliberately not a Gymnasium environment. It exposes the original
engine callback ``learn(state, historical_ego_reward)`` so a reproduction cannot
be confused with the active Gym API's fleet-speed reward contract.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .backend import engine_class


@dataclass(frozen=True)
class HistoricalTrainingResult:
    decisions: int
    training_frames: int
    training_blocks: int
    frames_per_training_block: int
    brain_seed: int
    traffic_seed_r: int
    traffic_seed_t: int
    final_observation: tuple


class HistoricalStarterRunner:
    """Run the original ``train_webworker`` schedule and ego reward callback."""

    reward_objective = "historical_ego_reward_(N-60)/20"
    interface = "historical_callback_runner"

    def __init__(self) -> None:
        self._engine_type = engine_class()

    def train(
        self,
        learn: Callable[[list, float], int],
        reset_brain_seed: Callable[[int], None],
        *,
        iterations: int = 30_000,
        seed: int = 0,
        progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> HistoricalTrainingResult:
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")

        total_frames = 30 * iterations
        # Exact train_webworker expression and JS loop behavior. For the official
        # 30,000 iterations this is exactly 10 blocks of 90,000 frames.
        requested_runs = total_frames / 100_000 + 1
        blocks = int(np.ceil(requested_runs))
        frames_per_block = int(np.ceil(total_frames / requested_runs))

        rng = np.random.default_rng(seed)
        brain_seed = int(rng.integers(10_000_000))
        traffic_seed_r, traffic_seed_t = map(int, rng.integers(10_000_000, size=2))
        reset_brain_seed(brain_seed)

        engine = self._engine_type(
            lanes_side=1,
            patches_ahead=10,
            patches_behind=0,
            other_agents=0,
        )
        engine.r = traffic_seed_r
        engine.t = traffic_seed_t
        decisions = 0

        def historical_callback(state, historical_ego_reward):
            nonlocal decisions
            decisions += 1
            return learn(state, historical_ego_reward)

        for block in range(blocks):
            engine.reset()
            for _ in range(frames_per_block):
                engine.V(historical_callback)
            if progress is not None:
                progress(block + 1, blocks, decisions)

        return HistoricalTrainingResult(
            decisions=decisions,
            training_frames=blocks * frames_per_block,
            training_blocks=blocks,
            frames_per_training_block=frames_per_block,
            brain_seed=brain_seed,
            traffic_seed_r=traffic_seed_r,
            traffic_seed_t=traffic_seed_t,
            final_observation=tuple(engine.ego_state()),
        )
