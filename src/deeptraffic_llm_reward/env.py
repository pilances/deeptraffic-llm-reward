"""Event-driven shared-policy Gymnasium adapter for the audited JS port.

A worker suspends exactly inside the engine policy callback. This avoids stale
observations, broadcasting one action to all agents, and the upstream wrapper's
one-interval reward delay. Simulation time does not advance during inference.
"""
from dataclasses import dataclass
from enum import IntEnum
from queue import Queue
from threading import Thread
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .backend import engine_class, REVISION

class Action(IntEnum):
    MAINTAIN = 0
    ACCELERATE = 1
    DECELERATE = 2
    CHANGE_LEFT = 3
    CHANGE_RIGHT = 4

@dataclass(frozen=True)
class DeepTrafficConfig:
    lanes_side: int = 3
    patches_ahead: int = 40
    patches_behind: int = 5
    other_agents: int = 10
    episode_frames: int = 2000

    def __post_init__(self):
        for key in ('lanes_side', 'patches_ahead', 'patches_behind', 'other_agents', 'episode_frames'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.other_agents > 10 or self.episode_frames < 1 or self.patches_ahead < 1:
            raise ValueError('Require other_agents <= 10, positive frames and lookahead')

class _Stopped(Exception):
    pass

class DeepTrafficEnv(gym.Env):
    metadata = {'render_modes': []}
    reward_objective = 'fleet_speed_integral'

    def __init__(self, config=None):
        self.config = config or DeepTrafficConfig()
        c = self.config
        self.observation_space = spaces.Box(0, 1,
            ((2*c.lanes_side+1)*(c.patches_ahead+c.patches_behind),), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self._engine_type = engine_class()
        self._thread = None
        self._done = True

    def _new_engine(self):
        c = self.config
        return self._engine_type(lanes_side=c.lanes_side, patches_ahead=c.patches_ahead,
            patches_behind=c.patches_behind, other_agents=c.other_agents)

    @property
    def controlled_agent_ids(self):
        return tuple(range(self.config.other_agents + 1))

    @property
    def decision_agent_ids(self):
        """Actual callback owners; historical car 10 has no decision phase."""
        return tuple(range(min(self.config.other_agents, 9) + 1))

    def validate_agent_policies(self, policies):
        if set(policies) != set(self.decision_agent_ids):
            raise ValueError(f'Policies required exactly for {self.decision_agent_ids}; '
                             'car 10 has no independent callback in the JS engine')
        if not all(callable(policy) for policy in policies.values()):
            raise TypeError('Each policy must be callable: policy(observation) -> action')

    def reset(self, *, seed=None, options=None):
        self.close()
        super().reset(seed=seed)
        self._eng = self._new_engine()
        anchor = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31-2))
        self._eng.r = self._eng.t = anchor
        self._eng.reset()
        self._actions, self._events = Queue(), Queue()
        self._sum = 0.0
        self._agent_speed_sums = {agent: 0.0 for agent in self.controlled_agent_ids}
        self._frame_rewards = []
        self._frames = self._last_frames = 0
        self._last_sum = 0.0
        self._done = False
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        obs, _, _, _, info = self._receive()
        return obs, info

    def _run(self):
        eng = self._eng
        def policy(state, historical_reward):
            phase = eng.G % 30
            agent = 0 if phase == 0 else phase // 3
            self._events.put(('decision', np.asarray(state, dtype=np.float32), agent))
            action = self._actions.get()
            if action is None:
                raise _Stopped()
            return action
        try:
            for _ in range(self.config.episode_frames):
                eng.V(policy)
                for agent in self.controlled_agent_ids:
                    car = eng.z[agent]
                    self._agent_speed_sums[agent] += max(0, car.c * car.a) * 20
                fleet_mph = sum(max(0, car.c*car.a)*20
                    for car in eng.z[:eng.nOtherAgents+1]) / (eng.nOtherAgents+1)
                self._sum += fleet_mph
                self._frame_rewards.append(fleet_mph / 80.0)
                self._frames += 1
            self._events.put(('end', np.asarray(eng.ego_state(), dtype=np.float32), None))
        except _Stopped:
            pass
        except BaseException as exc:
            self._events.put(('error', exc, None))

    def _receive(self):
        kind, obs, agent = self._events.get()
        if kind == 'error':
            self._done = True
            raise obs
        elapsed = self._frames - self._last_frames
        frame_rewards = tuple(self._frame_rewards[self._last_frames:self._frames])
        # Integral fleet reward: sum over an episode equals fleet mph * frames / 80.
        reward = (self._sum - self._last_sum) / 80.0
        self._last_frames, self._last_sum = self._frames, self._sum
        self._done = kind == 'end'
        info = {'agent_id': agent, 'frame': self._frames, 'elapsed_frames': elapsed,
            'controlled_vehicles': self._eng.nOtherAgents+1,
            'controlled_agent_ids': self.controlled_agent_ids,
            'decision_agent_ids': self.decision_agent_ids,
            'frame_rewards': frame_rewards,
            'episode_mean_speed_mph': self._sum/max(1, self._frames),
            'reward_objective': self.reward_objective,
            'environment_version': '0.4.0', 'backend_revision': REVISION}
        if self._done:
            # Only a closing snapshot, never presented as a new decision. Keep
            # the native engine grid semantics (same as ego_state()).
            final_states = {}
            for agent_id in self.decision_agent_ids:
                self._eng.H.fill_obs(agent_id, self._eng.K, self._eng)
                final_states[agent_id] = np.asarray(self._eng.K.to_state(), dtype=np.float32)
            info['final_observations'] = final_states
            info['agent_mean_speed_mph'] = {
                agent: total / self._frames for agent, total in self._agent_speed_sums.items()}
        return obs, reward, False, self._done, info

    def step(self, action):
        if self._done:
            raise gym.error.ResetNeeded('Call reset before step')
        if not self.action_space.contains(action):
            raise ValueError('Expected action 0..4')
        self._actions.put(int(action))
        return self._receive()

    def close(self):
        if self._thread is not None:
            if self._thread.is_alive():
                self._actions.put(None)
                self._thread.join()
            self._thread = None
        self._done = True

    def evaluate(self, policy, runs=500, frames=None):
        """Fresh-load JS protocol, with persistent RNG streams across runs.

        Independent of any training reward or current Gym episode.
        Uses upper-middle order statistic and per-run floor, exactly as JS.
        """
        return self.evaluate_agents({agent: policy for agent in self.decision_agent_ids},
                                    runs=runs, frames=frames)

    def evaluate_agents(self, policies, runs=500, frames=None):
        """Evaluate shared or different per-car policies in the SAME traffic world."""
        self.validate_agent_policies(policies)
        frames = self.config.episode_frames if frames is None else frames
        if runs < 1 or frames < 1:
            raise ValueError('Positive runs and frames required')
        eng = self._new_engine()
        eng.set_deterministic_seeds()
        scores = []
        for _ in range(runs):
            eng.reset()
            total = 0.0
            def callback(state, reward):
                phase = eng.G % 30
                agent_id = 0 if phase == 0 else phase // 3
                action = policies[agent_id](np.asarray(state, dtype=np.float32))
                if not self.action_space.contains(action):
                    raise ValueError('Policy returned invalid action')
                return int(action)
            for _ in range(frames):
                eng.V(callback)
                # Preserve upstream summation order and rounding.
                for car in eng.z[:eng.nOtherAgents+1]:
                    total += max(0.0, car.c*car.a)/(eng.nOtherAgents+1)
            scores.append(math.floor(total/frames*2000)/100)
        return {'median': sorted(scores)[len(scores)//2], 'mean': float(np.mean(scores)),
            'runs': scores, 'frames': frames, 'controlled_vehicles': eng.nOtherAgents+1,
            'decision_agent_ids': self.decision_agent_ids,
            'backend_revision': REVISION}
