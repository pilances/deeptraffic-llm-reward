"""Per-car experience collection for the native staggered JS decision schedule."""
from dataclasses import dataclass, replace
from collections import deque
import numpy as np


@dataclass(frozen=True)
class AgentTransition:
    agent_id: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    start_frame: int
    end_frame: int
    terminated: bool
    truncated: bool
    bootstrap_discount: float

    @property
    def elapsed_frames(self):
        return self.end_frame - self.start_frame


class ReplayBuffer:
    """Bounded replay of completed same-car transitions; reproducible sampling."""
    def __init__(self, capacity=10000, seed=0):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self._items = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._items)

    def add(self, transition):
        # Own arrays: mutation of a policy/episode return must not corrupt replay.
        self._items.append(replace(transition, state=transition.state.copy(),
                                   next_state=transition.next_state.copy()))

    def sample(self, batch_size):
        if batch_size < 1 or batch_size > len(self):
            raise ValueError('batch_size must be between 1 and replay size')
        indices = self._rng.choice(len(self), batch_size, replace=False)
        return [replace(self._items[int(index)],
            state=self._items[int(index)].state.copy(),
            next_state=self._items[int(index)].next_state.copy()) for index in indices]


@dataclass
class _Pending:
    state: np.ndarray
    action: int
    start_frame: int
    reward: float = 0.0
    discount: float = 1.0


class MultiAgentCollector:
    """Shared team reward, accumulated until the SAME car next decides.

    gamma_per_frame is explicitly a per-simulation-frame discount. For a usual
    gamma per 30 frames use gamma_per_frame=gamma**(1/30). Both discounted
    interval reward and bootstrap discount use this same time base.
    """
    def __init__(self, env, replay=None, gamma_per_frame=1.0):
        if not np.isfinite(gamma_per_frame) or not 0 < gamma_per_frame <= 1:
            raise ValueError('gamma_per_frame must be in (0,1]')
        self.env, self.replay = env, replay
        self.gamma_per_frame = gamma_per_frame

    def collect_episode(self, policies, *, seed=0, on_transition=None):
        """policies: a shared callable, or {decision_agent_id: callable}."""
        if callable(policies):
            policies = {agent: policies for agent in self.env.decision_agent_ids}
        self.env.validate_agent_policies(policies)
        pending, transitions = {}, []
        counts = {agent: 0 for agent in self.env.decision_agent_ids}
        episode_reward = 0.0

        def finish(agent, next_state, frame, terminated=False, truncated=False):
            previous = pending.pop(agent)
            transition = AgentTransition(agent, previous.state.copy(), previous.action,
                previous.reward, next_state.copy(), previous.start_frame, frame,
                terminated, truncated, 0.0 if terminated or truncated else previous.discount)
            transitions.append(transition)
            if self.replay is not None:
                self.replay.add(transition)
            if on_transition is not None:
                on_transition(transition)

        try:
            observation, info = self.env.reset(seed=seed)
            while True:
                agent, frame = info['agent_id'], info['frame']
                if agent in pending:
                    finish(agent, observation, frame)
                action = policies[agent](observation.copy())
                if not self.env.action_space.contains(action):
                    raise ValueError(f'Agent {agent} returned invalid action {action}')
                pending[agent] = _Pending(observation.copy(), int(action), frame)
                counts[agent] += 1
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                # A team reward is shared among all currently pending actions.
                # This replication is intentional, not a sum-to-one allocation.
                for frame_reward in info['frame_rewards']:
                    for previous in pending.values():
                        previous.reward += previous.discount * frame_reward
                        previous.discount *= self.gamma_per_frame
                if terminated or truncated:
                    for agent in list(pending):
                        finish(agent, info['final_observations'][agent], info['frame'],
                               terminated, truncated)
                    break
            return {'seed': seed, 'frames': info['frame'], 'decisions_per_agent': counts,
                'controlled_agent_ids': self.env.controlled_agent_ids,
                'decision_agent_ids': self.env.decision_agent_ids,
                'episode_mean_speed_mph': info['episode_mean_speed_mph'],
                'agent_mean_speed_mph': info['agent_mean_speed_mph'],
                'episode_team_reward': episode_reward, 'transitions': transitions}
        finally:
            self.env.close()
