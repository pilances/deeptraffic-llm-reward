"""Shared-policy DQN with same-car, time-aware replay and frozen evaluation."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import platform
import time
import numpy as np
import torch
from torch import nn
from .env import DeepTrafficConfig, DeepTrafficEnv
from .multiagent import MultiAgentCollector, ReplayBuffer
from .backend import REVISION


@dataclass(frozen=True)
class DQNConfig:
    episodes: int = 150
    hidden: tuple = (128, 128)
    learning_rate: float = 0.0003
    gamma_per_decision: float = 0.95
    reward_scale: float = 1 / 30
    replay_capacity: int = 30000
    batch_size: int = 64
    learning_starts: int = 2000
    train_every: int = 4
    target_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 30000
    gradient_clip: float = 10.0

    def __post_init__(self):
        for name in ('episodes', 'replay_capacity', 'batch_size', 'learning_starts',
                     'train_every', 'target_every', 'epsilon_decay'):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not self.hidden or any(not isinstance(v, int) or v < 1 for v in self.hidden):
            raise ValueError('hidden must contain positive layer sizes')
        if self.batch_size > self.replay_capacity or self.learning_starts < self.batch_size:
            raise ValueError('Require batch_size <= learning_starts and replay capacity')
        if not 0 < self.gamma_per_decision <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Invalid gamma or epsilon schedule')
        for name in ('learning_rate', 'reward_scale', 'gradient_clip'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')


def q_network(inputs, hidden):
    layers = []
    sizes = (inputs, *hidden, 5)
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DQNAgent:
    def __init__(self, observation_size, config=None, seed=0):
        self.config = config or DQNConfig()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = q_network(observation_size, self.config.hidden)
        self.target = q_network(observation_size, self.config.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.observation_size = observation_size
        self.decisions = self.updates = 0

    @property
    def epsilon(self):
        c = self.config
        fraction = min(1.0, self.decisions / c.epsilon_decay)
        return c.epsilon_start + fraction * (c.epsilon_end-c.epsilon_start)

    @torch.no_grad()
    def greedy(self, observation):
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        return int(self.online(state).argmax().item())

    def explore(self, observation):
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(5))
        else:
            action = self.greedy(observation)
        self.decisions += 1
        return action

    def update(self, transitions):
        states = torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        rewards = torch.tensor([t.reward * self.config.reward_scale for t in transitions], dtype=torch.float32)
        discounts = torch.tensor([t.bootstrap_discount for t in transitions], dtype=torch.float32)
        # Explicit episode-end mask is a second guard, even if a malformed
        # transition were passed with a nonzero terminal bootstrap discount.
        end_mask = torch.tensor([t.terminated or t.truncated for t in transitions])
        discounts = discounts.masked_fill(end_mask, 0)
        with torch.no_grad():
            targets = rewards + discounts * self.target(next_states).max(dim=1).values
        estimates = self.online(states).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(estimates, targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite DQN loss')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, env_config, training_seed):
        torch.save({'online': self.online.state_dict(), 'dqn_config': asdict(self.config),
            'env_config': asdict(env_config), 'training_seed': training_seed,
            'observation_size': self.observation_size, 'decisions': self.decisions,
            'updates': self.updates, 'backend_revision': REVISION}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['backend_revision'] != REVISION:
            raise ValueError('Checkpoint belongs to a different backend')
        agent = cls(checkpoint['observation_size'], DQNConfig(**checkpoint['dqn_config']), checkpoint['training_seed'])
        agent.online.load_state_dict(checkpoint['online'])
        agent.target.load_state_dict(checkpoint['online'])
        agent.decisions, agent.updates = checkpoint['decisions'], checkpoint['updates']
        return agent, DeepTrafficConfig(**checkpoint['env_config'])


def train_shared_dqn(seed, output, config=None, env_config=None):
    config = config or DQNConfig()
    env_config = env_config or DeepTrafficConfig(other_agents=4)
    if env_config.other_agents < 0 or env_config.other_agents > 9:
        raise ValueError('Require 1..10 controlled cars for the shared-policy trainer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_seeds = [1_000_000 + seed * 10000 + ep for ep in range(config.episodes)]
    manifest = {'training_seed': seed, 'training_scenario_seeds': train_seeds,
        'dqn_config': asdict(config), 'env_config': asdict(env_config),
        'backend_revision': REVISION, 'torch': str(torch.__version__),
        'numpy': np.__version__, 'python': platform.python_version(),
        'objective': 'fleet_speed_integral', 'checkpoint_selection': 'final only',
        'algorithm': 'DQN (target-network max, uniform replay, Huber loss, Adam)'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    env = DeepTrafficEnv(env_config)
    agent = DQNAgent(env.observation_space.shape[0], config, seed)
    replay = ReplayBuffer(config.replay_capacity, seed=seed)
    collector = MultiAgentCollector(env, replay, config.gamma_per_decision**(1/30))
    completed = 0
    losses = []
    def on_transition(transition):
        nonlocal completed
        completed += 1
        if completed >= config.learning_starts and completed % config.train_every == 0:
            losses.append(agent.update(replay.sample(config.batch_size)))
    started = time.monotonic()
    with (output/'training.jsonl').open('w') as log:
        for episode, scenario_seed in enumerate(train_seeds):
            losses.clear()
            result = collector.collect_episode(agent.explore, seed=scenario_seed, on_transition=on_transition)
            row = {'episode': episode+1, 'scenario_seed': scenario_seed,
                'fleet_mph': result['episode_mean_speed_mph'], 'epsilon': agent.epsilon,
                'decisions': agent.decisions, 'transitions': completed, 'updates': agent.updates,
                'replay_size': len(replay), 'mean_loss': float(np.mean(losses)) if losses else None}
            log.write(json.dumps(row)+'\n')
            log.flush()
            if (episode+1) % 10 == 0 or episode == 0:
                print(f'seed={seed} episode={episode+1}/{config.episodes} '
                      f'mph={row["fleet_mph"]:.2f} epsilon={agent.epsilon:.3f} updates={agent.updates}', flush=True)
    agent.save(output/'final.pt', env_config, seed)
    # Verify the saved policy before using it in held-out evaluation.
    restored, restored_config = DQNAgent.load(output/'final.pt')
    assert restored_config == env_config
    for left, right in zip(agent.online.parameters(), restored.online.parameters()):
        assert torch.equal(left, right)
    summary = {'seed': seed, 'decisions': agent.decisions, 'transitions': completed,
        'updates': agent.updates, 'training_frames': config.episodes*env_config.episode_frames,
        'elapsed_seconds': time.monotonic()-started, 'checkpoint': 'final.pt'}
    (output/'training_summary.json').write_text(json.dumps(summary, indent=2))
    return restored


def evaluate_fleet(policy, env_config, scenario_seeds):
    """Independent seeded Gym episodes; identical protocol for DQN and rules."""
    seeds = list(scenario_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Require nonempty unique evaluation seeds')
    env = DeepTrafficEnv(env_config)
    scores = []
    try:
        for index, seed in enumerate(seeds):
            obs, info = env.reset(seed=int(seed))
            while True:
                obs, _, terminated, truncated, info = env.step(policy(obs))
                if terminated or truncated:
                    break
            scores.append(info['episode_mean_speed_mph'])
            if (index+1) % 100 == 0:
                print(f'evaluation={index+1}/{len(seeds)}', flush=True)
    finally:
        env.close()
    return {'scenario_seeds': seeds, 'scores_mph': scores,
        'median_mph': float(np.median(scores)), 'mean_mph': float(np.mean(scores)),
        'iqr_mph': float(np.percentile(scores,75)-np.percentile(scores,25)),
        'env_config': asdict(env_config), 'backend_revision': REVISION,
        'protocol': 'independent reset(seed), unrounded fleet episode means, numpy median',
        'exploration': False}
