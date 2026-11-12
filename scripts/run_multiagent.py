"""Test multiple cars in one world; save compact reproducible JSON results."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv, MultiAgentCollector, ReplayBuffer
from deeptraffic_llm_reward.backend import REVISION
from deeptraffic_llm_reward.policies import RuleBasedPolicy, RandomPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=5, help='Controlled cars incl. ego, 1..11')
    parser.add_argument('--policy', choices=['rule', 'random', 'maintain', 'accelerate'], default='rule')
    parser.add_argument('--policy-mode', choices=['shared', 'per-agent'], default='shared')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--frames', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lanes-side', type=int, default=3)
    parser.add_argument('--patches-ahead', type=int, default=40)
    parser.add_argument('--patches-behind', type=int, default=5)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not 1 <= args.agents <= 11 or args.episodes < 1 or args.seed < 0:
        parser.error('Require agents=1..11, positive episodes, nonnegative seed')
    config = DeepTrafficConfig(other_agents=args.agents-1, episode_frames=args.frames,
        lanes_side=args.lanes_side, patches_ahead=args.patches_ahead,
        patches_behind=args.patches_behind)
    env = DeepTrafficEnv(config)
    replay = ReplayBuffer(seed=args.seed)
    collector = MultiAgentCollector(env, replay)
    reports = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        def make_policy(agent):
            if args.policy == 'rule':
                return RuleBasedPolicy(config.lanes_side, config.patches_ahead, config.patches_behind).act
            if args.policy == 'random':
                return RandomPolicy(seed=seed*100 + agent).act
            return lambda obs: 0 if args.policy == 'maintain' else 1
        policies = make_policy(0) if args.policy_mode == 'shared' else {
            agent: make_policy(agent) for agent in env.decision_agent_ids}
        result = collector.collect_episode(policies, seed=seed)
        transitions = result.pop('transitions')
        result['transition_count'] = len(transitions)
        result['final_transition_count'] = sum(t.truncated for t in transitions)
        reports.append(result)
    report = {'backend_revision': REVISION, 'config': asdict(config),
        'policy': args.policy, 'policy_mode': args.policy_mode,
        'protocol': 'independent reset(seed) rollout; NOT historical sequential evaluation',
        'reward_objective': env.reward_objective, 'gamma_per_frame': 1.0,
        'replay_size': len(replay), 'episodes': reports,
        'median_episode_mean_mph': float(np.median([r['episode_mean_speed_mph'] for r in reports]))}
    output = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + '\n')
    print(output)

if __name__ == '__main__':
    main()
