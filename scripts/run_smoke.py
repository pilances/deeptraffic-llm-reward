"""Deterministic JS-protocol evaluation of non-learning controls."""
import argparse
import json
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv, Action
from deeptraffic_llm_reward.policies import RandomPolicy, RuleBasedPolicy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['random', 'rule', 'maintain', 'accelerate'], default='rule')
    parser.add_argument('--runs', '--seeds', dest='runs', type=int, default=10)
    parser.add_argument('--frames', type=int, default=2000)
    parser.add_argument('--other-agents', type=int, default=10)
    args = parser.parse_args()
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=args.other_agents, episode_frames=args.frames))
    if args.policy in ('rule', 'random'):
        policy = RuleBasedPolicy() if args.policy == 'rule' else RandomPolicy(seed=0)
        act = policy.act
    else:
        action = Action.MAINTAIN if args.policy == 'maintain' else Action.ACCELERATE
        act = lambda obs: int(action)
    print(json.dumps({'policy': args.policy, **env.evaluate(act, runs=args.runs)}, indent=2))

if __name__ == '__main__':
    main()
