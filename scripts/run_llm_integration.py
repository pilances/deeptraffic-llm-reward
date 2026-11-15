"""Bounded live adapter checks: synthetic, one step, two short episodes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.llm_client import OllamaClient
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/llm_integration_v1'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    client = OllamaClient()
    cfg = DeepTrafficConfig(lanes_side=6, patches_ahead=52,
                            patches_behind=0, other_agents=10, episode_frames=6)
    policy = LLMPolicyAdapter(cfg, client, on_invalid='raise')
    with urlopen(client.base_url+'/api/tags', timeout=10) as response:
        metadata = json.load(response)
    report = {'config': asdict(cfg), 'client': asdict(client),
              'models': metadata, 'episodes': [], 'benchmark': False}
    def decide(obs, context):
        try:
            return policy(obs)
        finally:
            with (args.out/'decisions.jsonl').open('a') as log:
                log.write(json.dumps({'context': context, **policy.last_record})+'\n')
    try:
        report['synthetic_action'] = decide(np.ones(676, dtype=np.float32), 'synthetic')
        for label, seed, single in [('single_step', 42, True),
                                    ('smoke', 42, False), ('smoke', 43, False)]:
            env = DeepTrafficEnv(cfg)
            episode = {'label': label, 'seed': seed, 'steps': []}
            report['episodes'].append(episode)
            try:
                obs, info = env.reset(seed=seed)
                while True:
                    agent = info['agent_id']
                    action = decide(obs, {'label': label, 'seed': seed,
                                         'agent_id': agent, 'frame': info['frame']})
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode['steps'].append({'agent_id': agent, 'action': action,
                        'reward': reward, 'frame': info['frame'],
                        'terminated': terminated, 'truncated': truncated,
                        'next_state': obs.tolist()})
                    if single or terminated or truncated:
                        break
                episode['mean_speed_mph'] = info['episode_mean_speed_mph']
            finally:
                env.close()
        report['success'] = True
    finally:
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'success': True, 'episodes': [
        {'label': e['label'], 'seed': e['seed'], 'steps': len(e['steps'])}
        for e in report['episodes']]}, indent=2))


if __name__ == '__main__':
    main()
