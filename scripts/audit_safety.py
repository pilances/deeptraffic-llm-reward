"""Replay existing action journals to measure safety, without any LLM requests."""
import argparse
import json
from pathlib import Path
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.evaluation import observation_hash
from deeptraffic_llm_reward.safety_metrics import summarize_safety, format_safety_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    contract = json.loads((args.experiment/'manifest.json').read_text())['contract']
    reports, all_events = {}, {}
    for policy in ('dqn_submission', 'rule', 'llm'):
        config = contract.get('llm_config', contract['config']) if policy == 'llm' else contract['config']
        cfg = DeepTrafficConfig(**config)
        all_events[policy] = []
        for seed in contract['scenario_seeds']:
            path = args.experiment/f'{policy}_{seed}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            env = DeepTrafficEnv(cfg)
            events = []
            try:
                obs, info = env.reset(seed=seed)
                for index, row in enumerate(rows):
                    context = dict(seed=seed, index=index, agent_id=info['agent_id'],
                                   frame=info['frame'], state_sha256=observation_hash(obs))
                    if row['context'] != context:
                        raise ValueError(f'{path}: replay context mismatch')
                    obs, reward, terminated, truncated, info = env.step(row['action'])
                    after = dict(reward=reward, next_state_sha256=observation_hash(obs),
                        next_agent_id=info['agent_id'], frame=info['frame'],
                        terminated=terminated, truncated=truncated)
                    if row['after'] != after:
                        raise ValueError(f'{path}: replay outcome mismatch')
                    events.extend(info['safety_events'])
                if not rows or not (terminated or truncated):
                    raise ValueError(f'{path}: incomplete episode')
            finally:
                env.close()
            reports[f'{policy} / {seed}'] = summarize_safety(events)
            all_events[policy].extend(events)
    totals = {name: summarize_safety(events) for name, events in all_events.items()}
    output = args.out or args.experiment/'safety_audit'
    output.mkdir(parents=True, exist_ok=False)
    (output/'safety_report.json').write_text(json.dumps(dict(episodes=reports, totals=totals, events=all_events), indent=2))
    table = format_safety_table(totals) + '\n' + format_safety_table(reports)
    (output/'safety_overview.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
