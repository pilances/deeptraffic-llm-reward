"""Prespecified 5-car shared DQN versus rule-policy experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.dqn import DQNConfig, train_shared_dqn, evaluate_fleet
from deeptraffic_llm_reward.policies import RuleBasedPolicy


def summarize(root, baseline=None):
    rule = json.loads((root/'rule.json').read_text())
    rows = []
    manifests = []
    for seed in (0, 1, 2):
        directory = root/f'seed_{seed}'
        report = json.loads((directory/'evaluation.json').read_text())
        manifest = json.loads((directory/'manifest.json').read_text())
        manifests.append(manifest)
        for key in ('scenario_seeds', 'env_config', 'backend_revision', 'protocol', 'exploration'):
            if report[key] != rule[key]:
                raise ValueError(f'Incompatible comparison: {key}')
        if set(manifest['training_scenario_seeds']) & set(report['scenario_seeds']):
            raise ValueError('Training/test leakage')
        delta = np.asarray(report['scores_mph']) - np.asarray(rule['scores_mph'])
        rng = np.random.default_rng(912)
        bootstrap = delta[rng.integers(0, len(delta), size=(10000,len(delta)))].mean(axis=1)
        rows.append({'training_seed': seed, 'median_mph': report['median_mph'],
            'mean_mph': report['mean_mph'], 'iqr_mph': report['iqr_mph'],
            'median_delta_to_rule': report['median_mph']-rule['median_mph'],
            'paired_mean_delta': float(delta.mean()),
            'paired_mean_delta_ci95': np.percentile(bootstrap, [2.5,97.5]).tolist(),
            'scenario_win_fraction': float(np.mean(delta>0))})
    if any(m['dqn_config'] != manifests[0]['dqn_config'] for m in manifests):
        raise ValueError('Unequal DQN training configurations')
    training_seeds = [set(m['training_scenario_seeds']) for m in manifests]
    if any(training_seeds[i] & training_seeds[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Overlapping training scenarios across replicates')
    summary = {'rule': {k:rule[k] for k in ('median_mph','mean_mph','iqr_mph')},
        'dqn': rows, 'mean_dqn_median_mph': float(np.mean([r['median_mph'] for r in rows])),
        'std_dqn_median_mph': float(np.std([r['median_mph'] for r in rows], ddof=1)),
        'evaluation_scenarios': len(rule['scenario_seeds']), 'env_config': rule['env_config'],
        'dqn_config': manifests[0]['dqn_config'], 'backend_revision': rule['backend_revision'],
        'ci_scope': 'paired scenario bootstrap conditional on each trained checkpoint; not training-seed uncertainty'}
    if baseline is not None:
        architecture_rows = []
        for seed, manifest in enumerate(manifests):
            old_dir = baseline/f'seed_{seed}'
            old_manifest = json.loads((old_dir/'manifest.json').read_text())
            old_report = json.loads((old_dir/'evaluation.json').read_text())
            report = json.loads((root/f'seed_{seed}'/'evaluation.json').read_text())
            for key in ('training_seed', 'training_scenario_seeds', 'env_config',
                        'backend_revision', 'torch', 'numpy', 'python', 'objective',
                        'checkpoint_selection', 'algorithm'):
                if manifest[key] != old_manifest[key]:
                    raise ValueError(f'Architecture comparison differs in {key}')
            for key in ('scenario_seeds', 'env_config', 'backend_revision', 'protocol', 'exploration'):
                if report[key] != old_report[key]:
                    raise ValueError(f'Architecture evaluation differs in {key}')
            if ({k:v for k,v in manifest['dqn_config'].items() if k != 'hidden'} !=
                    {k:v for k,v in old_manifest['dqn_config'].items() if k != 'hidden'}):
                raise ValueError('Architecture comparison changes other hyperparameters')
            delta = np.asarray(report['scores_mph']) - np.asarray(old_report['scores_mph'])
            rng = np.random.default_rng(912)
            bootstrap = delta[rng.integers(0, len(delta), size=(10000,len(delta)))].mean(axis=1)
            architecture_rows.append({'training_seed': seed,
                'baseline_median_mph': old_report['median_mph'],
                'new_median_mph': report['median_mph'],
                'median_delta_mph': report['median_mph']-old_report['median_mph'],
                'paired_mean_delta_mph': float(delta.mean()),
                'paired_mean_delta_ci95': np.percentile(bootstrap,[2.5,97.5]).tolist(),
                'scenario_win_fraction': float(np.mean(delta>0))})
        summary['architecture_comparison'] = {
            'baseline_directory': str(baseline),
            'baseline_hidden': old_manifest['dqn_config']['hidden'],
            'new_hidden': manifests[0]['dqn_config']['hidden'],
            'seeds': architecture_rows,
            'mean_median_delta_mph': float(np.mean([r['median_delta_mph'] for r in architecture_rows])),
            'interpretation': 'Previously inspected benchmark reused for requested architecture comparison; not a new untouched final test.'}
    (root/'comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','eval','rule','compare'], required=True)
    parser.add_argument('--seed', type=int, choices=[0,1,2], default=0)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--runs', type=int, default=500)
    parser.add_argument('--hidden', type=int, nargs='+', default=[128,128],
                        help='ReLU hidden-layer widths, e.g. 128 128 64 64')
    parser.add_argument('--baseline', type=Path,
                        help='Compare paired seeds to an existing architecture experiment (compare mode)')
    parser.add_argument('--controlled-agents', type=int, choices=range(1, 11), default=5,
                        help='Number of controlled cars; default 5')
    parser.add_argument('--out', type=Path, default=Path('results/dqn_five_v1'))
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'compare':
        summarize(root, args.baseline.resolve() if args.baseline else None)
        return
    if args.runs < 1 or args.runs > 10000:
        parser.error('Require runs in 1..10000')
    env_config = DeepTrafficConfig(other_agents=args.controlled_agents-1, lanes_side=3, patches_ahead=40,
                                   patches_behind=5, episode_frames=2000)
    scenarios = list(range(2_000_000, 2_000_000+args.runs))
    if args.mode == 'rule':
        path = root/'rule.json'
        if path.exists():
            raise FileExistsError(path)
        policy = RuleBasedPolicy(3,40,5).act
    elif args.mode == 'eval':
        directory = root/f'seed_{args.seed}'
        from deeptraffic_llm_reward.dqn import DQNAgent
        agent, checkpoint_env = DQNAgent.load(directory/'final.pt')
        if checkpoint_env != env_config:
            raise ValueError('Checkpoint environment does not match evaluation environment')
        policy = agent.greedy
        path = directory/'evaluation.json'
    else:
        directory = root/f'seed_{args.seed}'
        policy = train_shared_dqn(args.seed, directory,
                                 DQNConfig(episodes=args.episodes, hidden=tuple(args.hidden)), env_config).greedy
        path = directory/'evaluation.json'
    report = evaluate_fleet(policy, env_config, scenarios)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({'mode':args.mode, 'seed':args.seed,
                     'median_mph':report['median_mph'], 'mean_mph':report['mean_mph']}, indent=2))

if __name__ == '__main__':
    main()
