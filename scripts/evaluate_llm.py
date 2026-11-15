"""Frozen LLM versus submitted DQN and rule policy on identical seeded scenarios."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.backend import REVISION
from deeptraffic_llm_reward.evaluation import run_seeded_episode, summarize_matched
from deeptraffic_llm_reward.llm_client import OllamaClient
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter
from deeptraffic_llm_reward.llm_state import prepare_llm_request
from deeptraffic_llm_reward.policies import RuleBasedPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/llm_comparison_short_v1'))
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2000100, 2000101, 2000102])
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('results/submission_eleven_v1/checkpoint.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Require unique nonnegative seeds')
    config = DeepTrafficConfig(lanes_side=6, patches_ahead=52, patches_behind=0,
                               other_agents=10, episode_frames=args.frames)
    client = OllamaClient(keep_alive_seconds=300)
    checkpoint = args.checkpoint.read_bytes()
    net = json.loads(checkpoint)
    layers = net['layers']
    if [l['layer_type'] for l in layers] != ['input', 'fc', 'regression']:
        raise ValueError('Expected the supplied linear 676 -> 5 checkpoint')
    def values(w):
        return [w[str(i)] for i in range(len(w))] if isinstance(w, dict) else w
    weights = np.array([values(f['w']) for f in layers[1]['filters']], dtype=np.float64)
    biases = np.array(values(layers[1]['biases']['w']), dtype=np.float64)
    if weights.shape != (5, 676) or biases.shape != (5,) or not np.isfinite(weights).all() or not np.isfinite(biases).all():
        raise ValueError('Invalid checkpoint dimensions/values')
    with urlopen(client.base_url+'/api/tags', timeout=10) as response:
        model = next(m for m in json.load(response)['models'] if m['name'] == client.model)
    with urlopen(client.base_url+'/api/version', timeout=10) as response:
        version = json.load(response)
    source_root = Path(__file__).resolve().parents[1]
    sources = ['scripts/evaluate_llm.py'] + ['src/deeptraffic_llm_reward/'+n+'.py'
        for n in ('env', 'backend', 'evaluation', 'policies', 'llm_policy', 'llm_state', 'llm_action', 'llm_client')]
    contract = {'config': asdict(config), 'scenario_seeds': args.seeds,
        'client': asdict(client), 'model_digest': model['digest'], 'server_version': version,
        'checkpoint_sha256': hashlib.sha256(checkpoint).hexdigest(), 'backend_revision': REVISION,
        'sources_sha256': {p: hashlib.sha256((source_root/p).read_bytes()).hexdigest() for p in sources},
        'prompt_sha256': prepare_llm_request(np.ones(676, dtype=np.float32), config)['prompt_sha256'],
        'target_mph': 76.6, 'exploration': False, 'fallback': 'raise',
        'protocol': 'fresh independently seeded Gym episodes; floored fleet mean; upper-middle median',
        'controlled_vehicles': 11, 'decision_agent_ids': list(range(10))}
    if args.resume:
        manifest = json.loads((args.out/'manifest.json').read_text())
        if manifest['contract'] != contract:
            raise ValueError('Resume contract differs; use a new directory')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out/'manifest.json').write_text(json.dumps({'contract': contract,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'installed_model': model}, indent=2))
        (args.out/'checkpoint.json').write_bytes(checkpoint)
    policies = {'dqn_submission': lambda obs: int(np.argmax(weights @ obs.astype(np.float64) + biases)),
                'rule': RuleBasedPolicy(6, 52, 0).act,
                'llm': LLMPolicyAdapter(config, client, on_invalid='raise')}
    results = {}
    progress = {'status': 'running', 'completed': {}}
    try:
        for name, policy in policies.items():
            results[name] = []
            progress['completed'][name] = []
            for seed in args.seeds:
                result = run_seeded_episode(config, seed, policy, args.out/f'{name}_{seed}.jsonl')
                results[name].append(result)
                progress['completed'][name].append(result)
                (args.out/'progress.json').write_text(json.dumps(progress, indent=2))
                print(f'{name} seed={seed}: {result["score_mph"]:.2f} mph', flush=True)
        summary = summarize_matched(results, frames=args.frames)
        summary['scenario_seeds'] = args.seeds
        summary['interpretation'] = ('Exploratory short-horizon comparison; historical target is contextual only'
            if args.frames != 2000 else 'Small matched sample; different scenario protocol from historical 500-run target')
        (args.out/'comparison.json').write_text(json.dumps(summary, indent=2))
        progress['status'] = 'complete'
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        progress['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        progress['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.out/'progress.json').write_text(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()
