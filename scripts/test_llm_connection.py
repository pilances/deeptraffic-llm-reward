"""One real prompt with a synthetic state; no simulator actions or training."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import urlopen
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.llm_client import OllamaClient
from deeptraffic_llm_reward.llm_state import prepare_llm_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:11434')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--out', type=Path, default=Path('results/llm_connection_v1'))
    args = parser.parse_args()
    client = OllamaClient(args.url, args.model)
    args.out.mkdir(parents=True, exist_ok=False)
    # Metadata requests do not perform inference. One chat call follows below.
    with urlopen(args.url.rstrip('/')+'/api/tags', timeout=10) as response:
        models = json.load(response)['models']
    model = next((m for m in models if m['name'] == args.model), None)
    if model is None:
        raise RuntimeError('Requested model is not installed')
    with urlopen(args.url.rstrip('/')+'/api/version', timeout=10) as response:
        server_version = json.load(response)
    config = DeepTrafficConfig(lanes_side=6, patches_ahead=52, patches_behind=0, other_agents=10)
    grid = np.ones((52,13), dtype=np.float32)
    grid[:, :3] = 0
    grid[:, 10:] = 0
    grid[45:49, 6] = 50/2000  # fictional slower vehicle ahead
    prepared = prepare_llm_request(grid.flatten(), config)
    (args.out/'prepared.json').write_text(json.dumps(prepared, indent=2))
    started = time.monotonic()
    exchange = client.chat(prepared['messages'])  # Exactly one inference request.
    report = {'success': True, 'synthetic_state': True, 'inference_requests': 1,
        'simulation_actions_executed': 0, 'utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.monotonic()-started, 'client': asdict(client),
        'server_version': server_version, 'installed_model': model,
        'state_config': asdict(config), **exchange}
    (args.out/'exchange.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'success': True, 'model': args.model,
        'elapsed_seconds': report['elapsed_seconds'],
        'reply': exchange['response']['message']['content'], 'artifacts': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
