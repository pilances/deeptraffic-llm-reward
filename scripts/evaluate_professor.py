"""Evaluate P4 with native legal actions, eleven controlled cars and exact-message caching."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.backend import REVISION
from deeptraffic_llm_reward.evaluation import run_seeded_episode, summarize_matched
from deeptraffic_llm_reward.llm_client import OllamaClient
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter
from deeptraffic_llm_reward.llm_professor import prepare_professor_request, SYS_P4, THRESHOLDS, PROMPT_VERSION
from deeptraffic_llm_reward.safety_metrics import format_safety_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/llm_compact/professor_p4_v2'))
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--variant', choices=['record'], default='record')
    parser.add_argument('--base-url', default='http://localhost:11434',
                        help='Ollama URL, including a local SSH tunnel endpoint')
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2000100, 2000101, 2000102])
    args = parser.parse_args()
    from deeptraffic_llm_reward.llm_professor_record import prepare_professor_request as renderer, PROMPT_VERSION as version
    if args.frames < 1 or not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Require positive frames and unique nonnegative seeds')
    config = DeepTrafficConfig(lanes_side=3, patches_ahead=40, patches_behind=5,
                               other_agents=10, episode_frames=args.frames)
    client = OllamaClient(base_url=args.base_url, model=args.model, keep_alive_seconds=300)
    policy = LLMPolicyAdapter(config, client, prepare_request=renderer,
                             cache_exact=True, include_lane_status=True)
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[Path(__file__).resolve().parents[1]/'src/deeptraffic_llm_reward'/f'{n}.py'
               for n in ('llm_professor', 'llm_professor_record', 'llm_policy', 'llm_client', 'env', 'evaluation', 'backend', 'safety_metrics')]]
    manifest = dict(config=asdict(config), client=asdict(client), seeds=args.seeds,
                    backend_revision=REVISION, original_professor_system_prompt=SYS_P4,
                    prompt_version=version, variant=args.variant, thresholds=THRESHOLDS,
                    cache='exact messages, valid responses only, in-memory', temperature=0,
                    sources_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    reconstruction='Visible P4 text transcribed; decoder reconstructed, not original professor source',
                    native_lane_status=True, action_mask='native gate + constrained JSON action enum; strict validation',
                    protocol='independent seeded episodes; controlled fleet speed, native scheduling; not historical competition protocol')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    progress = dict(status='running', completed=[])
    try:
        for seed in args.seeds:
            result = run_seeded_episode(config, seed, policy, args.out/f'llm_{seed}.jsonl')
            results.append(result)
            progress['completed'].append(seed)
            (args.out/'progress.json').write_text(json.dumps(progress, indent=2))
            print(f'P4 seed={seed}: {result["score_mph"]:.2f} mph', flush=True)
            print(format_safety_table({str(seed): result['safety_summary']}), flush=True)
        summary = summarize_matched({'llm': results}, frames=args.frames)
        (args.out/'comparison.json').write_text(json.dumps(summary, indent=2))
        progress['status'] = 'complete'
    except BaseException as exc:
        progress.update(status='failed', error=str(exc))
        raise
    finally:
        (args.out/'progress.json').write_text(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()
