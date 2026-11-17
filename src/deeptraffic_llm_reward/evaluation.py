"""Matched seeded fleet evaluation, with an auditable action replay journal."""
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from .env import DeepTrafficEnv
from .safety_metrics import summarize_safety


def observation_hash(observation):
    return hashlib.sha256(np.asarray(observation, dtype='<f4').tobytes()).hexdigest()


def run_seeded_episode(config, seed, policy, trace_path):
    """Resume by replaying recorded actions, checking every pre/post state.

    Each seed starts a fresh environment. Only successfully executed decisions
    enter the journal; failures are saved separately and abort the evaluation.
    No retries or replacement actions are introduced here.
    """
    path = Path(trace_path)
    previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    env = DeepTrafficEnv(config)
    started = perf_counter()
    rows = []
    try:
        obs, info = env.reset(seed=seed)
        initial_hash = observation_hash(obs)
        initial_traffic_hash = hashlib.sha256(json.dumps(
            [(car.b, car.y, car.a, car.c) for car in env._eng.z],
            allow_nan=False).encode()).hexdigest()
        index = 0
        while True:
            context = {'seed': seed, 'index': index, 'agent_id': info['agent_id'],
                       'frame': info['frame'], 'state_sha256': observation_hash(obs)}
            if index < len(previous):
                row = previous[index]
                if row['context'] != context:
                    raise ValueError('Replay state/context mismatch')
                action = row['action']
            else:
                before = perf_counter()
                try:
                    act_with_info = getattr(policy, 'act_with_info', None)
                    action = int(act_with_info(obs, info) if callable(act_with_info) else policy(obs))
                except Exception as exc:
                    with path.with_suffix('.errors.jsonl').open('a') as stream:
                        stream.write(json.dumps({'context': context,
                            'error': str(exc), 'decision': getattr(policy, 'last_record', None)})+'\n')
                    raise
                row = {'context': context, 'action': action,
                       'latency_seconds': perf_counter()-before,
                       'decision': getattr(policy, 'last_record', None)}
            obs, reward, terminated, truncated, info = env.step(action)
            after = {'reward': reward, 'next_state_sha256': observation_hash(obs),
                     'next_agent_id': info['agent_id'], 'frame': info['frame'],
                     'terminated': terminated, 'truncated': truncated}
            if index < len(previous):
                if row['after'] != after:
                    raise ValueError('Replay outcome mismatch')
                if 'safety_events' in row and row['safety_events'] != info['safety_events']:
                    raise ValueError('Replay safety events mismatch')
            else:
                row['safety_events'] = info['safety_events']
                row['after'] = after
                with path.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                if (index+1) % 10 == 0:
                    print(f'{path.stem}: {index+1} decisions, frame {info["frame"]}', flush=True)
            rows.append(row)
            index += 1
            if terminated or truncated:
                break
        if len(previous) > index:
            raise ValueError('Replay journal exceeds episode length')
        raw = info['episode_mean_speed_mph']
        result = {'seed': seed, 'frames': info['frame'], 'decisions': index,
                  'initial_state_sha256': initial_hash, 'raw_mean_speed_mph': raw,
                  'initial_traffic_sha256': initial_traffic_hash,
                  'score_mph': math.floor(raw*100)/100,
                  'agent_mean_speed_mph': info['agent_mean_speed_mph'],
                  'action_counts': {str(a): sum(r['action'] == a for r in rows) for a in range(5)},
                  'fallback_count': sum(bool((r['decision'] or {}).get('resolution', {}).get('fallback_used'))
                                        for r in rows),
                  'original_decision_seconds': sum(r['latency_seconds'] for r in rows),
                  'execution_seconds': perf_counter()-started, 'replayed_decisions': len(previous)}
        result['safety_summary'] = info['safety_summary']
        result['safety_events'] = env._safety_events.copy()
        path.with_suffix('.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


def summarize_matched(results, target=76.6, frames=2000):
    """Summarize only completed, paired scenarios; never silently drop failures."""
    names = list(results)
    reference = results[names[0]]
    scenario_hash = lambda r: r.get('initial_traffic_sha256', r['initial_state_sha256'])
    keys = [(r['seed'], r['frames'], scenario_hash(r)) for r in reference]
    if not keys or len({r['seed'] for r in reference}) != len(keys):
        raise ValueError('Require unique, nonempty scenarios')
    summary = {'frames': frames, 'historical_target_mph': target,
               'historical_protocol_match': False, 'policies': {}}
    for name, rows in results.items():
        if [(r['seed'], r['frames'], scenario_hash(r)) for r in rows] != keys:
            raise ValueError('Policies were not evaluated on matching scenarios')
        if any(r['frames'] != frames for r in rows):
            raise ValueError('Unexpected episode length')
        scores = [r['score_mph'] for r in rows]
        median = sorted(scores)[len(scores)//2]
        summary['policies'][name] = {'scores_mph': scores, 'median_mph': median,
            'mean_mph': float(np.mean(scores)), 'min_mph': min(scores), 'max_mph': max(scores),
            'delta_to_historical_target_mph': median-target,
            'decisions': sum(r['decisions'] for r in rows),
            'fallback_count': sum(r['fallback_count'] for r in rows),
            'decision_seconds': sum(r['original_decision_seconds'] for r in rows),
            'action_counts': {str(a): sum(r['action_counts'][str(a)] for r in rows) for a in range(5)}}
        if all('safety_events' in r for r in rows):
            summary['policies'][name]['safety_summary'] = summarize_safety(
                [event for r in rows for event in r['safety_events']])
    if 'llm' in results:
        summary['llm_paired_comparisons'] = {}
        llm = np.array([r['score_mph'] for r in results['llm']])
        for name in names:
            if name != 'llm':
                delta = llm - np.array([r['score_mph'] for r in results[name]])
                summary['llm_paired_comparisons'][name] = {
                    'scenario_deltas_mph': delta.tolist(), 'mean_delta_mph': float(delta.mean()),
                    'wins': int(np.sum(delta > 0)), 'ties': int(np.sum(delta == 0)),
                    'losses': int(np.sum(delta < 0))}
    return summary
