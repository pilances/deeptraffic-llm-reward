import json
from dataclasses import replace
import pytest
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.evaluation import run_seeded_episode, summarize_matched


def test_interrupted_episode_replays_actions_without_extra_policy_calls(tmp_path):
    cfg = DeepTrafficConfig(other_agents=10, episode_frames=12)
    calls = []
    def interrupted(obs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError('interrupted transport')
        return 1
    trace = tmp_path/'resume.jsonl'
    with pytest.raises(RuntimeError, match='interrupted transport'):
        run_seeded_episode(cfg, 42, interrupted, trace)
    assert len(trace.read_text().splitlines()) == 2
    resumed_calls = []
    def policy(obs):
        resumed_calls.append(1)
        return 1
    resumed = run_seeded_episode(cfg, 42, policy, trace)
    fresh = run_seeded_episode(cfg, 42, lambda obs: 1, tmp_path/'fresh.jsonl')
    assert resumed['decisions'] == fresh['decisions'] == 4
    assert len(resumed_calls) == 2
    assert resumed['score_mph'] == fresh['score_mph']
    assert resumed['agent_mean_speed_mph'] == fresh['agent_mean_speed_mph']
    def forbidden(obs):
        pytest.fail('Completed replay must not call the policy')
    assert run_seeded_episode(cfg, 42, forbidden, trace)['replayed_decisions'] == 4


def test_replay_rejects_different_seed_and_tampered_outcome(tmp_path):
    cfg = DeepTrafficConfig(episode_frames=4)
    path = tmp_path/'trace.jsonl'
    run_seeded_episode(cfg, 42, lambda obs: 1, path)
    with pytest.raises(ValueError, match='context mismatch'):
        run_seeded_episode(cfg, 43, lambda obs: 1, path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[0]['after']['reward'] = -999
    path.write_text(''.join(json.dumps(row)+'\n' for row in lines))
    with pytest.raises(ValueError, match='outcome mismatch'):
        run_seeded_episode(cfg, 42, lambda obs: 1, path)


def test_comparison_requires_matched_scenarios_and_reports_paired_deltas(tmp_path):
    cfg = DeepTrafficConfig(episode_frames=4)
    rows = [run_seeded_episode(cfg, s, lambda obs: 1, tmp_path/f'{s}.jsonl') for s in (42, 43, 44)]
    summary = summarize_matched({'dqn': rows, 'llm': rows}, frames=4)
    assert summary['llm_paired_comparisons']['dqn']['scenario_deltas_mph'] == [0, 0, 0]
    assert summary['llm_paired_comparisons']['dqn']['ties'] == 3
    assert summary['historical_protocol_match'] is False
    with pytest.raises(ValueError, match='matching scenarios'):
        summarize_matched({'dqn': rows, 'llm': rows[::-1]}, frames=4)
    with pytest.raises(ValueError, match='episode length'):
        summarize_matched({'llm': rows}, frames=2000)
