from dataclasses import replace
import numpy as np
import pytest
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.llm_compact import prepare_compact_request
from deeptraffic_llm_reward.evaluation import run_seeded_episode, summarize_matched


def test_compact_extracts_nearest_front_rear_and_keeps_zero_ambiguous():
    cfg = DeepTrafficConfig(lanes_side=3, patches_ahead=40, patches_behind=5)
    grid = np.ones((45, 7), dtype=np.float32)
    grid[33, 3] = 41 / 2000
    grid[19, 2] = 45 / 2000
    grid[42, 4] = 75 / 2000
    grid[40, 2] = 0
    obs = grid.flatten()
    before = obs.copy()
    result = prepare_compact_request(obs, cfg)
    text = result['messages'][1]['content']
    assert result['decoded_lanes']['CURRENT']['front']['distance_patches'] == 7
    assert result['decoded_lanes']['CURRENT']['front']['speed_mph'] == 41
    assert result['decoded_lanes']['LEFT']['front']['distance_patches'] == 21
    assert result['decoded_lanes']['RIGHT']['rear']['distance_patches'] == 2
    assert 'STATUS: BLOCKED' in text
    assert 'FAST traffic CLOSE, 2 patches away, 75 mph' in text
    assert 'zero-valued (off-road or stopped traffic; ambiguous)' in text
    assert 'grid' not in text and len(text) < 1200
    np.testing.assert_array_equal(obs, before)
    assert result == prepare_compact_request(obs.copy(), cfg)


def test_zero_and_missing_observations_do_not_invent_speed_or_rear_clearance():
    cfg = DeepTrafficConfig(lanes_side=0, patches_ahead=2, patches_behind=0)
    result = prepare_compact_request(np.zeros(2), cfg)
    hit = result['decoded_lanes']['CURRENT']['front']
    assert hit['speed_mph'] is None and hit['ambiguous_zero']
    assert not result['decoded_lanes']['LEFT']['observed']
    cfg = replace(cfg, lanes_side=1)
    assert 'behind: unknown' in prepare_compact_request(np.ones(6), cfg)['messages'][1]['content']
    with pytest.raises(ValueError):
        prepare_compact_request(np.ones(5), cfg)


def test_observation_window_changes_keep_identical_physics_for_fixed_actions(tmp_path):
    old = DeepTrafficConfig(lanes_side=6, patches_ahead=52, patches_behind=0, episode_frames=200)
    new = replace(old, lanes_side=3, patches_ahead=40, patches_behind=5)
    for seed in (2000100, 2000101, 2000102):
        a = run_seeded_episode(old, seed, lambda obs: 1, tmp_path/f'old_{seed}.jsonl')
        b = run_seeded_episode(new, seed, lambda obs: 1, tmp_path/f'new_{seed}.jsonl')
        assert a['initial_state_sha256'] != b['initial_state_sha256']
        assert a['initial_traffic_sha256'] == b['initial_traffic_sha256']
        assert a['agent_mean_speed_mph'] == b['agent_mean_speed_mph']
        assert summarize_matched({'dqn': [a], 'llm': [b]}, frames=200)['llm_paired_comparisons']['dqn']['ties'] == 1
