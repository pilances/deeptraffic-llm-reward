from dataclasses import replace
import numpy as np
import pytest
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.llm_compact import prepare_compact_request
from deeptraffic_llm_reward.evaluation import run_seeded_episode, summarize_matched


@pytest.mark.parametrize('current,left,right,expected', [
    (8, 20, 9, {'LEFT': 12, 'RIGHT': 1}),
    (8, 12, 11, {'LEFT': 4, 'RIGHT': 3}),
    (8, 1, None, {'LEFT': -7, 'RIGHT': 44}),
    (None, 20, None, {'LEFT': -32, 'RIGHT': 0}),
])
def test_gap_gains_use_observed_clearance_and_keep_overtake_out_of_messages(current, left, right, expected):
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=52, patches_behind=10)
    grid = np.ones((62, 3), dtype=np.float32)
    for col, distance in enumerate((left, current, right)):
        if distance is not None:
            grid[52-distance, col] = 60 / 2000
    result = prepare_compact_request(grid.flatten(), cfg)
    assert result['observed_gap_gains'] == expected
    text = result['messages'][1]['content']
    for side, gain in expected.items():
        assert f'{side} GAP GAIN: {gain:+d} patches.' in text
    for message in result['messages']:
        assert 'OVERTAKE' not in message['content']
        assert 'candidate_lanes' not in message['content']
        assert 'Candidate lanes' not in message['content']
    assert 'overtake' in result and 'candidate_lanes' in result['overtake']


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
    assert result['observed_gap_gains'] == {'LEFT': None, 'RIGHT': None}
    cfg = replace(cfg, lanes_side=1)
    assert 'behind: not observed; rely on native CHANGE LEGAL/BLOCKED status.' in prepare_compact_request(np.ones(6), cfg)['messages'][1]['content']
    with pytest.raises(ValueError):
        prepare_compact_request(np.ones(5), cfg)


@pytest.mark.parametrize('behind', [0, 1, 5])
def test_rear_description_distinguishes_unobserved_and_visible_clear(behind):
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=2, patches_behind=behind)
    grid = np.ones((2 + behind, 3), dtype=np.float32)
    request = prepare_compact_request(grid.flatten(), cfg)
    text = request['messages'][1]['content']
    for direction in ('LEFT', 'RIGHT'):
        if behind <= 1:
            assert f'{direction}: behind: not observed;' in text
        else:
            assert f'{direction}: behind: no rear vehicle observed in the visible rear region;' in text
    assert 'do not by themselves disqualify' in request['messages'][0]['content']
    if behind > 1:
        grid[3, 0] = 0
        text = prepare_compact_request(grid.flatten(), cfg)['messages'][1]['content']
        assert 'LEFT: behind: zero-valued region 1 patches away' in text


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


@pytest.mark.parametrize('distance,legal,zero,expected', [
    (8, True, False, 'REQUIRED'), (20, True, False, 'OPTIONAL'),
    (8, False, False, 'NOT USEFUL'), (8, True, True, 'NOT USEFUL'),
])
def test_overtake_uses_relative_speed_clearance_and_native_gate(distance, legal, zero, expected):
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=52, patches_behind=10)
    grid = np.ones((62, 3), dtype=np.float32)
    grid[52-distance, 1] = 0 if zero else 60 / 2000
    status = dict(current_lane_centered=True, ego_speed_mph=75, lane_number=4, lane_count=7,
                  left=dict(legal=False, reason='target_region_blocked'),
                  right=dict(legal=legal, reason=None if legal else 'off_road'))
    result = prepare_compact_request(grid.flatten(), cfg, lane_status=status)
    assert result['overtake']['status'] == expected
    assert result['overtake']['candidate_lanes'] == (['RIGHT'] if expected != 'NOT USEFUL' else [])
    assert 'EGO SPEED: 75.0 mph.' in result['messages'][1]['content']
    assert 'CURRENT LANE: 4 of 7' in result['messages'][1]['content']
    status['ego_speed_mph'] = 62
    slower = prepare_compact_request(grid.flatten(), cfg, lane_status=status)
    assert slower['overtake']['status'] == expected
    assert slower['state_sha256'] != result['state_sha256']
    status['ego_speed_mph'] = 75
    status['current_lane_centered'] = False
    assert prepare_compact_request(grid.flatten(), cfg, lane_status=status)['overtake']['status'] == 'NOT USEFUL'


def test_no_early_overtake_without_clearance_gain_or_known_blocker():
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=52, patches_behind=10)
    grid = np.ones((62, 3), dtype=np.float32)
    grid[32, 1] = 60 / 2000
    grid[27, 2] = 75 / 2000  # Only five extra patches: insufficient for OPTIONAL.
    status = dict(current_lane_centered=True, ego_speed_mph=75, lane_number=4, lane_count=7,
                  left=dict(legal=False, reason='off_road'), right=dict(legal=True, reason=None))
    assert prepare_compact_request(grid.flatten(), cfg, lane_status=status)['overtake']['status'] == 'NOT USEFUL'
    grid[:, 1] = 1
    assert prepare_compact_request(grid.flatten(), cfg, lane_status=status)['overtake']['status'] == 'NOT USEFUL'
    assert prepare_compact_request(grid.flatten(), cfg)['overtake']['status'] == 'UNKNOWN'


def test_slow_far_target_lane_is_candidate_when_legal():
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=52, patches_behind=10)
    grid = np.ones((62, 3), dtype=np.float32)
    grid[52-8, 1] = 60 / 2000
    grid[52-30, 2] = 60 / 2000
    status = dict(current_lane_centered=True, ego_speed_mph=75, lane_number=4, lane_count=7,
                  left=dict(legal=False, reason='off_road'), right=dict(legal=True, reason=None))
    result = prepare_compact_request(grid.flatten(), cfg, lane_status=status)
    assert result['overtake']['status'] == 'REQUIRED'
    assert result['overtake']['candidate_lanes'] == ['RIGHT']
