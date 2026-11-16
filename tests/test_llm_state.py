import json
import numpy as np
import pytest
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.llm_state import prepare_llm_request


def test_layout_round_trip_and_action_mapping():
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=2, patches_behind=1)
    obs = np.arange(9, dtype=np.float32)/10
    request = prepare_llm_request(obs, cfg)
    state = json.loads(request['messages'][1]['content'])
    np.testing.assert_array_equal(np.array(state['grid']).flatten(), obs)
    assert state['longitudinal_offsets'] == [-2,-1,0]
    assert state['lane_offsets_left_to_right'] == [-1,0,1]
    assert '"0": "MAINTAIN"' in request['messages'][0]['content']
    assert '"3": "CHANGE_LEFT"' in request['messages'][0]['content']
    assert 'deeptraffic-prompt-v3' in request['messages'][0]['content']
    assert request['prompt_version'] == 'deeptraffic-prompt-v3'
    assert 'No cells behind the controlled vehicle are supplied' in request['messages'][0]['content']
    assert request == prepare_llm_request(obs.copy(), cfg)


def test_no_rear_observation_is_not_described_as_visible():
    cfg = DeepTrafficConfig(lanes_side=6, patches_ahead=52, patches_behind=0)
    request = prepare_llm_request(np.ones(676, dtype=np.float32), cfg)
    prompt = request['messages'][0]['content']
    assert 'No cells behind the controlled vehicle are supplied' in prompt
    assert 'Rear traffic is unknown' in prompt
    assert 'a separate ego speed' in prompt
    assert 'whether another vehicle is slower than you' in prompt
    assert 'sufficiently clear region ahead and behind' not in prompt
    assert 'slower vehicle blocks' not in prompt


def test_positive_offsets_are_described_as_rear_observations():
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=2, patches_behind=3)
    request = prepare_llm_request(np.ones(15, dtype=np.float32), cfg)
    state = json.loads(request['messages'][1]['content'])
    assert state['longitudinal_offsets'] == [-2, -1, 0, 1, 2]
    assert '2 rows behind the vehicle are supplied' in request['messages'][0]['content']


@pytest.mark.parametrize('obs', [[0], [float('nan')]*3, [-.1]*3, [1.1]*3])
def test_invalid_observation(obs):
    with pytest.raises(ValueError):
        prepare_llm_request(obs, DeepTrafficConfig(lanes_side=1, patches_ahead=1))


def test_real_benchmark_observation_without_simulation_advance():
    cfg = DeepTrafficConfig(lanes_side=6, patches_ahead=52, patches_behind=0)
    env = DeepTrafficEnv(cfg)
    try:
        obs, info = env.reset(seed=42)
        before = obs.copy()
        request = prepare_llm_request(obs, cfg)
        grid = json.loads(request['messages'][1]['content'])['grid']
        assert np.asarray(grid).shape == (52,13)
        np.testing.assert_array_equal(before, obs)
    finally:
        env.close()
