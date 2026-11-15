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
    assert 'deeptraffic-prompt-v2' in request['messages'][0]['content']
    assert request['prompt_version'] == 'deeptraffic-prompt-v2'
    assert 'RAW-STATE EXAMPLES' in request['messages'][0]['content']
    assert request == prepare_llm_request(obs.copy(), cfg)


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
