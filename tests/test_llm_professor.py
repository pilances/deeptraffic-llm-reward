import json
import numpy as np
import pytest
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.llm_professor import prepare_professor_request, parse_professor_action, masked_argmax, legal_actions
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter
from deeptraffic_llm_reward.llm_action import ActionParseError


def status(lane=4, left=True, right=True, centered=True):
    return dict(lane_number=lane, agent_id=0, frame=0, current_lane_centered=centered,
                left=dict(legal=left, reason=None if left else 'target_region_blocked'),
                right=dict(legal=right, reason=None if right else 'target_region_blocked'))


def test_photo_example_and_rear_threshold():
    cfg = DeepTrafficConfig(lanes_side=3, patches_ahead=40, patches_behind=5)
    grid = np.ones((45, 7), dtype=np.float32)
    grid[33, 3] = 41/2000
    grid[19, 2] = 45/2000
    grid[42, 4] = 62/2000
    request = prepare_professor_request(grid.flatten(), cfg, lane_status=status())
    text = request['messages'][1]['content']
    assert 'You are in lane 4 of 7. STATUS: BLOCKED' in text
    assert 'Your lane: SLOW car CLOSE 7p ahead (41 mph).' in text
    assert 'LEFT lane: SLOW car far 21p ahead (45 mph).' in text
    assert 'RIGHT lane: OPEN (clear far ahead).' in text
    assert 'RIGHT lane: FAST car CLOSE 2p behind (62 mph).' in text
    assert 'GAP GAIN' not in text and 'CHANGE LEGAL' in text


@pytest.mark.parametrize('lane', range(1, 8))
def test_lane_inference_and_road_mask(lane):
    cfg = DeepTrafficConfig(lanes_side=3, patches_ahead=40, patches_behind=5)
    grid = np.ones((45, 7), dtype=np.float32)
    for col in range(7):
        if not 1 <= lane + col - 3 <= 7:
            grid[:, col] = 0
    request = prepare_professor_request(grid.flatten(), cfg, lane_status=status(lane, lane > 1, lane < 7))
    assert request['lane_number'] == lane
    legal = request['legal_actions']
    assert (3 in legal) == (lane > 1)
    assert (4 in legal) == (lane < 7)
    if lane == 1:
        assert masked_argmax([0, 1, 0, 100, 2], legal) == 4
        with pytest.raises(ActionParseError):
            parse_professor_action('{"reason":"go","action":"left"}', legal)


def test_named_response_and_exact_cache():
    class Client:
        calls = 0
        def chat(self, messages, *, response_schema=None):
            self.calls += 1
            return {'response': {'message': {'content': json.dumps(dict(reason='Open road.', action='accelerate'))}}}
    cfg = DeepTrafficConfig(lanes_side=3, patches_ahead=40, patches_behind=5)
    client = Client()
    policy = LLMPolicyAdapter(cfg, client, prepare_request=prepare_professor_request, cache_exact=True, include_lane_status=True)
    info = dict(agent_id=0, frame=0, lane_change_status=status())
    assert policy.act_with_info(np.ones(315), info) == 1
    assert policy.act_with_info(np.ones(315), info) == 1
    assert client.calls == 1 and policy.last_record['cache_hit']
    obs = np.ones(315)
    obs[669 % 315] = .03
    assert policy.act_with_info(obs, info) == 1
    assert client.calls == 2
    with pytest.raises(ActionParseError):
        parse_professor_action('{"reason":"go","action":1}', [0, 1, 2, 3, 4])


def test_native_gate_masks_occupied_lane_even_with_free_grid():
    cfg = DeepTrafficConfig(lanes_side=3, patches_ahead=40, patches_behind=5)
    request = prepare_professor_request(np.ones(315), cfg, lane_status=status(right=False))
    assert request['legal_actions'] == [0, 1, 3]
    assert 'RIGHT: CHANGE BLOCKED (target_region_blocked)' in request['messages'][1]['content']
    assert 'right' not in request['response_schema']['properties']['action']['enum']
    with pytest.raises(ActionParseError, match='unavailable_action'):
        parse_professor_action('{"reason":"open","action":"right"}', request['legal_actions'])
    assert legal_actions(status(centered=False)) == [0, 1]
    with pytest.raises(ValueError, match='native'):
        legal_actions(None)


def test_decelerate_excluded_from_message_schema_and_parser():
    cfg = DeepTrafficConfig(lanes_side=3, patches_ahead=40, patches_behind=5)
    request = prepare_professor_request(np.ones(315), cfg, lane_status=status())
    assert 2 not in request['legal_actions']
    assert 'decelerate' not in request['response_schema']['properties']['action']['enum']
    assert 'decelerate' not in request['messages'][1]['content']
    with pytest.raises(ActionParseError, match='unavailable_action'):
        parse_professor_action('{"reason":"wait","action":"decelerate"}', request['legal_actions'])
