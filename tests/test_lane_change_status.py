import json
import numpy as np
import pytest
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.backend import engine_class
from deeptraffic_llm_reward.safety_metrics import lane_change_status
from deeptraffic_llm_reward.llm_compact import prepare_compact_request
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter
from deeptraffic_llm_reward.evaluation import run_seeded_episode


def test_status_matches_native_gate_including_fractional_scan_and_map_boundaries():
    eng = engine_class()()
    eng.reset()
    car = eng.z[0]
    for lane, x in [(0, 4), (3, 64), (3, 65), (6, 124)]:
        for gas in (0, 1, 1.37, 2.01):
            for y in (-10, 300.6, 699):
                for obstacle in (False, True):
                    eng.H.reset()
                    if obstacle:
                        eng.H.set(2, 30, 1)
                    car.b, car.x, car.a, car.y = lane, x, gas, y
                    before = (car.b, car.x, car.a, car.y, eng.J)
                    status = lane_change_status(eng, 0)
                    assert before == (car.b, car.x, car.a, car.y, eng.J)
                    for key, direction in [('left', -1), ('right', 1)]:
                        car.b = lane
                        assert status[key]['legal'] == car.lane_change(direction)


@pytest.mark.parametrize('seed', [2000100, 2000101, 2000102])
def test_callback_status_matches_actual_application_for_each_agent(seed):
    env = DeepTrafficEnv(DeepTrafficConfig(episode_frames=200))
    try:
        obs, info = env.reset(seed=seed)
        index = 0
        while True:
            status = info['lane_change_status']
            agent, frame = info['agent_id'], info['frame']
            car = env._eng.z[agent]
            assert status['ego_speed_mph'] == max(0, car.c * car.a) * 20
            assert status['lane_number'] == car.b + 1
            assert status['lane_count'] == 7
            assert status['agent_id'] == agent and status['frame'] == frame
            action = 3 if index % 2 else 4
            obs, _, _, done, info = env.step(action)
            applied = [e for e in info['safety_events'] if e['decision_command']
                       and e['agent_id'] == agent and e['frame'] == frame]
            assert len(applied) == 1
            assert applied[0]['accepted'] == status['left' if action == 3 else 'right']['legal']
            index += 1
            if done:
                assert 'lane_change_status' not in info
                break
    finally:
        env.close()


def test_compact_pipeline_transmits_status_and_never_replaces_the_model_action(tmp_path):
    cfg = DeepTrafficConfig(episode_frames=200)
    class Client:
        def __init__(self):
            self.messages = []
        def chat(self, messages):
            self.messages.append(messages)
            # Deliberately ignore legality: the adapter must execute this unchanged.
            return {'response': {'message': {'content': '{"action":3}'}}}
    client = Client()
    policy = LLMPolicyAdapter(cfg, client, prepare_request=prepare_compact_request, include_lane_status=True)
    result = run_seeded_episode(cfg, 2000100, policy, tmp_path/'trace.jsonl')
    assert result['action_counts']['3'] == result['decisions'] == len(client.messages)
    rows = [json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
    for row, messages in zip(rows, client.messages):
        status = row['decision']['prepared']['lane_change_status']
        assert status['agent_id'] == row['context']['agent_id']
        assert status['frame'] == row['context']['frame']
        assert ('LEFT: CHANGE LEGAL.' if status['left']['legal'] else 'LEFT: CHANGE BLOCKED') in messages[1]['content']
        assert f'EGO SPEED: {status["ego_speed_mph"]:.1f} mph.' in messages[1]['content']
        assert f'CURRENT LANE: {status["lane_number"]} of 7' in messages[1]['content']
        assert 'OVERTAKE:' not in messages[1]['content']
        assert 'overtake' in row['decision']['prepared']
        gains = row['decision']['prepared']['observed_gap_gains']
        for side in ('LEFT', 'RIGHT'):
            assert f'{side} GAP GAIN: {gains[side]:+d} patches.' in messages[1]['content']
    assert result['safety_summary']['decision_commands']['lane_change']['rejected'] > 0


def test_status_is_required_and_changes_state_hash():
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=2, patches_behind=0)
    obs = np.ones(6, dtype=np.float32)
    class Client:
        def chat(self, messages):
            pytest.fail('Missing or stale status must not reach the LLM')
    policy = LLMPolicyAdapter(cfg, Client(), prepare_request=prepare_compact_request, include_lane_status=True)
    with pytest.raises(ValueError, match='Fresh'):
        policy(obs)
    status = dict(agent_id=0, frame=0, current_lane_centered=True,
                  left=dict(legal=True, reason=None), right=dict(legal=False, reason='off_road'))
    with pytest.raises(ValueError, match='Fresh'):
        policy.act_with_info(obs, dict(agent_id=1, frame=0, lane_change_status=status))
    a = prepare_compact_request(obs, cfg, lane_status=status)
    status['left'] = dict(legal=False, reason='target_region_blocked')
    b = prepare_compact_request(obs, cfg, lane_status=status)
    assert a['state_sha256'] != b['state_sha256']
