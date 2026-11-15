from copy import deepcopy
import json
import numpy as np
import pytest
from deeptraffic_llm_reward import Action, DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.llm_action import ActionParseError
from deeptraffic_llm_reward.llm_client import LLMCommunicationError
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter
from deeptraffic_llm_reward.llm_state import prepare_llm_request


class SimulatedClient:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def chat(self, messages):
        self.calls.append(deepcopy(messages))
        if isinstance(self.text, Exception):
            raise self.text
        return {'request': {'model': 'simulated', 'messages': deepcopy(messages)},
                'response': {'done': True, 'message': {'content': self.text}}}


@pytest.fixture(autouse=True)
def forbid_network_and_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Adapter tests must not call a network or simulator')
    monkeypatch.setattr('deeptraffic_llm_reward.llm_client.urlopen', forbidden)
    monkeypatch.setattr(DeepTrafficEnv, 'reset', forbidden)
    monkeypatch.setattr(DeepTrafficEnv, 'step', forbidden)


def small_state():
    return DeepTrafficConfig(lanes_side=1, patches_ahead=2, patches_behind=1), np.arange(9, dtype=np.float32)/10


@pytest.mark.parametrize('action', list(Action))
def test_composes_all_three_components(action):
    cfg, obs = small_state()
    before = obs.copy()
    client = SimulatedClient(json.dumps({'action': int(action)}))
    policy = LLMPolicyAdapter(cfg, client)
    assert policy.last_record is None and client.calls == []
    assert policy(obs) == int(action)
    assert client.calls == [prepare_llm_request(obs, cfg)['messages']]
    record = policy.last_record
    assert record['resolution']['action'] == int(action)
    assert record['status'] == 'ok' and record['elapsed_seconds'] >= 0
    assert record['exchange']['response']['message']['content'] == client.text
    assert json.loads(json.dumps(record)) == record
    np.testing.assert_array_equal(obs, before)


def test_benchmark_state_is_passed_in_full_without_history():
    cfg = DeepTrafficConfig(lanes_side=6, patches_ahead=52, patches_behind=0)
    client = SimulatedClient('{"action":1}')
    policy = LLMPolicyAdapter(cfg, client)
    first = np.ones(676, dtype=np.float32)
    policy(first)
    record = policy.last_record
    second = first.copy()
    second[0] = 0
    policy(second)
    state = json.loads(client.calls[-1][1]['content'])
    assert np.asarray(state['grid']).shape == (52,13)
    assert len(client.calls) == 2 and all(len(call) == 2 for call in client.calls)
    assert policy.last_record['prepared']['state_sha256'] != record['prepared']['state_sha256']


def test_parser_error_keeps_response_but_returns_no_action():
    cfg, obs = small_state()
    client = SimulatedClient('{"action":1}')
    policy = LLMPolicyAdapter(cfg, client)
    assert policy(obs) == 1
    client.text = '{"action":true}'
    with pytest.raises(ActionParseError):
        policy(obs)
    assert policy.last_record['status'] == 'error'
    assert policy.last_record['stage'] == 'parse'
    assert policy.last_record['error_reason'] == 'invalid_action_type'
    assert policy.last_record['resolution'] is None
    assert policy.last_record['exchange']['response']['message']['content'] == client.text


def test_explicit_fallback_is_recorded():
    cfg, obs = small_state()
    policy = LLMPolicyAdapter(cfg, SimulatedClient('invalid'), on_invalid='maintain')
    assert policy(obs) == int(Action.MAINTAIN)
    assert policy.last_record['status'] == 'fallback'
    assert policy.last_record['resolution']['fallback_used'] is True
    assert policy.last_record['resolution']['parsed_action'] is None


def test_transport_failure_is_not_a_fallback():
    cfg, obs = small_state()
    client = SimulatedClient(LLMCommunicationError('offline'))
    policy = LLMPolicyAdapter(cfg, client, on_invalid='maintain')
    with pytest.raises(LLMCommunicationError):
        policy(obs)
    assert len(client.calls) == 1
    assert policy.last_record['stage'] == 'chat'
    assert policy.last_record['resolution'] is None


def test_invalid_state_never_reaches_client():
    cfg, _ = small_state()
    client = SimulatedClient('{"action":1}')
    policy = LLMPolicyAdapter(cfg, client)
    with pytest.raises(ValueError):
        policy(np.array([np.nan]))
    assert not client.calls
    assert policy.last_record['stage'] == 'prepare'


def test_constructor_rejects_invalid_configuration():
    cfg, _ = small_state()
    with pytest.raises(ValueError):
        LLMPolicyAdapter(cfg, SimulatedClient(''), on_invalid='guess')
    with pytest.raises(TypeError):
        LLMPolicyAdapter(cfg, object())
