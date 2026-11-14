import json
import pytest
from deeptraffic_llm_reward.env import Action
from deeptraffic_llm_reward.llm_action import ActionParseError, parse_action, resolve_action


@pytest.mark.parametrize('action', list(Action))
def test_all_engine_actions(action):
    text = ' \n' + json.dumps({'action': int(action)}) + '\t'
    assert parse_action(text) == int(action)
    assert type(parse_action(text)) is int
    result = resolve_action(text, on_invalid='maintain')
    assert result.action == action and result.parsed_action == action
    assert result.raw_response == text
    assert not result.fallback_used and result.error_reason is None


@pytest.mark.parametrize('text,reason', [
    ('', 'invalid_json'), ('Go left', 'invalid_json'),
    ('```json\n{"action":3}\n```', 'invalid_json'),
    ('{"action":1} explanation', 'invalid_json'),
    ('{"action":1}{"action":2}', 'invalid_json'),
    ('{"action":', 'invalid_json'),
    ('{}', 'invalid_schema'), ('{"other":1}', 'invalid_schema'),
    ('{"action":1,"reason":"fast"}', 'invalid_schema'),
    ('[3]', 'invalid_schema'), ('3', 'invalid_schema'), ('null', 'invalid_schema'),
    ('{"action":"3"}', 'invalid_action_type'),
    ('{"action":true}', 'invalid_action_type'),
    ('{"action":false}', 'invalid_action_type'),
    ('{"action":3.0}', 'invalid_action_type'),
    ('{"action":3e0}', 'invalid_action_type'),
    ('{"action":null}', 'invalid_action_type'),
    ('{"action":[]}', 'invalid_action_type'),
    ('{"action":{}}', 'invalid_action_type'),
    ('{"action":NaN}', 'invalid_action_type'),
    ('{"action":Infinity}', 'invalid_action_type'),
    ('{"action":-1}', 'invalid_action_value'),
    ('{"action":5}', 'invalid_action_value'),
    ('{"action":1,"action":4}', 'duplicate_key'),
    ('{"action":1,"action":1}', 'duplicate_key'),
])
def test_rejection_and_explicit_fallback(text, reason):
    with pytest.raises(ActionParseError) as error:
        parse_action(text)
    assert error.value.reason == reason
    with pytest.raises(ActionParseError):
        resolve_action(text)
    result = resolve_action(text, on_invalid='maintain')
    assert result.action == int(Action.MAINTAIN)
    assert result.parsed_action is None
    assert result.fallback_used and result.error_reason == reason
    assert result.raw_response == text and result.on_invalid == 'maintain'
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def test_valid_maintain_is_distinct_from_fallback():
    valid = resolve_action('{"action":0}')
    invalid = resolve_action('invalid', on_invalid='maintain')
    assert valid.action == invalid.action == 0
    assert valid.parsed_action == 0 and not valid.fallback_used
    assert invalid.parsed_action is None and invalid.fallback_used


def test_programming_errors_are_not_hidden_by_fallback():
    with pytest.raises(ValueError, match='on_invalid'):
        resolve_action('{"action":1}', on_invalid='repair')
    with pytest.raises(TypeError):
        resolve_action({'action': 1}, on_invalid='maintain')


def test_fallback_emits_structured_warning(caplog):
    result = resolve_action('invalid', on_invalid='maintain')
    assert len(caplog.records) == 1
    assert caplog.records[0].action_resolution == result.to_dict()
    caplog.clear()
    resolve_action('{"action":0}', on_invalid='maintain')
    assert not caplog.records
