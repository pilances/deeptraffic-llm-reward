import io
import json
from unittest.mock import patch
from urllib.error import URLError
import pytest
from deeptraffic_llm_reward.llm_client import OllamaClient, LLMCommunicationError


def test_single_post_preserves_raw_reply():
    reply = {'done': True, 'message': {'content': '{"action":3}'}}
    with patch('deeptraffic_llm_reward.llm_client.urlopen',
               return_value=io.BytesIO(json.dumps(reply).encode())) as transport:
        exchange = OllamaClient().chat([{'role':'user', 'content':'fictional state'}])
    transport.assert_called_once()
    req = transport.call_args.args[0]
    assert req.get_method() == 'POST'
    payload = json.loads(req.data)
    assert payload['think'] is False and payload['stream'] is False
    assert payload['options']['temperature'] == 0
    assert exchange['response'] == reply


@pytest.mark.parametrize('reply', [[], {}, {'done':False},
    {'done':True, 'message':{'content':''}},
    {'done':True, 'done_reason':'length', 'message':{'content':'partial'}}])
def test_reject_incomplete_response(reply):
    with patch('deeptraffic_llm_reward.llm_client.urlopen',
               return_value=io.BytesIO(json.dumps(reply).encode())):
        with pytest.raises(LLMCommunicationError):
            OllamaClient().chat([{'role':'user', 'content':'state'}])


def test_failure_has_no_implicit_retry():
    with patch('deeptraffic_llm_reward.llm_client.urlopen', side_effect=URLError('offline')) as transport:
        with pytest.raises(LLMCommunicationError):
            OllamaClient().chat([{'role':'user', 'content':'state'}])
    transport.assert_called_once()
