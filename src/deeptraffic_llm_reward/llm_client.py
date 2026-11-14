"""Ollama message transport only; no simulation or action interpretation."""
from dataclasses import dataclass
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LLMCommunicationError(RuntimeError):
    """Request failed or server did not return a completed textual reply."""


@dataclass(frozen=True)
class OllamaClient:
    base_url: str = 'http://localhost:11434'
    model: str = 'qwen3:8b'
    timeout_seconds: float = 180.0

    def __post_init__(self):
        url = urlparse(self.base_url)
        if url.scheme not in ('http', 'https') or not url.netloc or url.query or url.fragment:
            raise ValueError('Expected an HTTP(S) server base URL')
        if not self.model.strip() or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('Require model name and finite positive timeout')

    def chat(self, messages):
        """Send exactly one POST, returning the full response without action parsing.

        No retries, history, fallback actions or implicit requests. Non-thinking,
        greedy decoding and a fixed seed aid repeatability, not bitwise guarantees.
        """
        if not messages or any(not isinstance(m, dict) or
                m.get('role') not in ('system', 'user', 'assistant') or
                not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Require nonempty role/content messages')
        payload = {
            'model': self.model, 'messages': messages, 'stream': False,
            'think': False, 'format': 'json', 'keep_alive': 0,
            'options': {'temperature': 0, 'seed': 42, 'num_predict': 64, 'num_ctx': 8192},
        }
        request = Request(self.base_url.rstrip('/')+'/api/chat',
                          data=json.dumps(payload, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise LLMCommunicationError(f'Ollama HTTP error {exc.code}') from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise LLMCommunicationError(f'Ollama request failed: {type(exc).__name__}') from exc
        if not isinstance(result, dict) or result.get('done') is not True:
            raise LLMCommunicationError('Ollama returned an incomplete response')
        message = result.get('message')
        if not isinstance(message, dict) or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise LLMCommunicationError('Ollama returned no textual reply')
        if result.get('done_reason') == 'length':
            raise LLMCommunicationError('Ollama reply exceeded the output token budget')
        return {'request': payload, 'response': result}
