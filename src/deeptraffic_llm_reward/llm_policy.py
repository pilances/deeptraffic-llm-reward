"""Compose state preparation, an injected chat client and strict action parsing."""
from copy import deepcopy
from dataclasses import asdict
from time import perf_counter
from .llm_action import resolve_action
from .llm_state import prepare_llm_request


class LLMPolicyAdapter:
    """Callable observation -> integer action, without executing that action.

    The caller must explicitly supply a client with chat(messages), returning
    the request/response envelope used by OllamaClient. Construction does no I/O.
    last_record holds only the latest attempt; callers own persistent logging.
    Instances are intended for sequential calls, not concurrent use.
    """
    def __init__(self, config, client, *, on_invalid='raise', prepare_request=prepare_llm_request):
        if on_invalid not in ('raise', 'maintain'):
            raise ValueError('on_invalid must be raise or maintain')
        if not callable(getattr(client, 'chat', None)):
            raise TypeError('Client must provide chat(messages)')
        self.config = config
        self.client = client
        self.on_invalid = on_invalid
        self.prepare_request = prepare_request
        self.last_record = None

    def __call__(self, observation) -> int:
        started = perf_counter()
        # Clear the previous decision before attempting new work: failures must
        # never expose an old successful action as the current result.
        record = {'config': asdict(self.config), 'on_invalid': self.on_invalid,
                  'status': 'pending', 'stage': 'prepare', 'resolution': None}
        self.last_record = record
        try:
            prepared = self.prepare_request(observation, self.config)
            record['prepared'] = deepcopy(prepared)
            record['stage'] = 'chat'
            exchange = self.client.chat(prepared['messages'])
            record['exchange'] = deepcopy(exchange)
            record['stage'] = 'parse'
            text = exchange['response']['message']['content']
            resolution = resolve_action(text, on_invalid=self.on_invalid)
            record['resolution'] = resolution.to_dict()
            record['status'] = 'fallback' if resolution.fallback_used else 'ok'
            return resolution.action
        except Exception as exc:
            # Record and re-raise; communication/programming errors are never
            # converted into a driving action by this adapter.
            record['status'] = 'error'
            record['error_type'] = type(exc).__name__
            record['error_reason'] = getattr(exc, 'reason', str(exc))
            raise
        finally:
            record['elapsed_seconds'] = perf_counter()-started
