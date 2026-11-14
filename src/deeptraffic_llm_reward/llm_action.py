"""Strict text-to-action parsing and explicit fallback, without external I/O."""
from dataclasses import asdict, dataclass
import json
import logging
from typing import Optional
from .env import Action

logger = logging.getLogger(__name__)


class ActionParseError(ValueError):
    """Invalid model output, with a stable reason for later benchmark logs."""
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ActionParseError('duplicate_key')
        result[key] = value
    return result


def parse_action(text: str) -> int:
    """Accept exactly one JSON object containing only an integer action 0..4.

    Surrounding whitespace is allowed. No extraction, coercion or repair is
    performed. Non-string input is a caller error, not a model-output fallback.
    """
    if not isinstance(text, str):
        raise TypeError('Expected model response text')
    try:
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ActionParseError('invalid_json') from exc
    if not isinstance(payload, dict) or set(payload) != {'action'}:
        raise ActionParseError('invalid_schema')
    action = payload['action']
    # bool is an int subclass in Python; JSON true/false must not become actions.
    if type(action) is not int:
        raise ActionParseError('invalid_action_type')
    if action not in {int(a) for a in Action}:
        raise ActionParseError('invalid_action_value')
    return action


@dataclass(frozen=True)
class ActionResolution:
    raw_response: str
    action: int
    parsed_action: Optional[int]
    fallback_used: bool
    error_reason: Optional[str]
    on_invalid: str

    def to_dict(self):
        """JSON-serializable record; persistence belongs to the later adapter."""
        return asdict(self)


def resolve_action(text: str, *, on_invalid: str = 'raise') -> ActionResolution:
    """Resolve text or explicitly select MAINTAIN; never execute the action.

    Default: propagate ActionParseError. Opt-in 'maintain': return action 0
    with the original response and error reason so fallback cannot be mistaken
    for a valid model prediction. Communication failures are handled elsewhere.
    """
    if on_invalid not in ('raise', 'maintain'):
        raise ValueError('on_invalid must be raise or maintain')
    try:
        action = parse_action(text)
    except ActionParseError as exc:
        if on_invalid == 'raise':
            raise
        result = ActionResolution(text, int(Action.MAINTAIN), None, True, exc.reason, on_invalid)
        logger.warning('Invalid LLM response; using MAINTAIN (%s)', exc.reason,
                       extra={'action_resolution': result.to_dict()})
        return result
    return ActionResolution(text, action, action, False, None, on_invalid)
