"""Pure observation-to-message preparation; no network calls or action execution."""
import hashlib
import json
import numpy as np
from .env import Action

VERSION = 'deeptraffic-state-v1'


def prepare_llm_request(observation, config):
    """Return JSON-serializable messages and provenance using only observations.

    Grid rows follow engine h=-patchesAhead .. patchesBehind-1; columns
    follow relative lane offsets -lanesSide .. +lanesSide. Values are the
    original normalized float32 values, without lossy text rounding.
    """
    rows = config.patches_ahead + config.patches_behind
    columns = 2 * config.lanes_side + 1
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (rows * columns,) or not np.isfinite(obs).all():
        raise ValueError('Expected a finite flat observation matching the configuration')
    if np.any(obs < 0) or np.any(obs > 1):
        raise ValueError('Observation values must be in 0..1')
    state = {
        'schema_version': VERSION,
        'lane_offsets_left_to_right': list(range(-config.lanes_side, config.lanes_side+1)),
        'longitudinal_offsets': list(range(-config.patches_ahead, config.patches_behind)),
        'grid': obs.reshape(rows, columns).tolist(),
    }
    content = json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False)
    instructions = (
        'You control one vehicle in DeepTraffic. Choose an action to improve average '
        'speed of the controlled fleet using only the supplied observation. '
        'Grid columns run left to right; lane offset 0 is your lane. Rows run from '
        'farthest ahead (negative longitudinal offset) toward the vehicle and behind '
        '(positive offsets). Offset 0 is at the vehicle. Values encode speed in mph '
        'using the native engine encoding: 1.0 means free space; occupied cells '
        'contain vehicle speed in mph divided by 2000. Zero may mean off-road '
        'or stopped traffic. The observation does not identify vehicles '
        'or provide a separate ego speed. Actions: ' +
        json.dumps({int(a): a.name for a in Action}, sort_keys=True) +
        '. Return only JSON with one integer action, e.g. {"action":0}.'
    )
    return {
        'schema_version': VERSION,
        'state_sha256': hashlib.sha256(content.encode()).hexdigest(),
        'prompt_sha256': hashlib.sha256(instructions.encode()).hexdigest(),
        'messages': [{'role': 'system', 'content': instructions},
                     {'role': 'user', 'content': content}],
    }
