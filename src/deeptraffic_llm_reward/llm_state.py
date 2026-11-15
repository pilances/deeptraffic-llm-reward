"""Pure observation-to-message preparation; no network calls or action execution."""
import hashlib
import json
import numpy as np
from .env import Action

VERSION = 'deeptraffic-state-v1'
PROMPT_VERSION = 'deeptraffic-prompt-v2'


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
        f'{PROMPT_VERSION}. You control one vehicle in DeepTraffic. Choose one '
        'action using only the supplied 676-value observation. The objective is '
        'high average speed of the controlled fleet while avoiding unsafe lane changes. '
        'The observation is unchanged and contains no extra derived features.\n\n'
        'READING THE GRID:\n'
        '- The JSON grid is a matrix, flattened row by row in the original observation.\n'
        '- Rows run from farthest ahead (negative longitudinal offset) toward the '
        'vehicle and then behind (positive offset). With patches_behind=0, row 0 '
        'is farthest ahead and the last row is nearest the vehicle.\n'
        '- Columns run left to right by relative lane offset. Offset 0 is always '
        'the controlled vehicle\'s current lane; absolute lane number is unavailable.\n'
        '- 1.0 means free space. A value below 1.0 marks an occupied cell and '
        'encodes that vehicle\'s speed as mph divided by 2000. Zero can mean a '
        'stopped vehicle or off-road space.\n'
        '- Vehicle identities, a separate ego speed and absolute lane number are '
        'unavailable. Infer a nearby vehicle from its contiguous occupied cells.\n\n'
        'DECISION ORDER:\n'
        '1. Inspect the current lane ahead. If it is clear, choose ACCELERATE.\n'
        '2. If a slower vehicle blocks the current lane, inspect both adjacent lanes.\n'
        '3. Choose CHANGE_LEFT or CHANGE_RIGHT only when the target lane has a '
        'sufficiently clear region ahead and behind in the grid. Prefer the safer '
        'target lane; if both are comparable, prefer CHANGE_LEFT.\n'
        '4. If both adjacent lanes are blocked or cannot be judged safe, choose MAINTAIN.\n'
        '5. Choose DECELERATE only when the current lane is blocked and no safe lane '
        'change is visible. Never change lanes merely because a lane is adjacent.\n\n'
        'RAW-STATE EXAMPLES:\n'
        '- Slow vehicle ahead, adjacent left region clear, adjacent right region blocked: CHANGE_LEFT.\n'
        '- Slow vehicle ahead, adjacent right region clear, adjacent left region blocked: CHANGE_RIGHT.\n'
        '- Slow vehicle ahead, both adjacent regions blocked or uncertain: MAINTAIN.\n'
        '- Current lane clear ahead: ACCELERATE.\n\n'
        'Actions: ' + json.dumps({int(a): a.name for a in Action}, sort_keys=True) +
        '. Analyze internally and return only one JSON object with an integer action, '
        'for example {"action":0}. Do not return explanations, markdown or extra keys.'
    )
    return {
        'schema_version': VERSION,
        'prompt_version': PROMPT_VERSION,
        'state_sha256': hashlib.sha256(content.encode()).hexdigest(),
        'prompt_sha256': hashlib.sha256(instructions.encode()).hexdigest(),
        'messages': [{'role': 'system', 'content': instructions},
                     {'role': 'user', 'content': content}],
    }
