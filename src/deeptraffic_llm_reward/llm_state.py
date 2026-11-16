"""Pure observation-to-message preparation; no network calls or action execution."""
import hashlib
import json
import numpy as np
from .env import Action

VERSION = 'deeptraffic-state-v1'
PROMPT_VERSION = 'deeptraffic-prompt-v3'


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
    rear_rows = max(0, config.patches_behind - 1)
    rear_visibility = (
        'No cells behind the controlled vehicle are supplied. Rear traffic is unknown; '
        'do not treat it as either clear or blocked. Judge lane changes from the '
        'visible space ahead, without claiming that the unseen rear is safe.\n\n'
        if rear_rows == 0 else
        f'{rear_rows} rows behind the vehicle are supplied. Use them '
        'when judging a lane change; traffic beyond the observed rows is unknown.\n\n'
    )
    instructions = (
        f'{PROMPT_VERSION}. You control one vehicle in DeepTraffic. Choose one '
        f'action using only the supplied {rows * columns}-value observation. The objective is '
        'high average speed of the controlled fleet. '
        'The observation is unchanged and contains no extra derived features.\n\n'
        'READING THE GRID:\n'
        '- The JSON grid is a matrix, flattened row by row in the original observation.\n'
        '- Negative longitudinal offsets run from farthest ahead toward the vehicle. '
        'Offset 0 is at the vehicle; positive offsets, if present, are behind it.\n'
        '- Columns run left to right by relative lane offset. Offset 0 is always '
        'the controlled vehicle\'s current lane; absolute lane number is unavailable.\n'
        '- 1.0 means free space. Values strictly between 0 and 1 mark occupied '
        'cells and encode vehicle speed as mph divided by 2000. Zero can mean a '
        'stopped vehicle or off-road space.\n'
        '- Vehicle identities, a separate ego speed and absolute lane number are '
        'unavailable. Do not claim to know whether another vehicle is slower than '
        'you. Infer a nearby vehicle from its contiguous occupied cells.\n'
        'There are safety measures in place to prevent collisions. Therefore, an action which would result '
        'in a direct collision with another vehicle or off-road space will be ignored by the simulator.\n'
        'However, you should still avoid choosing such actions, as they will result in a lower average speed.\n\n'
        + rear_visibility +
        'DECISION ORDER:\n'
        '1. If the visible current lane has no nearby obstruction ahead, choose ACCELERATE.\n'
        '2. If an occupied region is nearby ahead in the current lane, compare the '
        'visible forward clearance of the adjacent lanes.\n'
        '3. Choose CHANGE_LEFT or CHANGE_RIGHT when that adjacent lane is visibly '
        'more open ahead and has no visible obstruction at the lane-change area. '
        'Avoid a lane that appears off-road or occupied. Do not require knowledge '
        'of traffic outside the observation.\n'
        '4. If neither adjacent lane is visibly better, choose MAINTAIN; choose '
        'DECELERATE if an obstruction is immediately ahead.\n\n'
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
