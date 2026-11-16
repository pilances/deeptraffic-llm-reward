"""Observation-only semantic abstraction for the LLM, without action selection."""
import hashlib
import json
import numpy as np
from .env import Action

VERSION = 'deeptraffic-compact-state-v1'
PROMPT_VERSION = 'deeptraffic-compact-prompt-v1'
THRESHOLDS = dict(fast_mph=70, near_patches=14, rear_fast_mph=62, rear_near_patches=5)


def prepare_compact_request(observation, config):
    rows, cols = config.patches_ahead + config.patches_behind, 2 * config.lanes_side + 1
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (rows * cols,) or not np.isfinite(obs).all() or np.any((obs < 0) | (obs > 1)):
        raise ValueError('Expected a finite flat observation in 0..1 matching configuration')
    grid = obs.reshape(rows, cols)

    def nearest(col, offsets):
        for offset in offsets:
            value = float(grid[offset + config.patches_ahead, col])
            if value < 1:
                # Zero is ambiguous: never invent speed or road geometry for it.
                return dict(distance_patches=abs(offset), speed_mph=None if value == 0 else round(value * 2000, 1),
                            ambiguous_zero=value == 0)
        return None

    lanes = {}
    for name, delta in [('LEFT', -1), ('CURRENT', 0), ('RIGHT', 1)]:
        col = config.lanes_side + delta
        if not 0 <= col < cols:
            lanes[name] = dict(observed=False)
            continue
        lanes[name] = dict(observed=True,
            front=nearest(col, range(-1, -config.patches_ahead - 1, -1)),
            at_vehicle=float(grid[config.patches_ahead, col]) if config.patches_behind else None,
            rear=nearest(col, range(1, config.patches_behind)))

    def describe(hit, rear=False):
        if hit is None:
            return 'no obstruction in the observed region'
        distance = hit['distance_patches']
        if hit['ambiguous_zero']:
            return f'zero-valued region {distance} patches away (off-road or stopped traffic; ambiguous)'
        speed = hit['speed_mph']
        fast = THRESHOLDS['rear_fast_mph' if rear else 'fast_mph']
        near = THRESHOLDS['rear_near_patches' if rear else 'near_patches']
        return f'{"FAST" if speed >= fast else "SLOW"} traffic {"CLOSE" if distance <= near else "FAR"}, {distance} patches away, {speed:g} mph'

    lines = [f'Visible region: {config.patches_ahead} patches ahead, {max(0, config.patches_behind - 1)} patches behind.',
             'Own speed and absolute lane number: unknown. Distances are to occupied cells, not vehicle centers.']
    own = lanes['CURRENT']['front']
    blocked = own is not None and (own['ambiguous_zero'] or
        (own['distance_patches'] <= THRESHOLDS['near_patches'] and own['speed_mph'] < THRESHOLDS['fast_mph']))
    lines.append('STATUS: ' + ('BLOCKED by nearby slow traffic or an ambiguous zero region.' if blocked else 'No nearby slow obstruction detected.'))
    for name, lane in lanes.items():
        if not lane['observed']:
            lines.append(f'{name}: not observed.')
            continue
        lines.append(f'{name}: ahead: {describe(lane["front"])}.')
        if name != 'CURRENT':
            value = lane['at_vehicle']
            lines.append(f'{name} at vehicle position: ' + ('unknown.' if value is None else
                'free.' if value == 1 else 'zero-valued (off-road or stopped traffic; ambiguous).' if value == 0 else 'occupied.'))
            lines.append(f'{name}: behind: ' + (describe(lane['rear'], rear=True) + '.' if config.patches_behind > 1 else 'unknown.'))
    content = '\n'.join(lines)
    instructions = (
        f'{PROMPT_VERSION}. Control one car on a seven-lane highway. Maximize average speed of the controlled fleet; top speed is 80 mph. '
        'The user supplies a summary derived only from the observation. FAST means at least 70 mph ahead, '
        'CLOSE means at most 14 patches ahead; rear FAST means at least 62 mph and rear CLOSE at most 5 patches. '
        'SLOW is an absolute speed category, not a comparison with your unknown speed. '
        'Default to ACCELERATE when there is no nearby slow obstruction. '
        'When BLOCKED, consider overtaking into an adjacent lane with more forward clearance or faster traffic. '
        'Avoid a target with nearby slow traffic, occupied or ambiguous space at your position, or FAST CLOSE traffic behind. '
        'Unknown space is not proven clear or blocked. A clear observed region does not establish that the entire lane is empty. '
        'If no adjacent lane offers an improvement, MAINTAIN or DECELERATE as appropriate. '
        'Choose the action yourself using these observations. Actions: ' + json.dumps({int(a): a.name for a in Action}) +
        '. Return only JSON with one integer action, e.g. {"action":1}; no explanation or extra keys.'
    )
    return dict(schema_version=VERSION, prompt_version=PROMPT_VERSION,
        thresholds=THRESHOLDS.copy(), decoded_lanes=lanes,
        state_sha256=hashlib.sha256(obs.astype('<f4').tobytes()).hexdigest(),
        prompt_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
        messages=[dict(role='system', content=instructions), dict(role='user', content=content)])
