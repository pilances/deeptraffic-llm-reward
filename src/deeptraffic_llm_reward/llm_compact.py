"""Compact observation summary with optional native lane-change telemetry."""
import hashlib
import json
import numpy as np

VERSION = 'deeptraffic-compact-state-v6'
PROMPT_VERSION = 'deeptraffic-compact-prompt-v16'
THRESHOLDS = dict(fast_mph=70, near_patches=14, rear_fast_mph=70, rear_near_patches=5,
                  overtake_speed_delta_mph=5, optional_gap_gain_patches=14,
                  minimum_lane_change_gap_gain_patches=4)


def prepare_compact_request(observation, config, *, lane_status=None):
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

    ego_speed = lane_status.get('ego_speed_mph') if lane_status else None
    lane_number = lane_status.get('lane_number') if lane_status else None
    lines = [f'Visible region: {config.patches_ahead} patches ahead, {max(0, config.patches_behind - 1)} patches behind.',
             'Distances are to occupied cells, not vehicle centers.',
             f'EGO SPEED: {ego_speed:.1f} mph.' if ego_speed is not None else 'EGO SPEED: UNKNOWN.',
             f'CURRENT LANE: {lane_number} of {lane_status["lane_count"]} (1 = leftmost; lane number is the target lane during a change).'
             if lane_number is not None else 'CURRENT LANE: UNKNOWN.']
    if lane_status is None:
        lines.append('CURRENT: centering UNKNOWN. LEFT: CHANGE STATUS UNKNOWN. RIGHT: CHANGE STATUS UNKNOWN.')
    else:
        lines.append('CURRENT: ' + ('CENTERED.' if lane_status['current_lane_centered'] else 'NOT CENTERED; lane change in progress or vehicle off center.'))
        for name, key in [('LEFT', 'left'), ('RIGHT', 'right')]:
            gate = lane_status[key]
            lines.append(f'{name}: CHANGE LEGAL.' if gate['legal'] else
                         f'{name}: CHANGE BLOCKED ({gate["reason"]}).')
    own = lanes['CURRENT']['front']
    speed_delta = (ego_speed - own['speed_mph'] if ego_speed is not None and
                   own is not None and not own['ambiguous_zero'] else None)
    candidates = []
    overtake = 'UNKNOWN' if ego_speed is None or lane_status is None else 'NOT USEFUL'
    if speed_delta is not None:
        lines.append(f'FRONT SPEED DIFFERENCE: {speed_delta:+.1f} mph (positive = ego faster).')
        # A car already forced down to the blocker's speed still needs to overtake.
        impeded = speed_delta >= THRESHOLDS['overtake_speed_delta_mph'] or own['speed_mph'] < THRESHOLDS['fast_mph']
        if lane_status['current_lane_centered'] and impeded:
            for name, key in [('LEFT', 'left'), ('RIGHT', 'right')]:
                target = lanes[name]
                if not target['observed'] or not lane_status[key]['legal']:
                    continue
                front = target['front']
                # Missing traffic means a lower bound on clearance, not an empty lane.
                gap = config.patches_ahead if front is None else front['distance_patches']
                # A legal lane is useful unless it has a slow vehicle close ahead.
                # Target speed ranks candidates later; it must not suppress a
                # legal lane merely because its distant traffic is slower.
                speed_benefit = front is None or (not front['ambiguous_zero'] and not (
                    front['distance_patches'] <= THRESHOLDS['near_patches'] and
                    front['speed_mph'] < THRESHOLDS['fast_mph']))
                near = own['distance_patches'] <= THRESHOLDS['near_patches']
                gap_gain = 1 if near else THRESHOLDS['optional_gap_gain_patches']
                if speed_benefit and gap >= own['distance_patches'] + gap_gain:
                    candidates.append(name)
            if candidates:
                overtake = 'REQUIRED' if own['distance_patches'] <= THRESHOLDS['near_patches'] else 'OPTIONAL'
    blocked = own is not None and (own['ambiguous_zero'] or
        (own['distance_patches'] <= THRESHOLDS['near_patches'] and own['speed_mph'] < THRESHOLDS['fast_mph']))
    lines.append('STATUS: ' + ('BLOCKED by nearby slow traffic or an ambiguous zero region.' if blocked else 'No nearby slow obstruction detected.'))
    # Compare observed clearance within the same horizon. A clear window is
    # capped at the horizon; it never establishes an unbounded empty lane.
    current_gap = config.patches_ahead if own is None else own['distance_patches']
    gap_gains = {}
    lines.append(f'CURRENT FRONT GAP: {current_gap} patches (capped at visible ahead distance).')
    for name in ('LEFT', 'RIGHT'):
        lane = lanes[name]
        if not lane['observed']:
            gap_gains[name] = None
            lines.append(f'{name} GAP GAIN: UNKNOWN.')
            continue
        front = lane['front']
        gap = config.patches_ahead if front is None else front['distance_patches']
        gap_gains[name] = gap - current_gap
        lines.append(f'{name} FRONT GAP: {gap} patches; {name} GAP GAIN: {gap_gains[name]:+d} patches.')
    for name, lane in lanes.items():
        if not lane['observed']:
            lines.append(f'{name}: not observed.')
            continue
        lines.append(f'{name}: ahead: {describe(lane["front"])}.')
        if name != 'CURRENT':
            value = lane['at_vehicle']
            lines.append(f'{name} at vehicle position: ' + ('unknown.' if value is None else
                'free.' if value == 1 else 'zero-valued (off-road or stopped traffic; ambiguous).' if value == 0 else 'occupied.'))
            if config.patches_behind <= 1:
                rear_description = 'not observed; rely on native CHANGE LEGAL/BLOCKED status.'
            elif lane['rear'] is None:
                rear_description = ('no rear vehicle observed in the visible rear region; '
                                    'traffic farther behind is unknown.')
            else:
                rear_description = describe(lane['rear'], rear=True) + '.'
            lines.append(f'{name}: behind: {rear_description}')
    content = '\n'.join(lines)
    instructions = (
        f'{PROMPT_VERSION}. Maximize average fleet speed.\n'
        'Use the CURRENT centering status and the native LEFT/RIGHT CHANGE LEGAL/BLOCKED fields.\n'
        'Missing rear observations or traffic beyond the visible rear region do not by themselves '
        'disqualify a direction marked CHANGE LEGAL. This status describes current engine eligibility, '
        'not a guarantee of future safety or speed advantage. If change status is UNKNOWN, do not assume it is legal.\n'
        'Apply the FIRST matching rule, then stop:\n'
        '1. CURRENT is not CENTERED: choose ACCELERATE (1).\n'
        '2. When STATUS is BLOCKED, a direction qualifies when CURRENT is CENTERED, the direction is '
        'CHANGE LEGAL, and its GAP GAIN is positive (+1 or more). Change immediately when a qualifying '
        'direction exists; do not wait for a larger gap. When STATUS is not BLOCKED, a direction qualifies '
        'only when CURRENT is CENTERED, the direction is CHANGE LEGAL, and its GAP GAIN is at least '
        f'+{THRESHOLDS["minimum_lane_change_gap_gain_patches"]} patches. '
        'If only LEFT qualifies, choose CHANGE_LEFT (3). If only RIGHT qualifies, choose CHANGE_RIGHT (4). '
        'If both qualify, choose the larger GAP GAIN. Only when both qualifying gains are equal, choose CHANGE_RIGHT (4). '
        'If neither qualifies, choose ACCELERATE (1). '
        'Use the supplied GAP GAIN values (target observed gap minus CURRENT observed gap). '
        'A legal direction below +4 or with UNKNOWN GAP GAIN does not qualify.\n'
        '3. Otherwise: choose ACCELERATE (1).\n'
        'Never choose a lane marked CHANGE BLOCKED. Never choose DECELERATE. '
        'If you are blocked and neither side qualifies, ease off briefly and wait for a gap. '
        'Return only JSON: {"action": <integer>}.\n'
        'Example: Output: {"action":1}'
    )
    return dict(schema_version=VERSION, prompt_version=PROMPT_VERSION,
        lane_change_status=lane_status,
        thresholds=THRESHOLDS.copy(), decoded_lanes=lanes,
        observed_gap_gains=gap_gains,
        overtake=dict(status=overtake, candidate_lanes=candidates, front_speed_difference_mph=speed_delta),
        state_sha256=hashlib.sha256(obs.astype('<f4').tobytes() + json.dumps(lane_status, sort_keys=True).encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
        messages=[dict(role='system', content=instructions), dict(role='user', content=content)])
