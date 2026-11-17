"""P4 instruction transcribed from the supplied photograph; reconstructed renderer."""
import hashlib
import json
import numpy as np

ACTION_TO_ID = dict(maintain=0, accelerate=1, decelerate=2, left=3, right=4)
PROMPT_VERSION = 'professor-p4-record-frozen'
THRESHOLDS = dict(FAST=70, NEAR=14, REAR_NEAR=5, REAR_FAST=62)
SYS_P4 = """You control one car on a seven-lane highway. Maximize the controlled fleet's average speed.

Use only these actions:
- ACCELERATE
- CHANGE_LEFT
- CHANGE_RIGHT

Never choose DECELERATE or MAINTAIN.

ACCELERATE is the default action. If the current lane is not BLOCKED by a SLOW car CLOSE ahead, always choose ACCELERATE. A legal lane by itself is never a reason to change lanes.

You may change lanes only when all of these conditions hold:
1. STATUS is BLOCKED.
2. CURRENT is CENTERED.
3. The chosen direction says CHANGE LEGAL.
4. The target lane does not contain a SLOW car CLOSE ahead.
5. The target lane does not contain a FAST car CLOSE behind.
6. The target lane has at least 4 patches more forward distance than CURRENT.
   An OPEN target lane qualifies only when CURRENT is not OPEN.

If any one of these conditions fails, that lane is forbidden.

If both adjacent lanes qualify, choose the lane with the larger forward distance. If their forward distances differ by fewer than 4 patches, prefer the lane closer to the center lane 4.

If STATUS is BLOCKED but no adjacent lane satisfies every condition, choose ACCELERATE and wait for a better gap.

Never change lanes merely because:
- a lane is legal,
- a lane is closer to the center,
- a vehicle is far behind,
- the current lane contains a vehicle that is not SLOW and CLOSE.

Evaluate LEFT and RIGHT independently. Never infer that one direction is preferable without comparing both.

Return only JSON with exactly these two keys:
{"reason":"one short sentence","action":"accelerate"}

The action value must be one of the currently available actions. Do not output explanations or additional keys."""


def legal_actions(lane_status):
    """P4 experiment mask: no decelerate; native lane-change gate at callback."""
    if lane_status is None:
        raise ValueError('P4 requires current native lane-change status')
    actions = [0, 1]
    if lane_status['current_lane_centered']:
        for key, action in [('left', 3), ('right', 4)]:
            if lane_status[key]['legal']:
                actions.append(action)
    return actions


def prepare_professor_request(observation, config, *, lane_status=None):
    legal = legal_actions(lane_status)
    if (config.lanes_side, config.patches_ahead, config.patches_behind) != (3, 40, 5):
        raise ValueError('P4 requires lanes_side=3, patches_ahead=40, patches_behind=5')
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (315,) or not np.isfinite(obs).all() or np.any((obs < 0) | (obs > 1)):
        raise ValueError('Expected 315 finite observation values in [0,1]')
    grid = obs.reshape(45, 7)
    lane = lane_status['lane_number']

    def nearest(col, offsets):
        for d in offsets:
            value = float(grid[40 + d, col])
            if value < 1:
                return dict(distance=abs(d), speed=round(value * 2000, 1), ambiguous=value == 0)
        return None

    lanes = {}
    for name, col in [('LEFT', 2), ('CURRENT', 3), ('RIGHT', 4)]:
        lanes[name] = dict(on_road=1 <= lane + col - 3 <= 7,
                           front=nearest(col, range(-1, -41, -1)),
                           rear=nearest(col, range(1, 5)))
    def describe(hit):
        if hit is None:
            return 'OPEN (clear far ahead).'
        if hit['ambiguous']:
            return f'zero-valued region {hit["distance"]}p ahead (stopped traffic or off-road; ambiguous).'
        return (f'{"FAST" if hit["speed"] >= 70 else "SLOW"} car '
                f'{"CLOSE" if hit["distance"] <= 14 else "far"} '
                f'{hit["distance"]}p ahead ({hit["speed"]:g} mph).')
    own = lanes['CURRENT']['front']
    blocked = own is not None and not own['ambiguous'] and own['speed'] < 70 and own['distance'] <= 14
    lines = [f'You are in lane {lane if lane is not None else "UNKNOWN"} of 7. STATUS: '
             + ('BLOCKED by a slow car ahead.' if blocked else 'Not blocked by a slow car close ahead.'),
             'Your lane: ' + describe(own)]
    lines.append('CURRENT: ' + ('CENTERED.' if lane_status['current_lane_centered'] else 'NOT CENTERED; finish the current lane change.'))
    for name, action in [('LEFT', 3), ('RIGHT', 4)]:
        entry = lanes[name]
        if not entry['on_road']:
            lines.append(f'{name} lane: OFF ROAD (unavailable).')
            continue
        lines.append(f'{name} lane: ' + describe(entry['front']))
        gate = lane_status[name.lower()]
        lines.append(f'{name}: CHANGE LEGAL.' if action in legal else
                     f'{name}: CHANGE BLOCKED ({gate["reason"] or "not_centered"}).')
        rear = entry['rear']
        if rear is not None and not rear['ambiguous'] and rear['speed'] >= 62 and rear['distance'] <= 5:
            lines.append(f'{name} lane: FAST car CLOSE {rear["distance"]}p behind ({rear["speed"]:g} mph).')
        elif rear is not None and rear['ambiguous']:
            lines.append(f'{name} lane: zero-valued rear region (ambiguous).')
    lines.append('Available actions: ' + ', '.join(n for n in
                 ('accelerate', 'decelerate', 'maintain', 'left', 'right') if ACTION_TO_ID[n] in legal))
    instructions = "You drive ONE car on a 7-lane highway; your shared policy maximizes the controlled fleet AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road ahead allows.\nYour default action is ACCELERATE. Whenever your lane is OPEN or the car ahead is fast, accelerate — never 'maintain' (it wastes speed recovery) and never change lanes (a lane change costs speed unless it actually gets you past a slow car).\nIgnore cars behind you; never move aside for them.\nChange lanes ONLY to overtake when you are BLOCKED (a SLOW car CLOSE ahead in your lane). If STATUS is BLOCKED and at least one adjacent lane is CHANGE LEGAL, you MUST choose a CHANGE LEGAL action immediately. Do not choose ACCELERATE or MAINTAIN in this situation. Pick an adjacent lane that is OPEN or whose nearest car is fast or far; NEVER move into a lane that also has a slow car close ahead, and never cut into a lane where a fast car is close behind. If both sides qualify, prefer the lane closer to the centre (lane 4).\nIf you are blocked and neither side qualifies, ease off briefly and wait for a gap.\nThink in one short sentence, then choose one action.\nUse only Available actions. Native CHANGE BLOCKED excludes that direction, even if its lane looks OPEN. CHANGE LEGAL means the native gate permits the request now, not that it improves speed. Return only JSON with exactly reason and action: {\"reason\":\"one short sentence\",\"action\":\"accelerate\"}. Use an available action name.\nUse only Available actions. Native CHANGE BLOCKED excludes that direction, even if its lane looks OPEN. CHANGE LEGAL means the native gate permits the request now, not that it improves speed. When blocked, change only if the target has at least 4 patches more forward distance than CURRENT, or CURRENT is occupied and the target is OPEN. Otherwise choose ACCELERATE. Return only JSON with exactly reason and action: {\"reason\":\"one short sentence\",\"action\":\"accelerate\"}. Use an available action name."
    schema = dict(type='object', properties=dict(reason=dict(type='string', minLength=1),
                  action=dict(type='string', enum=[n for n, a in ACTION_TO_ID.items() if a in legal])),
                  required=['reason', 'action'], additionalProperties=False)
    return dict(schema_version='professor-p4-state-v2', prompt_version=PROMPT_VERSION,
                thresholds=THRESHOLDS.copy(), legal_actions=legal, decoded_lanes=lanes,
                lane_change_status=lane_status, response_schema=schema,
                lane_number=lane, response_format='professor_p4',
                state_sha256=hashlib.sha256(obs.astype('<f4').tobytes() + json.dumps(lane_status, sort_keys=True).encode()).hexdigest(),
                prompt_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
                messages=[dict(role='system', content=instructions), dict(role='user', content='\n'.join(lines))])


def parse_professor_action(text, legal_actions):
    from .llm_action import ActionParseError, _unique_object
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise ActionParseError('invalid_json') from exc
    if (not isinstance(obj, dict) or set(obj) != {'reason', 'action'}
            or not isinstance(obj['reason'], str) or not obj['reason'].strip()
            or not isinstance(obj['action'], str) or obj['action'] not in ACTION_TO_ID):
        raise ActionParseError('invalid_professor_schema')
    action = ACTION_TO_ID[obj['action']]
    if action not in legal_actions:
        raise ActionParseError('unavailable_action')
    return action


def masked_argmax(q_values, legal_actions):
    """For a student network: mask off-road outputs before greedy selection."""
    scores = np.asarray(q_values, dtype=np.float64).copy()
    if scores.shape != (5,) or not np.isfinite(scores).all() or not legal_actions:
        raise ValueError('Require five finite Q-values and legal actions')
    scores[[a for a in range(5) if a not in legal_actions]] = -np.inf
    return int(scores.argmax())

