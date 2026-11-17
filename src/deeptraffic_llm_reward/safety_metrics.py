"""Observe native action applications without changing the pinned engine."""
from collections import Counter
import math


def lane_change_status(engine, agent_id):
    """Read the native gate without calling the mutating lane_change method."""
    car = engine.z[agent_id]
    centered = abs(car.x - (20 * car.b + 4)) < 0.5
    def check(direction):
        reason = None
        if not centered:
            reason = 'not_centered'
        else:
            target = (car.x + 7.5) / 20 + direction
            if not 0 <= math.floor(target) < 7:
                reason = 'off_road'
            else:
                offset = -3 * car.a
                while offset < 4:
                    if not (100 <= engine.H.get(target, car.y / 10 + offset, 0)):
                        reason = 'target_region_blocked'
                        break
                    offset += 1
        return dict(legal=reason is None, reason=reason)
    return dict(agent_id=agent_id, frame=engine.G, current_lane_centered=centered,
                ego_speed_mph=max(0, car.c * car.a) * 20,
                lane_number=int(car.b) + 1, lane_count=7,
                left=check(-1), right=check(1), source='native_engine_gate')


def observe_actions(engine, events):
    car_type = type(engine.z[0])
    original = getattr(car_type, '_safety_original_act', car_type.act)
    car_type._safety_original_act = original
    identities = {id(car): i for i, car in enumerate(engine.z)}

    def act(car, action):
        if car._engine is not engine:
            return original(car, action)
        agent = identities[id(car)]
        phase = engine.G % 30
        owner = 0 if phase == 0 else phase // 3
        decision = phase % 3 == 0 and owner < 10 and agent == owner
        before_lane, before_gas, factor = car.b, car.a, car.c
        reason = None
        if action in (3, 4):
            reason = lane_change_status(engine, agent)['left' if action == 3 else 'right']['reason']
        original(car, action)
        if action not in (1, 3, 4):
            return
        event = dict(frame=engine.G, agent_id=agent, action=int(action),
                     decision_command=decision, safety_factor=factor)
        if action in (3, 4):
            accepted = car.b != before_lane
            event.update(accepted=accepted, rejection_reason=None if accepted else reason or 'unknown')
        else:
            event.update(throttle_increased=car.a > before_gas,
                         at_throttle_cap=before_gas >= 2,
                         safety_limited=factor < 2,
                         fully_stopped_by_safety=factor == 0)
        events.append(event)

    car_type.act = act


def summarize_safety(events):
    """Command counts use callback owners; propagated applications are separate."""
    def counts(items):
        lane = [e for e in items if e['action'] in (3, 4)]
        accel = [e for e in items if e['action'] == 1]
        rejected = sum(not e['accepted'] for e in lane)
        return dict(lane_change=dict(requested=len(lane), accepted=len(lane)-rejected,
            rejected=rejected, rejected_percent=100*rejected/len(lane) if lane else 0,
            rejection_reasons=dict(Counter(e['rejection_reason'] for e in lane if not e['accepted']))),
            accelerate=dict(requested=len(accel), rejected_by_safety=0,
                throttle_increased=sum(e['throttle_increased'] for e in accel),
                at_throttle_cap=sum(e['at_throttle_cap'] for e in accel),
                safety_limited_at_request=sum(e['safety_limited'] for e in accel),
                fully_stopped_at_request=sum(e['fully_stopped_by_safety'] for e in accel)))
    return dict(decision_commands=counts([e for e in events if e['decision_command']]),
                all_action_applications=counts(events))


def format_safety_table(reports):
    lines = ['| Policy | Lane requests | Accepted | Rejected | Rejected % | Accelerate | Safety limited at request | Throttle cap |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, report in reports.items():
        row = report['decision_commands']
        l, a = row['lane_change'], row['accelerate']
        lines.append(f'| {name} | {l["requested"]} | {l["accepted"]} | {l["rejected"]} | '
                     f'{l["rejected_percent"]:.1f} | {a["requested"]} | {a["safety_limited_at_request"]} | {a["at_throttle_cap"]} |')
    return '\n'.join(lines) + '\n'
