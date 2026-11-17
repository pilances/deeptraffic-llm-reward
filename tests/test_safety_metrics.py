import json
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.backend import engine_class
from deeptraffic_llm_reward.safety_metrics import observe_actions, summarize_safety
from deeptraffic_llm_reward.evaluation import run_seeded_episode


def test_actual_lane_acceptance_and_rejection_reasons():
    eng = engine_class()()
    eng.reset()
    events = []
    observe_actions(eng, events)
    car = eng.z[0]
    eng.H.reset()
    car.b, car.x, car.y = 3, 64, 300
    car.act(3)
    assert car.b == 2 and events[-1]['accepted']
    car.act(3)
    assert not events[-1]['accepted'] and events[-1]['rejection_reason'] == 'not_centered'
    car.b, car.x = 0, 4
    car.act(3)
    assert events[-1]['rejection_reason'] == 'off_road'
    car.b, car.x = 3, 64
    eng.H.set(2, 30, 1)
    car.act(3)
    assert events[-1]['rejection_reason'] == 'target_region_blocked'


def test_accelerate_is_not_rejected_by_safety_or_confused_with_cap():
    eng = engine_class()()
    eng.reset()
    events = []
    observe_actions(eng, events)
    car = eng.z[0]
    car.a, car.c = 1, 0
    car.act(1)
    assert car.a == 1.02
    car.a, car.c = 2, 2
    car.act(1)
    report = summarize_safety(events)['decision_commands']['accelerate']
    assert report == dict(requested=2, rejected_by_safety=0, throttle_increased=1,
        at_throttle_cap=1, safety_limited_at_request=1, fully_stopped_at_request=1)


def test_instrumentation_preserves_physics_and_reset_does_not_stack(monkeypatch):
    cfg = DeepTrafficConfig(episode_frames=200)
    env = DeepTrafficEnv(cfg)
    try:
        def episode():
            obs, info = env.reset(seed=2000100)
            observations = [obs.copy()]
            while True:
                obs, _, _, done, info = env.step(len(observations) % 5)
                observations.append(obs.copy())
                if done:
                    return observations, info
        a, ia = episode()
        b, ib = episode()
        assert ia['safety_summary'] == ib['safety_summary']
        monkeypatch.setattr('deeptraffic_llm_reward.env.observe_actions', lambda *args: None)
        # Restore the class to the original engine method for an unobserved run.
        cls = type(env._eng.z[0])
        cls.act = cls._safety_original_act
        c, ic = episode()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
        assert ia['agent_mean_speed_mph'] == ic['agent_mean_speed_mph']
    finally:
        env.close()


def test_commands_match_callback_counts_and_resume(tmp_path):
    cfg = DeepTrafficConfig(episode_frames=200)
    path = tmp_path/'trace.jsonl'
    first = run_seeded_episode(cfg, 2000100, lambda obs: 3, path)
    replay = run_seeded_episode(cfg, 2000100, lambda obs: 1, path)
    report = first['safety_summary']
    assert report['decision_commands']['lane_change']['requested'] == first['decisions']
    assert report['all_action_applications']['lane_change']['requested'] > first['decisions']
    assert first['safety_summary'] == replay['safety_summary']
    assert first['score_mph'] == replay['score_mph']
    assert 'safety_events' in json.loads(path.read_text().splitlines()[0])
