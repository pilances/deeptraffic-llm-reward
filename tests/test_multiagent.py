import json
import shutil
import subprocess
import numpy as np
import pytest
from deeptraffic_llm_reward import DeepTrafficEnv, DeepTrafficConfig, MultiAgentCollector, ReplayBuffer
from deeptraffic_llm_reward.backend import backend_root


@pytest.mark.parametrize('others', [0, 1, 4, 9, 10])
def test_per_car_transitions_and_exact_reward_intervals(others):
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=others, episode_frames=65))
    # First record the independently stepped frame stream and observations.
    observations, frame_rewards = {}, []
    obs, info = env.reset(seed=0)
    while True:
        observations[(info['agent_id'], info['frame'])] = obs.copy()
        obs, _, _, done, info = env.step(1)
        frame_rewards.extend(info['frame_rewards'])
        if done:
            final = info['final_observations']
            break
    replay = ReplayBuffer(capacity=1000)
    gamma = 0.97
    report = MultiAgentCollector(env, replay, gamma).collect_episode(lambda obs: 1, seed=0)
    assert report['frames'] == 65
    assert len(replay) == len(report['transitions']) == sum(report['decisions_per_agent'].values())
    assert report['episode_team_reward'] == pytest.approx(sum(frame_rewards))
    assert report['episode_mean_speed_mph'] == pytest.approx(sum(frame_rewards)*80/65)
    assert np.mean(list(report['agent_mean_speed_mph'].values())) == pytest.approx(report['episode_mean_speed_mph'])
    for transition in report['transitions']:
        a, start, end = transition.agent_id, transition.start_frame, transition.end_frame
        np.testing.assert_array_equal(transition.state, observations[(a, start)])
        expected_next = final[a] if transition.truncated else observations[(a, end)]
        np.testing.assert_array_equal(transition.next_state, expected_next)
        expected_reward = sum(r*gamma**i for i,r in enumerate(frame_rewards[start:end]))
        assert transition.reward == pytest.approx(expected_reward)
        assert not transition.terminated
        assert transition.bootstrap_discount == pytest.approx(0 if transition.truncated else gamma**(end-start))
        if not transition.truncated:
            assert transition.elapsed_frames == 30
    if others == 10:
        assert report['controlled_agent_ids'] == tuple(range(11))
        assert 10 not in report['decisions_per_agent']
    assert env._thread is None


def test_policy_mapping_validation_and_error_cleanup():
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=1, episode_frames=40))
    collector = MultiAgentCollector(env)
    with pytest.raises(ValueError):
        collector.collect_episode({0: lambda obs: 0})
    with pytest.raises(ValueError):
        collector.collect_episode({0: lambda obs: 0, 1: lambda obs: 99})
    assert env._thread is None
    # Can reuse collector for a different episode without pending-state leakage.
    report = collector.collect_episode({0: lambda obs: 1, 1: lambda obs: 2}, seed=1)
    assert set(report['decisions_per_agent']) == {0, 1}
    assert {t.action for t in report['transitions'] if t.agent_id == 0} == {1}
    assert {t.action for t in report['transitions'] if t.agent_id == 1} == {2}


def test_replay_capacity_sampling_and_array_ownership():
    replay = ReplayBuffer(capacity=3, seed=5)
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=1, episode_frames=61))
    report = MultiAgentCollector(env, replay).collect_episode(lambda obs: 1)
    assert len(replay) == 3
    samples = replay.sample(3)
    assert len({(t.agent_id,t.start_frame) for t in samples}) == 3
    report['transitions'][-1].state[:] = -99
    assert all(np.all(t.state >= 0) for t in replay.sample(3))
    with pytest.raises(ValueError):
        replay.sample(4)


@pytest.mark.skipif(shutil.which('node') is None, reason='Node required')
@pytest.mark.parametrize('others', [1, 4, 9, 10])
def test_collected_mixed_actions_match_js_final_trajectories(others):
    frames = 95
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=others, episode_frames=frames))
    # Node dump uses a global scripted action list. We supply different actions
    # per callback owner, including the truncated final decision cycle.
    schedule = [(frame, 0 if frame % 30 == 0 else (frame % 30)//3)
        for frame in range(frames)
        if frame % 30 == 0 or (frame % 30) % 3 == 0 and 1 <= (frame % 30)//3 <= min(others,9)]
    actions = [(agent+1)%5 for frame, agent in schedule]
    js = subprocess.run(['node', str(backend_root()/'reference/node_ref/run_ref.js'),
        'dump', '--actions='+','.join(map(str, actions)), '--frames='+str(frames),
        '--seed-offset=1', '--other-agents='+str(others), '--lanes-side=3',
        '--patches-ahead=40', '--patches-behind=5'], capture_output=True, text=True, check=True)
    reference = json.loads(js.stdout)
    report = MultiAgentCollector(env).collect_episode({
        agent: (lambda obs, action=(agent+1)%5: action) for agent in env.decision_agent_ids}, seed=0)
    assert sum(report['decisions_per_agent'].values()) == len(actions)
    np.testing.assert_allclose([[c.b,c.a,c.c,c.y,c.x] for c in env._eng.z],
                               reference['states'][-1]['cars'], rtol=0, atol=1e-9)


def test_shared_and_mapped_evaluation_agree():
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=4, episode_frames=65))
    shared = env.evaluate(lambda obs: 1, runs=3)
    mapped = env.evaluate_agents({a: lambda obs: 1 for a in env.decision_agent_ids}, runs=3)
    assert shared == mapped
