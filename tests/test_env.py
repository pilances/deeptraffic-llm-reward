import json
import shutil
import subprocess
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.backend import backend_root

def test_gym_and_lifecycle():
    env = DeepTrafficEnv(DeepTrafficConfig(episode_frames=32))
    try:
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=0)
        assert obs.shape == (315,)
        assert info['controlled_vehicles'] == 11
        assert info['reward_objective'] == 'fleet_speed_integral'
        assert info['agent_id'] == 0
        _, _, _, _, info = env.step(1)
        assert info['agent_id'] == 1
        assert info['frame'] == 3
        env.reset(seed=2)  # close a worker suspended inside the callback
    finally:
        env.close()
    assert env._thread is None

@pytest.mark.parametrize('others', [0, 4, 10])
def test_adapter_uses_exact_callback_observation_and_actions(others):
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=others, episode_frames=100))
    reference = env._new_engine()
    reference.r = reference.t = 0
    reference.reset()
    states = []
    actions = [1, 3, 2, 4, 0]
    def policy(obs, reward):
        states.append(np.asarray(obs, dtype=np.float32))
        return actions[(len(states)-1) % len(actions)]
    total = 0.0
    for _ in range(100):
        reference.V(policy)
        total += sum(max(0, c.c*c.a)*20 for c in reference.z[:others+1])/(others+1)
    try:
        obs, _ = env.reset(seed=0)
        rewards = 0.0
        for i, expected in enumerate(states):
            np.testing.assert_array_equal(obs, expected)
            obs, reward, terminated, truncated, info = env.step(actions[i % 5])
            rewards += reward
            assert not terminated
            assert truncated == (i == len(states)-1)
        assert info['episode_mean_speed_mph'] == pytest.approx(total/100)
        assert rewards == pytest.approx(total/80)
        np.testing.assert_array_equal(
            [[c.x, c.y, c.a, c.c] for c in env._eng.z],
            [[c.x, c.y, c.a, c.c] for c in reference.z])
    finally:
        env.close()

@pytest.mark.skipif(shutil.which('node') is None, reason='Node required for JS parity')
@pytest.mark.parametrize('others', [0, 10])
def test_evaluator_matches_javascript(others):
    args = ['node', str(backend_root()/'reference/node_ref/run_ref.js'), 'eval',
        '--policy=accel', '--runs=3', '--frames=2000', '--det=true',
        '--lanes-side=3', '--patches-ahead=40', '--patches-behind=5', f'--other-agents={others}']
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    js = json.loads(result.stdout)
    env = DeepTrafficEnv(DeepTrafficConfig(other_agents=others))
    py = env.evaluate(lambda obs: 1, runs=3)
    assert py['median'] == pytest.approx(js['median'], abs=1e-9)
    assert py['mean'] == pytest.approx(js['mean'], abs=1e-9)

def test_invalid_action_does_not_resume_worker():
    env = DeepTrafficEnv(DeepTrafficConfig(episode_frames=1))
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(9)
        assert env.step(0)[3]
    finally:
        env.close()
