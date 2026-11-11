"""Historical tests; imports refer to the former v0.3 API."""
import numpy as np
import pytest

from deeptraffic_llm_reward import Action, DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.policies import RandomPolicy, RuleBasedPolicy


def test_reset_matches_observation_contract() -> None:
    env = DeepTrafficEnv()
    observation, info = env.reset(seed=7)

    assert observation.shape == (3, 10)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["step"] == 0
    assert len(env.vehicles) == 20


def test_same_seed_and_actions_are_reproducible() -> None:
    first = DeepTrafficEnv(DeepTrafficConfig(episode_steps=20))
    second = DeepTrafficEnv(DeepTrafficConfig(episode_steps=20))
    first_observation, _ = first.reset(seed=123)
    second_observation, _ = second.reset(seed=123)
    np.testing.assert_array_equal(first_observation, second_observation)

    actions = [0, 4, 2, 3, 1] * 4
    for action in actions:
        first_step = first.step(action)
        second_step = second.step(action)
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]


def test_invalid_action_is_rejected() -> None:
    env = DeepTrafficEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(5)


def test_out_of_road_lane_change_is_overridden() -> None:
    env = DeepTrafficEnv()
    env.reset(seed=2)
    env._cars[0].x = 0.0
    env._cars[0].target_lane = 0

    _, _, _, _, info = env.step(int(Action.CHANGE_LEFT))

    assert env.vehicles[0].target_lane == 0
    assert info["requested_action"] == "CHANGE_LEFT"
    assert info["action_applied"] == "MAINTAIN"
    assert info["safety_override"] is True


def test_truncation_and_mean_speed_are_reported() -> None:
    env = DeepTrafficEnv(DeepTrafficConfig(episode_steps=3))
    observation, _ = env.reset(seed=3)
    policy = RuleBasedPolicy()

    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(
            policy.act(observation)
        )

    assert 0.0 <= reward <= 1.0
    assert terminated is False
    assert truncated is True
    assert 0.0 <= info["episode_mean_speed_mph"] <= 80.0


def test_random_policy_is_seeded() -> None:
    observation = np.full((3, 10), 80.0, dtype=np.float32)
    first = RandomPolicy(seed=9)
    second = RandomPolicy(seed=9)
    assert [first.act(observation) for _ in range(20)] == [
        second.act(observation) for _ in range(20)
    ]

