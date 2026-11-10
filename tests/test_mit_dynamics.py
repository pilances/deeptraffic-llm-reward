"""Regression scenarios for the source-based v0.2 reconstruction."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from deeptraffic_llm_reward import Action, DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.env import VehicleState


def isolated_env(**kwargs):
    env = DeepTrafficEnv(DeepTrafficConfig(vehicles=1, **kwargs))
    env.reset(seed=12)
    return env


def car(x, y, speed=65):
    return VehicleState(x, y, int(x // 20), speed, speed_mph=speed)


def test_frame_clock_partial_last_step_and_reward():
    env = isolated_env(episode_frames=35)
    _, reward, _, truncated, info = env.step(Action.MAINTAIN)
    assert info["frame"] == 30
    assert reward == 1 and not truncated
    _, reward, _, truncated, info = env.step(Action.DECELERATE)
    assert info["frame"] == 35 and truncated
    assert reward == pytest.approx(0.9)
    assert info["episode_mean_speed_mph"] == pytest.approx((30 * 80 + 5 * 72) / 35)


def test_relative_motion_sign_and_fixed_ego():
    env = isolated_env()
    env._cars += [car(0, 300, 60), car(120, 300, 80)]
    env._cars[0].speed_factor = 0.875  # 70 mph
    env._advance_frame()
    assert env._cars[0].y == 175
    assert env._cars[1].y > 300  # slower falls backwards
    assert env._cars[2].y < 300  # faster travels upwards


@pytest.mark.parametrize("exit_y,entry_y", [(-41, 700), (701, -40)])
def test_respawn_both_directions_randomizes_and_repeats(exit_y, entry_y):
    env = isolated_env()
    env._cars.append(car(0, exit_y, 10))
    env._respawn_exited()
    spawned = env.vehicles[1]
    assert spawned.y == entry_y
    assert 55 <= spawned.speed_mph <= 75
    assert spawned.x == spawned.target_lane * 20
    assert env._respawns == 1
    other = isolated_env()
    other._cars.append(car(0, exit_y, 10))
    other._respawn_exited()
    assert other.vehicles == env.vehicles


def test_respawn_uses_multiple_lanes():
    lanes = set()
    for seed in range(12):
        env = isolated_env()
        env.reset(seed=seed)
        env._cars.append(car(0, -41, 10))
        env._respawn_exited()
        lanes.add(env.vehicles[1].target_lane)
    assert len(lanes) > 1


def test_respawn_defers_if_boundary_is_full():
    env = isolated_env()
    env._cars += [car(0, -41)] + [car(lane * 20, 680) for lane in range(7)]
    env._respawn_exited()
    assert env._cars[1].y == -41
    assert env._respawns == 0


def test_vehicle_occupies_full_rectangle_and_both_lanes_in_transition():
    env = isolated_env()
    env._cars.append(car(10, 100, 60))
    grid = env._occupancy_grid()
    np.testing.assert_array_equal(grid[0:2, 10:14], np.full((2, 4), 60))
    assert np.count_nonzero(grid == 60) == 8


def test_observation_front_order_and_road_edges():
    env = isolated_env(patches_behind=2)
    env._cars[0].x = 0
    env._cars[0].target_lane = 0
    env._cars.append(car(0, 120, 60))  # rear ends at 160
    obs = env._observation()
    assert obs.shape == (3, 12)
    assert (obs[0] == 0).all()
    assert obs[1, 0] == 80  # cell 160..170
    assert (obs[1, 1:5] == 60).all()
    assert (obs[1, 10:] == 80).all()


def test_lane_change_is_continuous():
    env = isolated_env()
    env.step(Action.CHANGE_LEFT)
    ego = env.vehicles[0]
    assert ego.target_lane == 2
    assert 40 < ego.x < 60
    assert ego.x == pytest.approx(40 + 20 * (29 / 30) ** 30)


@pytest.mark.parametrize("y", [80, 180, 220])
def test_adjacent_front_side_and_rear_rectangles_block_change(y):
    env = isolated_env()
    env._cars.append(car(40, y))
    assert not env._lane_change_is_safe(0, 2)


@pytest.mark.parametrize("gap,expected", [(40, 60), (30, 30), (41, 80)])
def test_longitudinal_safety_formula(gap, expected):
    env = isolated_env()
    env._cars.append(car(60, 175 - 40 - gap, 60))
    env._advance_frame()
    assert env.vehicles[0].speed_mph == expected


def test_initial_conditions_and_nonoverlap_during_rollout():
    for seed in range(3):
        env = DeepTrafficEnv(DeepTrafficConfig(episode_frames=600))
        env.reset(seed=seed)
        assert env.vehicles[0].speed_mph == 80
        assert all(v.speed_mph == 65 for v in env.vehicles[1:])
        for i, a in enumerate(env.vehicles):
            assert all(not env._initial_conflict(a, b) for b in env.vehicles[i+1:])
        rng = np.random.default_rng(seed)
        # Check every frame, including intermediate lane-change geometry.
        for frame in range(600):
            if frame % 30 == 0:
                for i in range(len(env._cars)):
                    env._apply_action(i, Action(int(rng.integers(5))))
            env._advance_frame()
            cars = env.vehicles
            for i, a in enumerate(cars):
                assert 0 <= a.speed_mph <= 80
                assert all(not (env._lateral_overlap(a, b) and abs(a.y - b.y) < 40 - 1e-8)
                           for b in cars[i+1:])


def test_gymnasium_contract():
    check_env(DeepTrafficEnv(), skip_render_check=True)
