"""Frame-based reconstruction of the published DeepTraffic dynamics.

See docs/environment_spec.md for sources, ambiguities and remaining assumptions.
Coordinates: x is the left edge; y is the nose, increasing down the screen.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(IntEnum):
    ACCELERATE = 0
    DECELERATE = 1
    CHANGE_LEFT = 2
    CHANGE_RIGHT = 3
    MAINTAIN = 4


@dataclass(frozen=True)
class DeepTrafficConfig:
    lanes: int = 7
    road_cells: int = 70
    vehicles: int = 20
    lanes_side: int = 1
    patches_ahead: int = 10
    patches_behind: int = 0
    max_speed_mph: float = 80.0
    traffic_initial_speed_mph: float = 65.0
    lane_width_px: float = 20.0
    cell_length_px: float = 10.0
    vehicle_length_px: float = 40.0
    ego_nose_px: float = 175.0
    decision_frames: int = 30
    episode_frames: int = 10_000
    # Compatibility: an explicit old episode_steps overrides episode_frames.
    episode_steps: Optional[int] = None
    # The following constants are reconstruction assumptions, not MIT facts.
    pixels_per_mph_per_frame: float = 1.0 / 60.0
    speed_factor_delta: float = 0.1
    respawn_min_speed_mph: float = 55.0
    respawn_max_speed_mph: float = 75.0
    background_decision_frames: tuple = (30, 45, 60)

    def __post_init__(self):
        integer_fields = ("lanes", "road_cells", "vehicles", "patches_ahead",
                          "decision_frames", "episode_frames")
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.episode_steps is not None and (
            not isinstance(self.episode_steps, int) or self.episode_steps < 1
        ):
            raise ValueError("episode_steps must be a positive integer")
        for name in ("lanes_side", "patches_behind"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_speed_mph", "lane_width_px", "cell_length_px",
                     "vehicle_length_px", "pixels_per_mph_per_frame",
                     "speed_factor_delta"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.traffic_initial_speed_mph <= self.max_speed_mph:
            raise ValueError("Invalid initial traffic speed")
        if not 0 < self.respawn_min_speed_mph <= self.respawn_max_speed_mph <= self.max_speed_mph:
            raise ValueError("Invalid respawn speed range")
        if not 0 <= self.ego_nose_px < self.road_height_px - self.vehicle_length_px:
            raise ValueError("Ego rectangle must fit on the road")
        if not self.background_decision_frames or any(
            not isinstance(t, int) or t < 1 for t in self.background_decision_frames
        ):
            raise ValueError("Background decision intervals must be positive integers")

    @property
    def road_height_px(self):
        return self.road_cells * self.cell_length_px

    @property
    def horizon_frames(self):
        return (self.episode_steps * self.decision_frames
                if self.episode_steps is not None else self.episode_frames)


@dataclass
class VehicleState:
    x: float
    y: float
    target_lane: int
    max_speed_mph: float
    speed_factor: float = 1.0
    speed_mph: float = 0.0
    decision_frames: int = 30


class DeepTrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or DeepTrafficConfig()
        c = self.config
        self.observation_space = spaces.Box(
            0.0, c.max_speed_mph,
            (2 * c.lanes_side + 1, c.patches_ahead + c.patches_behind),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)
        self._cars = []
        self._frame = self._step_count = self._respawns = 0
        self._speed_sum = self._safety_frames = 0.0

    @property
    def vehicles(self):
        return tuple(replace(car) for car in self._cars)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        self._frame = self._step_count = self._respawns = 0
        self._speed_sum = self._safety_frames = 0.0
        self._cars = [VehicleState(
            (c.lanes // 2) * c.lane_width_px, c.ego_nose_px,
            c.lanes // 2, c.max_speed_mph, speed_mph=c.max_speed_mph,
            decision_frames=c.decision_frames,
        )]
        # Rejection sampling can jam even when a safe arrangement exists.
        # Restart the whole layout rather than misclassifying a single jam.
        ego = self._cars[0]
        for restart in range(100):
            self._cars = [ego]
            for attempt in range(2000):
                if len(self._cars) == c.vehicles:
                    return self._observation(), self._info(None, None, False, 0.0)
                lane = int(self.np_random.integers(c.lanes))
                car = VehicleState(
                    lane * c.lane_width_px,
                    float(self.np_random.uniform(0, c.road_height_px - c.vehicle_length_px)),
                    lane, c.traffic_initial_speed_mph,
                    speed_mph=c.traffic_initial_speed_mph,
                    decision_frames=int(self.np_random.choice(c.background_decision_frames)),
                )
                # Keep every initial pair out of the longitudinal and adjacent
                # lane safety envelopes, including the full vehicle rectangle.
                if all(not self._initial_conflict(car, other) for other in self._cars):
                    self._cars.append(car)
        raise ValueError("Could not initialize this density safely after 100 layouts")

    def _initial_conflict(self, a, b):
        c = self.config
        lane_distance = abs(a.target_lane - b.target_lane)
        required = c.vehicle_length_px + (6 if lane_distance == 1 else 4) * c.cell_length_px
        return lane_distance <= 1 and abs(a.y - b.y) <= required

    def step(self, action):
        if not self._cars or self._frame >= self.config.horizon_frames:
            raise gym.error.ResetNeeded("Call reset before stepping a new episode")
        if not self.action_space.contains(action):
            raise ValueError("action must be an integer from 0 through 4")
        requested = Action(int(action))
        applied = self._apply_action(0, requested)
        blocked = applied != requested
        action_blocked = blocked
        frame_speeds = []
        for _ in range(min(self.config.decision_frames,
                           self.config.horizon_frames - self._frame)):
            for i, car in enumerate(self._cars[1:], 1):
                if self._frame % car.decision_frames == 0:
                    self._apply_action(i, Action(int(self.np_random.integers(5))))
            overridden = self._advance_frame()
            frame_speeds.append(self._cars[0].speed_mph)
            self._safety_frames += bool(overridden or action_blocked)
            blocked = blocked or overridden
            self._frame += 1
        self._step_count += 1
        speed_sum = sum(frame_speeds)
        self._speed_sum += speed_sum
        mean_speed = speed_sum / len(frame_speeds)
        return (self._observation(), mean_speed / self.config.max_speed_mph,
                False, self._frame >= self.config.horizon_frames,
                self._info(requested, applied, blocked, mean_speed))

    def _apply_action(self, index, action):
        car, c = self._cars[index], self.config
        if action == Action.ACCELERATE:
            car.speed_factor = min(1.0, car.speed_factor + c.speed_factor_delta)
        elif action == Action.DECELERATE:
            car.speed_factor = max(0.0, car.speed_factor - c.speed_factor_delta)
        elif action in (Action.CHANGE_LEFT, Action.CHANGE_RIGHT):
            lane = car.target_lane + (-1 if action == Action.CHANGE_LEFT else 1)
            if not self._lane_change_is_safe(index, lane):
                return Action.MAINTAIN
            car.target_lane = lane
        return action

    def _lane_change_is_safe(self, index, lane):
        c, car = self.config, self._cars[index]
        if not 0 <= lane < c.lanes:
            return False
        left, right = lane * c.lane_width_px, (lane + 1) * c.lane_width_px
        top = car.y - 6 * c.cell_length_px
        bottom = car.y + c.vehicle_length_px + c.cell_length_px
        for j, other in enumerate(self._cars):
            if j == index:
                continue
            # Include the other driver's reserved destination to prevent two
            # simultaneous lane changes into the same gap.
            other_left = min(other.x, other.target_lane * c.lane_width_px)
            other_right = max(other.x, other.target_lane * c.lane_width_px) + c.lane_width_px
            if (left < other_right and right > other_left
                    and top < other.y + c.vehicle_length_px and bottom > other.y):
                return False
        return True

    def _lateral_overlap(self, a, b):
        return abs(a.x - b.x) < self.config.lane_width_px - 1e-9

    def _advance_frame(self):
        c = self.config
        # Euler integration of dx/dt=(x_target-x)/T; keep rectangles separate.
        lateral_blocked = False
        for i, car in enumerate(self._cars):
            old_x = car.x
            car.x += (car.target_lane * c.lane_width_px - car.x) / c.decision_frames
            if abs(car.x - car.target_lane * c.lane_width_px) < 1e-6:
                car.x = car.target_lane * c.lane_width_px
            if any(j != i and self._lateral_overlap(car, other)
                   and abs(car.y - other.y) < c.vehicle_length_px
                   for j, other in enumerate(self._cars)):
                car.x = old_x
                lateral_blocked |= i == 0

        # Resolve front to back so followers see the leader's already limited
        # speed. A small kinematic guard prevents crossing during a finite frame.
        ordered = sorted(range(len(self._cars)), key=lambda i: self._cars[i].y)
        ego_override = lateral_blocked
        for n, i in enumerate(ordered):
            car = self._cars[i]
            wanted = car.max_speed_mph * car.speed_factor
            speed = wanted
            for j in ordered[:n]:
                lead = self._cars[j]
                if not self._lateral_overlap(car, lead):
                    continue
                gap = car.y - lead.y - c.vehicle_length_px
                # Interpret four cells as bumper clearance (paper ambiguous).
                if gap <= 4 * c.cell_length_px:
                    cap = lead.speed_mph if gap >= 4 * c.cell_length_px - 1e-9 else lead.speed_mph / 2
                    speed = min(speed, cap)
                speed = min(speed, lead.speed_mph + max(0.0, gap) / c.pixels_per_mph_per_frame)
            car.speed_mph = max(0.0, speed)
            if i == 0:
                ego_override |= speed < wanted - 1e-9
        ego_speed = self._cars[0].speed_mph
        for car in self._cars[1:]:
            car.y -= (car.speed_mph - ego_speed) * c.pixels_per_mph_per_frame
        self._respawn_exited()
        return ego_override

    def _respawn_exited(self):
        c = self.config
        for i, car in enumerate(self._cars[1:], 1):
            if car.y + c.vehicle_length_px >= 0 and car.y <= c.road_height_px:
                continue
            # Fully leaves the top -> enters at bottom, and conversely.
            y = c.road_height_px if car.y < 0 else -c.vehicle_length_px
            for lane in self.np_random.permutation(c.lanes):
                candidate = VehicleState(float(lane * c.lane_width_px), y, int(lane), 65.0)
                if any(j != i and self._lateral_overlap(candidate, other)
                       and abs(y - other.y) < c.vehicle_length_px + 4 * c.cell_length_px
                       for j, other in enumerate(self._cars)):
                    continue
                speed = float(self.np_random.uniform(c.respawn_min_speed_mph, c.respawn_max_speed_mph))
                self._cars[i] = replace(
                    candidate, max_speed_mph=speed, speed_mph=speed,
                    decision_frames=int(self.np_random.choice(c.background_decision_frames)),
                )
                self._respawns += 1
                break
            # If all entry lanes are blocked, keep the car outside and retry.

    def _occupancy_grid(self):
        c = self.config
        grid = np.full((c.lanes, c.road_cells), c.max_speed_mph, np.float32)
        for car in self._cars[1:]:
            first_lane = max(0, int(np.floor(car.x / c.lane_width_px)))
            last_lane = min(c.lanes, int(np.ceil((car.x + c.lane_width_px - 1e-9) / c.lane_width_px)))
            start = max(0, int(np.floor(car.y / c.cell_length_px)))
            end = min(c.road_cells, int(np.ceil((car.y + c.vehicle_length_px) / c.cell_length_px)))
            if end > start:
                grid[first_lane:last_lane, start:end] = np.minimum(
                    grid[first_lane:last_lane, start:end], car.speed_mph)
        return grid

    def _observation(self):
        c, ego = self.config, self._cars[0]
        grid = self._occupancy_grid()
        obs = np.zeros(self.observation_space.shape, np.float32)
        # Front cells: nearest first; rear cells: nearest rear bumper first.
        front = int(np.floor(ego.y / c.cell_length_px))
        rear = int(np.ceil((ego.y + c.vehicle_length_px) / c.cell_length_px))
        columns = list(range(front - 1, front - c.patches_ahead - 1, -1))
        columns += list(range(rear, rear + c.patches_behind))
        lane = int(np.floor((ego.x + c.lane_width_px / 2) / c.lane_width_px))
        for row, road_lane in enumerate(range(lane - c.lanes_side, lane + c.lanes_side + 1)):
            if 0 <= road_lane < c.lanes:
                for col, road_cell in enumerate(columns):
                    if 0 <= road_cell < c.road_cells:
                        obs[row, col] = grid[road_lane, road_cell]
        return obs

    def _info(self, requested, applied, overridden, decision_mean):
        return {
            "environment_version": "0.2.0",
            "step": self._step_count, "frame": self._frame,
            "ego_speed_mph": float(self._cars[0].speed_mph),
            "decision_mean_speed_mph": float(decision_mean),
            "episode_mean_speed_mph": float(self._speed_sum / self._frame if self._frame else self._cars[0].speed_mph),
            "requested_action": requested.name if requested is not None else None,
            # Accepted high-level action, before per-frame speed intervention.
            "action_applied": applied.name if applied is not None else None,
            "safety_override": bool(overridden),
            "safety_override_rate": self._safety_frames / self._frame if self._frame else 0.0,
            "respawns": self._respawns,
        }
