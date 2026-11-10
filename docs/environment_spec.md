# DeepTraffic environment v0.2

This version replaces the circular point-vehicle v0.1. Its old 70.699 mph score
is not comparable. The complete browser simulator has not been recovered;
leaderboard parity remains unverified.

## Sources

[MIT README](https://github.com/lexfridman/deeptraffic#deeptraffic-simulation--game),
[paper v2, section 3](https://arxiv.org/html/1801.02805v2#S3),
and the local network_basic.js starter configuration.

Documented geometry: 140 by 700 pixels, seven lanes, twenty 20 by 40 pixel cars,
7 by 70 grid cells. Ego nose stays at y=175. Initial speeds are 80 mph for ego
and 65 mph for background cars. Decisions occur every 30 frames. The paper
specifies ego-relative longitudinal motion and gradual lateral movement.
Exiting cars return at the opposite end with randomized speed and lane.

Source contradictions: the README assigns 80 to free cells, the paper zero.
We choose the README for policy input. For following below four cells, the
paper's formula halves leader speed, whereas its prose describes a
distance-dependent reduction. We implement the explicit formula.

## Transition contract

Positions are continuous floating-point pixels. x is the left edge, y the nose;
y increases down the screen. Bodies span [x,x+20) and [y,y+40).
Each step accepts an ego action, then integrates 30 frames:

1. Due background drivers choose actions (frame modulo their decision interval).
2. x += (target_lane * lane_width - x) / decision_frames.
3. Requested speed = individual maximum speed * persistent throttle factor.
4. Resolve following limits from front to back.
5. Non-ego y -= (speed - ego_speed) * pixels_per_mph_per_frame.
6. Replace fully exited cars at the opposite boundary.

No distance or observation wraps around the screen. Respawning selects uniformly
among safe lanes. If every lane is blocked, keep the car outside and retry.

## State and actions

IDs: 0 accelerate, 1 decelerate, 2 left, 3 right, 4 maintain.
Throttle changes are bounded to [0,1]. Safety limits actual speed but preserves
the throttle request. Safe lane requests change target_lane, not instantaneous x.
action_applied reports the accepted high-level action; safety_override reports
any rejected request or per-frame intervention.

Default observation is float32 (3,10). Rows run left to right around the lane
containing the ego center. Columns run nearest to farthest ahead of the nose's
grid cell. Optional rear columns follow, nearest to farthest behind the bumper.
Cells are fixed in screen coordinates. Full vehicle rectangles occupy every
intersected cell, including two lanes during a lane change. Ego is excluded.
Free cells contain 80, occupied cells the minimum occupying speed, off-road
cells zero. Thus an 80 mph car and empty space share a value, as in the README.
Flatten row-major; normalization belongs in policies.
The smoke rule assumes patches_behind=0 and lanes_side=1.

## Safety reconstruction

Adjacent-lane tests cover six cells ahead, the four-cell body, and one behind.
Other drivers' destination lanes are reserved to prevent conflicting requests.
Following distance is interpreted as bumper clearance: cap to leader speed at
four cells and half leader speed below four. Never force acceleration above
requested speed. The paper does not uniquely establish this bumper convention.

Additional numerical guards reject lateral overlap and prevent forward crossing
of the leader within a frame. These conservative guards are local additions.
Initialization excludes pairwise same/adjacent-lane safety envelopes. Bounded
rejection sampling restarts jammed layouts and raises on infeasible density.

## Remaining assumptions and limits

| Choice | Reconstruction default, not a recovered MIT constant |
| --- | --- |
| Motion scale | 1/60 pixels per mph per frame |
| Throttle increment | 0.1 per action |
| Background policy | Uniform choice of the five actions |
| Background intervals | Seeded choice among 30,45,60 frames |
| Respawn speed | Uniform 55–75 mph, also individual maximum |
| Respawn throttle | 1.0 |
| Exit / insertion | Entire body exits; safe insertion or deferred retry |
| Observation alignment | Fixed screen grid and ordering defined above |
| Training reward | Decision-interval mean ego speed / 80 |
| Safety guards, gap convention and layout sampler | Local reconstruction |

Single learned agent matches otherAgents=0 in the starter. Optional additional
learned agents, browser DQN and rendering are not implemented.

## Evaluation and reproducibility

Default horizon: 10,000 simulation frames, 334 decisions, last interval 10 frames.
The paper's figure caption uses 10,000 simulation steps but later prose gives
different totals; this frame interpretation is explicit, not server parity.
episode_mean_speed_mph weights every frame equally, including short final
intervals. Score is the median episode mean over fixed seeds; report IQR too.
No warm-up. safety_override_rate counts affected ego frames; rejected decisions
count for their whole interval. Reward and evaluation score are separate.

Development evaluation seeds: 0..9. Use disjoint training and final test seeds.
Freeze policies and disable learned-policy exploration for evaluation.
Same seed, config and actions reproduce all transitions and respawns.
Different policies alter trajectories and spawn eligibility: same seeds do not
guarantee identical subsequent traffic. Save version, config, seeds and policy
with experiment results. Smoke output is printed, not persisted.

Explicit legacy episode_steps=N means N times decision_frames; otherwise the
episode_frames limit applies. The runner uses --frames. Reset after truncation.

## Files and running

env.py contains configuration, vehicle state, reset and transition functions.
tests/test_mit_dynamics.py covers geometry, respawn, safety and timing.
scripts/run_smoke.py runs seeded policies. .venv is generated, Git-ignored,
and can be deleted and recreated using the README setup instructions.

```sh
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python scripts/run_smoke.py --policy rule --seeds 10 --frames 10000
.venv/bin/python scripts/run_smoke.py --policy random --seeds 10 --frames 10000
```
