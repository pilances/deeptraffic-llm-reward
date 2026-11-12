# Environment v0.4: pinned code port, fleet objective

## Source and geometry

Backend: Quillstacks/DeepTrafficByLLM commit
`acd7c2c9157f5b480e15cf0bcd2c793de45c5b68`, engine.py only.
The complete checkout is an external dependency; we do not import its training
loop, Gym-like wrapper, LLM prompts or heuristics. `backend.py` verifies hashes.
This replaces the former paper-only normative specification following the user's
explicit authorization to prefer historical baseline comparability.

Seven lanes, 20 vehicles, 140x700 road; original sampled 15x34 footprint,
scroll control around y=525, speed `20*c*a`, original safety scans and respawn
at y=734/-34. Preserve JS operation order, RNG consumption and action leakage.
The paper remains background, not the sole executable specification.

## Observation and actions

Default flattened float32 vector: 7 columns x (40 ahead + 5 rear) = 315.
Reshape as `(45, 7)`: rows far-ahead to rear, columns left to right.
Free road is 1; off-road is 0; vehicle values are `(c*a)/100`, mph = value*2000.
Stopped vehicles and off-road both encode zero. A policy must not assume zero
means free. No absolute lane number or global state is added to the observation.

IDs: 0 maintain, 1 accelerate, 2 decelerate, 3 left, 4 right.
The original throttle increment is 0.02. Safety and historic shared-variable
effects may change other vehicles when one policy action is requested.

## Event-driven stepping

`reset(seed)` creates a fresh engine and seeded layout, then pauses INSIDE the
first true policy callback, after that frame's motion/safety preprocessing.
`step(action)` supplies that callback's action and runs until the next callback
or horizon. A thread and queues suspend the unmodified engine call stack.
Inference latency never advances simulation time. Call close when abandoning
an episode; reset automatically closes a previous worker; worker errors propagate.

Ego requests at G%30=0; controlled others at G%30=3*index. With 10 others,
vehicle 10 never satisfies that condition (30 is outside 0..29), but remains
controlled/scored and receives leaked actions. This is preserved JS behavior.
The engine cap is 9 at construction, 10 after reset: there really are 11
controlled/scored cars in the default configuration, not 11 independent callbacks.

`info.agent_id` identifies the NEXT requesting agent, not the agent just acted.
`info.frame` counts fully completed frames, `elapsed_frames` gives the interval.
The returned reward is the sum of completed-frame fleet mph divided by 80 since
the previous callback. Episode reward therefore equals total fleet mph/80 over
all frames. This is our team reward definition, not the historical ego reward.
The environment exposes this as `reward_objective='fleet_speed_integral'` and in
every reset/step info dictionary. New Gymnasium agents optimize this objective.
No artificial reward is assigned to preprocessing before the initial action.
Termination is false; the frame horizon causes truncation. Final obs is an ego
snapshot; it is not a next decision and must not be used as another agent's state.

For a shared per-car DQN, maintain pending transitions PER AGENT and accumulate
team rewards until that SAME agent next decides. Use elapsed simulation time for
discounting; handle final incomplete intervals explicitly. Do not feed consecutive
callbacks from different agents into an off-the-shelf single-agent replay buffer.
Agent 10 has no independent decisions. `MultiAgentCollector` now performs this
per-car collection, discounted frame-reward aggregation and finite-horizon closure;
`ReplayBuffer` stores and samples the resulting transitions. Neural-network
updates remain future work. See multiagent.md for the complete contract.
For an LLM reward experiment, replace/add the training signal while evaluating
with the unchanged fleet metric and identical scenarios/configuration.

The MIT starter reproduction is deliberately outside this Gym contract. Its
public `HistoricalStarterRunner` exposes the original callback reward
`(N-60)/20` for a single ego vehicle. It is a baseline reproduction utility,
not a training template for new RL agents. This separation prevents silent
reward changes while preserving the historical result.

## Evaluation

`evaluate(policy, runs=500)` creates an independent fresh engine, restores its
deterministic RNG anchor and keeps streams across runs, matching JS reset order.
Each run uses 2000 frames by default. For each completed frame sum
`max(0,c*a)/(controlled count)` per car; run score is
`floor(total/frames*2000)/100`. Report upper-middle sorted score for even run
counts, NOT the average of the two central values. Return all run scores too.
This protocol differs from independent `reset(seed)` episodes used in training.

No warm-up. No wall-clock speed requirement. No LLM/DQN-specific simulation changes.
Use disjoint development and held-out scenarios and multiple training seeds;
the fixed historical evaluation block alone must not serve as a prompt tuning set.
