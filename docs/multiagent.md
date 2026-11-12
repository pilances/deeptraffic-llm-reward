# Multiagent rollout and replay collector

The existing JS engine already moved all 20 cars in one world. The new layer
exposes who decides, accepts shared or per-car policies, and builds replay
experiences for the same car across its staggered decision times.
No neural network training or LLM feedback is performed by the collector.

## Run and artifacts

From the repository with the pinned backend installed:

```sh
.venv/bin/python scripts/run_multiagent.py --agents 5 --policy rule --episodes 3 --out results/multiagent/five_cars.json
.venv/bin/python scripts/run_multiagent.py --agents 5 --policy random --policy-mode per-agent --episodes 3
.venv/bin/python scripts/run_multiagent.py --agents 11 --policy rule --episodes 3
.venv/bin/python -m pytest -rA
```

`--agents` counts controlled vehicles INCLUDING ego, so `--agents 5` configures
`other_agents=4`. `--policy-mode shared` reuses one callable; `per-agent` creates
a callable for every actual decision owner. Random policies then have independent
seeded RNGs. Rule policies are stateless and identical in either mode.
Observation parameters remain configurable via --lanes-side, --patches-ahead,
--patches-behind. Defaults are (3,40,5). No sweeps or tuning are implicit.

JSON output records backend revision, config, seeds, policy, counts per car,
per-car speeds, fleet mean speed and transition count. These are generated
diagnostic artifacts, safely regenerated with the command; no raw checkpoints
are overwritten. CLI `--out` overwrites only the specified report file.
Rollouts use independent reset(seed) layouts, so their scores must not be
compared numerically with the historical sequential 500-run evaluation block.

## Python API

```python
from deeptraffic_llm_reward import (
    DeepTrafficConfig, DeepTrafficEnv, MultiAgentCollector, ReplayBuffer,
)
from deeptraffic_llm_reward.policies import RuleBasedPolicy

env = DeepTrafficEnv(DeepTrafficConfig(other_agents=4))
replay = ReplayBuffer(capacity=10_000, seed=0)
collector = MultiAgentCollector(env, replay, gamma_per_frame=0.99 ** (1/30))
policy = RuleBasedPolicy().act
episode = collector.collect_episode(policy, seed=0)
batch = replay.sample(32)

# Alternatively, different callables keyed by the actual decision owners:
policies = {agent: RuleBasedPolicy().act for agent in env.decision_agent_ids}
episode = collector.collect_episode(policies, seed=1)
score = env.evaluate_agents(policies, runs=500)
```

collect_episode closes its worker on success and error. Reusing it starts a
fresh episode with no pending-transition carryover. Replay intentionally persists.
Policies receive only their local observation, not global state or other cars'
observations. IDs route calls, but are not injected into model inputs.
Independent policies are an experimental extension; MIT's cloned-policy setup
is represented by a shared callable (or identical stateless clones).

## Same-car transitions and reward

At frames 0,3,6,... the engine asks different cars for actions. When car A acts
again at frame 30, its previous (state,action) is completed with A's current
observation. B's state is never used as A's next state.

The environment reports per-completed-frame normalized fleet rewards in
info.frame_rewards. The collector adds each frame reward to every currently
pending action. This duplicates a common team reward by design; summing replay
rewards across cars is NOT the episode's global reward. Episode team reward is
recorded separately and equals fleet mean mph * episode_frames / 80.

For elapsed d frames and per-frame discount g, each transition contains
`reward = sum(g**k * fleet_reward[k], k=0..d-1)` and
`bootstrap_discount = g**d`. A future trainer's target is
`reward + bootstrap_discount * max_next_Q`. For a discount of 0.99 per
30-frame decision interval, pass g=0.99**(1/30), NOT g=0.99.

Our chosen finite-horizon objective ends at episode_frames. Pending transitions
are closed with same-car final snapshots, truncated=True, terminated=False,
and bootstrap_discount=0. This deliberately disables bootstrapping beyond the
finite experiment horizon. If later studying a continuing task, truncation
handling must be redesigned explicitly. Final snapshots are not new decisions.

Replay is bounded, samples without replacement, and owns observation copies.
The collector emits one transition per actual action, including each car's last
partial interval. Zero-decision cars do not receive invented transitions.

## Native engine limits

All cars share the road at the same simulation time; action calls remain
staggered, as in the native engine. There is no artificial synchronous joint
action step. Native side effects are preserved: a car's action can leak into
following controlled cars and ego in that frame.

With other_agents=10 there are eleven controlled/scored vehicles (0..10), but
only ten callback owners (0..9). `G % 30 == 3*10` is impossible. Car 10 receives
leaked actions and is scored; it never independently chooses. The API exposes
controlled_agent_ids separately from decision_agent_ids. Policy mappings must
match decision_agent_ids exactly, and supplying an independent policy for car 10
is rejected. For all controlled cars to have callbacks use at most ten cars.
Action leakage still exists in that configuration. Correcting the JS behavior
would create a different benchmark and is outside this implementation.

## Evidence and alignment

The MIT paper's Table 1 defines other_agents as non-ego vehicles controlled by
a clone of the competitor's code:
https://arxiv.org/html/1801.02805v2#S3
Shared-policy mode implements that intention. The paper does not establish the
car-10 quirk; that finding comes from the pinned executable JS and parity tests.

The professor's supplied message discusses an LLM solving a multiagent system
and rules such as clearing a lane for faster cars. Multiple cars in one common
world and the fleet metric provide an experimental basis for that objective.
The message does not prescribe a particular agent count or this replay design.
Our team reward is a declared research choice; historical starter training keeps
its original ego reward. A lower/higher smoke score is not evidence that a new
learning method beats MIT or implements cooperative behavior successfully.

Tests verify exact same-car state pairing; per-frame reward accumulation and
discounting; final partial intervals; policy routing; invalid-policy cleanup;
replay ownership/capacity; and shared/mapped evaluation equivalence.
Mixed per-car actions are also checked against Node/JS final states of all
20 vehicles for 2,5,10,11 controlled cars. Existing full-trajectory parity and
historical starter tests continue to run.

## Executed acceptance results

22 project tests passed, including the nested 18-test upstream suite. The new
mixed-action Node trajectory comparisons used an absolute tolerance of 1e-9
for all 20 cars' final (lane, throttle, safety, y, x) fields.

Executed CLI diagnostics with seeds 0,1,2 and 2000 frames per episode:

| Controlled cars | Policy | Callback owners | Transitions over 3 episodes | Median fleet mean mph |
| ---: | --- | ---: | ---: | ---: |
| 1 | shared rule | 1 | 201 | 66.69848 |
| 5 | shared rule | 5 | 1005 | 72.97489 |
| 11 | shared rule | 10 | 2001 | 75.82351 |
| 5 | independent random | 5 | 1005 | 65.91030 |

Full JSONs are in results/multiagent/. No models were trained or hyperparameters
selected using these results. Three diagnostic episodes do not establish a
performance ranking. The historical 66.8 mph number used a different observation,
training policy and evaluation sequence, so these values are not a reproduction
or a claimed improvement over it.
