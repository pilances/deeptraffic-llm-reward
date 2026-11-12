# DeepTraffic LLM Reward

Version 0.4 uses a pinned Python port of the JavaScript DeepTraffic engine,
with our own event-driven Gymnasium adapter. The research objective is to
improve a shared DQN policy using LLM feedback, measured by fleet speed.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/setup_backend.py
.venv/bin/python -m pytest
.venv/bin/python scripts/run_smoke.py --policy rule --runs 10 --frames 2000
```

Node.js is required for acceptance tests against JavaScript, not for Python
simulation. The setup script downloads Quillstacks into ignored `.external/`;
it pins a commit and validates engine and JS hashes. This external dependency
is required at runtime; `DEEPTRAFFIC_BACKEND` can point to another pinned checkout.
Deleting `.external/quillstacks` is safe if it contains no local work: rerun setup.
The setup script refuses to replace an existing checkout at a different revision.

Default: 20 vehicles, 11 controlled, 315 observation values, five actions,
2000 simulation frames. `step(action)` advances to the next vehicle's policy
callback. `info['agent_id']` identifies the vehicle requesting the NEXT action.
Use `close()` (or a context manager) to stop the suspended simulation worker.

```python
from deeptraffic_llm_reward import DeepTrafficEnv
from deeptraffic_llm_reward.policies import RuleBasedPolicy

policy = RuleBasedPolicy()
with DeepTrafficEnv() as env:
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, reward, terminated, done, info = env.step(policy.act(obs))
    print(info['episode_mean_speed_mph'])
    print(env.evaluate(policy.act, runs=500)['median'])
```

The Gym adapter is an alternating-agent interface, not a ready-made independent
single-agent DQN training loop. See [the transition contract](docs/environment_spec.md)
before constructing replay transitions. Evaluation uses the JS scoring protocol
independently of the training reward. The historical single-car starter can now
be trained with original ConvNetJS on the Python engine:

```sh
.venv/bin/python scripts/evaluate_starter.py --seed 0 --runs 500
.venv/bin/python scripts/verify_starter.py --seed 0 --runs 500
```

Repeat the first command with seeds 1 and 2 for independent training replicates.
This uses unchanged `network_basic.js` and its historical ego reward, not the
new team reward. Results and exported weights go to `results/starter/`.
These are generated artifacts and can be regenerated with the same commands.
`starter_brain.js` runs the original ConvNetJS learner in Node; `evaluate_starter.py`
feeds it the Python engine's exact decision observations and historical rewards.
Frozen weights are evaluated in Python, with a separate JS cross-check.
LLM feedback training and the multiagent replay collector remain future work.

Our prior paper-only environment, policies, tests and documentation are preserved
in `archive/paper_v03/` as historical source, excluded from active test discovery.
Its checkpoints, action IDs, observation encoding and scores are incompatible.

Read [the audit](docs/quillstacks_audit.md) for evidence, known engine quirks,
source provenance limits and why we replaced the simulation. We do not yet claim
to reproduce a historical leaderboard record or to have beaten a DQN baseline.
