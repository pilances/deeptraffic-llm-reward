# Engine audit and migration decision (2026-11-11)

## Decision

Use the pinned Quillstacks engine as an external simulation backend, replacing
the active paper reconstruction. Its multi-car control and JS reference tests
are better aligned with the professor's fleet-speed / DQN transfer objective.
Retain our own Gymnasium and evaluation interfaces. Archive v0.3 intact except
for archival headings. Do not transfer its numeric results or checkpoints.

Sources:
- https://github.com/Quillstacks/DeepTrafficByLLM/tree/acd7c2c9157f5b480e15cf0bcd2c793de45c5b68
- https://github.com/lexfridman/deeptraffic
- Quillstacks reference/original_js/gameopt.js, reference/node_ref/run_ref.js,
  reference/VALIDATION.md and tests/test_fidelity.py at the pinned commit.

## Observed evidence

Executed upstream parity suite locally with Node v26.13.0 and our Python 3.9
environment: 18 passed (score sample reduced to 3 runs of 2000 frames).
Includes single- and multiple-controlled-car scripted trajectories, fixed-policy
scores and API tests. This independently confirms the sampled Python/JS parity;
it is not exhaustive verification of every possible state.

Our acceptance suite adds exact callback observation/action replay comparisons
for 1, 5 and 11 controlled cars, cumulative team reward accounting, final states,
Gymnasium checker, reset/close lifecycle, invalid action handling, and independent
Python-vs-Node evaluation for 1 and 11 controlled cars. Upstream's 18 tests are
also invoked from our suite; absence of Node does not count as full acceptance.

Migration verification originally recorded 8 project tests. The current suite
passes 9 project tests (including the nested 18-test
upstream suite). Smoke evaluation, 10 sequential deterministic runs x 2000 frames,
11 controlled cars: our observation-only rule median 76.38 mph; maintain-only
median 59.82 mph. These small-sample controls validate integration, not a claim
of beating a trained historical baseline. Reproduce with run_smoke.py using
`--policy rule --runs 10` or `--policy maintain --runs 10`.

## Findings affecting reuse

1. Upstream step returns the ego reward computed at the opening boundary from
   the preceding interval. Naively storing it with the newly supplied action
   misaligns credit assignment. We do not use that wrapper or its reward.
2. Upstream step broadcasts its argument to every policy callback in a 30-frame
   block. We pause at each callback and supply each action from its actual state.
3. Upstream reset returns a grid before the next V() motion/safety update. Our
   reset returns the actual first decision observation.
4. Original JS leaks a controlled-other's local action to subsequent cars and
   ego in the same frame. Index 10 never independently decides because G%30
   cannot equal 30. Preserve these facts for code parity; silently repairing
   them would define a different benchmark.
5. The original reward is ego-centric; a fleet score does not imply that the
   original DQN was trained with a shared fleet reward. Our new training reward
   is explicitly defined as a team objective. Baseline comparisons must state it.
6. Quillstacks' headline 76.50 uses 24 runs, while other comparisons use 500.
   We do not treat its headline as our reproduced result or a theoretical ceiling.

## Provenance and distribution limits

The original Fridman repository does not contain the full gameopt.js engine.
Quillstacks provides a JS copy and a reference harness, but the reviewed files
do not provide a complete independently authenticated acquisition chain tying
its exact bytes to the server version of a specific historical record.
Thus our claim is parity with this PINNED JS reference, not certification that
every historic MIT score used it. Re-evaluate an identified trained DQN on this
same engine before claiming a baseline win. Neither that full DQN reproduction
nor a new DQN/LLM training run is completed by this environment migration.

No top-level license granting reuse of all Quillstacks additions was found in
the inspected checkout. We keep it as an ignored, explicitly downloaded external
dependency and do not vendor or relicense its source under this project's MIT
license. Clarify redistribution permission before bundling third-party sources.

Pinned hashes (also enforced at runtime):
- engine.py SHA256: 17727a936beb0f76af29e09746bd5f0631a2057bb8b8e4fd434aa700e247c254
- gameopt.js SHA256: 661c1e6e23c337506d6588d76fe4fd9f679b9d266f0dc6a68b8fd4e9b674bf86

The backend is downloaded by scripts/setup_backend.py; no upstream scripts run
automatically during normal environment import. Only the verified engine module
executes. Tests explicitly execute the pinned Node harness and upstream tests.
