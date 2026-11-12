# MIT starter reproduction

## Measured results (2026-11-11)

Each row is an independent 30000-decision training run, followed by the same
500 x 2000-frame deterministic test protocol. No seed was selected or discarded.

| Training seed | Median episode speed (mph) | Mean episode speed (mph) | Delta to 66.8 mph |
| --- | ---: | ---: | ---: |
| 0 | 66.32 | 65.37686 | -0.48 |
| 1 | 66.35 | 65.99530 | -0.45 |
| 2 | 66.68 | 66.23358 | -0.12 |

The arithmetic mean of the three benchmark medians is 66.45 mph: 0.35 mph
(approximately 0.52%) below the approximate 66.8 mph reference. The range is
66.32-66.68 mph. Three training seeds are a small replication, not an estimate
of the full training distribution. The reference's use of an approximate value
and unspecified training seed precludes attributing the residual to one cause.

All three runs reached age 30000 and epsilon 0. The exported Python forward pass
matched ConvNetJS on 101 probes per network (maximum absolute Q error below
2.3e-15). The standard eight environment tests and the new bridge integration
test passed. Full raw score distributions and checkpoints are in results/starter/.

The full independent JavaScript evaluation of checkpoint 0 reproduced both
66.32 mph median and 65.37686 mph mean over 500 episodes exactly to reported
precision (asserted tolerance 1e-9). No score discrepancy was found at the
Python/JS boundary for that trained policy.

Matched controls (same 30 inputs, one controlled car, 500 x 2000 frames):

| Policy | Median mph | Mean mph |
| --- | ---: | ---: |
| Maintain only | 51.57 | 51.66732 |
| Our simple rule policy | 66.19 | 65.56954 |

The three starter medians all exceed this heuristic median (by 0.13, 0.16,
0.49 mph respectively). This is descriptive, not a statistical superiority
claim. The earlier 76.38 mph was a different, 11-controlled-car/315-input task.
These results show why configuration-matched comparisons are essential.

Reference: https://github.com/lexfridman/deeptraffic states approximately
66.8 mph for network_basic.js. It does not give a distribution across training
seeds or enough provenance to recover the exact historical trained checkpoint.
The previously discussed 40 mph is not the reference stated for this file.

## Fixed experiment

- Unchanged root network_basic.js, matching the user's supplied source.
- 30 inputs, hidden fully-connected 10 ReLU units, 5 output action values.
- Original ConvNetJS/deepqlearn from the pinned external backend; no substitution
  with Double DQN, a target network, a new optimizer or changed reward.
- Replay 3000, learning threshold 500, gamma .5, SGD .001, momentum 0,
  batch size 64, L2 .01; burn-in 1000, exploration decays to zero at age 10000.
  Learning continues after exploration reaches zero, as in ConvNetJS.
- Training callback calls backward(previous reward) then forward(current state).
  The engine supplies historical `(N-60)/20`. We deliberately bypass the new
  Gym adapter's fleet reward for this historical reproduction.
- lanesSide=1, patchesAhead=10, patchesBehind=0, otherAgents=0.
- Original train_webworker schedule: 30*30000 = 900000 frames, 10 blocks of
  90000 frames, 30000 policy decisions. Network and replay persist across resets.
  Historical cross-episode learner windows are preserved rather than repaired.
- Three prespecified training seeds (0,1,2); no selection by evaluation score.
- Each frozen network: fresh-engine deterministic 500 runs x 2000 frames,
  upper-middle median of floored per-run average speeds, exploration zero.

Browser Math.random is replaced by a documented seeded LCG in the Node worker.
Training scenario anchors use NumPy's seeded generator. These make OUR runs
reproducible; they do not recover the unknown historical initialization/seeds.
Engine observations during training retain JS-compatible double precision.
Our existing evaluation API exposes float32 observations; the checkpoint's
float64 forward pass is checked against ConvNetJS, and a complete evaluation
against the double-precision JS engine checks for score effects of this boundary.

## Artifacts and verification

Run scripts/evaluate_starter.py for each seed. It writes the network, final age,
epsilon, replay size, source hash, backend revision, timing, and every evaluation
episode's score. The checkpoint is for inference, not exact training resumption
(replay and optimizer state are not serialized).

scripts/verify_starter.py loads seed 0's trained weights into original JavaScript
and compares 500-episode median AND mean with our Python result. It also evaluates
maintain-only and our simple rule policy with the SAME 30-input, single-car setup.
The earlier 76.38 mph heuristic used 315 inputs and eleven controlled cars and
must not be used as this experiment's matched baseline.

The tiny automated bridge test uses only 30 requested iterations; due to the
original noninteger numRuns loop it actually executes 60 decisions. Its purpose
is serialization and JS/Python parity, not learning quality. Main runs use the
exact integral 30000-iteration schedule above.

The source-provenance limitations in quillstacks_audit.md still apply. A close
score would support empirical consistency, not prove historical engine identity.
