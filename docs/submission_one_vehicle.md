# Submitted pretrained policy: one controlled vehicle

Source: the user-supplied `Eingefügter Text.txt`, copied unchanged to
`results/submission_one_verified_v1/source.js`, with SHA256 in `results.json`.
The source sets `otherAgents=10`, but this experiment explicitly overrides it
to zero additional agents before loading the engine. Only the ego car is
controlled, with 20 vehicles still in the simulation.

The submission contains a complete pretrained checkpoint restored through
`brain.value_net.fromJSON`. We evaluate those weights without additional
training or exploration. The declared 50000 training iterations, Adagrad,
gamma .98888 and tiny learning rate describe its training configuration; they
are not executed in this frozen-checkpoint experiment. The unused `prime`
function and commented hidden-layer definition do not affect inference.

Architecture: 676 inputs (lanesSide=6, patchesAhead=52, patchesBehind=0), a
single fully connected layer with five outputs, then the ConvNetJS regression
output. No hidden layer or ReLU is active. This is a linear Q policy with
3385 parameters, not a deep network.

Evaluation uses the pinned backend and its JavaScript reference engine,
500 deterministic sequential episodes of 2000 frames each, historical
per-episode floor and upper-middle median. A headless shim adds a no-op
`setAttribute` for the submission's canvas resizing code. It changes no network
weights, action logic or dynamics. The original source is retained separately.
The reference harness labels its default `policy` field as `nop` even when
`--code` is supplied; the supplied code defines the actual policy here.

JavaScript results: median 70.59 mph, mean 69.404 mph, min 42.8, max 80.
The full Python verification completed with median 70.59 and mean 69.404;
both aggregate metrics agree with JavaScript within 1e-9. All 500 episodes
completed in each engine. This validates these metrics for this checkpoint,
not every possible policy or engine configuration.
This is 5.26 mph (6.93%) below the user-reported 75.85 mph and 3.79 mph above
the approximately 66.8 mph historical starter reference. The 75.85 figure's
evaluation protocol was not supplied or independently verified. Because the
source declares 10 other controlled cars, the one-car result cannot establish
failure to reproduce the original multi-car score.

This score is not directly comparable to our recent independently reset Gym
benchmark medians: those runs used different evaluation scenario construction
and observations. No conclusion that architecture alone caused the difference
is supported. This experiment measures an existing checkpoint, not whether
retraining from scratch with its hyperparameters reproduces the score.

Reproduce into a new directory (single active simulation process):

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/evaluate_submission.py results/submission_one_verified_v1/source.js --out results/submission_one_repeat
```

`evaluate_submission.py` parses the supplied checkpoint, checks dimensions and
finite weights, evaluates the source in JavaScript, and independently computes
its linear float64 Q values inside our Python engine. It checks agreement of
the final median and mean and saves the 500 Python episode scores. Existing
output directories are refused. `results/submission_one_v1/` contains the
initial incomplete attempt, which stopped at an unsupported headless canvas
method before any valid score; use the verified directory above for results.

## Follow-up: otherAgents = 10

The same supplied weights were evaluated with the submission's declared
`otherAgents=10` (ego plus ten additional controlled cars, twenty total cars).
No retraining was performed. The 500-episode JavaScript reference run returned
median **75.80 mph**, mean **75.54058 mph**: **0.05 mph below** the reported
75.85 mph. The source SHA256 matches the one-car experiment. This is a close
numerical reproduction, not exact equality; the original score's complete
evaluation conditions are unknown, so the residual difference is unexplained.

Python/JavaScript verification used a separate ten-episode sequential run in
each engine. Both returned median 75.94 and mean 75.766 mph (within 1e-9).
The full 500-episode score above comes from JavaScript; Python was verified
on ten episodes, not on all 500. Historic engine callback scheduling and action
propagation quirks are retained, including the missing own callback for car 10.

Artifacts: `results/submission_eleven_v1/results.json` and `javascript.json`.
The previous one-car measurements remain unchanged. Reproduce into a new folder:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/evaluate_submission.py results/submission_one_verified_v1/source.js --other-agents 10 --verify-runs 10 --out results/submission_eleven_repeat
```
