# Five-car DQN depth comparison

Requested architecture experiment: compare the existing 315 -> 128 -> 128 -> 5
network (57605 trainable parameters) with 315 -> 128 -> 128 -> 64 -> 64 -> 5
(69701 parameters). Every hidden layer uses ReLU; the output remains linear.
Both architectures share one policy across five controlled cars.

## Executed results

All three training runs and all 1500 new evaluation episodes completed.
Each run used 300000 frames, 50250 transitions and 12063 gradient updates,
identical to the reference. The automated metadata comparison passed.

| Training seed | Original median mph | Deeper median mph | Difference mph |
| --- | ---: | ---: | ---: |
| 0 | 65.43885 | 64.62394 | -0.81491 |
| 1 | 59.47715 | 54.97045 | -4.50670 |
| 2 | 56.65382 | 64.48225 | +7.82843 |
| Mean of three medians | 60.52327 | 61.35888 | +0.83560 |

The mean of benchmark medians rises by 1.38%, but two out of three seeds
regress. The sample standard deviation across medians increases from 4.48497
to 5.53300 mph. This is not evidence of a consistent improvement with depth.
The unchanged rule policy's median is 72.97723 mph; all new checkpoints
remain substantially below it.

| Seed | Paired mean change mph | Conditional 95% CI mph |
| --- | ---: | --- |
| 0 | -1.03073 | [-1.46795, -0.59011] |
| 1 | -3.94730 | [-4.51407, -3.37330] |
| 2 | +7.29007 | [+6.86000, +7.72798] |

These intervals compare the new and old policy on matched scenario indices.
They do not establish an architecture-level effect across training seeds.
No further tuning, extra training or best-seed selection was performed.
Both architectures and all negative results are retained.

Full metrics and raw scores: `results/dqn_five_deep_v1/comparison.json` and
`seed_N/evaluation.json`. Full project suite: 27 tests passed, including the
upstream parity coverage. The new comparison test also verifies that changed
learning rates are rejected as an architecture-only comparison. `git diff
--check` passed.

## Protocol and reproduction

Only `hidden` changes. The original final checkpoints and results remain under
`results/dqn_five_v1/`; the new experiment writes `results/dqn_five_deep_v1/`.
Seeds 0, 1, 2, their respective training scenarios, 150 episodes of 2000 frames,
reward, replay, optimizer, exploration and update schedules are unchanged.
Final checkpoints are evaluated greedily on the same 500 scenario seeds
2000000..2000499. The existing rule report is reused because its policy and
entire evaluation protocol are unchanged.

```sh
.venv/bin/python scripts/run_dqn_experiment.py --mode train --seed 0 --hidden 128 128 64 64 --out results/dqn_five_deep_v1
.venv/bin/python scripts/run_dqn_experiment.py --mode train --seed 1 --hidden 128 128 64 64 --out results/dqn_five_deep_v1
.venv/bin/python scripts/run_dqn_experiment.py --mode train --seed 2 --hidden 128 128 64 64 --out results/dqn_five_deep_v1
cp results/dqn_five_v1/rule.json results/dqn_five_deep_v1/rule.json
.venv/bin/python scripts/run_dqn_experiment.py --mode compare --out results/dqn_five_deep_v1 --baseline results/dqn_five_v1
```

The train processes may run concurrently. Run comparison after they finish.
The CLI defaults still reproduce the original architecture; `--hidden` makes
the architecture explicit in each new run's manifest and checkpoint.

Comparison checks matching training seeds/scenarios, environment, package
versions, algorithm, reward objective, checkpoint selection, all nonarchitecture
DQN hyperparameters, evaluation scenarios and evaluation protocol. It reports
all three seeds, the paired mean scenario differences with conditional 95%
bootstrap intervals, and the average of the three median differences.
Bootstrap uses 10000 scenario resamples and RNG seed 912, as in the first run.

The benchmark was already inspected before this architecture was suggested.
This is a requested follow-up comparison on that benchmark, not an untouched
final test. The intervals quantify scenario uncertainty conditional on each
checkpoint, not uncertainty over training seeds. Three seeds are a small sample.
No checkpoint selection or further tuning based on these results is performed.

Tests exercise deterministic training and checkpoint/evaluation round trips
for both a tiny network and the full new architecture. The complete algorithm
and environment protocol are documented in [dqn_five.md](dqn_five.md).
