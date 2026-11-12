# One-vehicle MIT benchmark

The shared DQN trainer now supports one through ten controlled cars. This
experiment uses `other_agents=0`, so only car 0 is controlled and the remaining
simulated traffic is environment-controlled. It targets the historical MIT
starter reference of approximately 66.8 mph.

## Results

All seeds use the same 150 episodes of 2000 frames, DQN settings, training
scenario construction and 500 held-out evaluation seeds. The final checkpoint
is evaluated greedily. The rule policy uses the same one-car environment and
evaluation protocol.

| Policy / seed | Median mph | Mean mph | Median difference to rule |
| --- | ---: | ---: | ---: |
| Rule | 66.69303 | 66.37868 | 0 |
| DQN 0 | 67.09352 | 66.52238 | +0.40049 |
| DQN 1 | 60.74820 | 60.42155 | −5.94483 |
| DQN 2 | 66.80585 | 65.89644 | +0.11282 |
| Mean of DQN medians | 64.88253 | — | — |

Two of three DQN checkpoints exceed the matched rule-policy median by a small
amount. Their paired mean differences are still uncertain: seed 0 is +0.14370
mph (95% scenario bootstrap CI [-0.61053, 0.89450]) and seed 2 is −0.48223 mph
(CI [-1.22414, 0.26174]). Seed 1 is substantially worse. The DQN therefore
reaches the MIT reference in two runs, but does not consistently beat the
one-car rule baseline or establish a robust improvement over 66.8 mph.

The one-car runs generate only about 2010 transitions per 150-episode training
run, compared with 50250 transitions in the five-car experiment. The fixed
episode count therefore gives the one-car DQN far fewer learning updates. A
follow-up should increase training episodes or define a fixed transition budget
before drawing a final conclusion.

## Deeper-network follow-up

The same three seeds were retrained with `315 -> 128 -> 128 -> 64 -> 64 -> 5`:

| Training seed | Shallow median mph | Deep median mph | Difference mph |
| --- | ---: | ---: | ---: |
| 0 | 67.09352 | 64.38731 | −2.70621 |
| 1 | 60.74820 | 62.16780 | +1.41960 |
| 2 | 66.80585 | 63.66545 | −3.14041 |
| Mean of medians | 64.88253 | 63.40685 | −1.47567 |

The deeper network is therefore worse on average in this one-car protocol and
does not provide a reliable way to exceed the MIT reference. Its median is below
the matched rule policy for all three seeds. The paired architecture comparison
is stored in `results/dqn_one_deep_v1/comparison.json`; the full raw scores and
checkpoints remain in both result directories.

## Reproduce

```sh
.venv/bin/python scripts/run_dqn_experiment.py --mode rule --controlled-agents 1 --out results/dqn_one_v1
.venv/bin/python scripts/run_dqn_experiment.py --mode train --seed 0 --controlled-agents 1 --out results/dqn_one_v1
.venv/bin/python scripts/run_dqn_experiment.py --mode eval --seed 0 --controlled-agents 1 --out results/dqn_one_v1
.venv/bin/python scripts/run_dqn_experiment.py --mode compare --controlled-agents 1 --out results/dqn_one_v1
```

Use `--mode eval` for seeds 1 and 2 after their checkpoints exist. Evaluation
is deliberately run one process at a time on the laptop to limit CPU load.
