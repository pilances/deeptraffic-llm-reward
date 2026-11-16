# Compact LLM experiment

Run from the project checkout with Ollama and `qwen3:8b` available:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/evaluate_llm.py --compact --out results/llm_comparison_compact_v1
```

This keeps seeds 2000100, 2000101, 2000102 and 200 frames per episode. The LLM
uses `lanes_side=3, patches_ahead=40, patches_behind=5` (315 raw values).
The first offset is -40, the last +4: offset zero is at the vehicle, so only
four rows are strictly behind it. These are starting settings from the supplied
professor example, not empirically optimal hyperparameters.

`llm_compact.prepare_compact_request` converts only the observation to text:
nearest occupied cells ahead in the current and adjacent lanes, adjacent cells
at the vehicle position and nearest rear occupancy. It describes distance in
patches and speed in mph; zero remains ambiguous between off-road and stopped
traffic. It does not invent an absolute lane or ego speed. Contiguous occupancy
need not identify a unique vehicle; distances refer to occupied cells. It reports
no occupancy in the observed region rather than claiming an entire lane is empty.

Categories use the example's fixed thresholds: front FAST >=70 mph, CLOSE <=14
patches; rear FAST >=62 mph, CLOSE <=5 patches. BLOCKED summarizes nearby slow
occupancy or an ambiguous zero region; no action is selected by the decoder.
These thresholds are feature engineering. The system prompt also supplies a
driving heuristic. All derived features and thresholds are logged with each
decision. The raw grid is not included in the messages sent to the LLM.

The frozen DQN needs 676 inputs and therefore keeps its original 6/52/0 window,
as does the original rule baseline. Physics, vehicle counts, scoring and seeds
are shared. Initial traffic hashes verify matching car positions and speeds
despite different observation shapes. A fixed-action regression checks identical
per-vehicle scores across all three seeds with both windows.

Results contain each policy's observation configuration and paired speed deltas.
For improvement over raw-state LLM runs, compare the corresponding seed scores
and action counts in `llm_comparison_short_v1` and `llm_comparison_prompt_v3`.
Representation and observation window both change, so this experiment measures
their combined effect. Three short scenarios are exploratory and must not be
used to claim that the chosen hyperparameters are optimal.

Resume a compact experiment using the same command plus `--resume`. The existing
background resume launcher also restores compact settings from the manifest.

## First completed measurement

The live `qwen3:8b` run in `results/llm_comparison_compact_v1` completed all
201 decisions with no fallback. Comparison against the original complete raw
LLM run on the same seeds (200 frames each):

| Seed | Raw LLM v1 mph | Compact LLM mph | Difference mph |
| --- | ---: | ---: | ---: |
| 2000100 | 70.58 | 71.75 | +1.17 |
| 2000101 | 64.82 | 58.12 | -6.70 |
| 2000102 | 68.42 | 61.49 | -6.93 |
| Mean | 67.94 | 63.79 | -4.15 |

The only completed v3 episode was seed 2000100 at 70.69 mph; the compact result
on that seed is +1.06 mph. No complete three-seed v3 comparison is available.
The current matched DQN mean is 70.87 mph and rule mean 74.19 mph.

The compact LLM chose accelerate 52 times, decelerate 86 times, left 22 times
and right 41 times. Thus lane changes now occur, but the overall speed result
is worse, alongside frequent deceleration. This measurement does not establish
that braking caused the entire difference or that the observation window is
optimal. Further prompt/threshold choices should be evaluated on additional
seeds rather than presenting these reused scenarios as an unseen test set.
