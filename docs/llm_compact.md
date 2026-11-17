# Compact LLM experiment

## Current experiment: GAP GAIN state v5 / prompt v14

The compact evaluation defaults are now 3 lanes per side, 52 patches ahead and
10 rows including the ego row behind (9 strictly behind). The short prompt uses
the first matching rule: accelerate when not centered; when STATUS is BLOCKED,
change into a native legal lane only if its forward gap is at least 4 patches
larger than CURRENT; otherwise accelerate. If both qualify, prefer the larger
gap, then RIGHT for equal gaps. Four examples cover useful RIGHT and LEFT
changes, insufficient gap gain and an ongoing maneuver. FRONT GAP and signed
GAP GAIN (target minus CURRENT) are explicitly supplied for each observed side.
Gaps are capped at the visible ahead horizon when no obstruction is observed;
an unobserved side has UNKNOWN gain. The four-patch threshold is unchanged.
`observed_gap_gains` is logged with each prepared decision. OVERTAKE and its
candidate list are retained only in the structured log, and are absent from
both messages sent to the model. The recorded v8.1 system prompt is
preserved verbatim in `llm_prompt_record_v8_1.md` with its verified SHA256.

Run from this checkout, using a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u scripts/evaluate_llm.py --compact --llm-lanes-side 3 --llm-patches-ahead 52 --llm-patches-behind 10 --frames 200 --seeds 2000100 2000101 2000102 --out results/llm_comparison_compact_v11
```

## Ego telemetry and overtake heuristic (state v4, prompt v10)

Compact runs now also supply `EGO SPEED` and `CURRENT LANE` for the current
callback agent, using native instantaneous speed `max(0, car.c * car.a) * 20`
and lane `car.b + 1` (1 = leftmost). During a maneuver this is the target lane;
`CURRENT: NOT CENTERED` prevents starting a second maneuver. These values travel
with the fresh gate telemetry and enter the logged state hash.

`OVERTAKE` is computed from this speed, the observed nearest cells and native
legality. A candidate must be legal, offer more forward clearance, and have
front traffic is not SLOW CLOSE or no observed front traffic. A blocker is at
least 5 mph slower than ego or below 70 mph; the latter also detects a car that
has already forced ego to match its speed. Target speed ranks multiple legal
candidates but does not disqualify a lane when its traffic is FAR.
`REQUIRED` means the blocker is within 14 patches. `OPTIONAL` means it is farther
away and the candidate offers at least 14 additional patches of observed
clearance. Otherwise it is `NOT USEFUL`; missing ego telemetry is `UNKNOWN`.
Ambiguous zero regions never supply invented speeds or positive recommendations.
No detected traffic provides only a clearance lower bound at the observation
horizon. The current heuristic does not recommend moving when the current lane
has no observed blocker.

The prompt prioritizes REQUIRED opportunities and considers OPTIONAL ones,
keeping the existing RIGHT tie-break for similarly useful candidates. It never
permits a blocked gate or DECELERATE. Recommendations, candidate lanes, relative
speed and thresholds are logged; the model still chooses the action without
masking or substitution. This is additional feature engineering and privileged
telemetry compared with the original DQN input; improvement requires measurement.
`STATUS: BLOCKED` remains an observation-derived summary, not a native gate.
Seeds, physics, decision schedule and observation window are unchanged. Use a
fresh output directory for the changed prompt/source contract.

## Native lane-change status (compact state/prompt v2)

`evaluate_llm.py --compact` now supplies the exact current engine gate as
additional input, for the callback's current agent and frame:

```text
CURRENT: CENTERED.
LEFT: CHANGE BLOCKED (target_region_blocked).
RIGHT: CHANGE LEGAL.
```

`not_centered`, `off_road`, and `target_region_blocked` use the same geometric
scan as the native engine. The check is read-only and never attempts a maneuver.
`CHANGE LEGAL` states that a maneuver can be started now, not that it is useful
or that it has finished. The prompt requires a legal status for lane-change
commands and a longitudinal action when the car is not centered.

These fields are privileged engine telemetry beyond the original observation
vector. DQN and rule baselines still receive their original observations. Report
this input difference when comparing scores. Native gate status and raw grid
both enter the compact state hash; the full status is logged in each decision.
Absent telemetry is rendered as UNKNOWN by standalone preparation; the compact
evaluation adapter fails before contacting the model if fresh telemetry is absent
or belongs to a different agent/frame. No action masking or substitution occurs.

Example next run, retaining the last tested 52/10 window and the original seeds:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u scripts/evaluate_llm.py --compact --llm-patches-ahead 52 --llm-patches-behind 10 --out results/llm_comparison_compact_lane_status_v1
```

Use a fresh output directory; source and prompt changes invalidate old resume
contracts. The following original measurement used compact state/prompt v1.

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
