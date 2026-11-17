# DeepTraffic checkpoint: static audit and experimental four-weight patch

The supplied checkpoint has been inspected, and `checkpoint.candidate.json` contains a localized experimental edit. **No driving-policy failure has been causally established or corrected by a simulator evaluation.** The prompt contains a placeholder for the observed behavior; this candidate explores its example of lane-change hesitation. Its action and feature semantics require confirmation in the current simulator. The original checkpoint is included separately for comparison and rollback.

## What the checkpoint actually contains

The layers are `input -> fully connected -> regression`. The input dimensions are `13 × 1 × 52`, followed by five filters of length 676 and five output biases. There are **3,380 weights and five biases, totaling 3,385 learned scalars**. Every parameter is finite. There are no hidden layers, nonlinear feature detectors, convolutional filters, or recurrent pathways in this serialized network.

At inference, the regression layer passes through the scores:

`Q_a(x) = sum_i W[a,i] * x[i] + b[a]`.

For a greedy policy, the relevant quantity is the difference between actions:

`Q_a(x) - Q_c(x) = sum_i (W[a,i] - W[c,i]) * x[i] + b[a] - b[c]`.

A large individual weight is not sufficient to identify a decision mechanism. A feature can raise every action's Q-value while barely changing their ranking. Actual feature values, pairwise contributions, preprocessing, exploration settings, and simulator overrides matter.

## Recovered provenance and remaining uncertainty

All parsed JSON fields exactly match the checkpoint embedded in [jnorthrup/deeptraffic, mph7585.js](https://github.com/jnorthrup/deeptraffic/blob/master/mph7585.js). That file's configuration is `lanesSide=6`, `patchesAhead=52`, `patchesBehind=0`, `temporal_window=0`, `otherAgents=10`, and `epsilon_test_time=0`. This identifies the accompanying source configuration, not the user's current runtime. The filename is not a reproduced speed measurement. In particular, this source configuration has no temporal history or rear patches.

The current simulator's action dispatcher, state-flattening implementation, and normalization were not provided or successfully recovered. Input shape alone does not specify flattening order. The pattern at indices separated by 13 is consistent with a longitudinal grid whose lanes vary fastest. That remains an inference. The source's tensor dimensions must not be used to silently transpose the input vector.

The candidate assumes output 3 is left, output 4 is right, and inputs 668/669/670 are corresponding left/center/right cells near the ego vehicle, with higher values indicating greater attainable speed. These labels are **not confirmed for the current runtime**. No labels are needed to verify the numerical effects below.

## Measured action differences

All indices in this report are zero-based. The output biases are:

| Output | Bias |
| --- | ---: |
| 0 | 0.7393138490051929 |
| 1 | 1.2760774771670804 |
| 2 | 0.7511842912561690 |
| 3 | 1.1373540805322566 |
| 4 | 1.1811862267116993 |

Output 1 has the largest zero-input intercept. Its bias exceeds output 3 by 0.1387233966 and output 4 by 0.0948912505. A zero input need not be a valid driving state, so this does not establish a behavioral preference or show that a bias is defective.

The most relevant local coefficients are:

| Output | Input 668 | Input 669 | Input 670 |
| --- | ---: | ---: | ---: |
| 1 | 0.0623818724 | 1.7075182485 | 0.0875566076 |
| 3 | 1.9899741595 | 1.3285861001 | 0.0602234693 |
| 4 | 0.0706067079 | 1.4215252515 | 2.1537177276 |

Increasing input 668 by one unit increases `Q3-Q1` by 1.9275922871 and `Q3-Q4` by 1.9193674516. Increasing input 670 increases `Q4-Q1` by 2.0661611200 and `Q4-Q3` by 2.0934942584. Output 1's strongest weight is at input 630: 1.9982632380. Its weights on indices 552, 565, 591, 604, 617, 643, 656, and 669 are also large.

These numbers reveal existing asymmetric responses compatible with steering and center-lane speed sensitivity. They do not demonstrate that the network lacks a steering detector. To identify the cause of a particular hesitation, compute the contributions to `Q_left-Q_selected` and `Q_right-Q_selected` on that exact network input, then check whether the selected action was actually executed. DeepTraffic's documented safety system can override speed and lane-change requests; visible hesitation can therefore originate outside the network. See the [official simulator description](https://github.com/lexfridman/deeptraffic).

## The candidate edit

Only the following four values change. No biases or architectural fields change.

| JSON path below `layers[1]` | Before | After |
| --- | ---: | ---: |
| `filters[3].w["668"]` | 1.9899741594965055 | 2.0899741594965056 |
| `filters[3].w["669"]` | 1.3285861001113697 | 1.2285861001113696 |
| `filters[4].w["669"]` | 1.4215252515419985 | 1.3215252515419984 |
| `filters[4].w["670"]` | 2.1537177276454944 | 2.2537177276454945 |

With alpha = 0.1, the exact mathematical correction, up to floating-point rounding, is:

```
delta_Q0 = delta_Q1 = delta_Q2 = 0
delta_Q3 = 0.1 * (x[668] - x[669])
delta_Q4 = 0.1 * (x[670] - x[669])
```

Under the assumed mapping, a better-valued adjacent cell raises that direction's score relative to the unmodified actions. An adjacent cell with a lower value receives a score reduction. Equal values on the three selected features produce no correction. This preserves the baseline's response to a common change in those three values and avoids adding a state-independent steering offset.

The step of 0.1 is an **uncalibrated experimental amplitude**, not an estimate fitted to a real failure margin. Four of 3,380 weights change (0.1183%). The Frobenius norm of the change is 0.2, approximately 1.2165% of the original weight-matrix norm. A small parameter change alone does not guarantee a small change in greedy actions or long-term trajectories.

## What is preserved, and what is not established

All five scores remain mathematically unchanged whenever `x[668]=x[669]=x[670]`, regardless of the other 673 features. This includes an idealized free-road input only if those cells actually receive equal values in the encoder.

If all three selected features lie in `[L,U]`, every pairwise Q-margin changes by at most `alpha*(U-L)`. The shared center term makes this bound tighter than a generic two-output error bound: `delta_Q3-delta_Q4 = alpha*(x[668]-x[670])`.

Thus a unique baseline greedy winner whose margin over every competitor exceeds this bound retains its action. For `[0,1]` inputs the bound is 0.1 Q-units; for `[-1,1]` it is 0.2; for raw `[0,80]` values it is 8. The current encoding and any off-road sentinels must be checked before applying a bound. Floating-point comparisons require tolerance near ties.

This is a guarantee about one forward pass on a fixed input, not a certificate of closed-loop stability. States near decision boundaries can change actions. An action change alters future states; a linear patch cannot limit its effect to the example failure scenario. Higher grid speed also does not necessarily mean an empty or usable gap. The edit does not add a gap-safety gate, rear visibility, temporal memory, or hysteresis. Speed, weaving, rejected lane changes, and slowdown frequency remain unmeasured.

## Validation performed

- Confirmed four changed weights, zero changed biases, and equality of every other JSON field.
- Checked the analytical score-change formula on 4,096 synthetic inputs.
- Checked equal-triplet invariance on 1,024 synthetic inputs and the pairwise bound at all eight unit-box corners.
- Confirmed action retention for the 3,797 synthetic inputs with baseline margin greater than 0.1.
- Loaded both checkpoints through the actual ConvNetJS `Net.fromJSON` and forward implementation; compared 48 synthetic inputs against an independent NumPy calculation. Maximum score error was 1.57e-13 or less.

These are numerical and serialization tests. **Zero simulator rollouts and zero real failure-state tests were performed.** No measured mph improvement, 76.6 mph result, or preserved high-speed driving behavior is claimed. `validation.json` records results and reference-code hashes.

## Reproduce and evaluate

The bundle contains the untouched `checkpoint.original.json`, the candidate, the exact change manifest, this report, validation results, and a standard-library Python utility.

To reproduce the candidate from the original bytes:

```bash
python wsp_patch.py patch checkpoint.original.json --output checkpoint.reproduced.json
```

To compare real recorded inputs:

```bash
python wsp_patch.py compare checkpoint.original.json checkpoint.candidate.json states.jsonl > comparison.jsonl
```

Each nonempty input line must be a JSON object with `x` containing the actual 676-value vector presented to the value network. Optional fields are `id`, an integer `target_action` in 0..4, and `preserve_action: true` for stable states whose decisions should be retained. Use actual observations; arbitrary synthetic vectors are unsuitable for estimating driving quality. The utility reports both Q-vectors, selected actions, changes, and target/preservation counts. It evaluates the unmasked greedy policy with the lowest-index tie rule; exploration and simulator action overrides must be inspected separately.

Before treating this as a fix, confirm the current index/action mapping and normalization, replace the placeholder with an observed scenario, and collect both failing and successful high-speed states. Fit the sparse correction against the observed action margins while constraining action retention on the successful states. If those constraints are incompatible, this four-weight support cannot solve the requested edit.

Run the original and candidate under identical simulator configuration, controlled-agent count, frozen weights, and exploration settings. Use paired seeds where available, a held-out evaluation set, and repeated runs. Report the mean speed difference and its uncertainty alongside lane changes, left-right reversals, rejected commands, and time spent trailing. Keep calibration runs separate from the final comparison. Roll back by loading the original checkpoint if performance regresses. The candidate is a network-only JSON object; loading it does not configure the surrounding Brain or simulator.
