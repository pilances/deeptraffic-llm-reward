# Shared-policy DQN, five controlled cars

This experiment trains one neural policy shared by cars 0..4. It uses the
existing native JS dynamics, same-car collector, shared replay and fleet reward.
It is a new fleet-objective baseline, not the historical ConvNetJS starter.

The follow-up architecture experiment with four hidden layers is documented in
[dqn_depth_comparison.md](dqn_depth_comparison.md). The original results below
remain the reference for that comparison.

## Executed first comparison

All three prespecified training runs completed. Each used 300000 frames,
50250 shared-policy actions / same-car transitions and 12063 gradient updates.
All 1500 DQN test episodes and 500 rule test episodes completed.

| Policy | Median fleet mph | Mean fleet mph | IQR mph | Median difference to rule |
| --- | ---: | ---: | ---: | ---: |
| Rule (fixed) | 72.97723 | 72.22473 | 4.34355 | 0 |
| DQN seed 0 | 65.43885 | 65.09233 | 6.25314 | -7.53838 |
| DQN seed 1 | 59.47715 | 58.99831 | 8.08801 | -13.50008 |
| DQN seed 2 | 56.65382 | 56.74210 | 6.13204 | -16.32341 |

Mean of DQN benchmark medians: 60.52327 mph; sample standard deviation across
the three training seeds: 4.48497 mph. Thus this first fixed DQN configuration
does not beat the heuristic and is sensitive to training initialization/scenarios.
This does not show that DQN in general cannot beat it.

Paired mean DQN-minus-rule differences and conditional 95% bootstrap CIs:

| Seed | Mean difference mph | 95% CI mph | Fraction of scenarios won |
| --- | ---: | --- | ---: |
| 0 | -7.13240 | [-7.60258, -6.66425] | 7.8% |
| 1 | -13.22642 | [-13.80870, -12.64324] | 2.4% |
| 2 | -15.48263 | [-15.96157, -15.00005] | 0.8% |

All three conditional intervals are below zero. This comparison measures the
performance of these trained checkpoints; it is not a causal diagnosis of which
hyperparameter or part of the reward explains the shortfall. No retuning,
additional training or selection of a better intermediate checkpoint was done
after examining the held-out scores. Future tuning should define a separate
validation scenario set and retain these results as the initial baseline.

25 project tests passed, including DQN math, target updates, tiny deterministic
training and save/load tests, plus the existing collector/JS/starter tests.
The embedded upstream parity suite still executes its 18 tests.

## Reproduce

```sh
.venv/bin/python -m pip install -e '.[dev,rl]'
.venv/bin/python scripts/run_dqn_experiment.py --mode rule
.venv/bin/python scripts/run_dqn_experiment.py --mode train --seed 0
.venv/bin/python scripts/run_dqn_experiment.py --mode train --seed 1
.venv/bin/python scripts/run_dqn_experiment.py --mode train --seed 2
.venv/bin/python scripts/run_dqn_experiment.py --mode compare
.venv/bin/python -m pytest -rA
```

The four rule/train processes can run concurrently; compare only after all have
finished. Defaults write results/dqn_five_v1/. Use a new --out directory for a
separate experiment. Existing seed directories and rule reports are not silently
overwritten. Seeds 0,1,2 are prespecified, all results are retained.

## Fixed protocol

- Engine: pinned Quillstacks revision from backend.py, native action leakage retained.
- Five controlled cars (other_agents=4), shared weights, 315 inputs (3,40,5),
  five discrete actions. Each actual callback owner has its own replay state.
- 150 training episodes x 2000 frames = 300000 simulation frames per seed.
  Each episode yields 335 actions/transitions; total 50250 per training seed.
- Training layouts: 1000000 + 10000*training_seed + episode_index.
- Test layouts: 2000000..2000499, identical reset(seed) anchors for every policy.
  None overlap training. Each policy changes subsequent traffic, so matched seeds
  do not imply identical later traffic. The final checkpoint is evaluated once;
  no test-based checkpoint choice or hyperparameter search is performed.
- Greedy evaluation (epsilon=0): 500 independent episodes x 2000 frames.
  Score per episode is unrounded frame-weighted fleet mean mph. Report NumPy
  median, mean, IQR and all raw scores. This is NOT the historic sequential-RNG
  evaluation block or its floored upper-middle median. Historical 66.8 mph and
  previous 3-episode diagnostics are not directly comparable to these test scores.
- Rule baseline is unchanged RuleBasedPolicy(3,40,5), shared across all five cars,
  evaluated with exactly the same engine, observations, seeds, horizon and metric.

## Learner

Standard DQN: MLP 315 -> 128 ReLU -> 128 ReLU -> 5, Adam learning rate .0003,
uniform replay capacity 30000, batch 64, Huber loss, global gradient norm clipped
to 10. One update every four completed transitions after 2000 transitions;
hard target-network sync every 500 gradient updates. This is target-network
max DQN, not Double DQN. Target weights receive no gradients.

Epsilon decays linearly from 1.0 to .05 over 30000 shared policy decisions and
remains .05 for the rest of training. All cars use the same evolving network;
per-car observation and last-action histories remain distinct in the collector.
Updates occur from the collector's on_transition callback after insertion in
replay and before requesting the next action. Five final partial intervals are
inserted at each episode horizon. No transitions cross episode boundaries.

The collector uses g=.95**(1/30) per simulation frame. It returns interval reward
sum(g**k * normalized_fleet_reward[k]) and bootstrap_discount=g**elapsed_frames.
The learner multiplies rewards by 1/30 as a fixed positive numerical scaling;
the environment's reward definition is unchanged. Thus Q is in scaled reward
units, while evaluation remains mph. No reward clipping, imitation labels,
LLM feedback, policy-specific safety changes or action masks are used.
Target = scaled_interval_reward + bootstrap_discount * max Q_target(next_state).
Finite-horizon truncated and terminated transitions bootstrap with zero.

Training runs on CPU with one PyTorch thread per process and deterministic
algorithms. Seeds control initialization, epsilon exploration, replay sampling
and scenario anchors. Torch 2.6.0 is an optional pinned rl dependency. Without
it the simulator remains usable and DQN-specific tests skip.

## Artifacts and interpretation

- dqn.py: model, exploration, Bellman update, checkpoint loading, train/eval loops.
- run_dqn_experiment.py: the prespecified experiment and compatible comparisons.
- seed_N/manifest.json: complete configs, package versions, training scenarios.
- seed_N/training.jsonl: learning curve, loss, epsilon and update counts.
- seed_N/final.pt: inference checkpoint with configs and backend revision.
  It omits replay/optimizer/RNG state and is not an exact-resume checkpoint.
- seed_N/training_summary.json: counts, training time.
- seed_N/evaluation.json and rule.json: all 500 scenario scores and configs.
- comparison.json: per-seed differences, scenario win rates and bootstrap CIs.

Artifacts are generated by the scripts, can be inspected independently and can
be regenerated into a new output directory. Keep the first run as a comparison
baseline. Do not delete trained checkpoints solely because a result is poor.

Comparison validates identical environment/config/protocol/test seeds, disjoint
train/test layouts and identical DQN hyperparameters across training seeds.
Paired bootstrap resamples the 500 matched scenario indices (10000 replicates,
fixed bootstrap seed 912) and reports CI for the mean mph difference. This CI
is conditional on one trained network, not uncertainty over all possible
training runs; the three training-seed medians are reported separately.

## Verification

Tests check known Bellman targets including terminal masking, actual online
weight updates, target sync timing, deterministic exploration and training,
checkpoint restoration and leakage-free evaluation. The existing same-car
reward and JS trajectory parity tests remain applicable. Learning does not
guarantee outperforming the rule baseline; negative results must be reported.

Algorithm reference: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
The code applies DQN to the project's time-aware per-car transitions rather
than directly treating alternating Gym callbacks as single-agent transitions.
