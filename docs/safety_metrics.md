# Safety overview

New `evaluate_llm.py` runs automatically record `safety_events` in each JSONL
step, `safety_summary` in episode JSON and `comparison.json`, and an aggregate
`safety_overview.md` table. The console prints an overview after each episode.
The observations supplied to policies contain no safety telemetry.

Counts distinguish callback-owner commands (`decision_commands`) from every
native action application (`all_action_applications`). The historical engine
propagates some commands to other controlled cars, so the latter counts can be
larger. Command counts match `action_counts`; propagated applications must not
be interpreted as additional LLM decisions.

Lane-change acceptance is measured from the actual target lane change in the
native `Car.act` call. Acceptance means the engine started the maneuver, not that
the lateral transition has already finished. Rejections are classified as
`not_centered` (including an ongoing lane change), `off_road`, or
`target_region_blocked`. The last category describes the engine's scanned region,
not an independently validated predicted collision. Reasons follow native check
order; a rejected action can have more than one underlying obstacle.

ACCELERATE is never directly rejected by the safety layer. The engine increases
the gas setting unless it is already at its limit. Therefore the overview reports:

- `rejected_by_safety`: always zero for this engine.
- `throttle_increased`: actual increase in the gas setting.
- `at_throttle_cap`: gas was already at least 2; no further increase occurred.
- `safety_limited_at_request`: native speed factor `c < 2` when the command was
  applied. Safety was already limiting effective speed despite the gas command.
- `fully_stopped_at_request`: `c == 0` when applied.

Cap and safety-limit counts may overlap. Safety limitation is a condition at the
instant of application; it is not proof that the command caused braking, that
future acceleration was impossible, or that a particular collision was prevented.

The pinned external engine remains intact. An observation wrapper calls the
original action method exactly once and records its result. Regression tests
compare complete observation sequences and scores with and without observation.

Existing completed runs can be measured by deterministic action replay without
calling the LLM or changing the original logs:

```sh
.venv/bin/python scripts/audit_safety.py results/llm_comparison_compact_v2
```

This checks every input/output state hash, agent, frame and reward, aborting on
any mismatch or incomplete episode. Reports are saved in a new `safety_audit/`
subdirectory. Use `--out` for a different destination; existing reports are never
overwritten. Future runs with the new instrumented source need a fresh experiment
directory because source hashes are part of the resume contract.
