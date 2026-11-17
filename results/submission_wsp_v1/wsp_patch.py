#!/usr/bin/env python3
"""Reproduce a four-weight EXPERIMENTAL patch or compare real network inputs.

The semantic feature/action mapping and driving benefit are NOT validated.
Python standard library only. No training, simulator, or network access.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

SOURCE_SHA256 = "dba1846eacc50a2c9581391ae3b18ac45191565656651ead4c842b09753d03de"
EDITS = ((3, 668, 1), (3, 669, -1), (4, 669, -1), (4, 670, 1))


def load(path):
    model = json.loads(Path(path).read_text())
    layers = model["layers"]
    if [x["layer_type"] for x in layers] != ["input", "fc", "regression"]:
        raise ValueError("Expected input -> fc -> regression with no hidden layers")
    if math.prod(layers[0][k] for k in ("out_sx", "out_sy", "out_depth")) != 676:
        raise ValueError("Expected 676 inputs")
    fc = layers[1]
    if fc["num_inputs"] != 676 or fc["out_depth"] != 5 or len(fc["filters"]) != 5:
        raise ValueError("Expected five fully connected output filters")
    expected = {str(i) for i in range(676)}
    for f in fc["filters"]:
        if set(f["w"]) != expected or f["depth"] != 676:
            raise ValueError("Unexpected filter layout")
    if set(fc["biases"]["w"]) != {str(i) for i in range(5)}:
        raise ValueError("Unexpected bias layout")
    values = [v for f in fc["filters"] for v in f["w"].values()]
    values += list(fc["biases"]["w"].values())
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Non-finite or nonnumeric parameter")
    return model


def q_values(model, x):
    if len(x) != 676 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in x):
        raise ValueError("x must contain 676 finite numeric values from the actual network input")
    fc = model["layers"][1]
    result = []
    for a, f in enumerate(fc["filters"]):
        # Same accumulation order as ConvNetJS FullyConnLayer.forward.
        total = 0.0
        for i in range(676):
            total += f["w"][str(i)] * x[i]
        result.append(total + fc["biases"]["w"][str(a)])
    return result


def best(q):
    return max(range(5), key=q.__getitem__)


def patch(args):
    src, dst = Path(args.source), Path(args.output)
    if src.resolve() == dst.resolve():
        raise ValueError("Choose a separate output path to retain the frozen baseline")
    if dst.exists():
        raise ValueError("Output already exists; choose a new output path")
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 0.1:
        raise ValueError("This candidate uses 0 < alpha <= 0.1; larger edits need calibration")
    actual = hashlib.sha256(src.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError("Input bytes differ from the supplied checkpoint; patch not applied")
    model = copy.deepcopy(load(src))
    for action, index, sign in EDITS:
        model["layers"][1]["filters"][action]["w"][str(index)] += sign * args.alpha
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(model, separators=(",", ":"), allow_nan=False) + "\n")
    load(dst)
    print(json.dumps({"status": "experimental_candidate", "changed_weights": 4,
                      "alpha": args.alpha, "output": str(dst),
                      "simulation_validated": False}))


def compare(args):
    original, candidate = load(args.original), load(args.candidate)
    count = flips = protected = protected_flips = targets = before_hits = after_hits = 0
    max_shift = 0.0
    with Path(args.states).open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            x = record["x"]
            qb, qa = q_values(original, x), q_values(candidate, x)
            ab, aa = best(qb), best(qa)
            delta = [v - u for u, v in zip(qb, qa)]
            competitors = [j for j in range(5) if j != ab]
            margin = min(qb[ab] - qb[j] for j in competitors)
            adverse_shift = max(delta[j] - delta[ab] for j in competitors)
            target = record.get("target_action")
            if target is not None and (type(target) is not int or not 0 <= target < 5):
                raise ValueError(f"Invalid target_action on line {line_no}")
            keep = bool(record.get("preserve_action", False))
            count += 1
            flips += ab != aa
            protected += keep
            protected_flips += keep and ab != aa
            targets += target is not None
            before_hits += target is not None and ab == target
            after_hits += target is not None and aa == target
            max_shift = max(max_shift, max(map(abs, delta)))
            print(json.dumps({"id": record.get("id", line_no), "q_before": qb,
                              "q_after": qa, "action_before": ab, "action_after": aa,
                              "q_delta": delta, "baseline_margin": margin,
                              "retention_certified_by_margin": margin > adverse_shift,
                              "target_action": target, "preserve_action": keep}))
    print(json.dumps({"summary": {"states": count, "action_changes": flips,
        "protected_states": protected, "protected_action_changes": protected_flips,
        "target_states": targets, "target_hits_before": before_hits,
        "target_hits_after": after_hits, "max_abs_q_change": max_shift,
        "simulation_validated": False}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("patch", help="Create an experimental copy of the supplied checkpoint")
    p.add_argument("source")
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.set_defaults(run=patch)
    p = sub.add_parser("compare", help="Evaluate actual recorded network inputs, without rollouts")
    p.add_argument("original")
    p.add_argument("candidate")
    p.add_argument("states", help="JSONL records: x[676], optional id, target_action, preserve_action")
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
