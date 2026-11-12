"""Evaluate the supplied pretrained ConvNetJS submission with configurable cars.

No training: read the attached checkpoint and verify native JS against Python.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.backend import backend_root, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/submission_one_v1'))
    parser.add_argument('--runs', type=int, default=500)
    parser.add_argument('--other-agents', type=int, choices=range(11), default=0)
    parser.add_argument('--verify-runs', type=int,
                        help='Python/JS parity subset; default verifies all runs')
    args = parser.parse_args()
    verify_runs = args.runs if args.verify_runs is None else args.verify_runs
    if not 1 <= verify_runs <= args.runs:
        parser.error('Require 1 <= verify-runs <= runs')
    source = args.source.read_text()
    marker = 'brain.value_net.fromJSON('
    net, _ = json.JSONDecoder().raw_decode(source.split(marker, 1)[1].lstrip())
    layers = net['layers']
    assert [l['layer_type'] for l in layers] == ['input', 'fc', 'regression']
    assert layers[1]['num_inputs'] == 676 and layers[1]['out_depth'] == 5
    def values(w):
        return [w[str(i)] for i in range(len(w))] if isinstance(w, dict) else w
    weights = np.asarray([values(f['w']) for f in layers[1]['filters']], dtype=np.float64)
    biases = np.asarray(values(layers[1]['biases']['w']), dtype=np.float64)
    assert weights.shape == (5,676) and biases.shape == (5,)
    assert np.isfinite(weights).all() and np.isfinite(biases).all()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'source.js').write_text(source)
    # The submission resizes a browser canvas. Supply the missing headless DOM
    # method without changing its policy, weights or simulation parameters.
    shim = ('var originalGetElement=document.getElementById;'
            'document.getElementById=function(id){var el=originalGetElement(id);'
            'el.setAttribute=function(){};return el;};\n')
    (args.out/'headless.js').write_text(shim + source)
    (args.out/'checkpoint.json').write_text(json.dumps(net))
    command = ['node', str(backend_root()/'reference/node_ref/run_ref.js'), 'eval',
        '--code='+str((args.out/'headless.js').resolve()), '--other-agents='+str(args.other_agents),
        '--lanes-side=6', '--patches-ahead=52', '--patches-behind=0',
        '--runs='+str(args.runs), '--frames=2000', '--det=true']
    print(f'Evaluating supplied weights in JavaScript: otherAgents={args.other_agents}, runs={args.runs}', flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    js = json.loads(result.stdout)
    (args.out/'javascript.json').write_text(json.dumps(js, indent=2))
    print(f"JavaScript median={js['median']} mean={js['mean']}", flush=True)
    reference = js
    if verify_runs != args.runs:
        subset_command = [('--runs='+str(verify_runs)) if arg.startswith('--runs=') else arg
                          for arg in command]
        subset = subprocess.run(subset_command, capture_output=True, text=True, check=True)
        reference = json.loads(subset.stdout)
    print('Verifying in Python engine under the same historical evaluation protocol', flush=True)
    cfg = DeepTrafficConfig(lanes_side=6, patches_ahead=52, patches_behind=0,
                           other_agents=args.other_agents, episode_frames=2000)
    env = DeepTrafficEnv(cfg)
    try:
        py = env.evaluate(lambda obs: int(np.argmax(weights @ obs.astype(np.float64) + biases)),
                          runs=verify_runs)
    finally:
        env.close()
    np.testing.assert_allclose(py['median'], reference['median'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(py['mean'], reference['mean'], atol=1e-9, rtol=0)
    report = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        backend_revision=REVISION, trained_in_this_experiment=False,
        original_other_agents=10, other_agents_tested=args.other_agents,
        controlled_cars_tested=args.other_agents+1,
        architecture=[676,5], observation=[6,52,0], runs=args.runs, frames=2000,
        protocol='historical deterministic sequential RNG, per-episode floor, upper-middle median',
        supplied_score_mph=75.85, median_delta_to_supplied=js['median']-75.85,
        python_javascript_parity=True, verification_runs=verify_runs,
        verification_javascript=reference, evaluation=py, javascript=js)
    (args.out/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('evaluation','javascript')}, indent=2))
    print(f"median={js['median']} mean={js['mean']}", flush=True)


if __name__ == '__main__':
    main()
