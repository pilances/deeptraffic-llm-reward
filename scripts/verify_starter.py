"""Cross-check a trained checkpoint against JS and evaluate matched controls."""
import argparse
import json
import subprocess
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.backend import backend_root
from deeptraffic_llm_reward.policies import RuleBasedPolicy
from evaluate_starter import ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=500)
    args = parser.parse_args()
    output = ROOT/'results/starter'
    checkpoint = json.loads((output/f'seed_{args.seed}_checkpoint.json').read_text())
    submission = output/f'seed_{args.seed}_submission.js'
    submission.write_text((ROOT/'network_basic.js').read_text() +
        '\nbrain.value_net.fromJSON(' + json.dumps(checkpoint['net']) + ');\n')
    result = subprocess.run(['node', str(backend_root()/'reference/node_ref/run_ref.js'),
        'eval', '--code='+str(submission), '--other-agents=0',
        '--runs='+str(args.runs), '--frames=2000', '--det=true'],
        capture_output=True, text=True, check=True)
    js = json.loads(result.stdout)
    saved = json.loads((output/f'seed_{args.seed}_results.json').read_text())
    assert len(saved['evaluation']['runs']) == args.runs
    assert abs(js['median']-saved['evaluation']['median']) < 1e-9
    assert abs(js['mean']-saved['evaluation']['mean']) < 1e-9
    print(f"JS checkpoint verification: median={js['median']}, mean={js['mean']}", flush=True)
    env = DeepTrafficEnv(DeepTrafficConfig(lanes_side=1, patches_ahead=10,
        patches_behind=0, other_agents=0))
    controls = {}
    for name, policy in [('maintain', lambda obs: 0),
                         ('rule', RuleBasedPolicy(1, 10, 0).act)]:
        controls[name] = env.evaluate(policy, runs=args.runs)
        print(f"{name}: {controls[name]['median']}", flush=True)
    (output/'verification_and_controls.json').write_text(json.dumps(
        {'checkpoint_seed':args.seed, 'javascript':js, 'controls':controls}, indent=2))

if __name__ == '__main__':
    main()
