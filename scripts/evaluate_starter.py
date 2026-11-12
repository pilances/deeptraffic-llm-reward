"""Train the unchanged MIT starter on our Python engine, evaluate frozen weights.

Run from the repository: .venv/bin/python scripts/evaluate_starter.py --seed 0
Artifacts are generated JSON checkpoints/results; delete and regenerate safely.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.backend import backend_root, REVISION

ROOT = Path(__file__).resolve().parents[1]

class StarterPolicy:
    """Float64 forward pass for exported ConvNetJS 30-10-ReLU-5 weights."""
    def __init__(self, net):
        self.layers = net['layers']
    @staticmethod
    def values(w):
        return list(w.values()) if isinstance(w, dict) else w
    def q(self, state):
        x = np.asarray(state, dtype=np.float64)
        for layer in self.layers:
            if layer['layer_type'] == 'fc':
                weights = np.asarray([self.values(f['w']) for f in layer['filters']])
                x = weights @ x + np.asarray(self.values(layer['biases']['w']))
            elif layer['layer_type'] == 'relu':
                x = np.maximum(x, 0)
            elif layer['layer_type'] not in ('input', 'regression'):
                raise ValueError('Unexpected layer')
        return x
    def __call__(self, obs):
        return int(np.argmax(self.q(obs)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=500)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--verify-js', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'results/starter')
    args = parser.parse_args()
    if args.seed < 0 or args.runs < 1 or args.iterations < 1:
        parser.error('Require nonnegative seed and positive runs/iterations')
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = DeepTrafficConfig(lanes_side=1, patches_ahead=10, patches_behind=0,
        other_agents=0, episode_frames=2000)
    env = DeepTrafficEnv(cfg)
    eng = env._new_engine()
    proc = subprocess.Popen(['node', str(ROOT/'scripts/starter_brain.js'),
        str(backend_root()), str(ROOT/'network_basic.js'), str(args.seed)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def request(payload):
        proc.stdin.write(json.dumps(payload)+'\n')
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError('ConvNetJS worker stopped')
        return json.loads(line)
    start = time.monotonic()
    try:
        # Original train_webworker: totalFrames=30*iterations;
        # numRuns=totalFrames/100000+1, with JS integer loop boundaries.
        total_frames = 30*args.iterations
        n_runs = total_frames/100000+1
        frames_per_run = int(np.ceil(total_frames/n_runs))
        # Training randomness is explicitly seeded; original webworker was not.
        rng = np.random.default_rng(args.seed)
        request({'op':'seed', 'seed':int(rng.integers(10000000))})
        eng.r, eng.t = map(int, rng.integers(10000000, size=2))
        decisions = 0
        def learn(state, reward):
            nonlocal decisions
            decisions += 1
            return request({'op':'learn', 'state':state, 'reward':reward})
        for i in range(int(np.ceil(n_runs))):
            eng.reset()
            for _ in range(frames_per_run):
                eng.V(learn)
            print(f'seed={args.seed} training block {i+1}/{int(np.ceil(n_runs))}, decisions={decisions}', flush=True)
        checkpoint = request({'op':'export'})
        policy = StarterPolicy(checkpoint['net'])
        # Independently verify the Python forward pass against original JS.
        probes = [eng.ego_state()] + rng.uniform(0, 1, (100, 30)).tolist()
        max_error = 0.0
        for state in probes:
            js = request({'op':'q', 'state':state})
            py = policy.q(state)
            np.testing.assert_allclose(py, js, rtol=1e-12, atol=1e-12)
            assert int(np.argmax(js)) == int(np.argmax(py))
            max_error = max(max_error, float(np.max(np.abs(py-js))))
        checkpoint_path = args.out/f'seed_{args.seed}_checkpoint.json'
        checkpoint_path.write_text(json.dumps(checkpoint))
    finally:
        proc.stdin.close()
        proc.wait()
    print(f'seed={args.seed} evaluating {args.runs} x 2000 frames', flush=True)
    score = env.evaluate(policy, runs=args.runs)
    js_score = None
    if args.verify_js:
        submission = args.out/f'seed_{args.seed}_submission.js'
        submission.write_text((ROOT/'network_basic.js').read_text() +
            '\nbrain.value_net.fromJSON(' + json.dumps(checkpoint['net']) + ');\n')
        command = ['node', str(backend_root()/'reference/node_ref/run_ref.js'), 'eval',
            '--code='+str(submission.resolve()), '--other-agents=0',
            '--runs='+str(args.runs), '--frames=2000', '--det=true']
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        js_score = json.loads(result.stdout)
        assert abs(js_score['median']-score['median']) < 1e-9, (js_score, score)
        assert abs(js_score['mean']-score['mean']) < 1e-9, (js_score, score)
    report = {'training_seed':args.seed, 'iterations_requested':args.iterations,
        'decisions':decisions, 'training_frames':int(np.ceil(n_runs))*frames_per_run,
        'training_blocks':int(np.ceil(n_runs)), 'frames_per_training_block':frames_per_run,
        'reward':'historical engine callback (N-60)/20, ego only',
        'backend_revision':REVISION, 'network_sha256':hashlib.sha256((ROOT/'network_basic.js').read_bytes()).hexdigest(),
        'checkpoint':str(checkpoint_path), 'age':checkpoint['age'], 'epsilon':checkpoint['epsilon'],
        'max_forward_error':max_error, 'evaluation':score, 'javascript_verification':js_score,
        'reference_mph':66.8, 'delta_mph':score['median']-66.8,
        'elapsed_seconds':time.monotonic()-start}
    (args.out/f'seed_{args.seed}_results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'evaluation'}, indent=2), flush=True)
    print(f"median={score['median']}, mean={score['mean']}", flush=True)

if __name__ == '__main__':
    main()
