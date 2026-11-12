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
from deeptraffic_llm_reward import (
    DeepTrafficConfig,
    DeepTrafficEnv,
    HistoricalStarterRunner,
)
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
    trainer = HistoricalStarterRunner()
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
        # Training randomness is explicitly seeded; original browser training was not.
        def learn(state, reward):
            return request({'op':'learn', 'state':state, 'reward':reward})
        def reset_brain_seed(seed):
            request({'op':'seed', 'seed':seed})
        def progress(block, blocks, decisions):
            print(f'seed={args.seed} training block {block}/{blocks}, decisions={decisions}', flush=True)
        training = trainer.train(
            learn,
            reset_brain_seed,
            iterations=args.iterations,
            seed=args.seed,
            progress=progress,
        )
        checkpoint = request({'op':'export'})
        policy = StarterPolicy(checkpoint['net'])
        # Independently verify the Python forward pass against original JS.
        rng = np.random.default_rng(args.seed)
        probes = [training.final_observation] + rng.uniform(0, 1, (100, 30)).tolist()
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
    evaluation_env = DeepTrafficEnv(cfg)
    score = evaluation_env.evaluate(policy, runs=args.runs)
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
        'decisions':training.decisions, 'training_frames':training.training_frames,
        'training_blocks':training.training_blocks,
        'frames_per_training_block':training.frames_per_training_block,
        'training_interface':trainer.interface,
        'training_reward_objective':trainer.reward_objective,
        'evaluation_objective':'episode mean speed; one controlled car',
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
