"""Launch the matched evaluation independently of the invoking terminal/chat."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/llm_comparison_short_v1'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root/args.out).resolve()
    if not args.resume:
        parser.error('Create the experiment with evaluate_llm.py first, then use --resume')
    if not (output/'manifest.json').is_file():
        parser.error('No existing experiment manifest')
    launch_path = output/'background.json'
    if launch_path.exists():
        old_pid = json.loads(launch_path.read_text())['pid']
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            pass
        else:
            parser.error(f'Previous background PID {old_pid} is still alive')
    contract = json.loads((output/'manifest.json').read_text())['contract']
    command = [sys.executable, '-u', str(root/'scripts/evaluate_llm.py'),
               '--out', str(output), '--resume',
               '--frames', str(contract['config']['episode_frames']),
               '--seeds', *map(str, contract['scenario_seeds']),
               '--checkpoint', str(output/'checkpoint.json')]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with (output/'background.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=root, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    launch = {'pid': process.pid, 'command': command, 'cwd': str(root),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'log': str(output/'background.log')}
    launch_path.write_text(json.dumps(launch, indent=2))
    print(json.dumps(launch, indent=2))


if __name__ == '__main__':
    main()
