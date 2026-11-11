"""Execute the pinned upstream parity suite as part of our acceptance tests."""
import os
import subprocess
import sys
from deeptraffic_llm_reward.backend import backend_root

def test_upstream_fidelity_suite():
    env = dict(os.environ, DT_SCORE_RUNS='3')
    result = subprocess.run([sys.executable, '-m', 'pytest',
        str(backend_root()/'tests/test_fidelity.py'), '-q'], env=env,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '18 passed' in result.stdout, result.stdout
