"""End-to-end bridge/export/evaluation check using an untrained tiny run."""
import json
from pathlib import Path
import subprocess
import sys

def test_starter_bridge_and_js_evaluation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'scripts/evaluate_starter.py'),
        '--seed', '42', '--iterations', '30', '--runs', '3', '--verify-js',
        '--out', str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((tmp_path/'seed_42_results.json').read_text())
    assert report['max_forward_error'] < 1e-12
    assert report['evaluation']['median'] == report['javascript_verification']['median']
    assert report['age'] == report['decisions']
