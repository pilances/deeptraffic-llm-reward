"""End-to-end bridge/export/evaluation check using an untrained tiny run."""
import json
from pathlib import Path
import subprocess
import sys
from deeptraffic_llm_reward import HistoricalStarterRunner


def test_historical_runner_has_explicit_separate_reward_contract():
    rewards = []
    brain_seeds = []
    runner = HistoricalStarterRunner()
    result = runner.train(
        lambda state, reward: rewards.append(reward) or 0,
        brain_seeds.append,
        iterations=30,
        seed=42,
    )
    assert runner.interface == 'historical_callback_runner'
    assert runner.reward_objective == 'historical_ego_reward_(N-60)/20'
    assert result.decisions == len(rewards) == 60
    assert result.training_frames == 1784
    assert len(brain_seeds) == 1
    assert any(reward < 0 for reward in rewards)

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
    assert report['training_interface'] == 'historical_callback_runner'
    assert report['training_reward_objective'] == 'historical_ego_reward_(N-60)/20'
