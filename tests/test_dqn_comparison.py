"""Guard the claim that only architecture differs between two experiments."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import pytest

pytest.importorskip('torch')
from deeptraffic_llm_reward.dqn import DQNConfig

spec = importlib.util.spec_from_file_location(
    'dqn_experiment', Path(__file__).parents[1]/'scripts/run_dqn_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_paired_architecture_comparison_rejects_other_changes(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    report = dict(scenario_seeds=[100,101], scores_mph=[60,62], median_mph=61.,
                  mean_mph=61., iqr_mph=1., env_config={}, backend_revision='test',
                  protocol='test', exploration=False)
    for root, hidden in [(old,(128,128)), (new,(128,128,64,64))]:
        root.mkdir()
        (root/'rule.json').write_text(json.dumps(report))
        for seed in range(3):
            directory = root/f'seed_{seed}'
            directory.mkdir()
            manifest = dict(training_seed=seed, training_scenario_seeds=[seed],
                dqn_config=asdict(DQNConfig(hidden=hidden)), env_config={},
                backend_revision='test', torch='test', numpy='test', python='test',
                objective='fleet', checkpoint_selection='final only', algorithm='DQN')
            (directory/'manifest.json').write_text(json.dumps(manifest))
            (directory/'evaluation.json').write_text(json.dumps(report))
    experiment.summarize(new, old)
    result = json.loads((new/'comparison.json').read_text())['architecture_comparison']
    assert result['mean_median_delta_mph'] == 0
    assert all(row['paired_mean_delta_ci95'] == [0,0] for row in result['seeds'])
    path = new/'seed_0/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['dqn_config']['learning_rate'] *= 2
    path.write_text(json.dumps(manifest))
    # Alter all new replicates so the within-experiment consistency check passes.
    for seed in range(3):
        path = new/f'seed_{seed}'/'manifest.json'
        row = json.loads(path.read_text())
        row['dqn_config']['learning_rate'] = manifest['dqn_config']['learning_rate']
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='other hyperparameters'):
        experiment.summarize(new, old)
