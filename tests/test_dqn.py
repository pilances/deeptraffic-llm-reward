import json
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from deeptraffic_llm_reward import DeepTrafficConfig, AgentTransition
from deeptraffic_llm_reward.dqn import DQNAgent, DQNConfig, train_shared_dqn, evaluate_fleet


def transition(truncated=False):
    return AgentTransition(0, np.zeros(3,np.float32), 1, 30., np.ones(3,np.float32),
                           0, 30, False, truncated, .5)


def test_bellman_target_and_end_mask():
    cfg = DQNConfig(hidden=(4,), target_every=2)
    agent = DQNAgent(3, cfg)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        for p in agent.target.parameters():
            p.zero_()
        agent.target[-1].bias.fill_(2)
    before = [p.clone() for p in agent.online.parameters()]
    target_before = [p.clone() for p in agent.target.parameters()]
    # r=1 after scaling; continuing target=1+.5*2=2 => Huber 1.5;
    # truncated target=1 => Huber .5; mean=1.0.
    assert agent.update([transition(),transition(True)]) == pytest.approx(1.0)
    assert any(not torch.equal(p,q) for p,q in zip(before,agent.online.parameters()))
    assert all(torch.equal(p,q) for p,q in zip(target_before,agent.target.parameters()))
    agent.update([transition()])
    assert all(torch.equal(p,q) for p,q in zip(agent.online.parameters(),agent.target.parameters()))


def test_exploration_schedule_and_greedy_does_not_change_state():
    a = DQNAgent(3, DQNConfig(hidden=(4,),epsilon_decay=10), seed=1)
    b = DQNAgent(3, a.config, seed=1)
    assert [a.explore(np.zeros(3)) for _ in range(20)] == [b.explore(np.zeros(3)) for _ in range(20)]
    assert a.epsilon == pytest.approx(.05)
    before = a.decisions
    assert a.greedy(np.zeros(3)) == a.greedy(np.zeros(3))
    assert a.decisions == before


@pytest.mark.parametrize('hidden', [(8,), (128,128,64,64)])
def test_small_training_reproducibility_checkpoint_and_fair_evaluation(tmp_path, hidden):
    config = DQNConfig(episodes=2, hidden=hidden, batch_size=4, learning_starts=8,
                       target_every=2, epsilon_decay=10)
    env_config = DeepTrafficConfig(other_agents=4, episode_frames=65)
    a = train_shared_dqn(0, tmp_path/'a', config, env_config)
    b = train_shared_dqn(0, tmp_path/'b', config, env_config)
    assert a.updates > 0
    assert all(torch.equal(p,q) for p,q in zip(a.online.parameters(),b.online.parameters()))
    scores_a = evaluate_fleet(a.greedy, env_config, [2000000,2000001])
    scores_b = evaluate_fleet(b.greedy, env_config, [2000000,2000001])
    assert scores_a == scores_b
    manifest = json.loads((tmp_path/'a/manifest.json').read_text())
    assert not set(manifest['training_scenario_seeds']) & set(scores_a['scenario_seeds'])
    assert a.decisions == 24  # 5 + 5 + 2 callback actions per 65-frame episode
    with pytest.raises(FileExistsError):
        train_shared_dqn(0, tmp_path/'a', config, env_config)
