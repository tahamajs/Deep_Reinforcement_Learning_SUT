import numpy as np
from src.config import Config
from src.train import DQNAgent


def test_agent_with_prioritized_replay_and_double_dqn():
    cfg = Config()
    cfg.replay = "prioritized"
    cfg.double_dqn = True
    cfg.batch_size = 4

    agent = DQNAgent(cfg, state_dim=4, action_dim=2)

    # Seed the buffer with batch_size experiences
    for i in range(8):
        s = np.zeros(4) + i
        agent.memory.push(s, i % 2, float(i), s + 0.1, False)

    # Capture priorities before optimization
    before_priorities = agent.memory.priorities.copy()

    # Optimization should run without error
    agent.optimize_model()

    # After optimization, priorities at some indices should have been updated
    assert np.any(agent.memory.priorities != before_priorities)
