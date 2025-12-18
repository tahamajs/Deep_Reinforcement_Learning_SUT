import torch

from ..src.agent import RAUOBACAgent
from ..src.config import Config


def test_agent_update_steps():
    cfg = Config()
    cfg.device = "cpu"
    cfg.buffer_size = 100
    cfg.batch_size = 8
    cfg.retrieval_k = 3
    cfg.retrieval_nn = 10

    state_dim = 5
    action_dim = 2
    agent = RAUOBACAgent(state_dim, action_dim, cfg)

    # create a fake trajectory and add to buffer
    states = torch.randn((10, state_dim))
    actions = torch.tanh(torch.randn((10, action_dim)))
    rewards = torch.randn((10,))
    agent.retrieval_buffer.add_trajectory(states, actions, rewards, gamma=cfg.gamma)

    # sample a batch for critic update
    s_batch, a_batch, rtg_batch = agent.retrieval_buffer.sample_batch(
        min(cfg.batch_size, 8)
    )
    loss_c = agent.update_critic(s_batch, a_batch, rtg_batch.squeeze(1))
    assert isinstance(loss_c, float) and loss_c >= 0.0

    # offline actor update
    advantages = torch.randn((s_batch.shape[0],))
    loss_off = agent.update_offline_actor(s_batch, a_batch, advantages=advantages)
    assert isinstance(loss_off, float) and loss_off >= 0.0

    # online actor update (should run without error)
    loss_online = agent.update_online_actor(s_batch)
    assert isinstance(loss_online, float)
