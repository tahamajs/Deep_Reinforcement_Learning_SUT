import torch
from ..src.retrieval_buffer import RetrievalBuffer


def test_add_and_retrieve():
    rb = RetrievalBuffer(max_size=100, state_dim=3, action_dim=2, device="cpu")
    # create two trajectories: one near [0,0,0] with high returns, one near [10,10,10] with low returns
    states1 = torch.zeros((5, 3))
    actions1 = torch.zeros((5, 2))
    rewards1 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    states2 = torch.ones((5, 3)) * 10.0
    actions2 = torch.ones((5, 2)) * -1.0
    rewards2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    rb.add_trajectory(states1, actions1, rewards1, gamma=0.99)
    rb.add_trajectory(states2, actions2, rewards2, gamma=0.99)

    # query near zeros should retrieve actions of the first traj
    query = torch.tensor([0.1, -0.1, 0.0])
    retrieved = rb.retrieve_best_k(query, k=3, nn=10)
    assert retrieved.shape[0] > 0
    # retrieved actions should be close to zeros
    assert torch.allclose(retrieved.mean(dim=0), torch.zeros(2), atol=1e-1)

    # sample batch should return tensors of correct shapes
    s, a, rtg = rb.sample_batch(batch_size=4)
    assert s.shape == (4, 3)
    assert a.shape == (4, 2)
    assert rtg.shape == (4, 1)

