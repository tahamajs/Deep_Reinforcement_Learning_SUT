import torch
import math

from paperAssignments.Assignments1_50.CA1.sinkhorn import AnnealedSinkhornLoss
from paperAssignments.Assignments1_50.CA1.model import ParticleHead
import torch.nn as nn


def test_single_update_shapes_and_gradients():
    torch.manual_seed(0)
    B = 8
    A = 3
    N = 16
    D = 1
    obs_dim = 4

    # simple MLP encoder
    enc = nn.Sequential(
        nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU()
    )
    head = ParticleHead(in_dim=128, num_actions=A, num_particles=N, particle_dim=D)

    class Net(nn.Module):
        def __init__(self, enc, head):
            super().__init__()
            self.enc = enc
            self.head = head

        def forward(self, x):
            feats = self.enc(x)
            return self.head(feats)

    model = Net(enc, head)
    loss_fn = AnnealedSinkhornLoss(
        n_iters=5, eps_start=0.5, eps_end=0.1, decay_steps=10
    )

    # synthetic batch
    obs = torch.randn(B, obs_dim)
    pred = model(obs)  # (B, A, N, D)
    assert pred.shape == (B, A, N, D)

    actions = torch.randint(0, A, (B,))
    idx = actions.view(-1, 1, 1, 1).expand(-1, 1, N, D)
    pred_sel = pred.gather(1, idx).squeeze(1)  # (B, N, D)

    # create target by adding small noise
    with torch.no_grad():
        targ = pred_sel + 0.1 * torch.randn_like(pred_sel)

    loss = loss_fn(pred_sel, targ)
    loss.backward()

    # check loss finite
    assert math.isfinite(float(loss))

    # check gradients exist for some params
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


if __name__ == "__main__":
    test_single_update_shapes_and_gradients()
    print("train step test passed")










