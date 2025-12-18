import math

import torch

from paperAssignments.Assignments1_50.CA1.sinkhorn import AnnealedSinkhornLoss


def test_identity_and_symmetry():
    torch.manual_seed(0)
    B, N, D = 4, 16, 1
    x = torch.randn(B, N, D)
    y = x.clone()

    loss_fn = AnnealedSinkhornLoss(n_iters=20)
    s_xx = loss_fn(x, x).item()
    s_xy = loss_fn(x, y).item()

    # Identity: S(X,X) should be approximately zero (small positive due to numerical eps)
    assert math.isfinite(s_xx)
    assert s_xx >= -1e-6
    assert abs(s_xx) < 1e-4

    # Symmetry: S(X,Y) == S(Y,X)
    s_yx = loss_fn(y, x).item()
    assert abs(s_xy - s_yx) < 1e-6


if __name__ == "__main__":
    test_identity_and_symmetry()
    print("tests passed")
