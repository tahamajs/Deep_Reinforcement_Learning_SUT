import torch
from src.losses import regression_loss, classification_loss


def test_regression_loss():
    pred = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([1.1, 1.9])
    l = regression_loss(pred, target)
    assert l.item() >= 0.0


def test_classification_loss():
    logits = torch.randn(3, 4)
    targets = torch.tensor([0, 1, 2])
    l = classification_loss(logits, targets)
    assert l.item() >= 0.0
