import pytest
import torch
import sys
from pathlib import Path


def _ensure_repo_in_path():
    repo_root = Path(__file__).resolve().parents[4]
    ca8_src = repo_root / "paperAssignments" / "Assignments1-50" / "CA8" / "src"
    if str(ca8_src) not in sys.path:
        sys.path.insert(0, str(ca8_src))


def test_import_modules():
    _ensure_repo_in_path()
    # Ensure modules import without heavy dependencies
    import importlib

    config = importlib.import_module("config")
    envs = importlib.import_module("envs")
    losses = importlib.import_module("losses")
    utils = importlib.import_module("utils")
    agent = importlib.import_module("agent")

    # basic sanity checks
    assert hasattr(config, "cfg")
    assert hasattr(envs, "ToTheMaxWrapper")
    assert hasattr(losses, "SinkhornWrapper")
    assert hasattr(utils, "set_seed")
    assert hasattr(agent, "MaxSinkAgent")


def test_sinkhorn_fallback():
    _ensure_repo_in_path()
    import importlib

    losses = importlib.import_module("losses")
    SinkhornWrapper = losses.SinkhornWrapper

    loss = SinkhornWrapper(blur=0.01)
    # create small random clouds [B, N, d]
    x = torch.randn(2, 4, 1)
    y = torch.randn(2, 4, 1)
    out = loss(x, y)
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()

