import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_smoke_one_chain(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    cmd = [
        "python",
        str(repo / "scripts" / "run_experiment.py"),
        "--chain",
        "value",
        "--algo",
        "dqn",
        "--env",
        "CartPole-v1",
        "--steps",
        "1000",
        "--seed",
        "0",
        "--out-dir",
        str(tmp_path / "run"),
    ]
    subprocess.check_call(cmd, cwd=repo, env=env)
    metrics = tmp_path / "run" / "metrics.json"
    assert metrics.exists()
    payload = json.loads(metrics.read_text())
    for k in ["run_id", "chain", "algo", "env", "seed", "budget", "train_curve", "eval", "timing"]:
        assert k in payload
