import subprocess
import sys
import os
from pathlib import Path


def test_unified_runner_smoke():
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-m",
        "projects.amasa_clean.scripts.run_experiment",
        "--mode",
        "offline_train",
        "--preset",
        "smoke",
        "--algo",
        "cql",
        "--dataset",
        "data/amasa_offline.npz",
        "--out_dir",
        "projects/amasa_clean/results/smoke_test",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) if not env.get("PYTHONPATH") else f"{root}:{env['PYTHONPATH']}"
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    dataset = root / "data" / "amasa_offline.npz"
    if dataset.exists():
        assert proc.returncode == 0, proc.stderr
    else:
        assert proc.returncode in {0, 1}
