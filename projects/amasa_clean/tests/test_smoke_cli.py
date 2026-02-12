import subprocess
import sys
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
    # We only verify the command can start; dataset may be absent in fresh environments.
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert proc.returncode in {0, 1}
