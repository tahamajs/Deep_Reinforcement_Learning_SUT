from pathlib import Path
import json


def test_compute_metrics_script_exists():
    repo = Path(__file__).resolve().parents[1]
    assert (repo / "scripts" / "compute_metrics.py").exists(), "compute_metrics.py must exist"


def test_metrics_json_exists_and_keys():
    repo = Path(__file__).resolve().parents[1]
    p = repo / "outputs" / "example_run" / "metrics.json"
    assert p.exists(), "example metrics.json must exist"
    data = json.loads(p.read_text())
    # check presence of commonly expected keys
    assert "val_mse" in data or "val_rmse" in data, "metrics should include val_mse or val_rmse"
    assert any(k in data for k in ["accuracy", "precision", "recall", "f1_macro"]), "classification metric key missing (if applicable)"
