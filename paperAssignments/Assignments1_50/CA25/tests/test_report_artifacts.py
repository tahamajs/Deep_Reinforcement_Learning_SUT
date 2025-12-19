from pathlib import Path


def test_report_templates_exist():
    repo = Path(__file__).resolve().parents[1]
    assert (repo / "REPORT.md").exists(), "REPORT.md must exist"
    assert (repo / "rep.tex").exists(), "rep.tex must exist"
    assert (repo / "scripts" / "make_placeholder_figures.py").exists(), "placeholder script must exist"


def test_example_figures_present():
    p = Path(__file__).resolve().parents[1] / "outputs" / "example_run" / "pictures"
    assert p.exists() and p.is_dir(), "example pictures directory should exist"
    # check for at least one known placeholder file
    assert (p / "loss.png").exists(), "loss.png placeholder should exist"
    assert (p / "predictions.png").exists(), "predictions.png placeholder should exist"
