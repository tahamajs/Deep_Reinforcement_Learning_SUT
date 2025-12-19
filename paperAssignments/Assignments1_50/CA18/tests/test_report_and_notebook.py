from pathlib import Path


def test_report_exists():
    root = Path(__file__).resolve().parents[2]
    assert (root / "report.tex").exists()


def test_demo_notebook_exists():
    root = Path(__file__).resolve().parents[2]
    assert (root / "notebooks" / "demo.ipynb").exists()
