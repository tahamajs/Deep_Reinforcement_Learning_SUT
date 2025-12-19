from pathlib import Path


def test_report_headings_exist():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "REPORT.md").read_text()
    # check for required sections
    for heading in ["Abstract", "Introduction", "Methods", "Experimental Setup", "Results", "Discussion", "Reproducibility", "Appendix"]:
        assert heading in text, f"REPORT.md missing heading: {heading}"


def test_rep_tex_contains_hyperparams_and_repro():
    repo = Path(__file__).resolve().parents[1]
    tex = (repo / "rep.tex").read_text()
    assert "Hyperparameters" in tex or "Hyperparameters used" in tex, "rep.tex should include hyperparameter table or caption"
    assert "Reproducibility" in tex, "rep.tex should include reproducibility section"
