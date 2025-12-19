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


def test_report_contains_metric_terms():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "REPORT.md").read_text()
    assert "RMSE" in text, "REPORT.md should document RMSE formula"
    assert "Accuracy" in text, "REPORT.md should document Accuracy formula"
    assert "metrics.json" in text, "REPORT.md should mention metrics.json schema or saving metrics"


def test_rep_tex_expanded_sections():
    repo = Path(__file__).resolve().parents[1]
    tex = (repo / "rep.tex").read_text()
    assert "pseudocode" in tex.lower() or "algorithm" in tex.lower(), "rep.tex should include training pseudocode"
    assert "hyperparameter sweep" in tex.lower(), "rep.tex should mention hyperparameter sweeps"
    assert "implementation notes" in tex.lower(), "rep.tex should include implementation notes section"
    assert "building the pdf" in tex.lower(), "rep.tex should include PDF building instructions"
    assert "limitations" in tex.lower(), "rep.tex should include a Limitations section"
    assert "future work" in tex.lower(), "rep.tex should include a Future Work section"
    assert "evaluation metrics" in tex.lower(), "rep.tex should include an Evaluation Metrics subsection"
