import pytest
pytest.importorskip("matplotlib")
import tempfile
from pathlib import Path
from matplotlib import pyplot as plt

from src.utils import save_figure, ensure_dir


def test_save_figure_and_ensure_dir(tmp_path: Path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "pics" / "plot.png"
    save_figure(fig, out, dpi=100)
    assert out.exists()

    # ensure_dir returns a Path and creates directories
    d = tmp_path / "nested" / "dir"
    p = ensure_dir(d)
    assert p.exists() and p.is_dir()
