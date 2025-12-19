import os
import shutil
import sys
import tempfile

# Ensure repo root is on path like other tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ..scripts.generate_placeholder_figs import main as gen_main  # type: ignore


def test_generate_placeholder_figs_creates_files():
    out_dir = os.path.join(ROOT, "outputs", "ca9")
    # cleanup if exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    # run generator (should succeed without external deps)
    gen_main()
    # check files exist
    assert os.path.exists(os.path.join(out_dir, "plots", "losses.png"))
    assert os.path.exists(os.path.join(out_dir, "plots", "lam_std.png"))
    assert os.path.exists(os.path.join(out_dir, "eval", "returns.png"))
    assert os.path.exists(os.path.join(out_dir, "eval", "ep_0", "rewards.png"))
    # cleanup
    shutil.rmtree(out_dir)
