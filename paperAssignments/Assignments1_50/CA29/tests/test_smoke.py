"""Optional smoke test for quick sanity checks.

This test is disabled by default. Enable by setting environment variable
`RUN_SMOKE=1` (e.g., `RUN_SMOKE=1 pytest -q`). The test runs a few
`SAC.update()` steps with randomized data to ensure the training loop
does not crash on CPUs in CI-like environments.
"""

import os
import numpy as np
import pytest
import torch

from src.config import SACConfig
from src.sac import SAC


@pytest.mark.skipif(os.getenv('RUN_SMOKE') != '1', reason='Enable smoke tests with RUN_SMOKE=1')
def test_sac_smoke_update_cpu():
    cfg = SACConfig()
    cfg.buffer_size = 128
    cfg.batch_size = 8
    device = torch.device('cpu')

    sac = SAC(state_dim=3, action_dim=1, config=cfg, device=device)

    # Populate the buffer with random but well-formed transitions
    for _ in range(32):
        s = np.random.randn(3).astype(np.float32)
        a = np.random.randn(1).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(3).astype(np.float32)
        d = False
        sac.buffer.add(s, a, r, ns, d)

    # Call update a few times to ensure it runs without errors
    for _ in range(5):
        sac.update()

    assert True
