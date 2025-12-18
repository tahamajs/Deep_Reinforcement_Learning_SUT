import torch
from planner.checkpoint_buffer import CheckpointBuffer


def test_push_sample_save_load(tmp_path):
    buf = CheckpointBuffer(capacity=4, device=torch.device("cpu"))
    z = torch.randn(1, 8)
    buf.push(z, score=1.0, step=0)
    assert len(buf) == 1
    samples = buf.sample(k=1, prioritized=False)
    assert samples and "z" in samples[0]
    p = tmp_path / "buf.pt"
    buf.save(str(p))
    buf2 = CheckpointBuffer(capacity=4)
    buf2.load(str(p))
    assert len(buf2) == 1














