from pathlib import Path
from ca30.model import BaseModel
from ca30.train import simple_train_epoch
from ca30.utils import set_seed


def test_train_smoke(tmp_path: Path):
    set_seed(0)
    m = BaseModel(input_dim=4, hidden_dim=8, output_dim=3, seed=0)
    out = simple_train_epoch(m, batch_size=4, input_dim=4)
    assert out.shape[0] == 4
