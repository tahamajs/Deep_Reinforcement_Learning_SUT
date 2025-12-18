import os
import tempfile
import torch

from src.config import get_default_config
from src.model import TWMSSDModel, TWMSSDImageModel
from src.tokenizer import ImageVQVAE


def one_step_training_and_checkpoint(tmp_path):
    cfg = get_default_config()
    device = torch.device("cpu")
    img_vq = ImageVQVAE(codebook_size=16, d_model=cfg.d_model, in_ch=3)
    backbone = TWMSSDModel(d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=1)
    model = TWMSSDImageModel(img_vq, backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # create dummy images matching downsample assumptions: H,W divisible by 4
    B, C, H, W = 2, 3, 32, 32
    images = torch.randn(B, C, H, W)
    actions = torch.randn(
        B, (H // 4) * (W // 4), cfg.d_model
    )  # not used but shape-compatible
    pred_obs, pred_reward, recon, indices = model(images, actions=None)
    # compute simple loss and step
    loss = torch.nn.functional.mse_loss(recon, torch.randn_like(recon))
    opt.zero_grad()
    loss.backward()
    opt.step()

    # save checkpoint
    ckpt_path = os.path.join(tmp_path, "ckpt.pt")
    torch.save(
        {"model_state": model.state_dict(), "opt_state": opt.state_dict()}, ckpt_path
    )

    # load into new model
    img_vq2 = ImageVQVAE(codebook_size=16, d_model=cfg.d_model, in_ch=3)
    backbone2 = TWMSSDModel(d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=1)
    model2 = TWMSSDImageModel(img_vq2, backbone2).to(device)
    opt2 = torch.optim.Adam(model2.parameters(), lr=cfg.lr)
    ckpt = torch.load(ckpt_path, map_location=device)
    model2.load_state_dict(ckpt["model_state"])
    opt2.load_state_dict(ckpt["opt_state"])
    return True


def test_one_step_training_and_checkpoint(tmp_path):
    assert one_step_training_and_checkpoint(str(tmp_path))

