import importlib.util
import pathlib

import torch


def load_module_from_path(path: str, name: str):
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "paperAssignments.Assignments1_50.CA10"
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_policy_loss_computation():
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    policy_path = str(
        repo_root
        / "paperAssignments"
        / "Assignments1-50"
        / "CA10"
        / "policy"
        / "efficientzero_v2_ma.py"
    )
    models_path = str(
        repo_root
        / "paperAssignments"
        / "Assignments1-50"
        / "CA10"
        / "models"
        / "ezv2_ma_net.py"
    )

    mod_policy = load_module_from_path(policy_path, "policy_mod")
    mod_models = load_module_from_path(models_path, "ezv2_ma_net")
    EfficientZeroV2Policy = mod_policy.EfficientZeroV2Policy
    MAEZ = mod_models.MAEZV2Network

    obs_dim = 6
    latent_dim = 32
    joint_action_dim = 8
    policy = EfficientZeroV2Policy(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        joint_action_dim=joint_action_dim,
        device="cpu",
    )
    policy.eval()

    obs = torch.randn(2, obs_dim)
    actions = torch.randn(2, joint_action_dim)
    pi_t = torch.softmax(torch.randn(2, joint_action_dim), dim=-1)
    v_t = torch.randn(2)
    r_t = torch.randn(2)
    z_t = torch.randn(2)
    h0 = policy.net.initial_latent(obs)
    loss_weights = {"pi": 1.0, "v": 1.0, "r": 1.0, "z": 0.5}
    loss, losses = policy.compute_losses(h0, actions, pi_t, v_t, r_t, z_t, loss_weights)
    assert isinstance(loss.item(), float)
    assert "loss_pi" in losses and "loss_v" in losses














