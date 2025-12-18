import importlib.util
import pathlib

import torch


def load_module_from_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_mcts_basic():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    models_path = str(repo_root / "paperAssignments" / "Assignments1-50" / "CA10" / "models" / "ezv2_ma_net.py")
    mcts_path = str(repo_root / "paperAssignments" / "Assignments1-50" / "CA10" / "mcts" / "search.py")

    mod_models = load_module_from_path(models_path, "ezv2_ma_net")
    mod_mcts = load_module_from_path(mcts_path, "mcts_search")
    MAEZ = mod_models.MAEZV2Network
    MCTS = mod_mcts.MCTS

    obs_dim = 6
    latent_dim = 32
    joint_action_dim = 8
    net = MAEZ(obs_dim=obs_dim, latent_dim=latent_dim, joint_action_dim=joint_action_dim)
    net.eval()

    obs = torch.randn(1, obs_dim)
    h0 = net.initial_latent(obs)
    mcts = MCTS(net, c_puct=1.0)
    visits, policy = mcts.run(h0, num_simulations=20, topk=4)
    # visits should be non-negative and sum <= sims
    assert visits.sum().item() <= 20 + 1e-6
    assert torch.all(visits >= 0)
    assert abs(float(policy.sum().item()) - 1.0) < 1e-6

