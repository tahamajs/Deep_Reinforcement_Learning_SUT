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


def test_network_shapes():
    obs_dim = 6
    latent_dim = 32
    joint_action_dim = 8
    per_agent = (4, 4)
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    module_path = str(
        repo_root
        / "paperAssignments"
        / "Assignments1-50"
        / "CA10"
        / "models"
        / "ezv2_ma_net.py"
    )
    mod = load_module_from_path(module_path, "ezv2_ma_net")
    MAEZV2Network = mod.MAEZV2Network
    net = MAEZV2Network(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        joint_action_dim=joint_action_dim,
        per_agent_action_dims=per_agent,
    )
    net.eval()

    obs = torch.randn(2, obs_dim)
    h0 = net.initial_latent(obs)
    assert h0.shape == (2, latent_dim)

    logits_joint, logits_agents, v, z = net.predict_from_latent(h0)
    assert logits_joint.shape == (2, joint_action_dim)
    assert isinstance(logits_agents, list) and len(logits_agents) == len(per_agent)
    assert v.shape == (2,)
    assert z.shape == (2,)

    # unroll dynamics
    actions = torch.randn(2, 3, joint_action_dim)
    outputs = net.unroll_dynamics(h0[0:1], actions[0:1], steps=3)
    assert len(outputs) == 3
    for h, r in outputs:
        assert h.shape[1] == latent_dim
        assert isinstance(r.item(), float)
