import sys
import types
import torch
import importlib.util
from types import SimpleNamespace


def _make_dummy_planner_module():
    mod = types.ModuleType("planner")

    class CheckpointBuffer:
        def __init__(self, capacity=1024, device=None):
            self._store = []

        def push(self, z, score=0.0, step=0):
            self._store.append({"z": z, "score": score, "step": step})

        def sample(self, k=1, prioritized=True):
            return [self._store[0]] if self._store else []

        def __len__(self):
            return len(self._store)

    def simulate_branches(rssm, actor, value_fn, z_saved, cfg):
        # create two simple branches of length 3
        branches = []
        for b in range(2):
            traj = []
            for t in range(3):
                z = torch.randn(z_saved.shape, device=z_saved.device).squeeze(0)
                a = torch.randn((1,), device=z_saved.device)
                r = 1.0
                gamma = 1.0
                traj.append((z, a, r, gamma))
            # simple return
            branches.append(types.SimpleNamespace(ret=3.0, traj=traj))
        return branches

    def should_trigger(td_error, unc, entropy, cfg, last_trigger, step):
        return True

    @SimpleNamespace
    class TriggerConfig:
        cooldown = 1
        trigger_td = 0.5
        trigger_unc = 0.1
        trigger_ent_low = 0.0
        trigger_ent_high = 10.0

    mod.CheckpointBuffer = CheckpointBuffer
    mod.simulate_branches = simulate_branches
    mod.should_trigger = should_trigger
    mod.TriggerConfig = TriggerConfig
    return mod


def test_dreamer_agent_planner_integration_smoke(tmp_path):
    # inject dummy planner into sys.modules before loading agent
    sys.modules["planner"] = _make_dummy_planner_module()

    # load dreamer_agent module by path
    path = "/Users/tahamajs/Documents/uni/DRL/archive/cleanup_20251213_135640Z/CAs/Solutions/CA11_World_Models_RSSM/agents/dreamer_agent.py"
    spec = importlib.util.spec_from_file_location("dreamer_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    # construct minimal global_config
    dreamer_cfg = SimpleNamespace(
        agent_config=SimpleNamespace(
            latent_dim=8, imagination_horizon=4, hidden_dim=16
        ),
        rssm_config=SimpleNamespace(
            latent_dim=8,
            hidden_dim=16,
            stochastic_size=8,
            deterministic_size=16,
            learning_rate=1e-3,
        ),
    )
    global_config = SimpleNamespace(
        dreamer_config=dreamer_cfg,
        device=torch.device("cpu"),
        planner={"buffer_size": 16, "cooldown": 1},
    )

    Agent = mod.DreamerAgent
    agent = Agent(obs_dim=4, action_dim=1, global_config=global_config)

    # Create fake branches and call update_actor_critic_from_branches
    branches = []
    for _ in range(3):
        traj = []
        for _ in range(4):
            z = torch.randn(8)
            a = torch.randn(1)
            r = 1.0
            gamma = 1.0
            traj.append((z, a, r, gamma))
        branches.append(types.SimpleNamespace(ret=4.0, traj=traj))

    out = agent.update_actor_critic_from_branches(branches, topk_frac=0.5, lambda_=0.9)
    assert isinstance(out, dict)
    assert "actor_loss" in out and "critic_loss" in out
