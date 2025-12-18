import torch
import numpy as np
from src.config import Config
from src.model import RecurrentCritic, StochasticActor
from src.data import SequenceReplayBuffer
from src.sac import sac_update


def test_actor_logprob_consistency():
    cfg = Config(obs_dim=5, action_dim=3, seq_len=6, batch_size=2, hidden_size=32)
    actor = StochasticActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)
    # random batch
    obs = np.random.randn(cfg.batch_size, cfg.seq_len, cfg.obs_dim).astype("float32")
    obs_t = torch.tensor(obs)
    actions, logp_sampled, _ = actor.sample(obs_t)
    logp_calc = actor.log_prob(obs_t, actions)
    # values should be finite and close
    assert torch.isfinite(logp_sampled).all()
    assert torch.isfinite(logp_calc).all()
    assert torch.allclose(logp_sampled, logp_calc, atol=1e-4, rtol=1e-3)


def test_sac_update_changes_actor():
    cfg = Config(obs_dim=5, action_dim=2, seq_len=6, batch_size=2, hidden_size=32)
    critic1 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)
    critic2 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)
    target1 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)
    target2 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)
    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())
    actor = StochasticActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)

    opt_c1 = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    opt_c2 = torch.optim.Adam(critic2.parameters(), lr=1e-3)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-3)

    buf = SequenceReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.seq_len, max_size=50)
    # populate
    for _ in range(20):
        obs = np.random.randn(cfg.seq_len, cfg.obs_dim).astype("float32")
        with torch.no_grad():
            acts_t, logp_t, _ = actor.sample(torch.tensor(obs[None]))
        acts = acts_t[0].numpy().astype("float32")
        logp = logp_t[0].numpy().astype("float32")
        rews = np.random.randn(cfg.seq_len).astype("float32") * 0.01
        dones = np.zeros(cfg.seq_len, dtype="float32")
        buf.add(obs, acts, rews, dones, beh_logp=logp)

    batch = buf.sample_batch(cfg.batch_size)
    # record actor param norm
    before = torch.nn.utils.parameters_to_vector(actor.parameters()).detach().clone()
    a_loss = sac_update(
        [critic1, critic2],
        [target1, target2],
        actor,
        [opt_c1, opt_c2],
        opt_a,
        batch,
        cfg,
    )
    after = torch.nn.utils.parameters_to_vector(actor.parameters()).detach().clone()
    assert not torch.allclose(before, after)
    assert isinstance(a_loss, float) or torch.is_tensor(a_loss)










