import torch
from src.config import Config
from src.model import RecurrentCritic, StochasticActor
from src.data import SequenceReplayBuffer
from src.losses import critic_loss_lambda


def test_critic_lambda_loss_backward():
    cfg = Config(batch_size=4, seq_len=8, obs_dim=6, action_dim=2, hidden_size=32)
    device = "cpu"
    critic = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target_critic = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(
        device
    )
    target_critic.load_state_dict(critic.state_dict())
    actor = StochasticActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)

    buf = SequenceReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.seq_len, max_size=100)
    # populate small buffer with actor-sampled actions
    for _ in range(20):
        import numpy as np

        obs = np.random.randn(cfg.seq_len, cfg.obs_dim).astype("float32")
        with torch.no_grad():
            obs_t = torch.tensor(obs[None], device=device)
            acts_t, logp_t, _ = actor.sample(obs_t)
        acts = acts_t[0].cpu().numpy().astype("float32")
        logp = logp_t[0].cpu().numpy().astype("float32")
        rews = np.random.randn(cfg.seq_len).astype("float32") * 0.01
        dones = np.zeros(cfg.seq_len, dtype="float32")
        buf.add(obs, acts, rews, dones, beh_logp=logp)

    batch = buf.sample_batch(cfg.batch_size, device=device)
    obs_b, acts_b, rews_b, dones_b, beh_logp_b = batch
    loss, returns = critic_loss_lambda(
        critic,
        target_critic,
        obs_b,
        acts_b,
        rews_b,
        dones_b,
        beh_logp_b,
        cfg.gamma,
        cfg.lam,
        c_rho=cfg.c_rho,
        policy=actor,
    )
    loss.backward()
    assert torch.isfinite(loss).item()
    assert returns.shape == (cfg.batch_size, cfg.seq_len)











