"""
BCQ implementation (simple VAE behaviour model + perturbation network + BCQ agent).
This file is intentionally compact and import-safe.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        return self.mu(h), self.log_std(h)

    def decode(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        mu, log_std = self.encode(state, action)
        std = (log_std * 0.5).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.decode(state, z)
        return recon, mu, log_std


class PerturbationNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, phi: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.phi = phi

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([state, action], dim=-1))
        return action + self.phi * delta


class BCQ:
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 16, lr: float = 3e-4):
        from cql_implementation import QNetwork  # reuse simple QNetwork

        self.vae = VAE(state_dim, action_dim, latent_dim)
        self.perturb = PerturbationNetwork(state_dim, action_dim)
        self.Q = QNetwork(state_dim, action_dim)

        self.vae_opt = optim.Adam(self.vae.parameters(), lr=lr)
        self.perturb_opt = optim.Adam(self.perturb.parameters(), lr=lr)
        self.q_opt = optim.Adam(self.Q.parameters(), lr=lr)

    def select_action(self, state: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        # sample actions from VAE by sampling z from N(0,1)
        batch = state.shape[0]
        z = torch.randn(batch * n_samples, self.vae.mu.out_features, device=state.device)
        s_rep = state.unsqueeze(1).repeat(1, n_samples, 1).reshape(batch * n_samples, -1)
        decoded = self.vae.decode(s_rep, z)
        decoded = decoded.reshape(batch, n_samples, -1)

        # perturb
        perturbed = []
        for i in range(n_samples):
            a = decoded[:, i, :]
            perturbed.append(self.perturb(state, a).unsqueeze(1))
        cand = torch.cat(perturbed, dim=1)  # [B, n_samples, A]

        # score with Q and pick best
        q_vals = self.Q(state.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, state.shape[1]), cand.reshape(-1, cand.shape[-1]))
        q_vals = q_vals.reshape(batch, n_samples)
        idx = q_vals.argmax(dim=1)
        return cand[torch.arange(batch), idx]


if __name__ == "__main__":
    print("bcq_implementation module: define BCQ components for notebooks.")
