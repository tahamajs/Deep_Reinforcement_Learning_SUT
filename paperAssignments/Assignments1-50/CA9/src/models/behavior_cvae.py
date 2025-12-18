from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, s_dim: int, latent: int, a_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + latent, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim),
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        x = torch.cat([s, z], dim=-1)
        return self.net(x)


class CVAE(nn.Module):
    """A small conditional VAE used as a behavior model to sample candidate actions."""

    def __init__(self, s_dim: int, a_dim: int, latent: int):
        super().__init__()
        self.enc = Encoder(s_dim, a_dim, latent)
        self.dec = Decoder(s_dim, latent, a_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(s, a)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(s, z)
        return recon, mu, logvar

    def sample(self, s: torch.Tensor, num: int = 1) -> torch.Tensor:
        """Sample num actions per state s => [B, num, a_dim]"""
        B = s.shape[0]
        zs = torch.randn(B * num, self.enc.mu.out_features, device=s.device)
        s_rep = s.unsqueeze(1).repeat(1, num, 1).reshape(B * num, -1)
        with torch.no_grad():
            a = self.dec(s_rep, zs).reshape(B, num, -1)
        return a

    def loss(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        recon, mu, logvar = self.forward(s, a)
        recon_loss = F.mse_loss(recon, a, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld



