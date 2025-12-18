from typing import Tuple

import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """Nature CNN encoder (Atari-style) producing 512-d features.

    Input expects (B, C=4, H=84, W=84) float tensors in [0,1] or [0,255].
    """

    def __init__(self, in_channels: int = 4, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
        )

        # optional projection
        if out_dim != 512:
            self.proj = nn.Linear(512, out_dim)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        if self.proj is not None:
            h = self.proj(h)
        return h


class ParticleHead(nn.Module):
    """Maps encoder features to per-action particle clouds.

    Output shape: (B, num_actions, num_particles, particle_dim)
    particle_dim defaults to 1 for scalar returns but can be >1 for multi-objective.
    """

    def __init__(
        self,
        in_dim: int,
        num_actions: int,
        num_particles: int = 128,
        particle_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_particles = num_particles
        self.particle_dim = particle_dim
        # final linear projects to num_actions * num_particles * particle_dim
        self.fc = nn.Linear(in_dim, num_actions * num_particles * particle_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        out = self.fc(features)
        out = out.view(B, self.num_actions, self.num_particles, self.particle_dim)
        return out


class ParticleQNetwork(nn.Module):
    """Combined encoder + particle head producing distributional outputs."""

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        num_particles: int = 128,
        particle_dim: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = NatureCNN(in_channels=in_channels, out_dim=512)
        self.head = ParticleHead(
            in_dim=512,
            num_actions=num_actions,
            num_particles=num_particles,
            particle_dim=particle_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        particles = self.head(feats)
        return particles


if __name__ == "__main__":
    net = ParticleQNetwork(in_channels=4, num_actions=6, num_particles=16)
    x = torch.randn(2, 4, 84, 84)
    out = net(x)
    print(out.shape)  # (B, A, N, D)















