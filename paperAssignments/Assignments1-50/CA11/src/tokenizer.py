from typing import Tuple
import torch
import torch.nn as nn


class SimpleVQVAE(nn.Module):
    """
    Small vector-quantized variational autoencoder working on sequence embeddings.
    Encoder/decoder use 1D convs over the sequence dimension to produce/reconstruct
    (B, L, D) tensors. Codebook quantization is applied per-position.
    """

    def __init__(self, codebook_size: int, d_model: int, hidden: int = 128):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        # encoder: (B, L, D) -> (B, L, D)
        self.encoder = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, d_model, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, d_model, kernel_size=3, padding=1),
        )
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model) * 0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D)
        Returns:
            recon: (B, L, D)
            indices: (B, L) int tensor of code indices
        """
        B, L, D = x.shape
        # conv1d expects (B, C, L)
        z = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # (B, L, D)
        flat = z.view(-1, D)  # (B*L, D)
        # compute squared distances to codebook
        # (N, K) = (B*L, codebook_size)
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1).unsqueeze(0)
        )
        indices = torch.argmin(dists, dim=1)  # (B*L,)
        codes = self.codebook[indices].view(B, L, D)
        # straight-through estimation
        quantized = codes + (z - z.detach())
        recon = self.decoder(quantized.transpose(1, 2)).transpose(1, 2)
        return recon, indices.view(B, L)

    def decode_codes(self, indices: torch.Tensor) -> torch.Tensor:
        B, L = indices.shape
        codes = self.codebook[indices.view(-1)].view(B, L, self.d_model)
        return codes
