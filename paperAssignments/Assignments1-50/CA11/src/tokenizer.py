from typing import Tuple
import torch
import torch.nn as nn


class SimpleVQVAE(nn.Module):
    """
    Lightweight vector-quantized module that maps continuous embeddings to a codebook.
    This is intentionally simple and works on (B, L, D) tensors by quantizing feature vectors.
    """

    def __init__(self, codebook_size: int, d_model: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model))
        # small encoder/decoder (identity projections to keep import-safe)
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D)
        Returns:
            quantized: (B, L, D) quantized embeddings (stop-grad applied to codebook usage)
            indices: (B, L) int indices of selected codes
        """
        z = self.encoder(x)  # (B, L, D)
        # compute distances to codebook
        B, L, D = z.shape
        flat = z.view(-1, D)  # (B*L, D)
        # distances: (N, K)
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1).unsqueeze(0)
        )
        indices = torch.argmin(dists, dim=1)
        codes = self.codebook[indices].view(B, L, D)
        # straight-through estimator
        quantized = codes + (z - z.detach())
        recon = self.decoder(quantized)
        return recon, indices.view(B, L)

    def decode_codes(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, L)
        B, L = indices.shape
        codes = self.codebook[indices.view(-1)].view(B, L, self.d_model)
        return codes
