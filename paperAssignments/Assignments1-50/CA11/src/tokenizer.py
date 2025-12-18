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
        flat = z.reshape(-1, D)  # (B*L, D)
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


class ImageVQVAE(nn.Module):
    """
    VQ-VAE for images that produces sequence token embeddings.
    Encoder downsamples the image and projects to `d_model` channels; the spatial
    grid is flattened to a sequence of length L = H'*W' with embeddings size d_model.
    """

    def __init__(
        self, codebook_size: int, d_model: int, in_ch: int = 3, hidden: int = 128
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        # encoder: images (B, C, H, W) -> (B, d_model, H', W')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(hidden, d_model, kernel_size=3, padding=1),
        )
        # decoder: (B, d_model, H', W') -> (B, C, H, W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                d_model, hidden, kernel_size=4, stride=2, padding=1
            ),  # x2
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden, hidden, kernel_size=4, stride=2, padding=1
            ),  # x2
            nn.ReLU(),
            nn.Conv2d(hidden, in_ch, kernel_size=3, padding=1),
        )
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model) * 0.1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W)
        z = self.encoder(images)  # (B, D, H', W')
        B, D, Hp, Wp = z.shape
        seq = z.permute(0, 2, 3, 1).contiguous().view(B, Hp * Wp, D)  # (B, L, D)
        return seq

    def quantize(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = seq.shape
        flat = seq.reshape(-1, D)
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1).unsqueeze(0)
        )
        indices = torch.argmin(dists, dim=1)
        codes = self.codebook[indices].view(B, L, D)
        quantized = codes + (seq - seq.detach())
        return quantized, indices.view(B, L)

    def decode_from_codes(self, codes: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        # codes: (B, L, D) where L = Hp*Wp
        B, L, D = codes.shape
        x = codes.view(B, Hp, Wp, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H', W')
        recon = self.decoder(x)  # (B, C, H, W)
        return recon

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns recon_images, quantized_seq (B,L,D), indices (B,L)
        """
        B, C, H, W = images.shape
        seq = self.encode(images)
        # compute H', W' from encoder output
        # assume downsample by 4 (two stride-2 convs)
        Hp = H // 4
        Wp = W // 4
        quantized, indices = self.quantize(seq)
        recon = self.decode_from_codes(quantized, Hp, Wp)
        return recon, quantized, indices
