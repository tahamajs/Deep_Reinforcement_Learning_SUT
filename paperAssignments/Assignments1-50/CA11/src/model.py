from typing import Optional
import math

import torch
import torch.nn as nn


class LinearAttention(nn.Module):
    """Simple linear attention implementation: phi(x) = elu(x)+1"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    @staticmethod
    def feature_map(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # reshape heads
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        qf = self.feature_map(q)
        kf = self.feature_map(k)

        # accumulate S = kf^T @ v  -> (B, H, Dh, Dh) if done naively; use einsum for simplicity
        # We compute context = (qf @ (kf^T @ v)) per head using associative property
        # Compute K^T V -> (B, H, Dh, Dh) by einsum over sequence
        KV = torch.einsum(
            "bhld,bhlv->bhlv", kf, v
        )  # this keeps last dim Dh; it's effectively sum over l
        # Now attention output per position: out_t = qf_t @ KV_t? For linear attention we use global KV
        # Reduce KV over sequence: sum over l of kf[l] * v[l]
        KV_sum = torch.einsum("bhld,bhlv->bhdv", kf, v)  # (B, H, Dh, Dh)
        # Now multiply qf (B,H,L,Dh) with KV_sum (B,H,Dh,Dh) -> (B,H,L,Dh)
        out = torch.einsum("bhld,bhdz->bhlz", qf, KV_sum)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


class MambaBlock(nn.Module):
    """
    Lightweight SSM-inspired recurrent block.
    For clarity and import-safety we implement a simple gated recurrent update
    that mimics selective state updates.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        # Apply per-token gating with residual-like recurrence along sequence dimension.
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            inp = x[:, t, :]
            proposal = torch.tanh(self.linear(inp))
            g = torch.sigmoid(self.gate(inp))
            h = g * h + (1 - g) * proposal
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)


class SSMBlock(nn.Module):
    """
    Simple state-space model block with a diagonal-stable A parameterization.
    Implements h_{t+1} = a * h_t + b * x_t where a is constrained to (-1,1).
    """

    def __init__(self, d_model: int):
        super().__init__()
        # parameterize a via tanh so spectral radius < 1 for stability
        self.logit_a = nn.Parameter(torch.zeros(d_model))
        self.b = nn.Parameter(torch.randn(d_model) * 0.01)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        a = torch.tanh(self.logit_a)  # (D,)
        b = self.b  # (D,)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            inp = self.proj(x[:, t, :])
            h = a * h + b * inp
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)


class SSDHybridBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ssm_cfg: Optional[dict] = None):
        super().__init__()
        # prefer the diagonal-parameterized SSM for stability, fall back to Mamba behavior if needed
        self.ssm = SSMBlock(d_model)
        self.attn = LinearAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x


class TWMSSDModel(nn.Module):
    """
    Minimal end-to-end model that stacks hybrid blocks and produces
    predicted next-step observations and rewards.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 8):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList(
            [SSDHybridBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.pred_obs = nn.Linear(d_model, d_model)
        self.pred_reward = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None):
        """
        Args:
            obs: (B, L, D) input observation tokens/embeddings
            actions: optional (B, L, D) action embeddings to concatenate or add.
        Returns:
            pred_obs: (B, L, D)
            pred_reward: (B, L, 1)
        """
        x = self.embed(obs)
        if actions is not None:
            x = x + actions
        for layer in self.layers:
            x = layer(x)
        return self.pred_obs(x), self.pred_reward(x)


class TWMSSDImageModel(nn.Module):
    """
    Wrapper that accepts images, runs them through an ImageVQVAE to get token embeddings,
    then forwards through the TWMSSDModel backbone.
    """

    def __init__(self, image_vq, backbone: TWMSSDModel):
        super().__init__()
        self.vq = image_vq
        self.backbone = backbone

    def forward(self, images: torch.Tensor, actions: Optional[torch.Tensor] = None):
        # images: (B, C, H, W)
        recon, quantized, indices = self.vq(images)
        # quantized: (B, L, D)
        pred_obs, pred_reward = self.backbone(quantized, actions)
        return pred_obs, pred_reward, recon, indices






