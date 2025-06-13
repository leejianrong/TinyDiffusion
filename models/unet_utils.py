"""
unet_utils.py - Building blocks shared by the diffusion UNet
"""

# ──────────────────────────────────────────────────────────────────────────────
# Std-lib & third-party imports
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat  # make sure einops is installed
from einops.layers.torch import Rearrange
from .attend import Attend
from .helpers import exists, default, divisible_by


# ──────────────────────────────────────────────────────────────────────────────
# Small NN layers
# ──────────────────────────────────────────────────────────────────────────────


def Upsample(in_channels: int, out_channels: int | None = None) -> nn.Sequential:
    """
    ×2 nearest-neighbour upsampling followed by 3×3 convolution.
    """
    out_channels = default(out_channels, in_channels)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    )


def Downsample(in_channels: int, out_channels: int | None = None) -> nn.Sequential:
    """
    2×2 spatial unshuffle (space-to-depth) then 1×1 convolution.
    """
    out_channels = default(out_channels, in_channels)
    return nn.Sequential(
        # rearrange H×W into (H/2)×(W/2) with 4× channels
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
    )


class RMSNorm(nn.Module):
    """RMS LayerNorm, eq. 4 from *Zhang et al. 2022* (scale only)."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels**0.5
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        normalized = F.normalize(x, dim=1)  # unit RMS along channel dim
        return normalized * self.gain * self.scale


# ──────────────────────────────────────────────────────────────────────────────
# ✧ Positional embeddings
# ──────────────────────────────────────────────────────────────────────────────


class SinusoidalPosEmb(nn.Module):
    """
    Classic fixed sinusoidal position embedding (Vaswani et al., 2017).

    Generates a ``(B, dim)`` embedding for an input **1-D** tensor of shape ``(B,)``.
    """

    def __init__(self, dim: int, *, theta: int = 10_000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time: torch.Tensor) -> torch.Tensor:  # shape (B,)
        device = time.device
        half = self.dim // 2

        # Compute the 1 / θ^(2i/d) term once
        exponent = torch.exp(
            torch.arange(half, device=device) * -(math.log(self.theta) / (half - 1))
        )
        angles = time[:, None] * exponent[None, :]  # (B, half)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Fourier feature time embedding (learned OR random, à la *Daniel et al.*).

    When ``is_random=False`` the frequencies are learned (gradient flow);
    when ``True`` they are fixed random draws.
    """

    def __init__(self, dim: int, *, is_random: bool = False):
        super().__init__()
        assert divisible_by(dim, 2), "`dim` must be even"
        half = dim // 2
        self.frequencies = nn.Parameter(
            torch.randn(half), requires_grad=not is_random
        )  # (half,)
        self.register_buffer("two_pi", torch.tensor(2 * math.pi), persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:  # (B,)
        time = rearrange(time, "b -> b 1")  # (B,1)
        freqs = time * self.frequencies[None, :] * self.two_pi  # (B, half)
        fourier = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return torch.cat([time, fourier], dim=-1)  # prepend original t


# ──────────────────────────────────────────────────────────────────────────────
# Core network blocks
# ──────────────────────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """
    Conv → RMSNorm → SiLU → Dropout

    Optional FiLM-style scale/shift modulation (from a time embedding).
    """

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = RMSNorm(out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # FiLM modulation

        return self.dropout(self.act(x))


class ResnetBlock(nn.Module):
    """
    Two -> ConvBlocks with an optional FiLM from *time_emb* and a skip-connection.

    If ``in_channels != out_channels`` a 1×1 Conv skip adapts dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_emb_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.block2 = ConvBlock(out_channels, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            # FiLM modulation from time embedding
            time_vec = self.time_mlp(time_emb)  # (B, 2*out_channels)
            time_vec = rearrange(time_vec, "b c -> b c 1 1")
            scale_shift = time_vec.chunk(2, dim=1)  # (scale, shift)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.skip(x)


# ──────────────────────────────────────────────────────────────────────────────
# Attention blocks
# ──────────────────────────────────────────────────────────────────────────────


class LinearAttention(nn.Module):
    """
    Memory-efficient linear attention (softmax along **one** axis).

    Includes a small bank of learned *num_mem_kv* global keys/values.
    """

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
    ):
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head
        self.scale = dim_head**-0.5

        self.norm = RMSNorm(channels)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))

        self.to_qkv = nn.Conv2d(channels, hidden * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, channels, kernel_size=1), RMSNorm(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # each (B, hidden, H, W)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h c) x y -> b h c (x y)", h=self.heads
            ),
            (q, k, v),
        )

        # Add learned global memory
        mem_k, mem_v = map(lambda t: repeat(t, "h c n -> b h c n", b=batch), self.mem_kv)
        k, v = torch.cat([mem_k, k], dim=-1), torch.cat([mem_v, v], dim=-1)

        # Linear attention: softmax(q) along *tokens*, softmax(k) along *keys*
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        # Context = KVᵀ  (einsum notation for clarity)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        out = rearrange(out, "b h c (x y) -> b (h c) x y", x=height, y=width)
        return self.to_out(out)


class Attention(nn.Module):
    """
    Full (quadratic) attention with optional FlashAttention backend.
    """

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        flash: bool = False,
    ):
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head

        self.norm = RMSNorm(channels)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(channels, hidden * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads),
            (q, k, v),
        )

        # Append learned global memory tokens
        mem_k, mem_v = map(lambda t: repeat(t, "h n d -> b h n d", b=batch), self.mem_kv)
        k, v = torch.cat([mem_k, k], dim=-2), torch.cat([mem_v, v], dim=-2)

        out = self.attend(q, k, v)  # (B, H, N, D)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=height, y=width)
        return self.to_out(out)
