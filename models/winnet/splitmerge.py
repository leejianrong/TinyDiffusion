"""
Implementation of Splitting/Merging operator used at the start of 
the Lifting Invertible Neural Network (LINN). The operators take in an 
input and split it into a coarse part and a detail part.

In this file there are these options for operators:
    * DCTSplitMerge
    * CayleySplitMerge (learnable)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .haar_splitmerge import (
    HaarSplitMerge, 
    MultiScaleStationaryHaar,
    StationaryHaarSplitMerge,
    LearnableHaarSplitMerge,
)
from .splitmerge_utils import match_spatial_dims


def _build_dct_filters(p: int, c: int, device=None, dtype=None):
    """
    Create the first `c` orthonormal 2D DCT II basis filters of size pxp.
    Returned tensor has shape (c, 1, p, p) so it can be fed to conv2d.
    """
    filters = torch.zeros((c, p, p), device=device, dtype=dtype)
    idx = 0
    for u in range(p):
        alpha_u = math.sqrt(1 / p) if u == 0 else math.sqrt(2 / p)
        for v in range(p):
            if idx >= c:
                break
            alpha_v = math.sqrt(1 / p) if v == 0 else math.sqrt(2 / p)
            for x in range(p):
                for y in range(p):
                    filters[idx, x, y] = (
                        alpha_u
                        * alpha_v
                        * math.cos(math.pi * (2 * x + 1) * u / (2 * p))
                        * math.cos(math.pi * (2 * y + 1) * v / (2 * p))
                    )
                idx += 1
            if idx >= c:
                break
        # reshape to (c, 1, p, p)
        return filters.unsqueeze(1)


# Discrete Cosine Transform (DCT) ------------------------------------------------

class DCTSplitMerge(nn.Module):
    """
    Invertible splitting / merging block based on the 2D DCT frame.

    Args
    ----
    patch_size : int
        Spatial size p of the square DCT filters.
    out_channels : int
        Number of analysis channels `c` (must be <= p**2)
    coarse_to_in_ch_ratio : int, optional
        Number of channels returned as the coarse part `h`.
    in_channels : int, optional
        Number of channels in the input image (default 1).
    """

    def __init__(
        self,
        in_channels: int = 1,
        coarse_to_in_ch_ratio: int = 1,
        patch_size: int = 4,
        out_channels: int | None = None,
    ):
        super().__init__()
        if out_channels is None:
            # by default, assign out_channels = p^2
            out_channels = patch_size ** 2
            
        if out_channels > patch_size ** 2:
            raise ValueError(
                f"out_channels (={out_channels}) cannot exceed p^2 (= {patch_size**2})."
            )
            
        if coarse_to_in_ch_ratio > out_channels:
            raise ValueError("coarse_to_in_ch_ratio must be less than out_channels.")
        self.p = patch_size
        self.c = out_channels
        self.h = coarse_to_in_ch_ratio
        self.in_ch = in_channels

        # build analysis filters once, register as buffer (not a parameter)
        dct_filters = _build_dct_filters(self.p, self.c)
        if in_channels > 1: # replicate for grouped conv
            dct_filters = dct_filters.repeat(in_channels, 1, 1, 1)
        self.register_buffer("analysis_filters", dct_filters)

        # synthesis filters are identical for a tight/orthonormal frame
        # conv_transpose2d expects (in_channels, out_channels/groups, p, p)
        # which is exactly analysis_filters' shape.
        self.pad = (self.p - 1) // 2 # SAME padding when p is odd

    def forward(self, x):
        """
        Split x into x_coarse, x_detail.

        Parameters
        ----------
        x: Tensor, shape (B, in_ch, H, W)

        Returns
        -------
        x_coarse: Tensor, shape (B, h, H, W)    (if in_channels == 1)
        x_detail: Tensor, shape (B, c-h, H, W)
        For in_channels > 1, the channel dimension is grouped per input channel.
        """
        B, C, H, W = x.shape
        if C != self.in_ch:
            raise ValueError(f"Expected {self.in_ch} input channels, got {C}.")
        f = F.conv2d(
            x, 
            self.analysis_filters,
            padding=self.pad,
            stride=1,
            groups=self.in_ch,
        )
        x_coarse = f[:, : self.h * C, :, :]
        x_detail = f[:, self.h * C :, :, :]

        x_coarse = match_spatial_dims(x_coarse, x)
        x_detail = match_spatial_dims(x_detail, x)
        
        return x_coarse, x_detail

    def inverse(self, x_coarse, x_detail):
        """
        Reconstruct the original tensor from (x_coarse, x_detail).

        Returns 
        -------
        x_reconstructed: Tensor, shape (B, in_channels, H, W)
        """
        f_hat = torch.cat([x_coarse, x_detail], dim=1)
        # conv_transpose2d with the same filters (tight frame) gives identity
        x_rec = F.conv_transpose2d(
            f_hat,
            self.analysis_filters,
            padding=self.pad,
            stride=1,
            groups=self.in_ch,
        )
        x_rec = match_spatial_dims(x_rec, x_coarse)
        return x_rec


# Learnable (Cayley) ---------------------------------------------------------

class CayleySplitMerge(nn.Module):
    """
    Trainable, orthonormal split / merge via a Cayley-parametrised matrix.

    Parameters
    ----------
    in_channels     : int, optional
        Colour planes of the input image (1 = gray, 3 = RGB, …). Default: 1.
    coarse_to_in_ch_ratio : int, optional
        How many low-frequency channels are exposed as the 'coarse' part.
        Must satisfy 1 ≤ coarse_to_in_ch_ratio < p**2.  Default: 1.
    patch_size      : int
        Spatial size `p` of each square kernel.
    
    """

    def __init__(
        self, 
        in_channels: int = 1,
        coarse_to_in_ch_ratio: int = 1,
        patch_size: int = 4, 
        **kwargs,
    ):
        super().__init__()
        self.p  = patch_size
        self.d  = patch_size ** 2                # transform dimension
        self.h  = coarse_to_in_ch_ratio
        self.C  = in_channels
        if not (1 <= self.h < self.d):
            raise ValueError("coarse_to_in_ch_ratio must be in [1, p²-1].")

        # unconstrained parameters ⇒ learnable Θ  (initialised near zero)
        self.theta = nn.Parameter(0.01 * torch.randn(self.d, self.d))

        # SAME padding for stride 1 and odd patch_size
        self.pad = (self.p - 1) // 2

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _cayley_kernel_bank(self, dtype, device):
        """
        Build the p² orthonormal kernels   shape (p², 1, p, p)
        replicated `C` times if in_channels > 1.
        """
        θ     = self.theta.to(dtype=dtype, device=device)
        A     = θ - θ.T                             # skew-symmetric
        I     = torch.eye(self.d, dtype=dtype, device=device)
        # (I + A)^{-1}(I - A)  -- solve is numerically stabler than inverse
        K     = torch.linalg.solve(I + A, I - A)    # (p² × p²)
        kernels = K.reshape(self.d, 1, self.p, self.p).contiguous()

        if self.C > 1:
            kernels = kernels.repeat(self.C, 1, 1, 1)   # grouped conv
        return kernels

    # ------------------------------------------------------------------
    # forward  = analysis / split
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Split `x` into `(x_coarse, x_detail)`.

        Shapes
        ------
        x              : (B, C, H, W)
        x_coarse       : (B, h*C, H, W)
        x_detail       : (B, (p² - h)*C, H, W)
        """
        if x.shape[1] != self.C:
            raise ValueError(f"Expected {self.C} channels, got {x.shape[1]}.")

        W = self._cayley_kernel_bank(x.dtype, x.device)
        coeffs = F.conv2d(
            x,
            W,
            padding=self.pad,
            stride=1,
            groups=self.C,
        )
        x_coarse, x_detail = coeffs[:, : self.h * self.C], coeffs[:, self.h * self.C :]
        x_coarse = match_spatial_dims(x_coarse, x)
        x_detail = match_spatial_dims(x_detail, x)
        return x_coarse, x_detail

    # ------------------------------------------------------------------
    # inverse  = synthesis / merge
    # ------------------------------------------------------------------
    def inverse(self, x_coarse, x_detail):
        """
        Reconstruct the input from `(x_coarse, x_detail)`.
        """
        W = self._cayley_kernel_bank(x_coarse.dtype, x_coarse.device)
        coeffs = torch.cat([x_coarse, x_detail], dim=1)
        x_rec = F.conv_transpose2d(
            coeffs,
            W,
            padding=self.pad,
            stride=1,
            groups=self.C,
        )
        x_rec = match_spatial_dims(x_rec, x_coarse)
        return x_rec


