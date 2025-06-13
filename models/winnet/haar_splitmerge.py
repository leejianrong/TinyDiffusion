import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .splitmerge_utils import match_spatial_dims, orthonormalize_groupwise

class HaarSplitMerge(nn.Module):
    """
    One-level 2-D orthonormal Haar wavelet transform
    – forward()  : image  →  (coarse, detail)
    – inverse()  : (coarse, detail) → image
    """
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def _haar_coeffs(x):
        """
        x : (B, C, H, W)   even H, W
        returns four sub-bands, each (B, C, H/2, W/2)
        """
        # 2×2 blocks
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        # Orthonormal Haar (divide by 2)
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def forward(self, x):
        """
        Returns
        coarse : (B, 3, H/2, W/2)
        detail : (B, 9, H/2, W/2)   (LH, HL, HH stacked channel-wise)
        """
        ll, lh, hl, hh = self._haar_coeffs(x)
        detail = torch.cat([lh, hl, hh], dim=1)        # 3 × 3  = 9 channels
        return ll, detail

    @staticmethod
    def inverse(coarse, detail):
        """
        Reconstruct image from sub-bands.
        coarse : (B, 3, H/2, W/2)
        detail : (B, 9, H/2, W/2)  channel order MUST match .forward()
        """
        # unpack detail
        lh, hl, hh = torch.chunk(detail, 3, dim=1)

        ll = coarse
        # invert Haar (multiply by 0.5 = divide by 2)
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        # interleave to original resolution
        B, C, H2, W2 = ll.shape
        H, W = H2 * 2, W2 * 2
        out = torch.empty((B, C, H, W), device=ll.device, dtype=ll.dtype)
        out[..., 0::2, 0::2] = x00
        out[..., 0::2, 1::2] = x01
        out[..., 1::2, 0::2] = x10
        out[..., 1::2, 1::2] = x11
        return out


def _haar_filters(device=None, dtype=None):
    """Return the 4 stationary Haar filters of shape (4,1,2,2)."""
    h = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
    g = torch.tensor([0.5, -0.5], device=device, dtype=dtype)
    ll = torch.outer(h, h)
    lh = torch.outer(h, g)
    hl = torch.outer(g, h)
    hh = torch.outer(g, g)
    filt = torch.stack([ll, lh, hl, hh], dim=0)         # (4, 2, 2)
    return filt.unsqueeze(1)                            # (4,1,2,2)


class StationaryHaarSplitMerge(nn.Module):
    """
    One-scale redundant Haar wavelet split / merge (tight frame).

    Parameters
    ----------
    coarse_channels : int
        Number of LL channels to expose as 'coarse' (1 by default).
    in_channels : int
        Image channels (1 = gray, 3 = RGB, …).
    """

    def __init__(
        self, 
        in_channels: int = 1
    ):
        super().__init__()
        coarse_to_in_ch_ratio = 1
        self.h = coarse_to_in_ch_ratio
        self.in_ch = in_channels

        filt = _haar_filters()
        if in_channels > 1:
            filt = filt.repeat(in_channels, 1, 1, 1)    # grouped
        self.register_buffer("analysis_filters", filt)
        self.pad = 1  # SAME padding for 2×2 filters

    # ---------- analysis ----------
    def forward(self, x):
        """
        Split into (x_coarse, x_detail).

        Shapes
        ------
        x              : (B, C, H, W)
        x_coarse       : (B, h*C, H, W)
        x_detail       : (B, (4-h)*C, H, W)
        """
        B, C, H, W = x.shape
        if C != self.in_ch:
            raise ValueError(f"Expected {self.in_ch} channels, got {C}.")
        coeffs = F.conv2d(
            x,
            self.analysis_filters,
            padding=self.pad,
            stride=1,
            groups=self.in_ch,
        )
        x_coarse, x_detail = coeffs[:, : self.h * C], coeffs[:, self.h * C :]
        x_coarse = match_spatial_dims(x_coarse, x)
        x_detail = match_spatial_dims(x_detail, x)
        return x_coarse, x_detail

    # ---------- synthesis ----------
    def inverse(self, x_coarse, x_detail):
        coeffs = torch.cat([x_coarse, x_detail], dim=1)
        x_rec = F.conv_transpose2d(
            coeffs,
            self.analysis_filters,
            padding=self.pad,
            stride=1,
            groups=self.in_ch,
        )
        x_rec = match_spatial_dims(x_rec, x_coarse)
        return x_rec


class MultiScaleStationaryHaar(nn.Module):
    def __init__(
        self, 
        num_scales: int = 1, 
        in_channels: int = 1, 
        coarse_to_in_ch_ratio: int = 1
    ):
        super().__init__()
        assert num_scales >= 1
        self.num_scales = num_scales
        self.in_ch = in_channels
        self.h = coarse_to_in_ch_ratio

        base = _haar_filters()
        self.register_buffer("base_filters",
                             base.repeat(in_channels, 1, 1, 1))  # (4C,1,2,2)
        self.pad = 1   # base padding for 2×2 filters

    # ------------ analysis -------------------------------------------------
    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.in_ch:
            raise ValueError(f"expected {self.in_ch} channels, got {C}")

        details = []
        current = x

        for j in range(1, self.num_scales + 1):
            d = 2 ** (j - 1)                 # dilation
            LL, LH, HL, HH = self._analysis_once(current, d)
            details.append(torch.cat([LH, HL, HH], dim=1))
            current = LL                     # recurse on LL

        coeffs = torch.cat([current] + details[::-1], dim=1)
        x_coarse, x_detail = coeffs[:, : self.h * C], coeffs[:, self.h * C :]
        return x_coarse, x_detail

    # ------------ synthesis ------------------------------------------------
    def inverse(self, x_coarse, x_detail):
        coeffs = torch.cat([x_coarse, x_detail], dim=1)
        splits = [self.in_ch] + [3 * self.in_ch] * self.num_scales
        LLs_and_details = list(torch.split(coeffs, splits, dim=1))

        current = LLs_and_details[0]         # LL^num_scales
        for j in range(self.num_scales, 0, -1):
            d = 2 ** (j - 1)
            details = LLs_and_details[self.num_scales - j + 1]
            LH, HL, HH = torch.split(details, self.in_ch, dim=1)
            current = self._synthesis_once(current, LH, HL, HH, d)

        return current

    # ------------ helpers --------------------------------------------------
    def _analysis_once(self, x, dilation):
        y = F.conv2d(
            x, self.base_filters,
            padding=self.pad * dilation,
            dilation=dilation,
            groups=self.in_ch,
        )
        y = match_spatial_dims(y, x)                 # keep size
        C = self.in_ch
        return torch.split(y, C, dim=1)      # LL,LH,HL,HH

    def _synthesis_once(self, LL, LH, HL, HH, dilation):
        
        coeffs = torch.cat([LL, LH, HL, HH], dim=1)
        y = F.conv_transpose2d(
            coeffs, self.base_filters,
            padding=self.pad * dilation,
            dilation=dilation,
            groups=self.in_ch,
        )
        y = match_spatial_dims(y, LL)                # keep size
        return y


class LearnableHaarSplitMerge(nn.Module):
    """
    Learnable over-complete split / merge layer.

    Parameters
    ----------
    in_channels : int
        # image channels (1=gray, 3=RGB, …).
    num_filters : int
        # analysis filters **per input channel** (≥4).
    coarse_to_in_ch_ratio : int
        How many of the num_filters filters should be treated as "coarse".
        If you want only the first Haar LL as coarse, set coarse_to_in_ch_ratio=1.
    kernel_size : int
        Either 2 (Haar‐compatible) or 3,5,… (learned filters can be larger).
    """
    def __init__(
        self,     
        in_channels: int = 1,
        num_filters: int = 16,
        coarse_to_in_ch_ratio: int = 4,
        kernel_size: int = 2
    ):
        super().__init__()
        assert num_filters >= 4, "need at least the 4 Haar kernels"
        assert 1 <= coarse_to_in_ch_ratio <= num_filters, "coarse_to_in_ch_ratio must be ≤ num_filters"

        self.in_ch = in_channels
        self.num_filters = num_filters
        self.coarse_to_in_ch_ratio = coarse_to_in_ch_ratio
        self.pad = kernel_size // 2            # SAME padding

        # -------- analysis ------------------------------------------------
        self.analysis = nn.Conv2d(in_channels, num_filters * in_channels,
                                  kernel_size=kernel_size,
                                  padding=self.pad,
                                  stride=1,
                                  groups=in_channels,
                                  bias=False)

        # -------- initialise weights --------------------------------------
        with torch.no_grad():
            W = self.analysis.weight.data             # (num_filters*Cin,1,k,k)

            # 1. put Haar in the first 4 filters of EVERY group
            haar = _haar_filters(dtype=W.dtype, device=W.device)          # (4,1,2,2)
            if kernel_size == 2:
                for g in range(in_channels):
                    W[g*num_filters + 0 : g*num_filters + 4].copy_(haar)

            # 2. random orthogonal-ish init for the rest
            if num_filters > 4:
                rest = W.view(in_channels, num_filters, -1)[:, 4:]   # (Cin,num_filters-4, k*k)
                nn.init.orthogonal_(rest)                            # per input ch.

        # -------- tie synthesis to analysis (transpose) -------------------
        self.synthesis = nn.ConvTranspose2d(num_filters * in_channels, in_channels,
                                            kernel_size=kernel_size,
                                            padding=self.pad,
                                            stride=1,
                                            groups=in_channels,
                                            bias=False)
        # weight tying
        self.synthesis.weight = self.analysis.weight

    # ------------------------------------------------------------------ #
    #  forward / inverse                                                 #
    # ------------------------------------------------------------------ #
    def forward(self, x):
        """
        Returns
        -------
        x_coarse : (B, in_ch * coarse_to_in_ch_ratio, H, W)
        x_detail : (B, in_ch * (num_filters - coarse_to_in_ch_ratio), H, W)
        """
        B, C, H, W = x.shape
        if C != self.in_ch:
            raise ValueError(f"expected {self.in_ch} channels, got {C}")

        coeffs = self.analysis(x)  # (B, num_filters*Cin, H, W)
        coeffs = match_spatial_dims(coeffs, x)
        x_coarse = coeffs[:, : self.in_ch * self.coarse_to_in_ch_ratio]
        x_detail = coeffs[:, self.in_ch * self.coarse_to_in_ch_ratio :]
        
        return x_coarse, x_detail

    def inverse(self, x_coarse, x_detail):
        coeffs = torch.cat([x_coarse, x_detail], dim=1)
        x_rec = self.synthesis(coeffs)
        x_rec = match_spatial_dims(x_rec, x_coarse)
        return x_rec

    # ------------------------------------------------------------------ #
    #  optional regulariser for orthogonality (add to loss)              #
    # ------------------------------------------------------------------ #
    def orthogonality_loss(self):
        """
        L2 penalty ‖ W Wᵀ – I ‖₂  per input channel.
        Call it in your training loop and add to total loss.
        """
        W = self.analysis.weight                                   # (num_filters*Cin,1,k,k)
        Cin, num_filters = self.in_ch, self.num_filters
        W = W.view(Cin, num_filters, -1)                                     # (Cin, num_filters, k*k)
        G = torch.matmul(W, W.transpose(2, 1))                     # (Cin, num_filters, num_filters)
        I = torch.eye(num_filters, device=W.device, dtype=W.dtype).expand_as(G)
        return F.mse_loss(G, I)
