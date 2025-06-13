"""clista_denoise.py
The CLISTADenoise block consists of an analysis convolution (A), a stack of shared 
``SoftThresholdActivation`` non-linearities, and a synthesis convolution (S).
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .soft_threshold import SoftThresholdActivation

class SiLUResBlock(nn.Module):
    """
    Standard 2-conv residual block with SiLU activations.

           x ──► Conv ─► BN ─► SiLU ─► Conv ─► BN ─► + ─► SiLU(out)
             ╰────────────────────────────────────────╯
    If `in_ch != out_ch` or `stride != 1`, the skip path is a 1×1 Conv+BN.

    Parameters
    ----------
    in_ch   : input channel count
    out_ch  : output channel count
    stride  : spatial stride for the first conv (supports downsampling)
    bias    : whether the convolutions use bias terms (default False
              because BatchNorm has affine parameters)
    """

    def __init__(
        self, 
        in_ch: int, 
        out_ch: int,
        kernel_size: int = 3, 
        stride: int = 1, 
        bias: bool = False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                               stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size,
                               stride=1, padding=1, bias=bias)
        # Skip connection
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(
                in_ch, 
                out_ch, 
                kernel_size=1,
                stride=stride, 
                bias=bias
            )
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU(inplace=True)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.act(self.conv1(x))
        out = self.conv2(out)

        # residual addition and final activation
        return self.act(out + residual)


class CLISTADenoise(nn.Module):
    """CLISTADenoise block with shared parameters along T iterations.
    CLISTA stands for Convolutional Learned Iterative Shrinkage and Thresholding Algorithm.
    
    Parameters
    ----------
    latent_channels:
        Number of channels inside the PUNet latent space (called p in the original paper).
    detail_channels:
        Number of channels of the detail representation that the block takes as inputs and outputs.
    kernel_size:
        Spatial size k of both analysis and synthesis kernels (k x k)
    num_iters:
        Number of Learned Iterative Shrinkage and Thresholding Algorithm (LISTA) iterations T.
        Corresponds to the number of sets of soft-threshold that are stored.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        detail_channels: int = 15,
        kernel_size: int = 3,
        num_iters: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_iters = num_iters

        # Soft threshold activations
        # each holds a set of theta parameters
        self.soft_thresholds = nn.ModuleList([
            SoftThresholdActivation(latent_channels) for _ in range(num_iters)
        ])

        # Res Blocks
        self.analysis_conv = SiLUResBlock(
            in_ch=detail_channels, 
            out_ch=latent_channels, 
            kernel_size=kernel_size,
        )
        self.synthesis_conv = SiLUResBlock(
            in_ch=latent_channels,
            out_ch=detail_channels,
            kernel_size=kernel_size,
        )

    def forward(self, detail: Tensor, noise_level: Tensor) -> Tensor:
        """Apply T LISTA iterations and return the reconstructed detail."""
        latent = self.analysis_conv(detail)

        for soft_th in self.soft_thresholds:
            residual = detail - self.synthesis_conv(latent)
            feedback = self.analysis_conv(residual)
            latent = soft_th(latent + feedback, noise_level)

        reconstructed = self.synthesis_conv(latent)
        return reconstructed

    def orthogonality_loss(self) -> Tensor:
        """
        Let A ≔ first conv weight of analysis block  (shape: C_lat × C_det × k × k)
            S ≔ first conv weight of synthesis block (shape: C_det × C_lat × k × k)

        We flatten each kernel to a row vector, so that
            A_flat ≔ A.view(C_lat, -1),  S_flat ≔ S.view(C_det, -1).

        For an ideal orthogonal pair   S_flat @ A_flatᵀ = I_(C_det).
        We therefore minimise  ‖S_flat @ A_flatᵀ − I‖²_F.
        """
        # NB: each SiLUResBlock begins with a conv named `conv1`
        Wa = self.analysis_conv.conv1.weight
        Ws = self.synthesis_conv.conv1.weight

        Ca, Cd, k, _ = Wa.shape  # Ca=latent_channels, Cd=detail_channels
        Wa_flat = Wa.view(Ca, -1)               #  Ca × (Cd·k²)
        Ws_flat = Ws.view(Cd, -1)               #  Cd × (Ca·k²)

        # project both onto unit-norm rows (optional but stabilises training)
        Wa_flat = F.normalize(Wa_flat, dim=1)
        Ws_flat = F.normalize(Ws_flat, dim=1)

        eye = torch.eye(Cd, device=Wa.device, dtype=Wa.dtype)
        prod = Ws_flat @ Wa_flat.T              # Cd × Cd

        return ((prod - eye) ** 2).mean()





































        