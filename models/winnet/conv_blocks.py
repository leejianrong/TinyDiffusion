import torch
from torch import nn, Tensor
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A small CNN denoising block: Conv -> Conv -> SiLU
    """

    def __init__(
        self, 
        in_ch: int, 
        out_ch: int,
        hidden_ch: int = 32, 
        kernel_size: int = 3

    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch,
                               kernel_size, padding="same")
        self.conv2 = nn.Conv2d(hidden_ch, out_ch,
                               kernel_size, padding="same")

    def forward(
        self,
        x: Tensor,
        sigma: Tensor = None,
    ) -> Tensor:
        """
        Args
        ----
        x     : (B, in_ch, H, W)  batch of noisy images
        sigma : (B,) | (B,1) | (B,1,1,1)  noise std-dev for *each* image

        Returns
        -------
        (B, out_ch, H, W)  predicted clean images
        """
        h = self.conv1(x)
        return F.silu(self.conv2(h))


class ResBlock(nn.Module):
    """
    Standard 2-conv residual block with SiLU activations.

           x ──► Conv ─► SiLU ─► Conv ─► + ─► SiLU(out)
             ╰───────────────────────────╯
    If `in_ch != out_ch` or `stride != 1`, the skip path is a 1×1 Conv+BN.
    """

    def __init__(
        self, 
        in_ch: int, 
        out_ch: int,
        hidden_ch: int = 32,
        kernel_size: int = 3,
        stride: int = 1, 
        bias: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size,
                               stride=stride, padding='same', bias=bias)

        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=kernel_size,
                               stride=stride, padding='same', bias=bias)

        # Skip connection
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=bias),
            )
        else:
            self.skip = nn.Identity()

    # ---------------------------------------------------------------------
    def forward(
        self, 
        x: Tensor, 
        sigma: Tensor = None,
    ) -> Tensor:
        residual = self.skip(x)

        out = F.silu(self.conv1(x))
        out = self.conv2(out)

        # residual addition and final activation
        return F.silu(out + residual)
