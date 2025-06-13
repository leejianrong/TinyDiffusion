import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS LayerNorm, eq. 4 from *Zhang et al. 2022* (scale only)."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels**0.5
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        normalized = F.normalize(x, dim=1)  # unit RMS along channel dim
        return normalized * self.gain * self.scale
        

class SoftThresholdActivation(nn.Module):
    """Channel-wise soft-thresholding whose threshold scales with the noise level.
    
    A learnable positive threshold tau_c is maintained per output channel c.
    During the forward pass the threshold is multiplied by the input noise level
    so that the amount of shrinkage adapts to the noise in the current sample.

    Mathematically the operation is y = sign(x) * max(|x| - tau, 0)

    where tau = softplus(raw_tau) * noise_level / 50.
    """

    def __init__(
        self, 
        num_ch: int, 
        softplus_beta: float = 20.0, 
    ) -> None:
        super().__init__()

        # we keep a log-threshold that is allowed to be negative. Passing it
        # through a high-beta Softplus guarantees a positive threshold while
        # behaving almost like ReLU at the origin.
        self._raw_threshold = nn.Parameter(-0.1 * torch.rand(num_ch))

        # softplus beta is a constant used to approximate ReLU with Softplus 
        # while keeping the derivative finite.
        self._softplus = nn.Softplus(beta=softplus_beta)


    def forward(self, activation: Tensor, noise_level: Tensor | float) -> Tensor:
        """Apply noise-scaled soft-thresholding to activation."""
        # Ensure noise_level is a tensor for broadcasting.
        if not torch.is_tensor(noise_level):
            noise_level = torch.tensor(noise_level, dtype=activation.dtype, device=activation.device)
            
        # positive threshold per channel
        threshold = self._softplus(self._raw_threshold)

        if noise_level.ndim < activation.ndim:
            # note: activation is (B, C, H, W) and threshold is (1, C, 1, 1)
            # reshape noise_level to (B, 1, 1, 1) to allow broadcasting
            noise_level = noise_level.view(-1, 1, 1, 1)

        # reshape for broadcasting and scale with the current noise level
        threshold = threshold.view(1, -1, 1, 1) * noise_level

        # soft-thresholding (a.k.a. shrinkage operator)
        mag = torch.abs(activation) # |x|
        mag.sub_(threshold)         # mag = mag - tau
        torch.relu_(mag)            # max(..., 0) in-place
        return torch.sign(activation) * mag


# class SoftThresholdActivation(nn.Module):
#     def __init__(self, num_ch):
#         super().__init__()
#         self.num_ch = num_ch
#         # One learnable θ per hidden channel.  Shape = (1,C,1,1) so it
#         # broadcasts over batch & spatial dimensions.
#         self.theta = nn.Parameter(torch.ones(1, num_ch, 1, 1))

#     def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
#         # λ = σ * θ  (broadcast to (B, hidden_ch, H, W))
#         lam = sigma.view(-1, 1, 1, 1) * self.theta
#         return torch.sign(x) * F.relu(torch.abs(x) - lam)
        

class SoftThresholdConv(nn.Module):
    """
    A small CNN denoising block: Conv ➔ soft-threshold ➔ Conv,
    with per-channel learnable thresholds modulated by the noise level σ.
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

        self.activation = SoftThresholdActivation(hidden_ch)

    def forward(self,
                x: Tensor,
                sigma: Tensor) -> Tensor:
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
        h = self.activation(h, sigma)
        return self.conv2(h)


class SoftThresholdResBlock(nn.Module):
    """
    Residual denoising block:
        x ──► Conv ─► RMSNorm ─► Soft-threshold(σ) ─► Conv ─► + ─► out
          ╰───────────────────────────────────────────────────╯
    The soft-threshold λ = θ * σ, with an independent learnable θ per
    hidden channel.  The block accepts (noisy_batch, sigma_batch) and
    returns a denoised batch of the same spatial size.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int = 32,
        kernel_size: int = 3,
        *,
        bias: bool = True,
    ):
        super().__init__()

        # main path
        self.conv1 = nn.Conv2d(
            in_ch, hidden_ch, kernel_size, padding="same", bias=bias
        )
        self.conv2 = nn.Conv2d(
            hidden_ch, out_ch, kernel_size, padding="same", bias=bias
        )
        self.norm1 = RMSNorm(channels=hidden_ch)
        self.norm2 = RMSNorm(channels=out_ch)

        self.soft_th1 = SoftThresholdActivation(hidden_ch)
        self.soft_th2 = SoftThresholdActivation(out_ch)

        if in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        x: (B, in_ch, H, W) noisy images
        sigma: (B,) | (B,1) | (B,1,1,1) noise level for each image

        Returns
        -------
        (B, out_ch, H, W) denoised images
        """
        # main branch
        h = self.norm1(self.conv1(x))
        h = self.soft_th1(h, sigma)
        h = self.norm2(self.conv2(h))

        # residual addition + soft threshold non-linearity
        out = h + self.skip(x)
        return self.soft_th2(out, sigma)