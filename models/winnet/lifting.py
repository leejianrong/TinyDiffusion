import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

from .predict_update import PredictUpdateNet

class FixedAnalyticLiftingPair(nn.Module):
    """
    Analytic (non-trainable) lifting pair with perfect reconstruction and
    *identical* spatial sizes for coarse and detail parts.

    • predict(C)  : depth-wise 3×3 mean  ➔ tiled to 9 channels  
    • update(D̂)  : ¼ × Laplacian(D̂)    ➔ averaged back to 3 channels
    """
    def __init__(self):
        super().__init__()

        # 3×3 “plus-shape” mean filter  (sum = 1)
        kernel_p = torch.tensor([[0., 1., 0.],
                                 [1., 1., 1.],
                                 [0., 1., 0.]]) / 5.0
        self.register_buffer('weight_p', kernel_p[None, None])   # (1,1,3,3)

        # 3×3 Laplacian
        kernel_u = torch.tensor([[0., 1., 0.],
                                 [1., -4., 1.],
                                 [0., 1., 0.]])
        self.register_buffer('weight_u', kernel_u[None, None])   # (1,1,3,3)

    # ---------- helper ops ---------- #
    def predict(self, coarse):
        """
        coarse  : (B, 3, H, W)
        returns : (B, 9, H, W)   ---  three copies (LH,HL,HH positions)
        """
        B, Cc, H, W = coarse.shape                # Cc = 3
        pc = F.conv2d(coarse,
                      self.weight_p.repeat(Cc, 1, 1, 1),
                      padding=1,
                      groups=Cc)                  # (B, 3, H, W)
        return pc.repeat_interleave(3, dim=1)     # (B, 9, H, W)

    def update(self, d_hat):
        """
        d_hat   : (B, 9, H, W)
        returns : (B, 3, H, W)  ——  correction to add to coarse
        """
        lap = F.conv2d(d_hat,
                       self.weight_u.repeat(9, 1, 1, 1),
                       padding=1,
                       groups=9)                  # (B, 9, H, W)

        # collapse every (LH,HL,HH) triplet → RGB channels
        B, _, H, W = lap.shape
        lap = lap.reshape(B, 3, 3, H, W).mean(2)  # (B, 3, H, W)

        return 0.25 * lap                         # scale factor ¼

    # ---------- lifting pair ---------- #
    def forward(self, coarse, detail, sigma=None):
        """
        coarse : (B, 3, H, W)
        detail : (B, 9, H, W)
        """
        d_hat = detail - self.predict(coarse)
        c_hat = coarse + self.update(d_hat)
        return c_hat, d_hat

    def inverse(self, coarse_hat, detail_hat, sigma=None):
        """
        Perfect-reconstruction inverse.
        """
        coarse = coarse_hat - self.update(detail_hat)
        detail = detail_hat + self.predict(coarse)
        return coarse, detail


class FiLM(nn.Module):
    """Maps scalar sigma to per-channel (gamma, beta) gains."""
    def __init__(self, n_ch, hidden_ch=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, 2 * n_ch),
        )

    def forward(self, sigma):
        sigma = sigma.view(-1, 1)
        sigma = sigma.to(self.mlp[0].weight.dtype)
        gamma, beta = self.mlp(sigma).chunk(2, dim=1)
        return gamma[:, :, None, None], beta[:, :, None, None]


class ConvFiLMBlock(nn.Module):
    """Conv -> GN -> FiLM(sigma) -> SiLU"""
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(groups, out_ch, affine=False)
        self.film = FiLM(out_ch)

    def forward(self, x, sigma):
        gamma, beta = self.film(sigma)
        x = self.conv(x)
        x = self.gn(x)
        return F.silu(gamma * x + beta)


class Conditioner(nn.Module):
    """Conditioner CNN that uses 3 FiLM conditioned conv layers."""
    def __init__(self, in_ch, out_ch, hidden_ch=96):
        super().__init__()
        self.b1 = ConvFiLMBlock(in_ch, hidden_ch)
        self.b2 = ConvFiLMBlock(hidden_ch, hidden_ch)
        self.b3 = ConvFiLMBlock(hidden_ch, out_ch, groups=out_ch)

    def forward(self, x, sigma):
        x = self.b1(x, sigma)
        x = self.b2(x, sigma)
        return self.b3(x, sigma)


class RevNetLiftingPair(nn.Module):
    """
    Invertible residual pair refining (coarse, detail).
    coarse: (B, C_coarse, H, W). E.g., C_coarse = 3 (LL band).
    detail: (B, C_detail, H, W). E.g., C_detail = 9 (LL + HL + HH)
    sigma: (B, 1). Noise level (FiLM-conditioning)
    """
    def __init__(self, coarse_ch=3, detail_ch=9, hidden_ch=96):
        super().__init__()
        self.f = Conditioner(detail_ch, coarse_ch, hidden_ch) # update coarse
        self.g = Conditioner(coarse_ch, detail_ch, hidden_ch) # update detail

    def forward(self, coarse, detail, sigma):
        y_detail = detail + self.g(coarse, sigma)
        y_coarse = coarse + self.f(y_detail, sigma)
        return y_coarse, y_detail

    def inverse(self, y_coarse, y_detail, sigma):
        coarse = y_coarse - self.f(y_detail, sigma)
        detail = y_detail - self.g(coarse, sigma)
        return coarse, detail


class PredictUpdateLiftingPair(nn.Module):
    """One predict-update pair for the learned lifting scheme."""

    def __init__(
        self, 
        coarse_ch: int,
        detail_ch: int,
        hidden_ch: int,
        kernel_size: int = 3,
        num_predict_update_blocks: int = 1,
        conv_block_choice: str = "soft_threshold_residual",
    ) -> None:
        super().__init__()

        # two direction specific PUNets
        self.predict_net = PredictUpdateNet(
            in_ch=coarse_ch,
            out_ch=detail_ch,
            hidden_ch=hidden_ch,
            kernel_size=kernel_size,
            num_blocks=num_predict_update_blocks,
            conv_block_choice=conv_block_choice,
        )
        self.update_net = PredictUpdateNet(
            in_ch=detail_ch,
            out_ch=coarse_ch,
            hidden_ch=hidden_ch,
            kernel_size=kernel_size,
            num_blocks=num_predict_update_blocks,
            conv_block_choice=conv_block_choice,
        )

    def forward(
        self, 
        coarse: Tensor, 
        detail: Tensor, 
        noise_level: Tensor | float
    ) -> Tuple[Tensor, Tensor, Tensor]:

        predicted_detail = self.predict_net(coarse, noise_level)
        detail = detail - predicted_detail # lifting subtraction

        update_term = self.update_net(detail, noise_level)
        coarse = coarse + update_term # lifting addition

        return coarse, detail

    def inverse(
        self, 
        coarse: Tensor, 
        detail: Tensor, 
        noise_level: Tensor | float
    ) -> Tuple[Tensor, Tensor, Tensor]:

        update_term = self.update_net(detail, noise_level)
        coarse = coarse - update_term

        predicted_detail = self.predict_net(coarse, noise_level)
        detail = detail + predicted_detail

        return coarse, detail

















































