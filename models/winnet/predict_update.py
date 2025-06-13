import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .soft_threshold import SoftThresholdConv, SoftThresholdResBlock
from .conv_blocks import ConvBlock, ResBlock
        
class PredictUpdateNet(nn.Module):
    """Implementation of Predictor / Updater networks in LINN."""
    
    CONV_BLOCK_OPTIONS = {
        "simple_conv": ConvBlock,
        "simple_res": ResBlock,
        "soft_threshold": SoftThresholdConv, 
        "soft_threshold_residual": SoftThresholdResBlock,
    }
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int = 32,
        kernel_size: int = 3,
        num_blocks: int = 4,
        conv_block_choice: str = "soft_threshold",
    ) -> None:
        super().__init__()

        conv_block = self.CONV_BLOCK_OPTIONS.get(conv_block_choice)
        
        if conv_block is None:
            raise ValueError(
                f"Unknown conv block type '{conv_block_choice}'. Choices: {list(self.CONV_BLOCK_OPTIONS)}"
            )

        self.in_conv = conv_block(
            in_ch=in_ch, 
            out_ch=hidden_ch, 
            hidden_ch=hidden_ch, 
            kernel_size=kernel_size,
        )

        self.out_conv = conv_block(
            in_ch=hidden_ch, 
            out_ch=out_ch, 
            hidden_ch=hidden_ch, 
            kernel_size=kernel_size,
        )
        if num_blocks > 2:
            self.hidden_conv_blocks = nn.ModuleList([
                conv_block(
                    in_ch=hidden_ch, 
                    out_ch=hidden_ch, 
                    hidden_ch=hidden_ch, 
                    kernel_size=kernel_size
                ) for _ in range(num_blocks - 2)
            ])
        else:
            self.hidden_conv_blocks = None

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:          
        h = self.in_conv(x, noise_level)
        
        if self.hidden_conv_blocks is not None:
            for hidden_conv_block in self.hidden_conv_blocks:
                h = hidden_conv_block(h, noise_level)
            
        return self.out_conv(h, noise_level)
