"""linn.py
Implementation of a Learned Invertible Neural Network (LINN) based on a stack
of lifting steps. Each step alternates between predicting the detail sub-band 
from the coarse sub-band and updating the coarse sub-band using that prediction.
The design mirrors traditional wavelet lifting schemes but the predict/udpate
operators are learned by small CNNs (here, :class:`PUNet`).

External dependencies expected on the Python path:
    * PUNet - see punet.py
    * Splitting operators DCTSplitMerge and CayleySplitMerge - both provide
        forward (analysis) and inverse (synthesis)
"""


from typing import Tuple

import torch
from torch import Tensor, nn

from winnet.splitmerge import DCTSplitMerge, CayleySplitMerge
from winnet.predict_update import PredictUpdateNet


# Single lifting step
class LiftingStep(nn.Module):
    """One predict-update pair for the learned lifting scheme."""

    def __init__(
        self, 
        coarse_channels: int,
        punet_channels: int,
        detail_channels: int,
        kernel_size: int = 3,
        num_predict_update_blocks: int = 4,
        conv_block_choice: str = "soft_threshold",
    ) -> None:
        super().__init__()

        # two direction specific PUNets
        self.predict_net = PredictUpdateNet(
            in_ch=coarse_channels,
            out_ch=detail_channels,
            hidden_ch=punet_channels,
            kernel_size=kernel_size,
            num_blocks=num_predict_update_blocks,
            conv_block_choice=conv_block_choice,
        )
        self.update_net = PredictUpdateNet(
            in_ch=detail_channels,
            out_ch=coarse_channels,
            hidden_ch=punet_channels,
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
        """Apply predict then update; return updated (coarse, detail)"""
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
        """Exact inverse of the forward method."""
        update_term = self.update_net(detail, noise_level)
        coarse = coarse - update_term

        predicted_detail = self.predict_net(coarse, noise_level)
        detail = detail + predicted_detail

        return coarse, detail


# -----------------------------------------------------------------------
# Full LINN - stack of lifting steps with a learnable frame decomposition
# -----------------------------------------------------------------------

class LINN(nn.Module):
    """Learned invertible network with a learnable frame decomposition."""
    FRAME_CHOICES = {"dct": DCTSplitMerge, "cayley": CayleySplitMerge}

    def __init__(
        self,
        input_channels: int,
        coarse_channels: int,
        *,
        split_merge_patch_size: int = 4,
        punet_channels: int = 32,
        kernel_size: int = 3,
        num_lifting_steps: int = 2,
        num_predict_update_blocks: int = 4,
        conv_block_choice: str = "soft_threshold",
        frame_mode: str = "dct",
        device=None
    ) -> None:
        super().__init__()

        # Split / Merge
        detail_channels = ((split_merge_patch_size ** 2) * input_channels) - coarse_channels
        self.detail_channels = detail_channels
        
        split_merge_operator = self.FRAME_CHOICES.get(frame_mode)
        
        if split_merge_operator is None:
            raise ValueError(
                f"Unknown frame_mode '{frame_mode}'. Choices: {list(self.FRAME_CHOICES)}"
            )

        # The splitter outputs (coarse, detail) and we can invert them back to x
        self.split_merge = split_merge_operator(
            in_channels=input_channels,
            coarse_to_in_ch_ratio=(coarse_channels // input_channels),
            patch_size=split_merge_patch_size,
        )

        # Lifting stack
        self.lifting_stack = nn.ModuleList([
            LiftingStep(
                coarse_channels,
                punet_channels,
                detail_channels,
                kernel_size,
                num_predict_update_blocks,
                conv_block_choice,
            )
            for _ in range(num_lifting_steps)
        ])

    
    def forward(
        self, 
        x: Tensor, 
        noise_level: Tensor | float
    ) -> Tuple[Tensor, Tensor]:
        """Decompose x and apply all lifting steps; return all sub-bands."""
        coarse, detail = self.split_merge.forward(x)

        for step in self.lifting_stack:
            coarse, detail = step(coarse, detail, noise_level)

        return coarse, detail
        
    def inverse(
        self, 
        coarse: Tensor, 
        detail: Tensor, 
        noise_level: Tensor | float
    ) -> Tensor:
        """Reverse every lifting step and merge the sub-bands."""
        for step in reversed(self.lifting_stack):
            coarse, detail = step.inverse(coarse, detail, noise_level)

        return self.split_merge.inverse(coarse, detail)










































        