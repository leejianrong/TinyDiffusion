import torch
import warnings
from torch import Tensor, nn


from models.winnet.splitmerge import (
    HaarSplitMerge, 
    DCTSplitMerge, 
    StationaryHaarSplitMerge,
    MultiScaleStationaryHaar,
    LearnableHaarSplitMerge,
)
from models.winnet.lifting import FixedAnalyticLiftingPair, RevNetLiftingPair, PredictUpdateLiftingPair
from models.winnet.clista_denoise import CLISTADenoise
from models.unet import UNet


def invalid_choice_message(module_type, invalid_input, choices) -> str:
    msg = f"Unknown {module_type} '{invalid_input}'. Choices: {list(choices)}."
    return msg


class LiftingDenoiser(nn.Module):
    """
    Splitting/Merging using Haar wavelet transform,
    followed by lifting step and CLISTA/UNet denoiser.
    """
    LIFTING_TYPE_CHOICES = {"revnet", "soft_threshold", "fixed"}
    DETAIL_DENOISER_CHOICES = {"clista", "unet"}
    SPLIT_MERGE_CHOICES = {
        "haar", 
        "haar_redundant", 
        "dct",
        "learnable",  
    }
    
    def __init__(
        self,
        input_channels: int = 3,
        coarse_channels: int = 3,
        hidden_channels: int = 128,
        num_lifting_steps: int = 4,
        lifting_type: str = "revnet",
        detail_denoiser: str = "clista",
        split_merge_type: str = "haar",
        do_convert_t_to_sigma: bool = False,
        dct_patch_size: int | None = None, 
        num_haar_scales: int | None = None,
        num_learnable_filters: int | None = None,
        **detail_denoiser_kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = input_channels
        if coarse_channels % input_channels == 0:
            self.coarse_channels = coarse_channels
        else:
            warnings.warn(
                f"Number of coarse channels must be multiple of input channels. " 
                f"Defaulting to `coarse_channels=input_channels={input_channels}`"
            )
            self.coarse_channels = input_channels

        self.coarse_to_in_ch_ratio = self.coarse_channels // self.in_channels        
        self.hidden_channels = hidden_channels
        self.num_haar_scales = num_haar_scales
        self.num_learnable_filters = num_learnable_filters
            
        # if do_convert_t_to_sigma is true the `self.sigma` tensor will be 
        # populated after beta values are calculated by the 
        # GaussianDiffusion wrapper class.
        self.do_convert_t_to_sigma = do_convert_t_to_sigma
        self.register_buffer("sigmas", None, persistent=False)  

        # SplitMerge Operator ---------------------------------------------------------------------
 
        # The splitter outputs (coarse, detail) and we can invert them back to x
        self.split_merge_type = split_merge_type
        if self.split_merge_type == "haar":
            if self.coarse_channels != self.in_channels:
                warnings.warn(
                    "Number of coarse channels must equal to input channels when decimated haar "
                    "operator is used. Defaulting to `coarse_channels=in_channels`"
                )
                self.coarse_channels = self.in_channels
            self.split_merge = HaarSplitMerge()
            self.detail_channels = self.coarse_channels * 3

        elif self.split_merge_type == "haar_redundant":
            if self.num_haar_scales is None:
                if self.coarse_channels != self.in_channels:
                    warnings.warn(
                        "Behaviour of redundant haar operator defaults to "
                        "`coarse_channels=in_channels`. To use different number of "
                        "coarse channels, provide an argument for `num_haar_scales.`"
                    )
                    self.coarse_channels = self.in_channels
                self.split_merge = StationaryHaarSplitMerge(
                    in_channels=self.in_channels,
                )
                self.detail_channels = self.coarse_channels * 3
            else:
                self.split_merge = MultiScaleStationaryHaar(
                    num_scales=self.num_haar_scales,
                    coarse_to_in_ch_ratio=self.coarse_to_in_ch_ratio, 
                    in_channels=self.in_channels,
                )
                total_channels = self.in_channels * (self.num_haar_scales * 3 + 1)
                self.detail_channels = total_channels - self.coarse_channels

        elif self.split_merge_type == "dct":
            if dct_patch_size is None:
                raise ValueError(
                    "`dct_patch_size must have integer value when `split_merge_type='dct'`."
                )
            self.split_merge = DCTSplitMerge(
                in_channels=self.in_channels,
                coarse_to_in_ch_ratio=self.coarse_to_in_ch_ratio,
                patch_size=dct_patch_size,
            )
            detail_channels = ((dct_patch_size ** 2) * input_channels) - self.coarse_channels
            self.detail_channels = detail_channels

        elif self.split_merge_type == "learnable":
            if self.num_learnable_filters is None:
                self.num_learnable_filters = 4
                
            self.split_merge = LearnableHaarSplitMerge(
                in_channels=self.in_channels,
                coarse_to_in_ch_ratio=self.coarse_to_in_ch_ratio,
                num_filters=self.num_learnable_filters,
            )
            total_channels = self.num_learnable_filters * self.in_channels
            self.detail_channels = total_channels - self.coarse_channels
            
        else:
            raise ValueError(
                invalid_choice_message(
                    module_type="split merge operator", 
                    invalid_input=split_merge_type, 
                    choices=self.SPLIT_MERGE_CHOICES,
                )
            )

        # Lifting Steps -------------------------------------------------------------------------------
            
        if lifting_type == "revnet":
            self.lifting_steps = nn.ModuleList([
                RevNetLiftingPair(
                    coarse_ch=self.coarse_channels, 
                    detail_ch=self.detail_channels, 
                    hidden_ch=self.hidden_channels
                ) for _ in range(num_lifting_steps)
            ])
        elif lifting_type == "soft_threshold":
            self.lifting_steps = nn.ModuleList([
                PredictUpdateLiftingPair(
                    coarse_ch=self.coarse_channels,
                    detail_ch=self.detail_channels,
                    hidden_ch=self.hidden_channels,
                ) for _ in range(num_lifting_steps)
            ])
        elif lifting_type == "fixed": # use fixed lifting pair
            self.lifting_steps = nn.ModuleList([
                FixedAnalyticLiftingPair() for _ in range(num_lifting_steps)
            ])
        else:
            raise ValueError(
                invalid_choice_message(
                    module_type="lifting type", 
                    invalid_input=lifting_type, 
                    choices=self.LIFTING_TYPE_CHOICES,
                )
            )

        # Denoiser (acts on detailed channels) -------------------------------------------------------

        self.detail_denoiser_type = detail_denoiser
        
        if self.detail_denoiser_type == "clista":
            # **kwargs provides number of `latent_channels`
            self.detail_denoiser = CLISTADenoise(
                detail_channels=self.detail_channels,   # used for CLISTADenoise
                **detail_denoiser_kwargs,
            )
        elif self.detail_denoiser_type == "unet":
            # by default `out_ch = in_ch` for this unet implementation
            # **kwargs provides `base_channels` and `channel_multipliers`
            self.detail_denoiser = UNet(
                in_channels=self.detail_channels, 
                **detail_denoiser_kwargs,
            )
        else:
            raise ValueError(
                invalid_choice_message(
                    module_type="denoiser", 
                    invalid_input=self.detail_denoiser_type, 
                    choices=self.DETAIL_DENOISER_CHOICES,
                )
            )

    def forward(self, x_noisy: Tensor, noise_cond: Tensor) -> Tensor:        
        if self.do_convert_t_to_sigma:
            if self.sigmas is None:
                raise ValueError("sigmas not set. Call self.update_sigmas_t() first.")
            # convert batch of timesteps t to batch of sigma_t
            sigmas_batch = self.sigmas[noise_cond]
        else:
            sigmas_batch = noise_cond

        # analysis
        coarse, detail = self.split_merge(x_noisy)                    
        for lifting_step in self.lifting_steps:
            coarse, detail = lifting_step(coarse, detail, sigmas_batch)
            
        # denoising performed on detailed channels
        if self.detail_denoiser_type == "unet":
            # unets are conditioned on timestep t
            detail = self.detail_denoiser(detail, noise_cond)
        else:
            # condition on noise stddev sigma
            detail = self.detail_denoiser(detail, sigmas_batch)  
        
        # synthesis
        for lifting_step in reversed(self.lifting_steps):
            coarse, detail = lifting_step.inverse(coarse, detail, sigmas_batch)
            
        x_hat = self.split_merge.inverse(coarse, detail)     
        return x_hat

    
    def update_sigmas_t(
        self, 
        sigmas: Tensor, 
        update_statement: None | str = None
    ) -> None:
        """
        Update the `self.sigmas` tensor after they have been
        calculated externally. `sigmas[t]` contains the standard deviation
        of the noise at a particular timestep t.
        """
        self.register_buffer("sigmas", sigmas)
        if update_statement is not None:
            print(update_statement)   

            