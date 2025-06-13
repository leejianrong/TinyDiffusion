import torch
import torch.nn as nn
from functools import partial
from typing import List, Optional, Tuple, Union

from .helpers import cast_tuple
from .unet_utils import ResnetBlock, Attention, LinearAttention, Downsample, Upsample
from .unet_utils import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb

# ──────────────────────────────────────────────────────────────────────────────
# Assume the following helper layers are defined elsewhere in your codebase:
#   • ResnetBlock
#   • Attention (full)            • LinearAttention (memory-efficient)
#   • Downsample / Upsample       • SinusoidalPosEmb
#   • RandomOrLearnedSinusoidalPosEmb
#   • cast_tuple (utility that broadcasts scalars → tuples)
# ──────────────────────────────────────────────────────────────────────────────


class UNet(nn.Module):
    """Generic 2-D U-Net backbone for diffusion models.

    The network follows an encoder–bottleneck–decoder design:

    ┌──────────────┐  (skip)  ┌──────────────┐
    │ Down blocks  │ ───────▶ │ Up   blocks  │
    └──────────────┘          └──────────────┘
             │                      ▲
             └───►  Bottleneck  ────┘

    Key features
    ------------
    * **Self-conditioning** (Karras et al., 2022).
    * **Fourier time embeddings** – fixed, learned, or random.
    * Configurable **linear vs. full** attention (with optional FlashAttention).
    * Optional **variance prediction** head for e.g. DDPM + learned σ.

    Parameters
    ----------
    base_channels : int
        Base channel count ``C`` (width of the first ResNet block).
    initial_channels : int, optional
        Channel count right after the input 7 × 7 convolution.
        Defaults to ``base_channels``.
    output_channels : int, optional
        Channels produced by the final 1 × 1 convolution.  If *None*,
        defaults to ``in_channels`` (or ``2 × in_channels`` when
        ``predict_variance`` is *True*).
    channel_multipliers : tuple[int, ...], default=(1, 2, 4, 8)
        Scales applied to ``base_channels`` at each resolution.
    in_channels : int, default=3
        Number of channels in the input image.
    use_self_conditioning : bool, default=False
        Concatenate the previous model prediction to the current input.
    predict_variance : bool, default=False
        Predict both mean and variance (doubles the output channels).
    learned_sinusoidal_embedding : bool, default=False
        Use a learned Fourier feature embedding for time.
    random_fourier_embedding : bool, default=False
        Use *random* Fourier features instead of learned/fixed ones.
    learned_embedding_dim : int, default=16
        Size of the (learned / random) Fourier feature vector.
    sinusoidal_theta : int, default=10000
        Base frequency for the fixed sinusoidal time embedding.
    dropout_rate : float, default=0.0
        Drop-out rate inside each ResNet block.
    attention_head_dim : int | tuple[int, ...], default=32
        Key/query dimension per attention head.
    attention_heads : int | tuple[int, ...], default=4
        Number of attention heads.
    full_attention : bool | tuple[bool, ...], optional
        Where to use quadratic full attention (vs. linear).  If *None*,
        uses full attention only at the bottleneck.
    use_flash_attention : bool, default=False
        Enable FlashAttention in the full/linear attention layers.
    """

    def __init__(
        self,
        base_channels: int,
        *,
        initial_channels: Optional[int] = None,
        output_channels: Optional[int] = None,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        in_channels: int = 3,
        use_self_conditioning: bool = False,
        predict_variance: bool = False,
        learned_sinusoidal_embedding: bool = False,
        random_fourier_embedding: bool = False,
        learned_embedding_dim: int = 16,
        sinusoidal_theta: int = 10_000,
        dropout_rate: float = 0.0,
        attention_head_dim: Union[int, Tuple[int, ...]] = 32,
        attention_heads: Union[int, Tuple[int, ...]] = 4,
        full_attention: Optional[Union[bool, Tuple[bool, ...]]] = None,
        use_flash_attention: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # ─── Input stem ───────────────────────────────────────────────────
        self.in_channels = in_channels
        self.use_self_conditioning = use_self_conditioning
        conditioned_in_channels = in_channels * (2 if use_self_conditioning else 1)

        initial_channels = initial_channels or base_channels
        self.input_conv = nn.Conv2d(
            conditioned_in_channels,
            initial_channels,
            kernel_size=7,
            padding=3,
        )

        # Channels at each resolution, e.g. (C, 2C, 4C, 8C, 16C)
        encoder_channels: List[int] = [
            initial_channels,
            *(base_channels * m for m in channel_multipliers),
        ]
        in_out_pairs = list(zip(encoder_channels[:-1], encoder_channels[1:]))

        # ─── Time embedding ───────────────────────────────────────────────
        time_emb_dim = base_channels * 4
        use_fourier = learned_sinusoidal_embedding or random_fourier_embedding

        if use_fourier:
            pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_embedding_dim, use_random=random_fourier_embedding
            )
            fourier_dim = learned_embedding_dim + 1
        else:
            pos_emb = SinusoidalPosEmb(base_channels, theta=sinusoidal_theta)
            fourier_dim = base_channels

        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ─── Attention config helpers ─────────────────────────────────────
        if full_attention is None:  # full attn only at bottleneck by default
            full_attention = (*[False] * (len(channel_multipliers) - 1), True)

        num_stages = len(channel_multipliers)
        full_attention = cast_tuple(full_attention, num_stages)
        attention_heads = cast_tuple(attention_heads, num_stages)
        attention_head_dim = cast_tuple(attention_head_dim, num_stages)

        # ─── Building-block aliases ──────────────────────────────────────
        FullAttention = partial(Attention, flash=use_flash_attention)
        resnet = partial(ResnetBlock, time_emb_dim=time_emb_dim, dropout=dropout_rate)

        # ─── Encoder / Down path ─────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        for stage, (
            (cin, cout),
            use_full_attn,
            n_heads,
            head_dim,
        ) in enumerate(zip(in_out_pairs, full_attention, attention_heads, attention_head_dim)):
            last_stage = stage == len(in_out_pairs) - 1
            Attn = FullAttention if use_full_attn else LinearAttention

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        resnet(cin, cin),
                        resnet(cin, cin),
                        Attn(cin, dim_head=head_dim, heads=n_heads),
                        Downsample(cin, cout)
                        if not last_stage
                        else nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                    ]
                )
            )

        # ─── Bottleneck ──────────────────────────────────────────────────
        mid_channels = encoder_channels[-1]
        self.mid_block1 = resnet(mid_channels, mid_channels)
        self.mid_attn = FullAttention(
            mid_channels,
            heads=attention_heads[-1],
            dim_head=attention_head_dim[-1],
        )
        self.mid_block2 = resnet(mid_channels, mid_channels)

        # ─── Decoder / Up path ───────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for stage, (
            (cin, cout),
            use_full_attn,
            n_heads,
            head_dim,
        ) in enumerate(
            zip(
                *map(
                    reversed,
                    (in_out_pairs, full_attention, attention_heads, attention_head_dim),
                )
            )
        ):
            last_stage = stage == len(in_out_pairs) - 1
            Attn = FullAttention if use_full_attn else LinearAttention

            self.up_blocks.append(
                nn.ModuleList(
                    [
                        resnet(cout + cin, cout),
                        resnet(cout + cin, cout),
                        Attn(cout, dim_head=head_dim, heads=n_heads),
                        Upsample(cout, cin)
                        if not last_stage
                        else nn.Conv2d(cout, cin, kernel_size=3, padding=1),
                    ]
                )
            )

        # ─── Output head ─────────────────────────────────────────────────
        predicted_channels = in_channels * (2 if predict_variance else 1)
        self.out_channels = output_channels or predicted_channels

        self.final_res_block = resnet(initial_channels * 2, initial_channels)
        self.final_conv = nn.Conv2d(initial_channels, self.out_channels, kernel_size=1)

    # ─── Convenience properties ──────────────────────────────────────────────
    @property
    def downsample_factor(self) -> int:
        """Total factor by which input spatial dims are reduced."""
        return 2 ** (len(self.down_blocks) - 1)

    # ─── Forward pass ────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape ``(B, C, H, W)``.
        timesteps : torch.Tensor
            Timestep tensor ``(B,)`` or ``(B, 1)``.
        x_self_cond : torch.Tensor, optional
            Previous model prediction (same shape as ``x``).  
            Only used when *self-conditioning* is enabled.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, out_channels, H, W)``.
        """
        # Verify input size ----------------------------------------------------
        if any(s % self.downsample_factor != 0 for s in x.shape[-2:]):
            raise ValueError(
                f"Spatial dims {x.shape[-2:]} must be divisible by "
                f"{self.downsample_factor}"
            )

        # Optional self-conditioning ------------------------------------------
        if self.use_self_conditioning:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat([x_self_cond, x], dim=1)

        # Input convolution & time emb ----------------------------------------
        x = self.input_conv(x)
        skip_input = x.clone()  # save for final global skip
        t = self.time_mlp(timesteps)

        # Encoder --------------------------------------------------------------
        skips: List[torch.Tensor] = []
        for block1, block2, attn, down in self.down_blocks:
            x = block1(x, t); skips.append(x)
            x = block2(x, t); x = attn(x) + x; skips.append(x)
            x = down(x)

        # Bottleneck -----------------------------------------------------------
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        # Decoder --------------------------------------------------------------
        for block1, block2, attn, up in self.up_blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, t)

            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x, t); x = attn(x) + x
            x = up(x)

        # Output head ----------------------------------------------------------
        x = torch.cat([x, skip_input], dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
