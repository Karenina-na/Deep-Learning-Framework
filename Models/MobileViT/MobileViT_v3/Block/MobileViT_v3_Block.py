import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence

from Models.MobileViT.MobileViT_v3.Block.Module.LinearAttention import LinearAttnFFN
from Models.MobileViT.MobileViT_v3.Block.Module.BaseLayers import LayerNorm
from Models.MobileViT.MobileViT_v3.Block.Module.util import module_profile, build_normalization_layer


class MobileViTBlockv3(nn.Module):
    """
    This class defines the `MobileViTv3 block <>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
            self,
            in_channels: int,
            attn_unit_dim: int,
            ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
            n_attn_blocks: Optional[int] = 2,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            patch_h: Optional[int] = 8,
            patch_w: Optional[int] = 8,
            conv_ksize: Optional[int] = 3,
            dilation: Optional[int] = 1,
            attn_norm_layer: Optional[str] = "layer_norm_2d",
            *args,
            **kwargs
    ) -> None:
        cnn_out_dim = attn_unit_dim

        conv_3x3_in = nn.Sequential(nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            dilation=dilation,
            groups=in_channels,
            padding=1,
        ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(negative_slope=0.1))

        conv_1x1_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        super(MobileViTBlockv3, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        # MobileViTv3: input changed from just global to local+global
        self.conv_proj = nn.Sequential(nn.Conv2d(
            in_channels=2 * cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
        ),
        nn.BatchNorm2d(num_features=in_channels))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize
        self.enable_coreml_compatible_fn = True

        if self.enable_coreml_compatible_fn:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name="unfolding_weights",
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w)
        )
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(
            self,
            d_model: int,
            ffn_mult: Union[Sequence, int, float],
            n_layers: int,
            attn_dropout: float,
            dropout: float,
            ffn_dropout: float,
            attn_norm_layer: str,
            *args,
            **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                    np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(
            LayerNorm(normalized_shape=d_model)
        )

        return nn.Sequential(*global_rep), d_model

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        repr_str += "\n)"
        return repr_str

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def unfolding_coreml(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # im2col is not implemented in Coreml, so here we hack its implementation using conv2d
        # we compute the weights

        # [B, C, H, W] --> [B, C, P, N]
        batch_size, in_channels, img_h, img_w = feature_map.shape
        #
        patches = F.conv2d(
            feature_map,
            self.unfolding_weights,
            bias=None,
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        # col2im is not supported on coreml, so tracing fails
        # We hack folding function via pixel_shuffle to enable coreml tracing
        batch_size, in_dim, patch_size, n_patches = patches.shape

        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w

        feature_map = patches.reshape(
            batch_size, in_dim * self.patch_h * self.patch_w, n_patches_h, n_patches_w
        )
        assert (
                self.patch_h == self.patch_w
        ), "For Coreml, we need patch_h and patch_w are the same"
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        fm_conv = self.local_rep(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm_conv)
        else:
            patches, output_size = self.unfolding_pytorch(fm_conv)

        # learn global representations on all patches
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)

        # MobileViTv3: local+global instead of only global
        fm = self.conv_proj(torch.cat((fm, fm_conv), dim=1))

        # MobileViTv3: skip connection
        fm = fm + x

        return fm

    def forward_temporal(
            self, x: Tensor, x_prev: Tensor, *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.resize_input_if_needed(x)

        fm_conv = self.local_rep(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm_conv)
        else:
            patches, output_size = self.unfolding_pytorch(fm_conv)

        # learn global representations
        for global_layer in self.global_rep:
            if isinstance(global_layer, LinearAttnFFN):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)

        # MobileViTv3: local+global instead of only global
        fm = self.conv_proj(torch.cat((fm, fm_conv), dim=1)
                            )

        # MobileViTv3: skip connection
        fm = fm + x

        return fm, patches

    def forward(
            self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            # for spatio-temporal data (e.g., videos)
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            # for image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0
        input = self.resize_input_if_needed(input)

        res = input
        out, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, output_size = self.unfolding_pytorch(feature_map=out)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        out, p, m = module_profile(module=self.conv_proj, x=torch.cat((fm, out), dim=1))
        params += p
        macs += m

        return res, params, macs
