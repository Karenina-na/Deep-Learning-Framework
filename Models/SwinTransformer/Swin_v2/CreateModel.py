from Models.SwinTransformer.Swin_v2.SwinTransformer_v2 import SwinTransformer_v2
from typing import Tuple


def swin_transformer_v2_t(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformer_v2:
    """
    Function returns a tiny Swin Transformer V2 (SwinV2-T: C = 96, layer numbers = {2, 2, 6, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformer_v2) Tiny Swin Transformer V2
    """
    return SwinTransformer_v2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=96,
                              depths=(2, 2, 6, 2),
                              number_of_heads=(3, 6, 12, 24),
                              **kwargs)


def swin_transformer_v2_s(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformer_v2:
    """
    Function returns a small Swin Transformer V2 (SwinV2-S: C = 96, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformer_v2) Small Swin Transformer V2
    """
    return SwinTransformer_v2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=96,
                              depths=(2, 2, 18, 2),
                              number_of_heads=(3, 6, 12, 24),
                              **kwargs)


def swin_transformer_v2_b(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformer_v2:
    """
    Function returns a base Swin Transformer V2 (SwinV2-B: C = 128, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformer_v2) Base Swin Transformer V2
    """
    return SwinTransformer_v2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=128,
                              depths=(2, 2, 18, 2),
                              number_of_heads=(4, 8, 16, 32),
                              **kwargs)


def swin_transformer_v2_l(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformer_v2:
    """
    Function returns a large Swin Transformer V2 (SwinV2-L: C = 192, layer numbers ={2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformer_v2) Large Swin Transformer V2
    """
    return SwinTransformer_v2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=192,
                              depths=(2, 2, 18, 2),
                              number_of_heads=(6, 12, 24, 48),
                              **kwargs)


def swin_transformer_v2_h(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformer_v2:
    """
    Function returns a large Swin Transformer V2 (SwinV2-H: C = 352, layer numbers = {2, 2, 18, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformer_v2) Large Swin Transformer V2
    """
    return SwinTransformer_v2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=352,
                              depths=(2, 2, 18, 2),
                              number_of_heads=(11, 22, 44, 88),
                              **kwargs)


def swin_transformer_v2_g(input_resolution: Tuple[int, int],
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          **kwargs) -> SwinTransformer_v2:
    """
    Function returns a giant Swin Transformer V2 (SwinV2-G: C = 512, layer numbers = {2, 2, 42, 2}) for feature extraction
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size to be utilized
    :param in_channels: (int) Number of input channels
    :param use_checkpoint: (bool) If true checkpointing is utilized
    :param sequential_self_attention: (bool) If true sequential self-attention is performed
    :return: (SwinTransformer_v2) Giant Swin Transformer V2
    """
    return SwinTransformer_v2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=512,
                              depths=(2, 2, 42, 2),
                              number_of_heads=(16, 32, 64, 128),
                              **kwargs)
