from Models.SwinTransformer.Swin_v1.SwinTransformer_v1 import SwinTransformer_v1


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=7,
                               embed_dim=96,
                               depths=(2, 2, 6, 2),
                               num_heads=(3, 6, 12, 24),
                               num_classes=num_classes,
                               **kwargs)
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=7,
                               embed_dim=96,
                               depths=(2, 2, 18, 2),
                               num_heads=(3, 6, 12, 24),
                               num_classes=num_classes,
                               **kwargs)
    return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=7,
                               embed_dim=128,
                               depths=(2, 2, 18, 2),
                               num_heads=(4, 8, 16, 32),
                               num_classes=num_classes,
                               **kwargs)
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=12,
                               embed_dim=128,
                               depths=(2, 2, 18, 2),
                               num_heads=(4, 8, 16, 32),
                               num_classes=num_classes,
                               **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=7,
                               embed_dim=128,
                               depths=(2, 2, 18, 2),
                               num_heads=(4, 8, 16, 32),
                               num_classes=num_classes,
                               **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=12,
                               embed_dim=128,
                               depths=(2, 2, 18, 2),
                               num_heads=(4, 8, 16, 32),
                               num_classes=num_classes,
                               **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=7,
                               embed_dim=192,
                               depths=(2, 2, 18, 2),
                               num_heads=(6, 12, 24, 48),
                               num_classes=num_classes,
                               **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer_v1(in_chans=3,
                               patch_size=4,
                               window_size=12,
                               embed_dim=192,
                               depths=(2, 2, 18, 2),
                               num_heads=(6, 12, 24, 48),
                               num_classes=num_classes,
                               **kwargs)
    return model
