from typing import Union, Optional, Tuple
import torch
from torch import nn, Tensor


def make_divisible(
        v: Union[float, int],
        divisor: Optional[int] = 8,
        min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def bound_fn(min_val: Union[float, int], max_val: Union[float, int], value: Union[float, int]):
    return max(min_val, min(max_val, value))


def module_profile(module, x: Tensor, *args, **kwargs) -> Tuple[Tensor, float, float]:
    """
    Helper function to profile a module.

    .. note::
        Module profiling is for reference only and may contain errors as it solely relies on user implementation to
        compute theoretical FLOPs
    """

    if isinstance(module, nn.Sequential):
        n_macs = n_params = 0.0
        for l in module:
            try:
                x, l_p, l_macs = l.profile_module(x)
                n_macs += l_macs
                n_params += l_p
            except Exception as e:
                print(e, l)
                pass
    else:
        x, n_params, n_macs = module.profile_module(x)
    return x, n_params, n_macs


def build_normalization_layer(
        num_features: int,
        norm_type: Optional[str] = None,
        num_groups: Optional[int] = None,
        *args,
        **kwargs
) -> torch.nn.Module:
    """
    Helper function to build the normalization layer.
    The function can be used in either of below mentioned ways:
    Scenario 1: Set the default normalization layers using command line arguments. This is useful when the same normalization
    layer is used for the entire network (e.g., ResNet).
    Scenario 2: Network uses different normalization layers. In that case, we can override the default normalization
    layer by specifying the name using `norm_type` argument
    """
    norm_type = (
        "batch_norm"
        if norm_type is None
        else norm_type
    )
    num_groups = (
        1
        if num_groups is None
        else num_groups
    )
    momentum = 0.1
    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None

    if norm_type in NORM_LAYER_REGISTRY:
        if torch.cuda.device_count() < 1 and norm_type.find("sync_batch") > -1:
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            norm_type = norm_type.replace("sync_", "")
        norm_layer = NORM_LAYER_REGISTRY[norm_type](
            normalized_shape=num_features,
            num_features=num_features,
            momentum=momentum,
            num_groups=num_groups,
        )
    elif norm_type == "identity":
        norm_layer = Identity()
    else:
        print(
            "Supported normalization layer arguments are: {}. Got: {}".format(
                SUPPORTED_NORM_FNS, norm_type
            )
        )
    return norm_layer
