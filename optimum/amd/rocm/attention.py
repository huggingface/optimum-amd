# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
import importlib
from typing import Dict, Optional, Union

import torch
from packaging import version

from transformers import PretrainedConfig
from transformers.utils import logging
from transformers.utils.import_utils import _is_package_available, is_torch_available


logger = logging.get_logger(__name__)


def is_flash_attn_2_available_rocm():
    if not is_torch_available():
        return False

    # Let's add an extra check to see if cuda is available
    import torch

    # NOTE@AMD: We do not put a requirement on 2.1.0 here, as this version does not exist in ROCmSoftwarePlatform package.
    _flash_attn_2_available = _is_package_available("flash_attn") and version.parse(
        importlib.metadata.version("flash_attn")
    ) >= version.parse("2.0.4")

    return _flash_attn_2_available and torch.cuda.is_available()


@classmethod
def _check_and_enable_flash_attn_2(
    cls, config, torch_dtype: Optional[torch.dtype] = None, device_map: Optional[Union[str, Dict[str, int]]] = None
) -> PretrainedConfig:
    if not cls._supports_flash_attn_2:
        raise ValueError(
            "The current architecture does not support Flash Attention 2.0. Please open an issue on GitHub to "
            "request support for this architecture: https://github.com/huggingface/transformers/issues/new"
        )

    if not is_flash_attn_2_available_rocm():
        raise ImportError(
            "Flash Attention 2 is not available. Please refer to the documentation of https://github.com/Dao-AILab/flash-attention for"
            " installing it. Make sure to have at least the version 2.1.0"
        )
    else:
        flash_attention_version = version.parse(importlib.metadata.version("flash_attn"))

        # NOTE@AMD: We do not put a requirement on 2.1.0 here, as this version does not exist in ROCmSoftwarePlatform package.
        is_flash_greater_than_2 = flash_attention_version >= version.parse("2.0.4")
        if not is_flash_greater_than_2:
            raise ValueError(
                f"You need flash_attn package version to be greater or equal than 2.0.4. Make sure to have that version installed - detected version {flash_attention_version}."
            )

    _is_bettertransformer = getattr(cls, "use_bettertransformer", False)

    if _is_bettertransformer:
        raise ValueError(
            "Flash Attention 2 and BetterTransformer API are not compatible. Please make sure to disable BetterTransformers by doing model.reverse_bettertransformer()"
        )

    if torch_dtype is None:
        logger.warning(
            "You are attempting to use Flash Attention 2.0 without specifying a torch dtype. This might lead to unexpected behaviour"
        )
    elif torch_dtype is not None and torch_dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes. You passed {torch_dtype}, this might lead to"
            " unexpected behaviour."
        )

    if device_map is None:
        if torch.cuda.is_available():
            logger.warning(
                "You are attempting to use Flash Attention 2.0 with a model initialized on CPU. Make sure to move the model to GPU"
                " after initializing it on CPU with `model.to('cuda')`."
            )
        else:
            raise ValueError(
                "You are attempting to use Flash Attention 2.0 with a model initialized on CPU and with no GPU available. "
                "This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map "
                "or initialising the model on CPU and then moving it to GPU."
            )
    elif (
        device_map is not None
        and isinstance(device_map, dict)
        and ("cpu" in device_map.values() or "disk" in device_map.values())
    ):
        raise ValueError(
            "You are attempting to use Flash Attention 2.0 with a model dispatched on CPU or disk. This is not supported. Please make sure to "
            "initialise the model on a GPU by passing a device_map that contains only GPU devices as keys."
        )
    config._flash_attn_2_enabled = True
    return config
