# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
import importlib

from packaging import version

from transformers.utils.import_utils import _is_package_available


def is_flash_attention_triton_available():
    _flash_attn_2_available = _is_package_available("flash_attn") and version.parse(
        importlib.metadata.version("flash_attn")
    ) >= version.parse("2.0.4")

    if not _flash_attn_2_available:
        return False

    try:
        return True
    except Exception:
        return False
