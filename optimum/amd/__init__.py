# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
import contextlib

import transformers

from .rocm.import_utils import is_flash_attention_triton_available


# TODO: fix linting
if is_flash_attention_triton_available():
    from .rocm.attention import _check_and_enable_flash_attn_2, is_flash_attn_2_available_rocm
    from .rocm.models.llama import _llama_flash_attention_forward


@contextlib.contextmanager
def patcher(*args, **kwds):
    if not is_flash_attention_triton_available():
        raise ImportError()

    transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = (
        _llama_flash_attention_forward
    )

    transformers.utils.import_utils.is_flash_attn_2_available = is_flash_attn_2_available_rocm
    transformers.modeling_utils.PreTrainedModel._check_and_enable_flash_attn_2 = _check_and_enable_flash_attn_2
    try:
        yield
    finally:
        # TODO: unpatch
        pass
