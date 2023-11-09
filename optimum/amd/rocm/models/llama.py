# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import math

from flash_attn.flash_attn_triton import attention

from transformers.utils import logging


logger = logging.get_logger(__name__)


def _llama_flash_attention_forward(
    self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
):
    if attention_mask is not None:
        raise ValueError(
            "Flash Attention for RoCm using Triton does not support variable length inputs. Please do not use use_flash_attention_2=True for cases as batched inference, or training with padding."
        )
    else:
        softmax_scale = 1 / math.sqrt(query_states.shape[-1]) if softmax_scale is None else softmax_scale

        attn_output = attention(query_states, key_states, value_states, self.is_causal, softmax_scale)

    return attn_output
