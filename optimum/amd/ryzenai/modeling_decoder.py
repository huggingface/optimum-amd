# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""RyzenAIModelForXXX classes, allowing to run ONNX Models with ONNX Runtime VITIS-AI EP using the same API as Transformers."""

import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from onnx import shape_inference
from onnx.tools import update_model_dims

from optimum.exporters import TasksManager
from optimum.modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from optimum.onnx.utils import _get_external_data_paths
from optimum.utils.save_utils import maybe_load_preprocessors
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    PretrainedConfig,
)
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import ImageClassifierOutput, ModelOutput

from .utils import (
    ONNX_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME_STATIC,
    validate_provider_availability,
)


logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"

class RyzenAIModelForCausalLM(RyzenAIModel, GenerationMixin):
    """
    Runs model with causal language modeling head using ONNX Runtime VITIS-AI EP.
    """

    model_type = "onnx_model"
    auto_model_class = AutoModel

    def __init__(
        self,
        model: ort.InferenceSession,
        config: PretrainedConfig,
        vaip_config: Union[str, Path] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(model, config, vaip_config, model_save_dir, preprocessors, **kwargs)
        
        self.num_pkv = 2
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.key_value_input_names = [key for key in self.inputs_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_torch = isinstance(input_ids, torch.Tensor)

        inputs = {}
        known_output_shapes = {}

        if self.use_cache:
            # Create dummy past_key_values for decoder first generation step if none given
            use_cache_branch, past_key_values, known_output_shapes = self.prepare_past_key_values(
                input_ids, past_key_values, use_torch
            )

        inputs["input_ids"] = input_ids.cpu().detach().numpy() if use_torch else input_ids

        if "attention_mask" in self.inputs_names:
            inputs["attention_mask"] = attention_mask.cpu().detach().numpy() if use_torch else attention_mask

        if "position_ids" in self.inputs_names:
            if position_ids is None:
                raise ValueError("position_ids was not passed but is a required input for this ONNX model.")
            inputs["position_ids"] = position_ids.cpu().detach().numpy() if use_torch else position_ids

        # Add the past_key_values to the decoder inputs
        if past_key_values is not None:
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                inputs[input_name] = past_key_value.cpu().detach().numpy() if use_torch else past_key_value

        outputs = self.model.run(None, inputs)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 for the self-attention)
            past_key_values = tuple(
                torch.from_numpy(outputs[self.output_names[key]]).to(self.device)
                for key in self.key_value_output_names
            )

        logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self.device)

    if self.use_cache and self.model_type != "gpt_bigcode":
        # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
        # per decoder layer
        past_key_values = tuple(
            past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
        )

    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)

    def prepare_past_key_values(
        self,
        input_ids: Union[None, torch.LongTensor, np.ndarray],
        past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]],
        use_torch: bool,
    ):
        if past_key_values is not None:
            return past_key_values, {}

        sequence_length = input_ids.shape[1]

        constructor = torch if use_torch else np

        pkv_output_shape = {}
        # Generate dummy past for the first forward
        batch_size = input_ids.shape[0]
        embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        if self.model_type == "gemma":
            num_attention_heads = self.normalized_config.num_key_value_heads
            embed_size_per_head = self.normalized_config.head_dim
        elif self.model_type in {"mistral", "llama"}:
            num_attention_heads = self.normalized_config.num_key_value_heads
        else:
            num_attention_heads = self.normalized_config.num_attention_heads

        dtype = constructor.float16 if self.use_fp16 else constructor.float32

        num_key_value_heads = num_attention_heads

        shape = (batch_size, num_key_value_heads, 0, embed_size_per_head)
        key_or_value = constructor.zeros(shape, dtype=dtype)

        if use_torch:
            key_or_value = key_or_value.to(self.device)

        past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))

        for name, value in zip(self.key_value_output_names, past_key_values):
            shape = [*value.shape]
            shape[2] += sequence_length
            pkv_output_shape[name] = shape

        return past_key_values, pkv_output_shape

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        vaip_config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        provider: str = "VitisAIExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        init_cls: Optional[RyzenAIModel] = None,
        **kwargs,
    ) -> "RyzenAIModelForCausalLM":
        if config.model_type == "bloom":
            init_cls = RyzenAIBloomForCausalLM
        elif config.model_type == "falcon":
            init_cls = RyzenAIFalconForCausalLM
        elif config.model_type == "mpt":
            init_cls = RyzenAIMPTForCausalLM
        elif config.model_type == "opt":
            init_cls = RyzenAIOptForCausalLM
        elif config.model_type == "gpt_bigcode":
            init_cls = RyzenAIGPTBigCodeForCausalLM
        else:
            init_cls = RyzenAIModelForCausalLM

        model = super()._from_pretrained(
            model_id,
            config,
            use_auth_token,
            revision,
            force_download,
            cache_dir,
            file_name,
            subfolder,
            use_cache,
            local_files_only,
            use_merged,
            provider,
            session_options,
            provider_options,
            use_io_binding,
            model_save_dir,
            init_cls,
            **kwargs,
        )


        return model

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
