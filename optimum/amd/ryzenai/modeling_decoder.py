# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""RyzenAIModelForXXX classes, allowing to run ONNX Models with ONNX Runtime VITIS-AI EP using the same API as Transformers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch

from optimum.utils import NormalizedConfigManager, check_if_transformers_greater
from transformers import (
    AutoModel,
    GenerationConfig,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling import RyzenAIModel


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if check_if_transformers_greater("4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin


model_types = {}
# model_types = {
#     "bloom": RyzenAIBloomForCausalLM,
#     "falcon": RyzenAIFalconForCausalLM,
#     "mpt": RyzenAIMPTForCausalLM,
#     "opt": RyzenAIOptForCausalLM,
#     "gpt_bigcode": RyzenAIGPTBigCodeForCausalLM,
# }


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

        self._initialize_params(use_cache, generation_config)

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
            )

    def _get_key_value_names(self):
        key_names = [key for key in self.inputs_names if (".key" in key) or (".value" in key)]
        value_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        return key_names, value_names

    def _initialize_params(self, use_cache, generation_config):
        self.num_pkv = 2
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(self.config.model_type)(
            self.config
        )
        self.key_value_input_names, self.key_value_output_names = self._get_key_value_names()
        self.use_cache = len(self.key_value_input_names) > 0
        self.generation_config = generation_config or GenerationConfig.from_model_config(self.config)
        self._validate_cache_param(use_cache)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        use_torch = isinstance(input_ids, torch.Tensor)

        inputs, known_output_shapes = self._prepare_inputs(
            input_ids, attention_mask, position_ids, past_key_values, use_torch
        )

        # run inference
        outputs = self.model.run(None, inputs)

        logits, past_key_values = self._process_outputs(outputs, use_torch)
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=past_key_values)

    def _prepare_inputs(self, input_ids, attention_mask, position_ids, past_key_values, use_torch):
        inputs = {"input_ids": self._convert_to_numpy(input_ids, use_torch)}

        if self.use_cache:
            use_cache_branch, past_key_values = self.prepare_past_key_values(input_ids, past_key_values, use_torch)

        for name in ["attention_mask", "position_ids"]:
            if name in self.inputs_names:
                inputs[name] = self._convert_to_numpy(locals()[name], use_torch)

        if "attention_mask" in self.inputs_names:
            inputs["attention_mask"] = self._convert_to_numpy(attention_mask, use_torch)

        if "position_ids" in self.inputs_names:
            if position_ids is None:
                raise ValueError("position_ids was not passed but is a required input for this ONNX model.")
            inputs["position_ids"] = self._convert_to_numpy(attention_mask, position_ids)

        if past_key_values is not None:
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                inputs[input_name] = self._convert_to_numpy(past_key_value, use_torch)

        return inputs

    def _process_outputs(self, outputs, use_torch):
        logits = self._convert_to_tensor(outputs[self.output_names["logits"]], use_torch)

        if self.use_cache:
            past_key_values = tuple(
                self._convert_to_tensor(outputs[self.output_names[key]], use_torch)
                for key in self.key_value_output_names
            )

            if self.model_type != "gpt_bigcode":
                past_key_values = tuple(
                    past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
                )
        else:
            past_key_values = None

        return logits, past_key_values

    def _convert_to_numpy(self, value, use_torch):
        return value.cpu().detach().numpy() if use_torch else value

    def _convert_to_tensor(self, value, use_torch):
        return torch.from_numpy(value).to(self.device) if use_torch else torch.from_numpy(value)

    def prepare_past_key_values(
        self,
        input_ids: Union[None, torch.LongTensor, np.ndarray],
        past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]],
        use_torch: bool,
    ):
        if past_key_values is not None:
            return past_key_values

        sequence_length = input_ids.shape[1]

        constructor = torch if use_torch else np

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

        return past_key_values

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
        init_cls: Optional["RyzenAIModel"] = None,
        **kwargs,
    ) -> RyzenAIModel:
        if init_cls is None:
            init_cls = model_types.get(config.model_type, RyzenAIModelForCausalLM)

        model = super()._from_pretrained(
            cls,
            model_id,
            config,
            vaip_config,
            use_auth_token,
            revision,
            force_download,
            cache_dir,
            file_name,
            subfolder,
            local_files_only,
            provider,
            session_options,
            provider_options,
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

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1) if past_key_values else position_ids

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", None),
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
