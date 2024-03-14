# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""RyzenAIModelForXXX classes, allowing to run ONNX Models with ONNX Runtime VITIS-AI EP using the same API as Transformers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ort as ort
import torch

from optimum.utils import NormalizedConfigManager, check_if_transformers_greater
from transformers import (
    AutoModelForCausalLM,
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


class RyzenAIModelForCausalLM(RyzenAIModel, GenerationMixin):
    """
    Runs model with causal language modeling head using ONNX Runtime VITIS-AI EP.
    """

    model_type = "onnx_model"
    auto_model_class = AutoModelForCausalLM

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

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        use_torch = isinstance(input_ids, torch.Tensor)

        inputs = self._prepare_inputs(input_ids, attention_mask, position_ids, past_key_values, use_torch)

        # run inference
        outputs = self.model.run(None, inputs)

        logits, past_key_values = self._process_outputs(outputs, use_torch)
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=past_key_values)

    def _prepare_inputs(self, input_ids, attention_mask, position_ids, past_key_values, use_torch):
        inputs = {"input_ids": self._convert_to_numpy(input_ids, use_torch)}

        if "attention_mask" in self.inputs_names:
            inputs["attention_mask"] = self._convert_to_numpy(attention_mask, use_torch)

        if "position_ids" in self.inputs_names:
            if position_ids is None:
                raise ValueError("`position_ids` was not passed but is a required input for this ONNX model.")
            inputs["position_ids"] = self._convert_to_numpy(attention_mask, position_ids)

        if self.use_cache:
            past_key_values = self.prepare_past_key_values(input_ids, past_key_values, use_torch)

            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                inputs[input_name] = self._convert_to_numpy(past_key_value, use_torch)

        return inputs

    def _process_outputs(self, outputs, use_torch):
        logits = self._convert_to_tensor(outputs[self.output_names["logits"]], use_torch)

        past_key_values = None
        if self.use_cache:
            past_key_values = tuple(
                self._convert_to_tensor(outputs[self.output_names[key]], use_torch)
                for key in self.key_value_output_names
            )

            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )

        return logits, past_key_values

    def prepare_past_key_values(
        self,
        input_ids: Union[None, torch.LongTensor, np.ndarray],
        past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]],
        use_torch: bool,
    ):
        if past_key_values is not None:
            constructor = torch if use_torch else np
            dtype = constructor.float16 if self.use_fp16 else constructor.float32

            params = self.get_params_from_normalized_config()
            params["batch_size"] = input_ids.shape[0]
            params["sequence_length"] = input_ids.shape[1]

            # Generate dummy past for the first forward
            past_key_values = self.generate_past_key_values(constructor, params, dtype)

        return past_key_values

    def get_params_from_normalized_config(self):
        num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads

        return {
            "num_key_value_heads": num_attention_heads,
            "embed_size_per_head": embed_size_per_head,
        }

    def generate_past_key_values(self, constructor: Any, params: Dict[str, int], dtype: Any):
        shapes = self.get_params_from_normalized_config()
        shape = (params["batch_size"], shapes["num_key_value_heads"], 0, shapes["embed_size_per_head"])

        key_or_value = constructor.zeros(shape, dtype=dtype)
        past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))

        for _, value in zip(self.key_value_output_names, past_key_values):
            shape = [*value.shape]
            shape[2] += params["sequence_length"]

        return past_key_values

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
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
            init_cls = model_type_to_class.get(config.model_type, RyzenAIModelForCausalLM)

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
            tuple(past_state.index_select(0, beam_idx.to(past_state)) for past_state in layer_past)
            for layer_past in past
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True


class RyzenAIMistralForCausalLM(RyzenAIModelForCausalLM):
    def get_params_from_normalized_config(self):
        num_key_value_heads = self.normalized_config.num_key_value_heads
        embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        return {
            "num_key_value_heads": num_key_value_heads,
            "embed_size_per_head": embed_size_per_head,
        }


class RyzenAILlamaForCausalLM(RyzenAIModelForCausalLM):
    def get_params_from_normalized_config(self):
        num_key_value_heads = self.normalized_config.num_key_value_heads
        embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        return {
            "num_key_value_heads": num_key_value_heads,
            "embed_size_per_head": embed_size_per_head,
        }


class RyzenAIOPTForCausalLM(RyzenAIModelForCausalLM):
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
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }


class RyzenAIGPTBigCodeForCausalLM(RyzenAIModelForCausalLM):
    def _process_outputs(self, outputs, use_torch):
        logits = self._convert_to_tensor(outputs[self.output_names["logits"]], use_torch)

        past_key_values = None
        if self.use_cache:
            past_key_values = tuple(
                self._convert_to_tensor(outputs[self.output_names[key]], use_torch)
                for key in self.key_value_output_names
            )

        return logits, past_key_values

    def generate_past_key_values(self, constructor: Any, params: Dict[str, int], dtype: Any):
        # GPT BigCode uses muti-query attention, and has the specificity of putting both key and value in the same cache tensor.
        shape_key_and_value = (params["batch_size"], 0, params["embed_size_per_head"] * 2)
        key_and_value = constructor.zeros(shape_key_and_value, dtype=dtype)

        past_key_values = tuple(key_and_value for _ in range(len(self.key_value_input_names)))

        for _, value in zip(self.key_value_output_names, past_key_values):
            shape = [*value.shape]
            shape[1] += params["sequence_length"]

        return past_key_values

    # Adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Omit tokens covered by past_key_values
        if past_key_values:
            if self.config.multi_query:
                past_length = past_key_values[0].shape[1]
            else:
                past_length = past_key_values[0].shape[2]

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
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # Copied from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM._reorder_cache
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(layer_past.index_select(0, beam_idx) for layer_past in past_key_values)


model_type_to_class = {
    "opt": RyzenAIOPTForCausalLM,
    "gpt_bigcode": RyzenAIGPTBigCodeForCausalLM,
    "mistral": RyzenAIMistralForCausalLM,
    "llama": RyzenAILlamaForCausalLM,
}
