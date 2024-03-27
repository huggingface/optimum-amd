# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""RyzenAIModelForCausalLM classes, allowing to run ONNX Models with ONNX Runtime VITIS-AI EP using the same API as Transformers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch

from optimum.utils import NormalizedConfigManager
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling import RyzenAIModel
from .utils import set_builtins, set_environment_variables


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class RyzenAIModelForCausalLM(RyzenAIModel, GenerationMixin):
    """
    Runs model with causal language modeling head using ONNX Runtime VITIS-AI EP.
    """

    main_input_name = "input_ids"
    auto_model_class = AutoModelForCausalLM

    def __init__(
        self,
        model: ort.InferenceSession,
        config: "PretrainedConfig",
        vaip_config: Union[str, Path] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(model, config, vaip_config, model_save_dir, preprocessors, **kwargs)

        self._initialize_params(use_cache, generation_config)

        # need for generate
        self.device = torch.device("cpu")

    def _get_key_value_names(self):
        key_names = [key for key in self.inputs_names if (".key" in key) or (".value" in key)]
        value_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        return key_names, value_names

    def _initialize_params(self, use_cache, generation_config):
        if self.config is None:
            raise ValueError("The model config must be provided to instantiate the model.")

        self.num_pkv = 2
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(self.config.model_type)(
            self.config
        )
        self.key_value_input_names, self.key_value_output_names = self._get_key_value_names()
        self.use_cache = len(self.key_value_input_names) > 0
        self.generation_config = generation_config or GenerationConfig.from_model_config(self.config)

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with past-key-values."
            )

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
            inputs["position_ids"] = self._convert_to_numpy(position_ids, use_torch)

        if self.use_cache:
            if past_key_values is None:
                # Generate dummy past for the first forward
                batch_size, sequence_length = input_ids.shape
                past_key_values = self.prepare_past_key_values(batch_size, sequence_length, use_torch)
            else:
                past_key_values = self.process_input_past_key_values(past_key_values)

            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                inputs[input_name] = self._convert_to_numpy(past_key_value, use_torch)

        return inputs

    def process_input_past_key_values(self, past_key_values: Tuple[torch.Tensor]):
        # Override this method in subclasses to adapt to the specific model's configuration
        past_key_values = tuple(
            past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
        )

        return past_key_values

    def _process_outputs(self, outputs, use_torch):
        # Override this method in subclasses to adapt to the specific model's configuration
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

    def get_shape_params_from_normalized_config(self):
        # Override this method in subclasses to adapt to the specific model's configuration
        num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads

        return {
            "num_key_value_heads": num_attention_heads,
            "embed_size_per_head": embed_size_per_head,
        }

    def prepare_past_key_values(
        self,
        batch_size: int,
        sequence_length: int,
        use_torch: bool,
    ):
        # Override this method in subclasses to adapt to the specific model's configuration
        params = self.get_shape_params_from_normalized_config()
        key_or_value_shape = (batch_size, params["num_key_value_heads"], 0, params["embed_size_per_head"])

        constructor, dtype = self.get_constructor(use_torch)
        key_or_value = constructor.zeros(key_or_value_shape, dtype=dtype)

        past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))
        for _, value in zip(self.key_value_output_names, past_key_values):
            shape = [*value.shape]
            shape[2] += sequence_length

        return past_key_values

    def get_constructor(self, use_torch):
        constructor = torch if use_torch else np
        return constructor, constructor.float32

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
        use_cache: bool = True,
        **kwargs,
    ) -> RyzenAIModel:
        # set environment variables
        set_environment_variables()
        set_builtins()

        init_cls = model_type_to_class.get(config.model_type, RyzenAIModelForCausalLM)

        model, vaip_config, model_save_dir, preprocessors = cls._load_model_and_processors(
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
            **kwargs,
        )

        return init_cls(
            model,
            config=config,
            vaip_config=vaip_config,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
            use_cache=use_cache,
        )

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
    def get_shape_params_from_normalized_config(self):
        num_key_value_heads = self.normalized_config.num_key_value_heads
        embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        return {
            "num_key_value_heads": num_key_value_heads,
            "embed_size_per_head": embed_size_per_head,
        }


class RyzenAILlamaForCausalLM(RyzenAIModelForCausalLM):
    def get_shape_params_from_normalized_config(self):
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
    def process_input_past_key_values(self, past_key_values: Tuple[torch.Tensor]):
        return past_key_values

    def _process_outputs(self, outputs, use_torch):
        logits = self._convert_to_tensor(outputs[self.output_names["logits"]], use_torch)

        past_key_values = None
        if self.use_cache:
            past_key_values = tuple(
                self._convert_to_tensor(outputs[self.output_names[key]], use_torch)
                for key in self.key_value_output_names
            )

        return logits, past_key_values

    def prepare_past_key_values(
        self,
        batch_size: int,
        sequence_length: int,
        use_torch: bool,
    ):
        # GPT BigCode uses muti-query attention, and has the specificity of putting both key and value in the same cache tensor.
        params = self.get_shape_params_from_normalized_config()
        key_or_value_shape = (batch_size, 0, params["embed_size_per_head"] * 2)

        constructor, dtype = self.get_constructor(use_torch)
        key_or_value = constructor.zeros(key_or_value_shape, dtype=dtype)

        past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))
        for _, value in zip(self.key_value_output_names, past_key_values):
            shape = [*value.shape]
            shape[1] += sequence_length

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
