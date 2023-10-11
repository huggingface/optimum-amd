#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Defines the base classes that are used to perform inference with ONNX Runtime of Transformers models."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
from onnxruntime import InferenceSession

from transformers.modeling_outputs import Seq2SeqLMOutput

from ..utils import NormalizedConfigManager
from .utils import get_ordered_input_names, logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from .modeling import RyzenAIModel


class RyzenAIModelPart:
    """
    For multi-file ONNX models, such as encoder-decoder models, represents a part of the model.
    It has its own `onnxruntime.InferenceSession`, and can perform a forward pass.
    """

    def __init__(
        self,
        session: InferenceSession,
        parent_model: "RyzenAIModel",
    ):
        self.session = session
        self.parent_model = parent_model
        self.main_input_name = self.parent_model.main_input_name

        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(
            self.parent_model.config.model_type
        )(self.parent_model.config)

        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

        self._ordered_input_names = get_ordered_input_names(self.input_names.keys(), func=self.forward)

        input_tensor = session.get_inputs()[0]
        self.batch_size = input_tensor.shape[0]

    @property
    def device(self):
        return self.parent_model.device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class RyzenAIEncoder(RyzenAIModelPart):
    """
    Encoder model for ONNX Runtime inference.
    """


class RyzenAIModelDecoder(RyzenAIModelPart):
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.
    """

    def __init__(
        self,
        session: InferenceSession,
        parent_model: "RyzenAIModel",
    ):
        super().__init__(session, parent_model)
        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]

        if len(self.key_value_output_names) == 0:
            raise RuntimeError("Could not find the past key values in the provided model.")


class ORTDecoderForSeq2Seq(RyzenAIModelPart):
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.
    """

    def __init__(
        self,
        session: InferenceSession,
        parent_model: "RyzenAIModel",
    ):
        super().__init__(session, parent_model)

        self.num_pkv = 4
        self.first_gen = True
        self.end_of_generation = False

        self.max_decoder_sequence_length = self._infer_max_decoder_sequence_length()
        self.encoder_sequence_length = self._infer_encoder_sequence_length()
        self._initialize_generation_inputs()

    def _infer_max_decoder_sequence_length(self):
        output_tensor = self.session.get_outputs()[1]
        max_decoder_sequence_length = output_tensor.shape[2]
        return max_decoder_sequence_length

    def _infer_encoder_sequence_length(self):
        output_tensor = self.session.get_outputs()[3]
        encoder_sequence_length = output_tensor.shape[2]
        return encoder_sequence_length

    def _initialize_generation_inputs(self):
        self.decoder_attention_mask = np.zeros((self.batch_size, self.max_decoder_sequence_length)).astype(np.int64)
        self.decoder_attention_mask[0, 0] = 1
        self.position_ids = np.array([[0]]).astype(np.int64)

    def _update_inputs_after_inference(self):
        if self.position_ids[0][0] < self.max_decoder_sequence_length - 1:
            self.decoder_attention_mask[:, self.position_ids[0][0] + 1] = 1
        self.position_ids += 1

        if self.position_ids[0][0] == self.max_decoder_sequence_length:
            self.end_of_generation = True

    def prepare_inputs_for_decoder(
        self,
        past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]],
        use_torch: bool,
    ):
        # Generate dummy past for the first forward
        if past_key_values is None:
            constructor = torch if use_torch is True else np

            encoder_num_attention_heads = self.normalized_config.num_attention_heads
            decoder_num_attention_heads = self.normalized_config.num_attention_heads

            encoder_shape = (
                self.batch_size,
                encoder_num_attention_heads,
                self.encoder_sequence_length,
                self.normalized_config.hidden_size // encoder_num_attention_heads,
            )
            decoder_shape = (
                self.batch_size,
                decoder_num_attention_heads,
                self.max_decoder_sequence_length,
                self.normalized_config.hidden_size // decoder_num_attention_heads,
            )

            encoder_key_value = constructor.zeros(encoder_shape, dtype=constructor.float32)
            decoder_key_value = constructor.zeros(decoder_shape, dtype=constructor.float32)

            if use_torch is True:
                encoder_key_value = encoder_key_value.to(self.device)
                decoder_key_value = decoder_key_value.to(self.device)

            past_key_values = (decoder_key_value, decoder_key_value, encoder_key_value, encoder_key_value) * (
                len(self.key_value_input_names) // 4
            )

        return past_key_values

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Seq2SeqLMOutput:
        if past_key_values is None:
            self._initialize_generation_inputs()

        if self.end_of_generation is True:
            logits = torch.zeros((len(input_ids), 1, self.normalized_config.vocab_size))
            logits[:, :, self.normalized_config.eos_token_id] = 1

            return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

        use_torch = isinstance(input_ids, torch.Tensor)

        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )

        past_key_values = self.prepare_inputs_for_decoder(input_ids, past_key_values, use_torch=use_torch)

        if use_torch:
            onnx_inputs = {
                "input_ids": input_ids.cpu().detach().numpy(),
            }

            # Add the encoder_hidden_states inputs when needed
            if "encoder_hidden_states" in self.input_names:
                onnx_inputs["encoder_hidden_states"] = encoder_hidden_states.cpu().detach().numpy()

            # Add the encoder_attention_mask inputs when needed
            if "encoder_attention_mask" in self.input_names:
                onnx_inputs["encoder_attention_mask"] = encoder_attention_mask.cpu().detach().numpy()

            # Add the past_key_values to the decoder inputs
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()
        else:
            onnx_inputs = {
                "input_ids": input_ids,
            }

            # Add the encoder_hidden_states inputs when needed
            if "encoder_hidden_states" in self.input_names:
                onnx_inputs["encoder_hidden_states"] = encoder_hidden_states

            # Add the encoder_attention_mask inputs when needed
            if "encoder_attention_mask" in self.input_names:
                onnx_inputs["encoder_attention_mask"] = encoder_attention_mask

            # Add the past_key_values to the decoder inputs
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                onnx_inputs[input_name] = past_key_value

        if "position_ids" in self.input_names:
            onnx_inputs["position_ids"] = self.position_ids

        if "decoder_attention_mask" in self.input_names:
            onnx_inputs["decoder_attention_mask"] = self.decoder_attention_mask

        # Run inference
        outputs = self.session.run(None, onnx_inputs)

        out_past_key_values = tuple(
            torch.from_numpy(outputs[self.output_names[key]]).to(self.device) for key in self.key_value_output_names
        )

        logits = outputs[self.output_names["logits"]]
        if use_torch:
            logits = torch.from_numpy(logits).to(self.device)

        if self.first_gen == 0:
            out_past_key_values = [
                out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
            ]
            self.first_gen = False
        else:
            out_past_key_values = [
                torch.cat((out_past_key_values[i : i + 2], past_key_values[i + 2 : i + 4]))
                for i in range(0, len(out_past_key_values), self.num_pkv)
            ]

        self._update_inputs_after_inference()

        return Seq2SeqLMOutput(logits=logits, past_key_values=out_past_key_values)
