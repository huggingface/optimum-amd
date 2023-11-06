import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as onnxrt
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    GPTBigCodeForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions


model_name =  "bigcode/starcoderbase-1b"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.pad_token ="[PAD]"
tokenizer.padding_side = "left"

onnx_model_path = "./starcoderbase-1b-static-shapes"
decoder_model_path = "./starcoderbase-1b-static-shapes/decoder_model.onnx"
decoder_model_with_past_path = "./starcoderbase-1b-static-shapes/decoder_model_with_past.onnx"


class ORTModelForGPTBigCode(GPTBigCodeForCausalLM):
    """
    ONNX model with a causal language modeling head for ONNX Runtime inference. This class officially supports bloom, codegen, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gptj, llama.
    """

    def __init__(self, *args, **kwargs):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)

        sess_options = onnxrt.SessionOptions()
        self.provider = "CPUExecutionProvider"
        sess_options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.decoder = onnxrt.InferenceSession(
            decoder_model_path,
            providers=[self.provider],
            sess_options=sess_options,
        )
        self.decoder_with_past = onnxrt.InferenceSession(
            decoder_model_with_past_path,
            providers=[self.provider],
            sess_options=sess_options,
        )
        self.generation_config = GenerationConfig.from_model_config(config)

        self.inputs_names = {input_key.name: idx for idx, input_key in enumerate(self.decoder_with_past.get_inputs())}
        self.output_names = {
            output_key.name: idx for idx, output_key in enumerate(self.decoder_with_past.get_outputs())
        }
        self.key_value_input_names = [key for key in self.inputs_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        onnx_inputs = {"input_ids": input_ids.cpu().detach().numpy()}
        onnx_inputs["attention_mask"] = attention_mask.cpu().detach().numpy()
        onnx_inputs["position_ids"] = position_ids.cpu().detach().numpy()
        # from pdb import set_trace; set_trace()

        if past_key_values is not None and self.use_cache is True:
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

            outputs = self.decoder_with_past.run(None, onnx_inputs)
        else:
            outputs = self.decoder.run(None, onnx_inputs)

        if self.use_cache:
            past_key_values = tuple(
                torch.from_numpy(outputs[self.output_names[key]]).to(self.device)
                for key in self.key_value_output_names
            )

        logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self.device)

        return CausalLMOutputWithCrossAttentions(logits=logits, past_key_values=past_key_values)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)

        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, -1 * past_key_values[0].size(-2) :]

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
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


text = "Write hello world code in c++"
model = ORTModelForGPTBigCode()
inputs = tokenizer(text, padding="max_length", max_length=512, return_tensors="pt")

outputs = model.generate(**inputs, use_cache=True, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
