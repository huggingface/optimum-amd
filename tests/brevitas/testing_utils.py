# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import Optional

from optimum.amd.brevitas import BrevitasQuantizationConfig, BrevitasQuantizer
from optimum.amd.brevitas.data_utils import get_dataset_for_model
from transformers import AutoTokenizer


SUPPORTED_MODELS_TINY = {
    # "llama": {"fxmarty/tiny-llama-fast-tokenizer": ["text-generation"]},
    "opt": {"hf-internal-testing/tiny-random-OPTForCausalLM": ["text-generation", "text-generation-with-past"]}
}

# TODO: test sequence_length=1 and past_key_value_length once available in optimum.
VALIDATE_EXPORT_ON_SHAPES = {
    # "past_key_values_length": [0, 4, 9],
    "sequence_length": [2, 4, 10],
}


def get_quantized_model(
    model_name: str,
    is_static: bool,
    apply_gptq: bool,
    apply_weight_equalization: bool,
    activations_equalization: Optional[str],
):
    qconfig = BrevitasQuantizationConfig(
        is_static=is_static,
        apply_gptq=apply_gptq,
        apply_weight_equalization=apply_weight_equalization,
        activations_equalization=activations_equalization,
    )
    quantizer = BrevitasQuantizer.from_pretrained(model_name)

    calibration_dataset = None
    if is_static:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        calibration_dataset = get_dataset_for_model(
            model_name,
            qconfig=qconfig,
            dataset_name="wikitext2",
            tokenizer=tokenizer,
            nsamples=128,
            seqlen=64,
            split="train",
        )

    model = quantizer.quantize(qconfig, calibration_dataset=calibration_dataset)

    return model
