# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import Dict

import onnxruntime as ort
import torch


ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_WEIGHTS_NAME_STATIC = "model_static.onnx"
ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ENCODER_ONNX_FILE_PATTERN = r"(.*)?encoder(.*)?\.onnx"
DECODER_ONNX_FILE_PATTERN = r"(.*)?decoder((?!(with_past|merged)).)*?\.onnx"


def get_device_for_provider(provider: str, provider_options: Dict) -> torch.device:
    """
    Gets the PyTorch device (CPU/CUDA) associated with an ONNX Runtime provider.
    """
    # if provider in ["VitisAIExecutionProvider"]:
    #     return NotImplementedError("VitisAIExecutionProvider is not supported!")
    # else:
    return torch.device("cpu")


def validate_provider_availability(provider: str):
    """
    Ensure the ONNX Runtime execution provider `provider` is available, and raise an error if it is not.

    Args:
        provider (str): Name of an ONNX Runtime execution provider.
    """
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise ValueError(
            f"Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}."
        )
