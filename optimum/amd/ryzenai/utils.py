# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import os.path as osp

import onnxruntime as ort


ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_WEIGHTS_NAME_STATIC = "model_static.onnx"

DEFAULT_VAIP_CONFIG = osp.normpath(osp.join(osp.dirname(__file__), "./configs/vaip_config.json"))


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
