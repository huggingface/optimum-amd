# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "configuration": ["RyzenAIConfig", "QuantizationConfig", "AutoQuantizationConfig"],
    "modeling": [
        "RyzenAIModel",
        "RyzenAIModelForCustomTasks",
        "RyzenAIModelForImageClassification",
        "RyzenAIModelForImageSegmentation",
        "RyzenAIModelForImageToImage",
        "RyzenAIModelForObjectDetection",
    ],
    "modeling_decoder": [
        "RyzenAIModelForCausalLM",
    ],
    "quantization": ["RyzenAIOnnxQuantizer"],
    "pipelines": ["pipeline"],
}


# Direct imports for type-checking
if TYPE_CHECKING:
    from .configuration import AutoQuantizationConfig, QuantizationConfig, RyzenAIConfig
    from .modeling import (
        RyzenAIModel,
        RyzenAIModelForCustomTasks,
        RyzenAIModelForImageClassification,
        RyzenAIModelForImageSegmentation,
        RyzenAIModelForImageToImage,
        RyzenAIModelForObjectDetection,
    )
    from .modeling_decoder import RyzenAIModelForCausalLM
    from .pipelines import pipeline
    from .quantization import RyzenAIOnnxQuantizer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
