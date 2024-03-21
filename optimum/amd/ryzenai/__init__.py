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
        "RyzenAIModelForSemanticSegmentation",
        "RyzenAIModelForImageToImage",
        "RyzenAIModelForObjectDetection",
    ],
    "quantization": ["RyzenAIOnnxQuantizer"],
    "pipelines": ["pipeline"],
    "utils": ["DEFAULT_VAIP_CONFIG"],
}


# Direct imports for type-checking
if TYPE_CHECKING:
    from .configuration import AutoQuantizationConfig, QuantizationConfig, RyzenAIConfig
    from .modeling import (
        RyzenAIModel,
        RyzenAIModelForCustomTasks,
        RyzenAIModelForImageClassification,
        RyzenAIModelForImageToImage,
        RyzenAIModelForObjectDetection,
        RyzenAIModelForSemanticSegmentation,
    )
    from .pipelines import pipeline
    from .quantization import RyzenAIOnnxQuantizer
    from .utils import DEFAULT_VAIP_CONFIG
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
