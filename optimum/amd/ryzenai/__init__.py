# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import TYPE_CHECKING

from transformers.pipelines import PIPELINE_REGISTRY
from transformers.utils import OptionalDependencyNotAvailable, _LazyModule

from .modeling import RyzenAIModelForImageClassification
from .pipelines import TimmImageClassificationPipeline


PIPELINE_REGISTRY.register_pipeline(
    "image-classification-timm",
    pipeline_class=TimmImageClassificationPipeline,
    pt_model=RyzenAIModelForImageClassification,
)

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
    "quantization": ["RyzenAIOnnxQuantizer"],
    "version": ["__version__"],
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
    from .quantization import RyzenAIOnnxQuantizer
    from .version import __version__
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
