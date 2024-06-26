# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "yolov3": ["YoloV3ImageProcessor"],
    "yolov5": ["YoloV5ImageProcessor"],
    "yolov8": ["YoloV8ImageProcessor"],
    "yolox": ["YoloXImageProcessor"],
    "semanticfpn": ["SemanticFPNImageProcessor"],
    "hrnet": ["HRNetImageProcessor"],
}


# Direct imports for type-checking
if TYPE_CHECKING:
    from .hrnet import HRNetImageProcessor
    from .semanticfpn import SemanticFPNImageProcessor
    from .yolov3 import YoloV3ImageProcessor
    from .yolov5 import YoloV5ImageProcessor
    from .yolov8 import YoloV8ImageProcessor
    from .yolox import YoloXImageProcessor
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
