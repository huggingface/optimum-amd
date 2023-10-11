# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule


_import_structure = {
    "configuration": [
        "ORTConfig",
    ],
    "modeling_seq2seq": [
        "RyzenAIModelForSpeechSeq2Seq",
    ],
}


# Direct imports for type-checking
if TYPE_CHECKING:
    from .configuration import RyzenAIConfig
    from .modeling_seq2seq import (
        RyzenAIModelForSpeechSeq2Seq,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
