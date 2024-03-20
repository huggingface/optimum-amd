from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode

from optimum.exporters.onnx import onnx_export_from_model
from optimum.exporters.onnx.base import OnnxConfig
from transformers.modeling_utils import PreTrainedModel


def onnx_export_from_quantized_model(
    quantized_model: Union["PreTrainedModel"],
    output: Union[str, Path],
    opset: Optional[int] = None,
    optimize: Optional[str] = None,
    monolith: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_onnx_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    _variant: str = "default",
    preprocessors: List = None,
    device: str = "cpu",
    no_dynamic_axes: bool = False,
    task: str = "text-generation-with-past",
    use_subprocess: bool = False,
    do_constant_folding: bool = True,
    **kwargs_shapes,
):
    with torch.no_grad(), brevitas_proxy_export_mode(quantized_model, export_manager=StdQCDQONNXManager):
        onnx_export_from_model(
            quantized_model,
            output,
            opset=opset,
            monolith=monolith,
            optimize=optimize,
            model_kwargs=model_kwargs,
            custom_onnx_configs=custom_onnx_configs,
            fn_get_submodels=fn_get_submodels,
            _variant=_variant,
            preprocessors=preprocessors,
            device=device,
            no_dynamic_axes=no_dynamic_axes,
            use_subprocess=use_subprocess,
            do_constant_folding=do_constant_folding,
            task=task,
            do_validation=False,
            no_post_process=True,
            **kwargs_shapes,
        )
