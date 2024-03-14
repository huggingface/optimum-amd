import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from optimum.exporters.onnx import onnx_export_from_model


def export_quantized_model(quantized_model, path, task="text-generation-with-past"):
    with torch.no_grad(), brevitas_proxy_export_mode(quantized_model, export_manager=StdQCDQONNXManager):
        onnx_export_from_model(quantized_model, path, task=task, do_validation=False, no_post_process=True)
