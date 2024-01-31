from argparse import ArgumentParser

import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode

from optimum.amd import BrevitasQuantizationConfig, BrevitasQuantizer
from optimum.amd.brevitas.accelerate_utils import offload_model, remove_hooks
from optimum.amd.brevitas.data_utils import get_dataset, model_eval_accelerate
from optimum.exporters.onnx.__main__ import onnx_export  # TODO: move this method elsewhere (not __main__.py)
from transformers import AutoTokenizer


parser = ArgumentParser(description="Quantize OPT from 🤗 Transformers with AMD Brevitas")
parser.add_argument(
    "--model",
    type=str,
    default="facebook/opt-125m",
    help="Model checkpoint to quantize. Can either be a local path to a model folder, or a model on Hugging Face Hub. Example: facebook/opt-125m",
)
parser.add_argument(
    "--apply-gptq",
    action="store_true",
    default=False,
    help="Apply the GPTQ algorithm during quantization (Note, currently slow!)",
)
parser.add_argument(
    "--apply-weight-equalization", action="store_true", default=False, help="Apply the weight equalization algorithm"
)

# TODO: explain what fx and layerwise are.
parser.add_argument(
    "--apply-act-equalization",
    type=str,
    choices=[None, "fx", "layerwise"],
    default=None,
    help="Apply the activation equalization (SmoothQuant) algorithm (choices: [%(choices)s], default: %(default)s)",
)
parser.add_argument(
    "--replace-mha-with-quantizable",
    action="store_true",
    default=False,
    help="Replace attention with standard PyTorch implementation (default: %(default)s)",
)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

qconfig = BrevitasQuantizationConfig(
    apply_gptq=args.apply_gptq,
    apply_weight_equalization=args.apply_weight_equalization,
    activations_equalization=args.apply_act_equalization,
    replace_mha_with_quantizable=args.replace_mha_with_quantizable,
)

quantizer = BrevitasQuantizer.from_pretrained(args.model)

# calibration_dataloader = quantizer.get_calibration_dataloader(
#     tokenizer, dataset_name='wikitext2-raw', num_samples=qconfig.nsamples, seqlen=qconfig.seqlen)

calibration_dataloader = get_dataset(
    dataset_name="wikitext2", tokenizer=tokenizer, nsamples=128, seqlen=2048, split="train"
)

print("Apply quantization")
# TODO: model should ideally not be passed to `quantize`.
model = quantizer.quantize(qconfig, calibration_dataloader)
print("Quantization applied")

model = offload_model(model)
print("Model eval...")

validation_dataloader = get_dataset(
    dataset_name="wikitext2", tokenizer=tokenizer, nsamples=128, seqlen=2048, split="validation"
)

perplexity = model_eval_accelerate(model, validation_dataloader, qconfig.seqlen)
print(f"C4 perplexity: {perplexity}")

print("Exporting the model to ONNX...")
# When exporting to ONNX, Accelerate's hooks need to be removed otherwise we have unwanted Cast nodes in the ONNX graph.
remove_hooks(model)

# Export to ONNX through optimum.exporters.
with torch.no_grad(), brevitas_proxy_export_mode(model, export_manager=StdQCDQONNXManager):
    onnx_export(model, "opt_quantized_onnx", task="text-generation-with-past", do_validation=False)
