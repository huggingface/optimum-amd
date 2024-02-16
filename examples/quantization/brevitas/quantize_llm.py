from argparse import ArgumentParser

import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode

from optimum.amd import BrevitasQuantizationConfig, BrevitasQuantizer
from optimum.amd.brevitas.accelerate_utils import accelerate_offload, calc_gpu_device_map, calc_cpu_device_map
from optimum.amd.brevitas.data_utils import compute_perplexity, get_dataset_for_model
from optimum.exporters.onnx import onnx_export_from_model
from transformers import AutoTokenizer


parser = ArgumentParser(description="Quantize LLMs from 🤗 Transformers with AMD Brevitas")
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
    help="Apply the GPTQ algorithm during quantization (Note, currently slow!). This option requires a calibration dataset.",
)
parser.add_argument(
    "--apply-weight-equalization", action="store_true", default=False, help="Apply the weight equalization algorithm."
)
parser.add_argument(
    "--apply-bias-correction", action="store_true", default=False, help="Apply the bias correction algorithm."
)
parser.add_argument(
    "--activations-equalization",
    type=str,
    choices=[None, "cross_layer", "layerwise"],
    default=None,
    help="Apply the activation equalization (SmoothQuant) algorithm (choices: [%(choices)s], default: %(default)s). This option requires a calibration dataset.",
)
parser.add_argument(
    "--is-static",
    action="store_true",
    default=False,
    help="Whether to do static quantization of the activations (default: %(default)s), with pre-computed quantizaton parameters ahead of inference. This option requires a calibration dataset.",
)
parser.add_argument(
    "--seqlen",
    type=int,
    default=128,
    help="Sequence length to use during calibration (default: %(default)s).",
)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

# Specify how much of each device should set aside for accelerate's offload functions
# The absolute margin is in bytes & the relative margin is a ratio
# The margins are the portions of the device which should be reserved for other functions
# (not accelerate)
gpu_device_map = calc_gpu_device_map(absolute_mem_margin=2.0*1e9, relative_mem_margin=0.3)
cpu_device_map = calc_cpu_device_map(absolute_mem_margin=2.0*1e9, relative_mem_margin=0.3)

# Prepare the quantizer, specifying its configuration and loading the model.
qconfig = BrevitasQuantizationConfig(
    apply_gptq=args.apply_gptq,
    apply_weight_equalization=args.apply_weight_equalization,
    activations_equalization=args.activations_equalization,
    is_static=args.is_static,
    weights_symmetric=True,
    activations_symmetric=args.is_static,  # ONNX export only supports unsigned for dynamic quantization
    gpu_device_map=gpu_device_map,
    cpu_device_map=cpu_device_map,
)

quantizer = BrevitasQuantizer.from_pretrained(args.model)

# Load the data for calibration and evaluation.
calibration_dataset = get_dataset_for_model(
    args.model,
    qconfig=qconfig,
    dataset_name="wikitext2",
    tokenizer=tokenizer,
    nsamples=128,
    seqlen=args.seqlen,
    split="train",
)

validation_dataset = get_dataset_for_model(
    args.model,
    qconfig=qconfig,
    dataset_name="wikitext2",
    tokenizer=tokenizer,
    nsamples=128,
    seqlen=args.seqlen,
    split="validation",
)

# Evaluation of the non-quantized model.
with accelerate_offload(quantizer.model, qconfig.gpu_device_map, qconfig.cpu_device_map) as model:
    perplexity = compute_perplexity(model, validation_dataset, context_length=args.seqlen // 2, tokenizer=tokenizer)
print(f"Perplexity (original model): {perplexity}")

model = quantizer.quantize(qconfig, calibration_dataset)

# Evaluation of the quantized model.
with accelerate_offload(model, qconfig.gpu_device_map, qconfig.cpu_device_map):
    perplexity = compute_perplexity(model, validation_dataset, context_length=args.seqlen // 2, tokenizer=tokenizer)
print(f"Perplexity (quantized model): {perplexity}")

print("Exporting the model to ONNX...")

# Export to ONNX through optimum.exporters.
with torch.no_grad(), brevitas_proxy_export_mode(model, export_manager=StdQCDQONNXManager):
    onnx_export_from_model(
        model, "llm_quantized_onnx", task="text-generation-with-past", do_validation=False, no_post_process=True
    )
