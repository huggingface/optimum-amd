from argparse import ArgumentParser

import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode

from optimum.amd import BrevitasQuantizationConfig, BrevitasQuantizer
from optimum.amd.brevitas.accelerate_utils import remove_hooks
from optimum.amd.brevitas.data_utils import compute_perplexity, get_dataset_for_model
from optimum.exporters.onnx import onnx_export_from_model
from optimum.utils import recurse_setattr
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
parser.add_argument(
    "--cpu-offload",
    action="store_true",
    default=False,
    help="Use Accelerate library to offload part of the model to RAM. This is useful when handling large models on a limited amount of GPU memory.",
)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

# Prepare the quantizer, specifying its configuration and loading the model.
qconfig = BrevitasQuantizationConfig(
    apply_gptq=args.apply_gptq,
    apply_weight_equalization=args.apply_weight_equalization,
    activations_equalization=args.activations_equalization,
    is_static=args.is_static,
    weights_symmetric=True,
    activations_symmetric=args.is_static,  # ONNX export only supports unsigned for dynamic quantization
)

quantizer = BrevitasQuantizer.from_pretrained(
    args.model, device_map="auto" if args.cpu_offload else "cuda:0", torch_dtype="auto"
)

# Load the data for calibration and evaluation.
calibration_dataset = get_dataset_for_model(
    args.model,
    qconfig=qconfig,
    dataset_name="wikitext2",
    tokenizer=tokenizer,
    nsamples=128,
    seqlen=args.seqlen,
    split="train",
    device="cuda:0" if not args.cpu_offload else None,
)

validation_dataset = get_dataset_for_model(
    args.model,
    qconfig=qconfig,
    dataset_name="wikitext2",
    tokenizer=tokenizer,
    nsamples=128,
    seqlen=args.seqlen,
    split="validation",
    device="cuda:0" if not args.cpu_offload else None,
)

perplexity = compute_perplexity(
    quantizer.model, validation_dataset, context_length=args.seqlen // 2, tokenizer=tokenizer
)

print(f"Perplexity (original model): {perplexity}")

quantized_model = quantizer.quantize(qconfig, calibration_dataset)

# Evaluation of the quantized model.
perplexity = compute_perplexity(
    quantized_model, validation_dataset, context_length=args.seqlen // 2, tokenizer=tokenizer
)
print(f"Perplexity (quantized model): {perplexity}")

print("Exporting the model to ONNX...")
if args.cpu_offload:
    # When exporting to ONNX, Accelerate's hooks need to be removed otherwise we have unwanted Cast nodes in the ONNX graph.
    remove_hooks(quantized_model)

quantized_model = quantized_model.to("cpu")
for name, param in quantized_model.named_parameters():
    if param.dtype in [torch.float16, torch.bfloat16]:
        recurse_setattr(quantized_model, name, torch.nn.Parameter(param.to(torch.float32)))
for name, param in quantized_model.named_buffers():
    if param.dtype in [torch.float16, torch.bfloat16]:
        recurse_setattr(quantized_model, name, torch.nn.Parameter(param.to(torch.float32)))

# Export to ONNX through optimum.exporters.
with torch.no_grad(), brevitas_proxy_export_mode(quantized_model, export_manager=StdQCDQONNXManager):
    onnx_export_from_model(
        quantized_model,
        "llm_quantized_onnx",
        task="text-generation-with-past",
        do_validation=False,
        no_post_process=True,
    )
