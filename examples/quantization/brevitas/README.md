# Example of quantizing OPT model with Brevitas from Transformers checkpoints

The example shows an example on how to quantize an OPT model through Brevitas:

- Definition and instantiation of a parametrizable `BrevitasQuantizer`.
- Optional conversion of the OPT model to an FX representation, leveraging Hugging Face transformers' tracer.
- Support of executing post-training quantization (PTQ) algorithms and validation (SmoothQuant, GPTQ), while leveraging CPU offload from Hugging Face accelerate.
- Validation of the quantized model.
- Export of the quantized model as ONNX, QDQ-style.

## Prerequisites

The examples were tested using:
 - PyTorch v2.1.2
 - transformers installed from main
 - accelerate install from main
 - optimum installed from main

## Running the Example

```bash
python quantize_opt.py --model facebook/opt-125m --activations-equalization layerwise
```
To quantize OPT with SmoothQuant post-training quantization algorithm.

For all the options, please check:

```bash
python quantize_opt.py --help
```

Most options can be applied independently.
