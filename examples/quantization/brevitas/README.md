# Example of quantizing OPT model with Brevitas from Transformers checkpoints

The example with shows an example on how to quantize an OPT model through Brevitas: 
 - Instantiation of facebook's OPT model from HuggingFace transformers
 - The definition and instantiation of a parametrizable `BrevitasQuantizer` which extends `OptimumQuantizer`
 - Optional conversion of the OPT model to an FX representation, leveraging HuggingFace transformers' tracer
 - Prototype support of executing PTQ algorithms and validation, while leveraging CPU offload from HuggingFace accelerate
 - Quantization of the OPT model using Brevitas' PTQ algorithms
   - Optionally converting the `OPTAttention` layer to `torch.nn.MultiheadAttention` for finer-grained quantization of MHA layers
   - Optionally applying: SmoothQuant, GPTQ, weight equalization algorithms
 - Validation of the quantized model
 - (WIP) Export to quantized ONNX (QDQ-style), leveraging HuggingFace optimum's ONNX export

## Prerequisites

The examples were tested using:
 - PyTorch v2.1.2
 - transformers installed from main
 - accelerate install from main
 - optimum installed from main

## Running the Example

```bash
python quantize_opt.py --apply-act-equalization fx --with-fx
```
To quantize OPT with the graph-based SmoothQuant PTQ algorithm enabled.

For all the options, please check:

```bash
python quantize_opt.py --help
```

Most options can be applied independently.
