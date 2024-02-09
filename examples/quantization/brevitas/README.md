# Example of quantizing decoder class LLMs with Brevitas from Transformers checkpoints

The example shows an example on how to quantize a decoder-class LLM model through Brevitas:

- Definition and instantiation of a parametrizable `BrevitasQuantizer`.
- Optional conversion of the LLM model to an FX representation, leveraging Hugging Face transformers' tracer.
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
python quantize_llm.py --model facebook/opt-125m --activations-equalization layerwise
```
To quantize OPT-125M with SmoothQuant post-training quantization algorithm.

For all the options, please check:

```bash
python quantize_llm.py --help
```

Most options can be applied independently.
For optimal results, we suggest using the `--activations-equalization layerwise --apply-gtpq`,
but GPTQ may take a long time, depending on your available hardware.
