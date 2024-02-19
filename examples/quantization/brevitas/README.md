# Example of quantizing decoder class LLMs with Brevitas from Transformers checkpoints

The example shows an example on how to quantize a decoder-class LLM model through Brevitas:

- Definition and instantiation of a parametrizable `BrevitasQuantizer`.
- Optional conversion of the LLM model to an FX representation, leveraging Hugging Face transformers' tracer.
- Support of executing post-training quantization (PTQ) algorithms and validation (SmoothQuant, GPTQ), while leveraging CPU offload from Hugging Face accelerate.
- Validation of the quantized model.
- Export of the quantized model as ONNX, QDQ-style.

## Prerequisites

The examples were tested using:
- `brevitas>=0.10.2`
- `torch>=2.1.2`
- `transformers` installed from main (`pip install git+https://github.com/huggingface/transformers.git@4b236aed7618d90546cd2e8797dab5b4a24c5dce`)
- `optimum>=1.17.0`
- Optionally, `accelerate` installed from main (`pip install git+https://github.com/huggingface/accelerate.git`)

## Running the Example

```bash
python quantize_llm.py --model facebook/opt-125m --activations-equalization layerwise
```
To quantize OPT-125M with SmoothQuant post-training quantization algorithm.

For all the options, please check:

```bash
python quantize_llm.py --help
```

Most options can be applied independently. For optimal results, we suggest using the `--activations-equalization layerwise --apply-gtpq`, but GPTQ may take a long time, depending on your available hardware.

## RAM offloading

If quantizing large models, we recommend using the option `--cpu-offload` to offload the model to RAM using [Accelerate](https://huggingface.co/docs/accelerate/index), which loads the model's submodules dynamically to GPU.
