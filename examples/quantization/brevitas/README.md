# Example of quantizing decoder class LLMs with Brevitas from Transformers checkpoints

The example shows an example on how to quantize a decoder-class LLM model through Brevitas:

- Definition and instantiation of a parametrizable `BrevitasQuantizer`.
- Optional conversion of the LLM model to an FX representation, leveraging Hugging Face transformers' tracer.
- Support of executing post-training quantization (PTQ) algorithms and validation (SmoothQuant, GPTQ), while leveraging CPU offload from Hugging Face accelerate.
- Validation of the quantized model.
- Export of the quantized model as ONNX, QOP-style.

## Prerequisites

The examples were tested using:
- `python>=3.9` (required for QOP ONNX export)
- `brevitas` installed from dev (`pip install git+https://github.com/Xilinx/brevitas.git@dev`)
- `torch>=2.2`
- `transformers>=4.38.0`
- `optimum>=1.17.0`
- `optimum-amd` installed from main (`pip install git+https://github.com/huggingface/optimum-amd`)
- `accelerate>=0.30.0`

Note, you can install all the prerequisites with:

```bash
pip install 'brevitas @ git+https://github.com/Xilinx/brevitas.git@dev' 'optimum-amd[brevitas] @ git+https://github.com/huggingface/optimum-amd.git@main'
```

## Running the Example

To quantize OPT-125M with SmoothQuant post-training quantization algorithm, use:

```bash
python quantize_llm.py --model facebook/opt-125m --activations-equalization layerwise
```

For all the options, please check:

```bash
python quantize_llm.py --help
```

Most options can be applied independently. For optimal results, we suggest using the `--activations-equalization layerwise --apply-gptq`, but GPTQ may take a long time, depending on your available hardware.

## RAM offloading

If quantizing large models, we recommend using the option `--device auto` to offload the model to RAM using [Accelerate](https://huggingface.co/docs/accelerate/index), which loads the model's submodules dynamically to GPU.
