# Optimum-AMD

🤗 Optimum-AMD is an extension to Hugging Face libraries enabling performance optimizations for [ROCm for AMD GPUs](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html) and [Ryzen AI for AMD](https://ryzenai.docs.amd.com/en/latest/index.html) NPU accelerator.

## Install

Optimum-AMD library can be installed through pip:

```bash
pip install --upgrade-strategy eager optimum[amd]
```

Installation is possible from source as well:

```bash
git clone https://github.com/huggingface/optimum-amd.git
cd optimum-amd
pip install -e .
```

## ROCm support for AMD GPUs

Hugging Face libraries natively support AMD GPUs through [PyTorch for ROCm](https://pytorch.org/get-started/locally/) with zero code change.

🤗 Transformers natively supports [Flash Attention 2](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2), [GPTQ quantization](https://huggingface.co/docs/transformers/main_classes/quantization#autogptq-integration) with ROCm. [🤗 Text Generation Inference](https://huggingface.co/docs/text-generation-inference/quicktour) library for LLM deployment has native ROCm support, with Flash Attention 2, [Paged Attention](https://huggingface.co/docs/text-generation-inference/conceptual/paged_attention), fused positional encoding & layer norm kernels support.

[Find out more about these integrations in the documentation](https://huggingface.co/docs/optimum/main/en/amd/amdgpu/overview)!

In the future, Optimum-AMD may host more ROCm-specific optimizations.

### How to use it: Text Generation Inference

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) library for LLM deployment supports AMD Instinct MI210/MI250 GPUs. Deployment can be done as follow:

1. Install [ROCm5.7](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) to the host machine
2. Example LLM server setup: launch a Falcon-7b model server on the ROCm-enabled docker.
```bash
model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.2-rocm --model-id $model
```
3. Client setup: Open another shell and run:
```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

### How to use it: ONNX Runtime with ROCm

Optimum ONNX Runtime integration [supports ROCm for AMD GPUs](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/amdgpu). Usage is as follow:

1. Install [ROCm 5.7](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) on the host machine.
2. Use the example [Dockerfile](https://github.com/huggingface/optimum-amd/blob/main/docker/onnx-runtime-amd-gpu/Dockerfile) or install `onnxruntime-rocm` package locally from source. Pip wheels are not available at the time.
3. Run a BERT text classification ONNX model by using `ROCMExecutionProvider`:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

ort_model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True,
    provider="ROCMExecutionProvider",
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pipe = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipe("Both the music and visual were astounding, not to mention the actors performance.")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9997727274894714}]
```

## Ryzen AI

AMD's [Ryzen™ AI](https://www.amd.com/en/products/ryzen-ai) family of laptop processors provide users with an integrated Neural Processing Unit (NPU) which offloads the host CPU and GPU from AI processing tasks. Ryzen™ AI software consists of the Vitis™ AI execution provider (EP) for ONNX Runtime combined with quantization tools and a [pre-optimized model zoo](https://huggingface.co/models?other=RyzenAI). All of this is made possible based on Ryzen™ AI technology built on AMD XDNA™ architecture, purpose-built to run AI workloads efficiently and locally, offering a host of benefits for the developer innovating the next groundbreaking AI app.

Optimum-AMD provides easy interface for loading and inference of Hugging Face models on Ryzen AI accelerator.

### Ryzen AI Environment setup
A Ryzen AI environment needs to be enabled to use this library. Please refer to Ryzen AI's [Installation](https://ryzenai.docs.amd.com/en/latest/inst.html) and [Runtime Setup](https://ryzenai.docs.amd.com/en/latest/runtime_setup.html).

### How to use it?

* Quantize the ONNX model with Optimum or using the RyzenAI quantization tools

For more information on quantization refer to [Model Quantization](https://ryzenai.docs.amd.com/en/latest/modelport.html) guide.

* Load model with Ryzen AI class

To load a model and run inference with RyzenAI, you can just replace your `AutoModelForXxx` class with the corresponding `RyzenAIModelForXxx` class. 

```diff
import requests
from PIL import Image

- from transformers import AutoModelForImageClassification
+ from optimum.amd.ryzenai import RyzenAIModelForImageClassification
from transformers import AutoFeatureExtractor, pipeline

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model_id = <path of the model>
- model = AutoModelForImageClassification.from_pretrained(model_id)
+ model = RyzenAIModelForImageClassification.from_pretrained(model_id, vaip_config=<path to config file>)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
cls_pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
outputs = cls_pipe(image)
```

### Tests

An extensive test suite is included to test the library's behavior. The test suite can be found in the tests folder. To run the tests, navigate to the root of the repository and specify a path to a subfolder or a specific test file.

Before running the tests, make sure to install the necessary dependencies by using the following command:

```bash
pip install .[tests]
```

and then run,

```bash
pytest -s -v ./tests/ryzenai/
```

You can also specify a smaller set of tests in order to test only the feature you're working on.

#### Running Slow Tests

By default, slow tests are skipped, but you can set the `RUN_SLOW` environment variable to `1` to run them. 

```bash
RUN_SLOW=1 pytest -s -v ./tests/ryzenai/
```

> **_NOTE:_**  Enabling slow tests will involve downloading several gigabytes of models. Ensure you have enough disk space and a good internet connection!

**Windows Powershell**

For Windows Powershell, use the following command to run tests with slow tests enabled:

```bash
$env:RUN_SLOW=1; pytest -s -v ./tests/ryzenai/
```

If you find any issue while using those, please open an issue or a pull request.
