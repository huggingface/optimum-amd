# optimum-amd
AMD related optimizations for transformer models

## Ryzen AI
The AMD Ryzen™ AI Software Platform enables developers to take machine learning models trained in PyTorch or TensorFlow and run them on laptops powered by [Ryzen AI](https://www.amd.com/en/products/ryzen-ai). Ryzen AI is a dedicated AI accelerator integrated on-chip with the CPU cores. The Ryzen AI software platform intelligently optimizes tasks and workloads, freeing-up CPU and GPU resources, and ensuring optimal performance at lower power.

`optimum-amd` provides easy interface for loading and inference of Hugging Face models on Ryzen AI accelerator. 

### Ryzen AI Environment setup
A Ryzen AI environment needs to be enabled to use this library. Please refer to Ryzen AI's [Installation](https://ryzenai.docs.amd.com/en/latest/inst.html) and [Runtime Setup](https://ryzenai.docs.amd.com/en/latest/runtime_setup.html).

## Install
Optimum AMD is a fast-moving project, and you may want to install from source.

`pip install git+https://github.com/huggingface/optimum-amd.git`

### Installing in developer mode

If you are working on the `optimum-amd` code then you should use an editable install
by cloning and installing `optimum` and `optimum-amd`:

```
git clone https://github.com/huggingface/optimum
git clone https://github.com/huggingface/optimum-amd
pip install -e optimum -e optimum-amd
```

Now whenever you change the code, you'll be able to run with those changes instantly.


## How to use it?

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
+ model = RyzenAIModelForImageClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
cls_pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
outputs = cls_pipe(image)
```

If you find any issue while using those, please open an issue or a pull request.