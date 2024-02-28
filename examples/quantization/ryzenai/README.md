# Quantization
Optimum-amd provides a tool that enables you to apply quantization on many models hosted on the Hugging Face Hub using our RyzenAIOnnxQuantizer.

## Static quantization
The quantization process is abstracted via the AutoQuantizationConfig and the RyzenAIOnnxQuantizer classes. The former allows you to specify how quantization should be done, while the latter effectively handles quantization.

You can read the [VAI_Q_ONNX user guide](https://gitenterprise.xilinx.com/VitisAI-CP/vai_q_onnx/blob/dev/README.md) to learn about VAI_Q_ONNX quantization.

### Creating an AutoQuantizationConfig
The AutoQuantizationConfig class is used to specify how quantization should be done. The class can be initialized using the ipu_cnn_config() method.
```python
from optimum.amd.ryzenai import AutoQuantizationConfig
quantization_config = AutoQuantizationConfig.ipu_cnn_config()

```

### Creating an RyzenAIOnnxQuantizer
The RyzenAIOnnxQuantizer class is used to quantize your ONNX model. The class can be initialized using the from_pretrained() method.
Using a local ONNX model from a directory.
```python
from optimum.amd.ryzenai import RyzenAIOnnxQuantizer
quantizer = RyzenAIOnnxQuantizer.from_pretrained("path/to/model")
```

### Example
The RyzenAIOnnxQuantizer class can be used to quantize statically your ONNX model. When applying post-training static quantization, we need to generate the calibration dataset in order to perform the calibration step.
Below you will find an easy end-to-end example on how to quantize statically timm/resnet50.a1_in1k.

```python
from functools import partial
import timm
from optimum.amd.ryzenai import (
    AutoQuantizationConfig,
    RyzenAIModelForImageClassification,
    RyzenAIOnnxQuantizer,
)

model_id = "timm/resnet50.a1_in1k"

onnx_model = RyzenAIModelForImageClassification.from_pretrained(model_id, export=True)
# preprocess config
data_config = timm.data.resolve_data_config(pretrained_cfg=onnx_model.config)
transforms = timm.data.create_transform(**data_config, is_training=False)

def preprocess_fn(ex, transforms):
    image = ex["image"]
    if image.mode == "L":
        # Three channels.
        image = image.convert("RGB")
    pixel_values = transforms(image)

    return {"pixel_values": pixel_values}

# quantize
quantizer = RyzenAIOnnxQuantizer.from_pretrained(onnx_model)
quantization_config = AutoQuantizationConfig.ipu_cnn_config()

calibration_dataset = quantizer.get_calibration_dataset(
    "imagenet-1k",
    preprocess_function=partial(preprocess_fn, transforms=transforms),
    num_samples=10,
    dataset_split="train",
    preprocess_batch=False,
    streaming=True,
)
quantizer.quantize(
    quantization_config=quantization_config, dataset=calibration_dataset, save_dir="quantized_model"
)
```