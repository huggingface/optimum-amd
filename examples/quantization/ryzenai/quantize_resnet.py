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