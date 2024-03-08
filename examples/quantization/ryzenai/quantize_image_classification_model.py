from functools import partial
import timm
from argparse import ArgumentParser
from optimum.amd.ryzenai import (
    AutoQuantizationConfig,
    RyzenAIModelForImageClassification,
    RyzenAIOnnxQuantizer,
)

def parse_args():
    parser = ArgumentParser("RyzenAIQuantization")
    parser.add_argument("--model_id", type=str, default='timm/resnet50.a1_in1k', help='Model id, default to "timm/resnet50.a1_in1k"')
    parser.add_argument("--dataset", type=str, default='imagenet-1k', help='Calibration dataset, default to ""imagenet-1k""')
    args, _ = parser.parse_known_args()
    return args

def main(args):
    model_id = args.model_id

    onnx_model = RyzenAIModelForImageClassification.from_pretrained(model_id, export=True)
    # preprocess config
    data_config = timm.data.resolve_data_config(pretrained_cfg=onnx_model.config.to_dict())
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
        args.dataset,
        preprocess_function=partial(preprocess_fn, transforms=transforms),
        num_samples=10,
        dataset_split="train",
        preprocess_batch=False,
        streaming=True,
    )
    quantizer.quantize(
        quantization_config=quantization_config, dataset=calibration_dataset, save_dir="quantized_model"
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)