#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import time
from argparse import ArgumentParser
from functools import partial

import numpy as np
import onnxruntime
import timm
from timm.data import create_dataset, create_loader
from timm.utils import AverageMeter

from optimum.amd.ryzenai import (
    AutoQuantizationConfig,
    RyzenAIModelForImageClassification,
    RyzenAIOnnxQuantizer,
)


def parse_args():
    parser = ArgumentParser("RyzenAIQuantization")
    parser.add_argument("--data-path", metavar="DIR", required=True, help="path to dataset")
    parser.add_argument(
        "--model_id", type=str, default="timm/resnet50.a1_in1k", help='Model id, default to "timm/resnet50.a1_in1k"'
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet-1k", help='Calibration dataset, default to "imagenet-1k"'
    )
    parser.add_argument(
        "--onnx-output-opt", default="", type=str, metavar="PATH", help="path to output optimized onnx graph"
    )
    parser.add_argument("--profile", action="store_true", default=False, help="Enable profiler output.")
    parser.add_argument(
        "-j", "--workers", default=2, type=int, metavar="N", help="number of data loading workers (default: 2)"
    )
    parser.add_argument("-b", "--batch-size", default=1, type=int, metavar="N", help="mini-batch size (default: 1)")
    args, _ = parser.parse_known_args()
    return args


def main(args):
    model_id = args.model_id

    onnx_model = RyzenAIModelForImageClassification.from_pretrained(
        model_id, export=True, provider="CPUExecutionProvider"
    )
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
    quantization_config = AutoQuantizationConfig.cpu_cnn_config()

    calibration_dataset = quantizer.get_calibration_dataset(
        args.dataset,
        preprocess_function=partial(preprocess_fn, transforms=transforms),
        num_samples=200,
        dataset_split="validation",
        preprocess_batch=False,
        streaming=True,
    )
    quantizer.quantize(
        quantization_config=quantization_config, dataset=calibration_dataset, save_dir="quantized_model"
    )

    # Set graph optimization level
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.profile:
        sess_options.enable_profiling = True
    if args.onnx_output_opt:
        sess_options.optimized_model_filepath = args.onnx_output_opt

    session = onnxruntime.InferenceSession("quantized_model/model_quantized.onnx", sess_options)

    data_config = timm.data.resolve_data_config(pretrained_cfg=onnx_model.config.to_dict(), use_test_size=True)

    loader = create_loader(
        create_dataset("", args.data_path),
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=data_config["crop_pct"],
    )

    input_name = session.get_inputs()[0].name

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for i, (input, target) in enumerate(loader):
        # run the net and return prediction
        output = session.run([], {input_name: input.data.numpy()})
        output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy_np(output, target.numpy())
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f" * Prec@1 {top1.avg:.3f} ({100-top1.avg:.3f}) Prec@5 {top5.avg:.3f} ({100.-top5.avg:.3f})")


def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


if __name__ == "__main__":
    args = parse_args()
    main(args)
