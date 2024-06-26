#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import shutil
import tarfile
import time
from argparse import ArgumentParser

import numpy as np
import onnxruntime
import timm
import torch
from datasets import Dataset
from timm.data import create_dataset, create_loader
from timm.models import create_model
from timm.utils import AverageMeter
from tqdm import tqdm

from optimum.amd.ryzenai import (
    AutoQuantizationConfig,
    RyzenAIModelForImageClassification,
    RyzenAIOnnxQuantizer,
)


"""
If you already have an ImageNet datasets, you can directly use your dataset path with' --calib-data-path' and '--eval-data-path'.

To prepare the test data, please check the download section of the main website:
https://huggingface.co/datasets/imagenet-1k/tree/main/data.
You need to register and download **val_images.tar.gz**.

For example:
python test_timm_quant_and_eval.py -c $PATH/calib_100 -e $PATH/val_data -m timm/resnetv2_50.a1h_in1k
or
python test_timm_quant_and_eval.py -v $PATH/val_images.tar.gz -m timm/resnetv2_50.a1h_in1k

Float Accuracy of resnet50.tv_in1k:
- Prec@1: 76.128%
- Prec@5: 92.858%

Quantization Accuracy of resnet50.tv_in1k:
- Prec@1: 74.384%
- Prec@5: 91.968%
"""


def parse_args():
    parser = ArgumentParser("RyzenAIQuantization")
    parser.add_argument("-v", "--val-path", metavar="DIR", required=False, help="path to dataset")
    parser.add_argument("-c", "--calib-data-path", metavar="DIR", required=False, help="path to dataset")
    parser.add_argument("-e", "--eval-data-path", metavar="DIR", required=False, help="path to dataset")
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        default="timm/resnet50.tv_in1k",
        help='Model id, default to "timm/resnet50.tv_in1k"',
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
    if args.val_path is None and (args.calib_data_path is None and args.eval_data_path is None):
        parser.error("You must either provide --calib-data-path and --eval-data-path, or --val-path")

    return args


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    # prepare val data and calib data
    if (args.calib_data_path is None and args.eval_data_path is None) and args.val_path is not None:
        source_folder = "val_data"
        calib_data_path = "calib_data"
        if os.path.isdir(source_folder) and os.path.isdir(calib_data_path):
            print(
                f"Detected that {source_folder} and {calib_data_path} already exist, skipping the creation of the calibration dataset."
            )
        else:
            os.makedirs(source_folder, exist_ok=True)
            with tarfile.open(args.val_path, "r:gz") as tar:
                tar.extractall(path=source_folder)

            if not os.path.exists(source_folder):
                raise ValueError("The val_data does not exist.")
            files = os.listdir(source_folder)
            for filename in files:
                if not filename.startswith("ILSVRC2012_val_") or not filename.endswith(".JPEG"):
                    continue

                n_identifier = filename.split("_")[-1].split(".")[0]
                folder_name = n_identifier
                folder_path = os.path.join(source_folder, folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = os.path.join(source_folder, filename)
                destination = os.path.join(folder_path, filename)
                shutil.move(file_path, destination)

            print("File organization complete.")

            if not os.path.exists(calib_data_path):
                os.makedirs(calib_data_path)

            destination_folder = calib_data_path

            subfolders = os.listdir(source_folder)
            cnt = 0
            for subfolder in subfolders:
                source_subfolder = os.path.join(source_folder, subfolder)
                destination_subfolder = os.path.join(destination_folder, subfolder)
                os.makedirs(destination_subfolder, exist_ok=True)

                files = os.listdir(source_subfolder)

                if files:
                    file_to_copy = files[0]
                    source_file = os.path.join(source_subfolder, file_to_copy)
                    destination_file = os.path.join(destination_subfolder, file_to_copy)

                    shutil.copy(source_file, destination_file)
                    cnt += 1
                if cnt >= 200:
                    break

            print("Creating calibration dataset complete.")
        args.calib_data_path = calib_data_path
        args.eval_data_path = source_folder

    model_id = args.model_id

    onnx_model = RyzenAIModelForImageClassification.from_pretrained(
        model_id, export=True, provider="CPUExecutionProvider"
    )
    # # preprocess config
    model = create_model(model_id, pretrained=False)
    data_config = timm.data.resolve_data_config(model=model, use_test_size=True)

    # # quantize
    quantizer = RyzenAIOnnxQuantizer.from_pretrained(onnx_model)
    quantization_config = AutoQuantizationConfig.cpu_cnn_config()
    quantization_config.extra_options = {
        "FastFinetune": {
            "BatchSize": 2,
            "FixedSeed": 1705472343,
            "NumBatches": 1,
            "NumIterations": 10000,
            "LearningRate": 0.1,
            "OptimAlgorithm": "adaround",
            "OptimDevice": "cuda:0",  # or 'cpu'
            "LRAdjust": (),
            "SelectiveUpdate": False,
            "EarlyStop": True,
            "DropRatio": 0.75,
            "RegParam": 0.01,  # default
            "BetaRange": (20, 2),  # default
            "WarmStart": 0.2,  # default
        },
        "Percentile": 99.9999,
    }

    calib_loader = create_loader(
        create_dataset("", args.calib_data_path),
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=data_config["crop_pct"],
    )

    data_list = []

    for batch in calib_loader:
        data = batch[0]
        data_list.append(data)

    data_list = torch.cat(data_list, dim=0)

    data_np = data_list.numpy()

    data_dict = {
        "pixel_values": data_np,
    }

    calibration_dataset = Dataset.from_dict(data_dict)

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

    loader = create_loader(
        create_dataset("", args.eval_data_path),
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
    for input, target in tqdm(loader, desc="Processing"):
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
