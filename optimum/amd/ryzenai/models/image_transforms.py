# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps


def letterbox(
    image,
    target_shape=(640, 640),
    padding_color=(114, 114, 114),
    auto=True,
    stretch_fill=False,
    allow_scale_up=True,
    stride=32,
):
    # Get the current shape of the image [height, width]
    current_shape = image.shape[:2]

    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)

    # Compute the scaling ratio (new / old)
    scaling_ratio = min(target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])

    # Scale ratio (new / old)
    if not allow_scale_up:
        scaling_ratio = min(scaling_ratio, 1.0)

    # Compute padding ratios for width and height
    padding_ratios = scaling_ratio, scaling_ratio

    # Calculate the dimensions after scaling (without padding)
    unscaled_dimensions = int(round(current_shape[1] * scaling_ratio)), int(round(current_shape[0] * scaling_ratio))

    # Calculate the padding amounts in width and height
    padding_width, padding_height = target_shape[1] - unscaled_dimensions[0], target_shape[0] - unscaled_dimensions[1]

    # Adjust padding to ensure stride multiple if auto is True
    if auto:
        padding_width, padding_height = np.mod(padding_width, stride), np.mod(padding_height, stride)
    elif stretch_fill:
        # If stretch_fill is True, set padding to zero and unscaled_dimensions to target_shape
        padding_width, padding_height = 0.0, 0.0
        unscaled_dimensions = (target_shape[1], target_shape[0])
        padding_ratios = target_shape[1] / current_shape[1], target_shape[0] / current_shape[0]

    padding_width /= 2
    padding_height /= 2

    if current_shape[::-1] != unscaled_dimensions:
        image = cv2.resize(image, unscaled_dimensions, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(padding_height - 0.1)), int(round(padding_height + 0.1))
    left, right = int(round(padding_width - 0.1)), int(round(padding_width + 0.1))

    # Apply border padding to the image
    image = ImageOps.expand(Image.fromarray(image), border=(left, top, right, bottom), fill=padding_color)
    image = np.array(image)

    return image, padding_ratios, (padding_width, padding_height)
