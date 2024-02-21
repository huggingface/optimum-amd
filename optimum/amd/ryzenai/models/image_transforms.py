# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import cv2

from transformers.image_transforms import pad
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
)


def letterbox_image(
    image,
    target_shape=(640, 640),
    padding_value=114,
    resample=PILImageResampling.BILINEAR,
    input_data_format=None,
):
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    if input_data_format == ChannelDimension.FIRST:
        current_shape = image.shape[1:]
    else:
        current_shape = image.shape[:2]
    target_shape = (target_shape, target_shape) if isinstance(target_shape, int) else target_shape
    scaling_ratio = min(target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])

    unscaled_dimensions = tuple(round(dim * scaling_ratio) for dim in current_shape)
    padding_height, padding_width = [
        (target - unscaled) / 2 for target, unscaled in zip(target_shape, unscaled_dimensions)
    ]

    image = cv2.resize(image, (unscaled_dimensions[1], unscaled_dimensions[0]), interpolation=cv2.INTER_LINEAR)

    # TODO: Replace the cv2 resize with PIL based resize after accuracy tests
    # image = resize(
    #     image,
    #     unscaled_dimensions,
    #     resample=resample,
    #     input_data_format=input_data_format,
    # )

    top, bottom = int(round(padding_height - 0.1)), int(round(padding_height + 0.1))
    left, right = int(round(padding_width - 0.1)), int(round(padding_width + 0.1))
    padding = ((top, bottom), (left, right))

    image = pad(
        image,
        padding=padding,
        constant_values=padding_value,
        input_data_format=input_data_format,
    )

    return image
