# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import (
    get_resize_output_image_size,
    rescale,
    resize,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
)
from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, TensorType


class SemanticFPNImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255.0,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        **kwargs,
    ):
        size = size if size is not None else {"height": 256, "width": 512}

        super().__init__(**kwargs)
        self.size = size
        self.data_format = ChannelDimension.LAST
        self.rescale_factor = rescale_factor

        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        size = get_size_dict(size, default_to_square=False)
        if "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        size = get_resize_output_image_size(
            input_image=image,
            size=size,
            default_to_square=False,
            input_data_format=input_data_format,
        )
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        data_format = data_format if data_format is not None else self.data_format
        self.data_format = data_format

        images = make_list_of_images(images)

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]
        # We assume that all images have the same channel dimension format.

        preprocessed_images = []
        target_sizes = []
        for image in images:
            input_data_format = infer_channel_dimension_format(images[0])

            if input_data_format == ChannelDimension.FIRST:
                image = image.transpose((2, 0, 1))
                input_data_format = ChannelDimension.LAST

            target_sizes.append(tuple(image.shape[:2]))

            image = self.resize(image, size=self.size, input_data_format=input_data_format)

            image = rescale(
                image=image, scale=self.rescale_factor, data_format=data_format, input_data_format=input_data_format
            )

            image = self.normalize(
                image,
                mean=self.image_mean,
                std=self.image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )

            image = np.ascontiguousarray(image, dtype=np.float32)

            preprocessed_images.append(image)

        data = {"pixel_values": preprocessed_images, "target_sizes": target_sizes}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: Union[TensorType, List[Tuple]] = None,
    ):
        outputs = list(outputs.values())

        if not isinstance(outputs[0], torch.Tensor):
            outputs = [torch.tensor(out) for out in outputs]

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        if self.data_format == ChannelDimension.LAST:
            outputs = torch.permute(outputs, (0, 3, 1, 2))

        if target_sizes is not None:
            semantic_segmentation = []
            for idx in range(len(target_sizes)):
                resized_logits = torch.nn.functional.interpolate(
                    outputs[idx].unsqueeze(dim=0), size=tuple(target_sizes[idx]), mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = outputs.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation
