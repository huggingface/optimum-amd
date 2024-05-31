# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    rescale,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
)
from transformers.utils import TensorType

from ..detection_utils import scale_coords
from ..image_transforms import letterbox_image


class YoloV10ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255.0,
        num_classes: int = 80,
        **kwargs,
    ):
        size = size if size is not None else {"height": 640, "width": 640}

        super().__init__(**kwargs)
        self.size = size
        self.data_format = ChannelDimension.FIRST
        self.rescale_factor = rescale_factor
        self.num_classes = num_classes

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
            # image = image[..., ::-1]
            target_sizes.append(image.shape)

            image = letterbox_image(
                image,
                [self.size["height"], self.size["width"]],
                input_data_format=input_data_format,
            )

            image = rescale(
                image=image, scale=self.rescale_factor, data_format=data_format, input_data_format=input_data_format
            )
            image = np.ascontiguousarray(image, dtype=np.float32)

            preprocessed_images.append(image)

        data = {"pixel_values": preprocessed_images, "target_sizes": target_sizes}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.25,
        nms_threshold: float = 0.45,
        target_sizes: Union[TensorType, List[Tuple]] = None,
        agnostic_nms=False,
        merge_nms=False,
        max_detections=1000,
        data_format: Union[str, ChannelDimension] = None,
    ):
        data_format = data_format if data_format is not None else self.data_format

        outputs = list(outputs.values())

        if not isinstance(outputs[0], torch.Tensor):
            outputs = [torch.tensor(out) for out in outputs]

        predictions = outputs[0]

        # if data_format == ChannelDimension.LAST:
        #     outputs = [torch.permute(out, (0, 3, 1, 2)) for out in outputs]
        has_confidence = predictions[..., 4] > threshold  # Candidates
        dets = [pred[has_confidence[idx]] for idx, pred in enumerate(predictions)]

        results = []
        for i, det in enumerate(dets):
            if target_sizes is not None:
                det[:, :4] = scale_coords(
                    (self.size["height"], self.size["width"]),
                    target_sizes[i],
                    det[:, :4],
                ).round()

            results.append({"scores": det[:, 4], "labels": det[:, 5].to(torch.int64), "boxes": det[:, :4]})

        return results
