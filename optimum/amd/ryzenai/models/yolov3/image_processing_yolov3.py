# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    flip_channel_order,
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

from ..detection_utils import non_max_suppression, scale_coords
from ..image_transforms import letterbox


def make_grid(anchor, nx=20, ny=20):
    d = anchor.device
    t = anchor.dtype

    shape = 1, 1, ny, nx, 2

    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv = torch.meshgrid(y, x, indexing="ij")

    grid = torch.stack((xv, yv), 2).expand(shape)
    anchor_grid = (anchor).reshape(3, 2).view((1, len(anchor) // 2, 1, 1, 2))

    return grid, anchor_grid


def postprocess(inputs, anchors, num_classes=80, stride=[8, 16, 32], shapes=[80, 40, 20]):
    nl = len(anchors)
    no = num_classes + 5

    outputs = []
    for i in range(nl):
        bs, _, ny, nx = inputs[i].shape
        grid, anchor_grid = make_grid(anchors[2 - i], nx, ny)

        inputs[i] = inputs[i].view(bs, nl, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        xy = (torch.sigmoid(inputs[i][..., 0:2]) + grid) * stride[2 - i]
        wh = (torch.exp(inputs[i][..., 2:4])) * anchor_grid

        conf = torch.sigmoid_(inputs[i][..., 4:])
        y = torch.cat((xy, wh, conf), -1)
        outputs.append(y.view(bs, -1, no))
    return torch.cat(outputs, 1)


class YoloV3ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255.0,
        num_classes: int = 80,
        stride: List[int] = [8, 16, 32],
        anchors: List[List[int]] = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
        ],
        **kwargs,
    ):
        size = size if size is not None else {"height": 416, "width": 416}

        super().__init__(**kwargs)
        self.size = size
        self.data_format = ChannelDimension.LAST
        self.stride = stride
        self.rescale_factor = rescale_factor
        self.anchors = anchors
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
        input_data_format = infer_channel_dimension_format(images[0])

        preprocessed_images = []
        target_sizes = []
        for image in images:
            image = flip_channel_order(image, input_data_format=input_data_format)
            if input_data_format == ChannelDimension.FIRST:
                image = image.transpose((2, 0, 1))
                input_data_format = ChannelDimension.LAST

            target_sizes.append(image.shape)

            image = letterbox(image, [self.size["height"], self.size["width"]], stride=self.stride, auto=False)[0]
            image = image.transpose((2, 0, 1))
            input_data_format = ChannelDimension.FIRST

            image = flip_channel_order(image, input_data_format=input_data_format)

            image = np.ascontiguousarray(image, dtype=np.float32)

            image = rescale(
                image=image, scale=self.rescale_factor, data_format=data_format, input_data_format=input_data_format
            )

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

        if merge_nms:
            raise ValueError("Merge NMS is not yet supported!")

        outputs = list(outputs.values())

        if not isinstance(outputs[0], torch.Tensor):
            outputs = [torch.tensor(out) for out in outputs]

        if data_format == ChannelDimension.LAST:
            outputs = [torch.permute(out, (0, 3, 1, 2)) for out in outputs]

        anchors = torch.tensor(self.anchors)
        predictions = postprocess(outputs, anchors, self.num_classes, self.stride)
        has_confidence = predictions[..., 4] > threshold

        dets = non_max_suppression(
            predictions,
            has_confidence,
            threshold,
            nms_threshold,
            agnostic=agnostic_nms,
            max_detections=max_detections,
        )

        results = []
        for i, det in enumerate(dets):
            if target_sizes is not None:
                det[:, :4] = scale_coords(
                    (self.size["height"], self.size["width"]),
                    target_sizes[i],
                    det[:, :4],
                ).round()

            results.append({"scores": det[:, 4], "labels": det[:, 5], "boxes": det[:, :4]})

        return results
