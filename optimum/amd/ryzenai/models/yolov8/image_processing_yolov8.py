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

from ..detection_utils import non_max_suppression, scale_coords
from ..image_transforms import letterbox_image


def make_anchor(input, ny, nx, grid_cell_offset=0.5):
    t, d = input.dtype, input.device

    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    y, x = torch.meshgrid(y + grid_cell_offset, x + grid_cell_offset, indexing="ij")

    return torch.stack((x, y), -1).view(-1, 2)


def dfl(x, c1=16):
    b, c, a = x.shape

    weights = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
    inter = x.view(b, 4, c1, a).transpose(2, 1).softmax(1)
    return (inter * weights).sum(dim=1, keepdim=True).view(b, 4, a)


def postprocess(inputs, reg_max=16, num_classes=80, stride=[8, 16, 32]):
    nl = len(stride)
    no = num_classes + reg_max * 4

    box, cls = torch.cat([xi.view(inputs[0].shape[0], no, -1) for xi in inputs], 2).split(
        (reg_max * 4, num_classes), 1
    )
    distance = dfl(box).chunk(2, 1)

    anchors, strides = [], []
    for i in range(nl):
        _, _, ny, nx = inputs[i].shape
        anchor = make_anchor(inputs[i], ny, nx)
        ustride = torch.full((ny * nx, 1), stride[i], dtype=inputs[i].dtype, device=inputs[i].device)

        anchors.append(anchor)
        strides.append(ustride)

    anchors = torch.cat(anchors).transpose(0, 1).unsqueeze(0)
    strides = torch.cat(strides).transpose(0, 1)
    distance = dfl(box).chunk(2, 1)

    x1_y1 = anchors - distance[0]
    x2_y2 = anchors + distance[1]

    dbox = torch.cat(((x2_y2 + x1_y1) / 2, x2_y2 - x1_y1), dim=1) * strides

    y = torch.cat((dbox, cls.sigmoid()), 1)

    return y


class YoloV8ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255.0,
        num_classes: int = 80,
        stride: List[int] = [8, 16, 32],
        reg_max: int = 16,
        **kwargs,
    ):
        size = size if size is not None else {"height": 640, "width": 640}

        super().__init__(**kwargs)
        self.size = size
        self.data_format = ChannelDimension.LAST
        self.rescale_factor = rescale_factor
        self.num_classes = num_classes
        self.stride = stride
        self.reg_max = reg_max

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
            if input_data_format == ChannelDimension.FIRST:
                image = image.transpose((2, 0, 1))
                input_data_format = ChannelDimension.LAST

            target_sizes.append(image.shape)

            image = letterbox_image(
                image,
                [self.size["height"], self.size["width"]],
                input_data_format=input_data_format,
            )

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
        nms_threshold: float = 0.7,
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

        predictions = postprocess(outputs, num_classes=self.num_classes, reg_max=self.reg_max, stride=self.stride)

        dets = non_max_suppression(
            predictions.transpose(2, 1),
            threshold,
            nms_threshold,
            agnostic=agnostic_nms,
            class_conf_start_index=4,
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
