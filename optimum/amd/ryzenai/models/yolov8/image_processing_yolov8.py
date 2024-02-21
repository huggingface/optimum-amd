# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

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


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def postprocess(inputs, reg_max=16, num_classes=80, stride=[8, 16, 32]):
    dfl = DFL(reg_max)

    no = num_classes + reg_max * 4
    stride = torch.tensor(stride)

    box, cls = torch.cat([xi.view(inputs[0].shape[0], no, -1) for xi in inputs], 2).split(
        (reg_max * 4, num_classes), 1
    )
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(inputs, stride, 0.5))

    dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)

    return y


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


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
            image = flip_channel_order(image, input_data_format=input_data_format)
            if input_data_format == ChannelDimension.FIRST:
                image = image.transpose((2, 0, 1))
                input_data_format = ChannelDimension.LAST

            target_sizes.append(image.shape)

            image = letterbox(image, [self.size["height"], self.size["width"]], auto=False)[0]
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

        has_confidence = predictions[:, 4 : 4 + self.num_classes].amax(1) > threshold

        dets = non_max_suppression(
            predictions.transpose(2, 1),
            has_confidence,
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
