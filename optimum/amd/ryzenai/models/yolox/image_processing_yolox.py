# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    PaddingMode,
    flip_channel_order,
    pad,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
)
from transformers.utils import TensorType

from ..detection_utils import non_max_suppression


def postprocess(outputs, img_size, strides):
    grids = []
    expanded_strides = []
    device = strides.device
    dtype = strides.dtype

    outputs = [out.reshape(*out.shape[:2], -1).transpose(2, 1) for out in outputs]
    outputs = torch.cat(outputs, axis=1)
    outputs[..., 4:] = outputs[..., 4:].sigmoid()

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = torch.meshgrid(
            torch.arange(wsize, device=device, dtype=dtype),
            torch.arange(hsize, device=device, dtype=dtype),
            indexing="xy",
        )
        grid = torch.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(torch.full((*shape, 1), stride, dtype=dtype, device=device))

    grids = torch.cat(grids, 1)
    expanded_strides = torch.cat(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


class YoloXImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size=None,
        stride: List[int] = [8, 16, 32],
        **kwargs,
    ):
        size = size if size is not None else {"height": 640, "width": 640}

        super().__init__(**kwargs)
        self.size = size
        self.resample = cv2.INTER_LINEAR
        self.data_format = ChannelDimension.LAST
        self.stride = stride

    def resize(
        self,
        image: np.ndarray,
        size: Tuple[int, int],
        resample=cv2.INTER_LINEAR,
    ) -> np.ndarray:
        image = cv2.resize(
            image,
            (size[1], size[0]),
            interpolation=resample,
        ).astype(np.uint8)

        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image
    def pad(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 114,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

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

        target_sizes = []
        padded_images = []
        for image in images:
            image = flip_channel_order(image, input_data_format=input_data_format)
            input_height, input_width = get_image_size(image, channel_dim=input_data_format)
            target_sizes.append(image.shape)

            ratio = min(self.size["height"] / input_height, self.size["width"] / input_width)

            size = (int(ratio * input_height), int(ratio * input_width))

            resized_image = self.resize(image, size=size, resample=self.resample)

            padded_img = self.pad(
                resized_image,
                output_size=(self.size["height"], self.size["width"]),
                data_format=data_format,
                input_data_format=input_data_format,
            )

            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

            padded_images.append(padded_img)

        data = {"pixel_values": padded_images, "target_sizes": target_sizes}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.1,
        nms_threshold: float = 0.45,
        target_sizes: Union[TensorType, List[Tuple]] = None,
        data_format: Union[str, ChannelDimension] = None,
        agnostic_nms=True,
        merge_nms=False,
        multi_label=False,
        max_detections=1000,
        classes=None,
    ):
        data_format = data_format if data_format is not None else self.data_format

        outputs = list(outputs.values())

        if not isinstance(outputs[0], torch.Tensor):
            outputs = [torch.Tensor(out) for out in outputs]

        if data_format == ChannelDimension.LAST:
            outputs = [torch.permute(out, (0, 3, 1, 2)) for out in outputs]

        predictions = postprocess(outputs, (self.size["height"], self.size["width"]), torch.Tensor(self.stride))

        has_confidence = predictions[..., 4] > threshold  # Candidates

        dets = non_max_suppression(
            predictions,
            has_confidence,
            threshold,
            nms_threshold,
            classes,
            agnostic_nms,
            multi_label=multi_label,
            max_detections=max_detections,
            merge_nms=merge_nms,
        )

        results = []
        for i, det in enumerate(dets):
            if target_sizes is not None:
                input_height, input_width, _ = target_sizes[i]
                ratio = min(self.size["height"] / input_height, self.size["width"] / input_width)
                det[:, :4] /= ratio

            outputs = []
            for *box, score, cls in reversed(det):
                label = int(cls)
                outputs.append({"score": score.item(), "label": label, "box": np.array(box)})
            results.append(outputs)

        return results
