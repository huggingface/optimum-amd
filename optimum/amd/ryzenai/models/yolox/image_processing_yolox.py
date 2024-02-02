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


def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)
    return dets


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(outputs, img_size, strides):
    grids = []
    expanded_strides = []

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


class YoloXImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size=None,
        **kwargs,
    ):
        size = size if size is not None else {"height": 640, "width": 640}

        super().__init__(**kwargs)
        self.size = size
        self.resample = cv2.INTER_LINEAR
        self.data_format = ChannelDimension.LAST
        self.p6 = False
        self.strides_wout_p6 = [8, 16, 32]
        self.strides_with_p6 = [8, 16, 32, 64]

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

        ratios = []
        padded_images = []
        for image in images:
            image = flip_channel_order(image, input_data_format=input_data_format)
            input_height, input_width = get_image_size(image, channel_dim=input_data_format)

            ratio = min(self.size["height"] / input_height, self.size["width"] / input_width)
            ratios.append(ratio)

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

        data = {"pixel_values": padded_images, "ratios": ratios}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        ratios: List[Tuple],
        nms_threshold: float = 0.45,
        score_threshold: float = 0.1,
        data_format: Union[str, ChannelDimension] = None,
        p6: Optional[bool] = None,
    ):
        data_format = data_format if data_format is not None else self.data_format
        p6 = p6 if p6 is not None else self.p6

        outputs = list(outputs.values())

        if isinstance(outputs[0], torch.Tensor):
            outputs = [out.cpu().detach().numpy() for out in outputs]
            ratios = [ratio.cpu().detach().numpy() for ratio in ratios]

        if data_format == ChannelDimension.LAST:
            outputs = [np.transpose(out, (0, 3, 1, 2)) for out in outputs]

        outputs = [out.reshape(*out.shape[:2], -1).transpose(0, 2, 1) for out in outputs]
        outputs = np.concatenate(outputs, axis=1)
        outputs[..., 4:] = sigmoid(outputs[..., 4:])

        strides = self.strides_with_p6 if p6 else self.strides_wout_p6
        results = []

        for i in range(len(ratios)):
            predictions = postprocess(outputs[i : i + 1], (self.size["height"], self.size["width"]), strides)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            boxes_xyxy /= ratios[i]

            dets = multiclass_nms(boxes_xyxy, scores, nms_threshold, score_threshold)

            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            outputs = [
                {"score": score, "label": label, "box": box}
                for score, label, box in zip(final_scores, final_cls_inds, final_boxes)
            ]
            results.append(outputs)

        return results
