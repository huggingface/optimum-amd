# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import time
from typing import Dict, Optional, Union

import cv2
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
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return image, padding_ratios, (padding_width, padding_height)


def make_grid(anchor, nx=20, ny=20):
    d = anchor.device
    t = anchor.dtype

    shape = 1, len(anchor) // 2, ny, nx, 2

    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv = torch.meshgrid(y, x, indexing="ij")

    grid = torch.stack((xv, yv), 2).expand(shape)
    anchor_grid = (anchor).view((1, len(anchor) // 2, 1, 1, 2)).expand(shape)

    return grid, anchor_grid


def postprocess(inputs, anchors, num_classes=80, stride=[8, 16, 32], shapes=[80, 40, 20]):
    nl = len(anchors)
    no = num_classes + 5

    outputs = []
    for i in range(nl):
        bs, _, ny, nx = inputs[i].shape
        grid, anchor_grid = make_grid(anchors[i], nx, ny)

        inputs[i] = inputs[i].view(bs, nl, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        y = inputs[i].sigmoid()
        xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride[i]
        wh = (y[..., 2:4] * 2) ** 2 * anchor_grid
        conf = y[..., 4:]
        y = torch.cat((xy, wh, conf), -1)
        outputs.append(y.view(bs, -1, no))

    return torch.cat(outputs, 1)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
    predictions,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_detections=300,
):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Args:
        predictions: Tensor with shape (batch_size, num_anchors, num_classes + 5)
        confidence_threshold: Confidence threshold for filtering predictions
        iou_threshold: Intersection over Union threshold for NMS
        classes: List of classes to filter predictions (optional)
        agnostic: Whether to perform class-agnostic NMS
        multi_label: Whether to allow multiple labels per box
        labels: A list of tensors representing ground truth labels for autolabelling
        max_detections: Maximum number of detections to keep after NMS
    Returns:
        List of detections, each with a tensor of shape (num_detections, 6) [xyxy, confidence, class]
    """

    num_classes = predictions.shape[2] - 5  # Number of classes
    has_confidence = predictions[..., 4] > confidence_threshold  # Candidates

    # Checks
    assert (
        0 <= confidence_threshold <= 1
    ), f"Invalid Confidence threshold {confidence_threshold}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_threshold <= 1, f"Invalid IoU {iou_threshold}, valid values are between 0.0 and 1.0"

    # Settings
    _, max_box_size = 2, 4096  # Minimum and maximum box width and height
    max_nms = 30000  # Maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # Seconds to quit after
    redundant = True  # Require redundant detections
    multi_label &= num_classes > 1  # Multiple labels per box (adds 0.5ms/img)
    merge_nms = False  # Use merge-NMS

    start_time = time.time()
    output = [torch.zeros((0, 6), device=predictions.device)] * predictions.shape[0]

    for image_idx, prediction in enumerate(predictions):
        prediction = prediction[has_confidence[image_idx]]

        # Concatenate apriori labels if autolabelling
        if labels and len(labels[image_idx]):
            ground_truth_labels = labels[image_idx]
            new_labels = torch.zeros((len(ground_truth_labels), num_classes + 5), device=prediction.device)
            new_labels[:, :4] = ground_truth_labels[:, 1:5]  # Box
            new_labels[:, 4] = 1.0  # Confidence
            new_labels[range(len(ground_truth_labels)), ground_truth_labels[:, 0].long() + 5] = 1.0  # Class
            prediction = torch.cat((prediction, new_labels), 0)

        # If no predictions remain, process the next image
        if not prediction.shape[0]:
            continue

        # Compute confidence
        prediction[:, 5:] *= prediction[:, 4:5]  # Confidence = obj_conf * class_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(prediction[:, :4])

        # Detections matrix nx6 (xyxy, confidence, class)
        if multi_label:
            indices, class_indices = (prediction[:, 5:] > confidence_threshold).nonzero(as_tuple=False).T
            prediction = torch.cat(
                (boxes[indices], prediction[indices, class_indices + 5, None], class_indices[:, None].float()), 1
            )
        else:  # Best class only
            confidence_values, class_indices = prediction[:, 5:].max(1, keepdim=True)
            prediction = torch.cat((boxes, confidence_values, class_indices.float()), 1)[
                confidence_values.view(-1) > confidence_threshold
            ]

        # Filter by class
        if classes is not None:
            prediction = prediction[(prediction[:, 5:6] == torch.tensor(classes, device=prediction.device)).any(1)]

        # Check shape
        num_boxes = prediction.shape[0]  # Number of boxes
        if not num_boxes:  # No boxes
            continue
        elif num_boxes > max_nms:  # Excess boxes
            prediction = prediction[prediction[:, 4].argsort(descending=True)[:max_nms]]  # Sort by confidence

        # Batched NMS
        class_offsets = prediction[:, 5:6] * (0 if agnostic else max_box_size)  # Class offsets
        nms_boxes, nms_scores = prediction[:, :4] + class_offsets, prediction[:, 4]  # Boxes (offset by class), Scores
        nms_indices = nms(nms_boxes.cpu().detach().numpy(), nms_scores.cpu().detach().numpy(), iou_threshold)  # NMS
        nms_indices = torch.tensor(nms_indices)
        if nms_indices.shape[0] > max_detections:  # Limit detections
            nms_indices = nms_indices[:max_detections]
        if merge_nms and (1 < num_boxes < 3e3):  # Merge NMS (boxes merged using weighted mean)
            iou_matrix = box_iou(nms_boxes[nms_indices], nms_boxes) > iou_threshold  # IoU matrix
            weights = iou_matrix * nms_scores[None]  # Box weights
            prediction[nms_indices, :4] = torch.mm(weights, prediction[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # Merged boxes
            if redundant:
                nms_indices = nms_indices[iou_matrix.sum(1) > 1]  # Require redundancy

        output[image_idx] = prediction[nms_indices]
        if (time.time() - start_time) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # Time limit exceeded

    return output


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


class YoloV5ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255.0,
        **kwargs,
    ):
        size = size if size is not None else {"height": 640, "width": 640}

        super().__init__(**kwargs)
        self.size = size
        self.data_format = ChannelDimension.LAST
        self.stride = [8, 16, 32]
        self.rescale_factor = rescale_factor
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.num_classes = 80

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
        target_size = []
        for image in images:
            image = flip_channel_order(image, input_data_format=input_data_format)
            if input_data_format == ChannelDimension.FIRST:
                image = image.transpose((2, 0, 1))
                input_data_format = ChannelDimension.LAST

            target_size.append(image.shape)

            image = letterbox(image, [self.size["height"], self.size["width"]], stride=self.stride, auto=False)[0]
            image = image.transpose((2, 0, 1))
            input_data_format = ChannelDimension.FIRST

            image = flip_channel_order(image, input_data_format=input_data_format)

            image = np.ascontiguousarray(image, dtype=np.float32)

            image = rescale(
                image=image, scale=self.rescale_factor, data_format=data_format, input_data_format=input_data_format
            )

            preprocessed_images.append(image)

        data = {"pixel_values": preprocessed_images, "target_size": target_size}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        target_size,
        nms_threshold: float = 0.45,
        score_threshold: float = 0.25,
        data_format: Union[str, ChannelDimension] = None,
        anchors=None,
        classes=None,
        num_classes=80,
        agnostic_nms=False,
        max_detections=1000,
        stride=None,
    ):
        data_format = data_format if data_format is not None else self.data_format
        anchors = anchors if anchors is not None else self.anchors

        if classes:
            num_classes = len(classes)

        stride = stride if stride is not None else self.stride

        outputs = list(outputs.values())

        if not isinstance(outputs[0], torch.Tensor):
            outputs = [torch.tensor(out) for out in outputs]

        if data_format == ChannelDimension.LAST:
            outputs = [torch.permute(out, (0, 3, 1, 2)) for out in outputs]

        anchors = torch.tensor(anchors)
        predictions = postprocess(outputs, anchors, num_classes, stride)

        dets = non_max_suppression(
            predictions, score_threshold, nms_threshold, classes, agnostic_nms, max_detections=max_detections
        )

        results = []

        for i, det in enumerate(dets):
            det[:, :4] = scale_coords((self.size["height"], self.size["width"]), det[:, :4], target_size[i]).round()

            outputs = []
            for *box, score, cls in reversed(det):
                label = int(cls)
                outputs.append({"score": score.item(), "label": label, "box": np.array(box)})
            results.append(outputs)

        return results
