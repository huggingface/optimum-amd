# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import time

import numpy as np
import torch
import torchvision


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
    has_confidence,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_detections=300,
    merge_nms=False,  # True
    conf_index=5,
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

    num_classes = predictions.shape[2] - conf_index  # Number of classes
    # has_confidence1 = predictions[..., 4] > confidence_threshold  # Candidates
    # has_confidence = predictions.transpose(2,1)[:, 4:4+num_classes].amax(1) > confidence_threshold  # candidates

    # Checks
    assert (
        0 <= confidence_threshold <= 1
    ), f"Invalid Confidence threshold {confidence_threshold}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_threshold <= 1, f"Invalid IoU {iou_threshold}, valid values are between 0.0 and 1.0"

    # Settings
    min_box_size, max_box_size = 2, 7680  # Minimum and maximum box width and height
    max_nms = 30000  # Maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # Seconds to quit after
    redundant = True  # Require redundant detections
    multi_label &= num_classes > 1  # Multiple labels per box (adds 0.5ms/img)

    start_time = time.time()
    output = [torch.zeros((0, 6), device=predictions.device)] * predictions.shape[0]

    for image_idx, prediction in enumerate(predictions):
        prediction[((prediction[..., 2:4] < min_box_size) | (prediction[..., 2:4] > max_box_size)).any(1), 4] = 0

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
        if conf_index == 5:
            prediction[:, 5:] *= prediction[:, 4:5]  # Confidence = obj_conf * class_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(prediction[:, :4])

        # Detections matrix nx6 (xyxy, confidence, class)
        if multi_label:
            indices, class_indices = (prediction[:, conf_index:] > confidence_threshold).nonzero(as_tuple=False).T
            prediction = torch.cat(
                (
                    boxes[indices],
                    prediction[indices, class_indices + conf_index, None],
                    class_indices[:, None].float(),
                ),
                1,
            )
        else:  # Best class only
            confidence_values, class_indices = prediction[:, conf_index:].max(1, keepdim=True)

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

        prediction = prediction[prediction[:, 4].argsort(descending=True)[:max_nms]]  # Sort by confidence

        # Batched NMS
        class_offsets = prediction[:, 5:6] * (0 if agnostic else max_box_size)  # Class offsets
        nms_boxes, nms_scores = prediction[:, :4] + class_offsets, prediction[:, 4]  # Boxes (offset by class), Scores
        nms_indices = torchvision.ops.nms(nms_boxes, nms_scores, iou_threshold)  # NMS

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
