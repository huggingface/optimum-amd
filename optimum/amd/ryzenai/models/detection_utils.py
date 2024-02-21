# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import torch
import torchvision

from transformers.image_transforms import center_to_corners_format


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def non_max_suppression(
    predictions,
    has_confidence,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    class_conf_start_index=5,
    agnostic=False,
    max_detections=300,
):
    output = [torch.zeros((0, 6), device=predictions.device)] * predictions.shape[0]

    for image_idx, _ in enumerate(predictions):
        prediction = predictions[image_idx]

        prediction = prediction[has_confidence[image_idx]]

        if not prediction.shape[0]:
            continue

        boxes = center_to_corners_format(prediction[:, :4])

        if class_conf_start_index == 5:
            prediction[:, 5:] *= prediction[:, 4:5]

        confidence_values, class_indices = prediction[:, class_conf_start_index:].max(1, keepdim=True)
        prediction = torch.cat((boxes, confidence_values, class_indices.float()), 1)[
            confidence_values.view(-1) > confidence_threshold
        ]
        prediction = prediction[prediction[:, 4].argsort(descending=True)]

        boxes = prediction[:, :4]
        scores = prediction[:, 4]
        class_indices = prediction[:, 5]

        if agnostic:
            nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        else:
            nms_indices = torchvision.ops.batched_nms(boxes, scores, class_indices, iou_threshold)

        if nms_indices.shape[0] > max_detections:
            nms_indices = nms_indices[:max_detections]

        output[image_idx] = prediction[nms_indices]

    return output


def scale_coords(current_shape, target_shape, coords):
    scaling_ratio = max(target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])

    padding_height, padding_width = [
        (current - target / scaling_ratio) / 2 for target, current in zip(target_shape, current_shape)
    ]

    coords[:, [0, 2]] = (coords[:, [0, 2]] - padding_width) * scaling_ratio
    coords[:, [1, 3]] = (coords[:, [1, 3]] - padding_height) * scaling_ratio

    coords[:, 0].clamp_(0, target_shape[1])
    coords[:, 1].clamp_(0, target_shape[0])
    coords[:, 2].clamp_(0, target_shape[1])
    coords[:, 3].clamp_(0, target_shape[0])

    return coords
