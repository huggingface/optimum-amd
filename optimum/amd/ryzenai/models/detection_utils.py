# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import torch

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


def box_iou(boxes1, boxes2, area1, area2):
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    _, sorted_indices = scores.sort(descending=True)
    areas = box_area(boxes)

    keep = []
    while len(sorted_indices) > 0:
        best_box_index = sorted_indices[0]
        keep.append(best_box_index.item())

        ious = box_iou(
            boxes[best_box_index].unsqueeze(0),
            boxes[sorted_indices[1:]],
            areas[best_box_index].unsqueeze(0),
            areas[sorted_indices[1:]],
        )

        filtered_indices_mask = ious.squeeze(0) <= threshold
        filtered_indices = torch.nonzero(filtered_indices_mask).squeeze().tolist()

        sorted_indices = sorted_indices[1:][filtered_indices]

    return torch.tensor(keep)


def non_max_suppression(
    predictions,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    class_conf_start_index=5,
    agnostic=False,
    max_detections=300,
):
    outputs = []
    for image_idx, _ in enumerate(predictions):
        prediction = predictions[image_idx]
        if class_conf_start_index == 5:
            scores = prediction[:, 4:5] * prediction[:, 5:]
        else:
            scores = prediction[:, class_conf_start_index:]

        boxes = center_to_corners_format(prediction[:, :4])
        output = agnostic_nms(boxes, scores, iou_threshold, confidence_threshold, agnostic=agnostic)

        if len(output) > max_detections:
            output = output[:max_detections]

        outputs.append(output)

    return outputs


def agnostic_nms(boxes, scores, iou_threshold, confidence_threshold, agnostic=True):
    final_dets = []
    num_classes = scores.shape[1] if not agnostic else 1

    for cls_ind in range(num_classes):
        if agnostic:
            cls_scores = scores.max(dim=1).values
        else:
            cls_scores = scores[:, cls_ind]

        valid_score_mask = cls_scores > confidence_threshold
        if valid_score_mask.sum() == 0:
            continue

        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]

        keep = nms(valid_boxes, valid_scores, iou_threshold)
        if keep.numel() > 0:
            kept_boxes = valid_boxes[keep]
            kept_scores = valid_scores[keep]

            if agnostic:
                kept_classes = torch.argmax(scores[valid_score_mask][keep], dim=1, keepdim=True).float()
            else:
                kept_classes = torch.full((len(keep), 1), cls_ind, dtype=torch.float32)

            dets = torch.cat([kept_boxes, kept_scores[:, None], kept_classes], dim=1)
            final_dets.append(dets)

    if len(final_dets) == 0:
        return torch.zeros((0, 6))

    return torch.cat(final_dets, 0)


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
