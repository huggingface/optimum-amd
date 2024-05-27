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


def nms(boxes, scores, nms_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(descending=True)
    keep = []
    while order.size(0) > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= nms_threshold)[0]
        order = order[inds + 1]

    return torch.tensor(keep)


def multiclass_nms(
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

        if agnostic:
            nms_method = multiclass_nms_class_agnostic
        else:
            nms_method = multiclass_nms_class_aware

        output = nms_method(boxes, scores, iou_threshold, confidence_threshold)

        if len(output) > max_detections:
            output = output[:max_detections]

        if output is not None:
            outputs.append(output)
        else:
            outputs.append(torch.zeros((0, 6)))

    return outputs


def multiclass_nms_class_aware(boxes, scores, nms_threshold, score_threshold):
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_threshold
        if valid_score_mask.sum() == 0:
            continue

        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep = nms(valid_boxes, valid_scores, nms_threshold)
        if len(keep) > 0:
            cls_inds = torch.ones((len(keep), 1), dtype=torch.float32) * cls_ind
            dets = torch.cat([valid_boxes[keep], valid_scores[keep, None], cls_inds], dim=1)
            final_dets.append(dets)

    if len(final_dets) == 0:
        return None

    return torch.cat(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_threshold, score_threshold):
    cls_inds = torch.argmax(scores, dim=1)
    cls_scores = scores[torch.arange(len(cls_inds)), cls_inds]
    valid_score_mask = cls_scores > score_threshold

    if valid_score_mask.sum() == 0:
        return None

    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]

    keep = nms(valid_boxes, valid_scores, nms_threshold)
    if keep.numel() > 0:
        dets = torch.cat([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], dim=1)
        return dets
    else:
        return None


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
