# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import torch
import torchvision

from transformers.image_transforms import center_to_corners_format


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
        scores, idxs = scores.max(1)

        valid_mask = scores > confidence_threshold
        if valid_mask.sum() == 0:
            outputs.append(torch.empty((0, 6)))
            continue

        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        idxs = idxs[valid_mask]

        scores, sorted_indices = scores.sort(descending=True)
        boxes = boxes[sorted_indices]
        idxs = idxs[sorted_indices]

        nms_indices = torch.zeros_like(idxs) if agnostic else idxs
        kept_classes = torchvision.ops.batched_nms(boxes, scores, nms_indices, iou_threshold)[:max_detections]

        output = torch.cat([boxes[kept_classes], scores[kept_classes][:, None], idxs[kept_classes][:, None]], dim=1)

        outputs.append(output)

    return outputs


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
