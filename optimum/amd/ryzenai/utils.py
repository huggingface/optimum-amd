# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import logging
import os
import random

import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)

ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_WEIGHTS_NAME_STATIC = "model_static.onnx"

DEFAULT_VAIP_CONFIG = os.path.normpath(os.path.join(os.path.dirname(__file__), "./configs/vaip_config.json"))


def validate_provider_availability(provider: str):
    """
    Ensure the ONNX Runtime execution provider `provider` is available, and raise an error if it is not.

    Args:
        provider (str): Name of an ONNX Runtime execution provider.
    """
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise ValueError(
            f"Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}."
        )


def plot_bbox(image, detections, output_path="plot_bbox_output.png"):
    """
    Plots labels and bounding boxes on an image.

    Args:
        image_path (str): Path to the image.
        detections (list): List of detections where each detection is a dictionary with keys 'label', 'bbox'.
                        The 'bbox' should be a list or tuple of the form [x_min, y_min, x_max, y_max].

    Returns:
        PIL.Image: Image with bounding boxes plotted.
    """
    if isinstance(image, str):
        image = Image.open(image)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Generate a unique color for each label
    colors = {}

    for detection in detections:
        label = f"{detection['label']} {detection['score']:.2f}"
        label_color_txt = f"{detection['label']}"
        bbox = detection["box"]

        if label_color_txt not in colors:
            colors[label_color_txt] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        color = colors[label_color_txt]

        # Draw the bounding box
        box = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
        draw.rectangle(box, outline=color, width=2)

        # Determine the text color (black or white) based on the brightness of the bounding box color
        brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        text_color = (0, 0, 0) if brightness > 186 else (255, 255, 255)

        # Draw the label background
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        text_bg_bbox = [box[0], box[1] - (text_bbox[3] - text_bbox[1]), box[0] + (text_bbox[2] - text_bbox[0]), box[1]]
        draw.rectangle(text_bg_bbox, fill=color)

        # Draw the label text
        draw.text((box[0], box[1] - (text_bbox[3] - text_bbox[1])), label, fill=text_color, font=font)

    image.save(output_path)
    logger.info(f"Image with bounding boxes saved to {output_path}")

    return image
