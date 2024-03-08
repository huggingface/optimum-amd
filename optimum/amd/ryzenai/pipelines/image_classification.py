# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import timm

from transformers import Pipeline
from transformers.image_utils import load_image


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class TimmImageClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)

        data_config = timm.data.resolve_data_config(pretrained_cfg=self.model.config.pretrained_cfg)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        model_inputs = transforms(image).unsqueeze(0)

        return {"pixel_values": model_inputs}

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, top_k=5):
        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()

        scores = softmax(outputs)

        dict_scores = [{"label": i, "score": score.item()} for i, score in enumerate(scores)]

        dict_scores.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            dict_scores = dict_scores[:top_k]

        return dict_scores
