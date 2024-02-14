# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


from transformers import Pipeline
from transformers.image_utils import load_image


class YoloObjectDetectionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        postprocess_params = {}
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        if "nms_threshold" in kwargs:
            postprocess_params["nms_threshold"] = kwargs["nms_threshold"]
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]
        if "data_format" in kwargs:
            preprocess_params["data_format"] = kwargs["data_format"]
            postprocess_params["data_format"] = kwargs["data_format"]

        return preprocess_params, {}, postprocess_params

    def preprocess(self, image, timeout=None, data_format=None):
        image = load_image(image, timeout=timeout)

        image_features = self.image_processor(image, return_tensors=self.framework, data_format=data_format)

        return image_features

    def _forward(self, model_inputs):
        target_sizes = model_inputs.pop("target_sizes")
        outputs = self.model(**model_inputs)
        model_outputs = {"target_sizes": target_sizes, **outputs}

        return model_outputs

    def postprocess(self, model_outputs, nms_threshold=0.45, threshold=0.25, data_format=None, top_k=None):
        results = []
        target_sizes = model_outputs.pop("target_sizes")
        outputs = self.image_processor.post_process_object_detection(
            outputs=model_outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            nms_threshold=nms_threshold,
            data_format=data_format,
        )[0]

        results = sorted(outputs, key=lambda x: x["score"], reverse=True)
        if top_k:
            results = results[:top_k]

        return results
