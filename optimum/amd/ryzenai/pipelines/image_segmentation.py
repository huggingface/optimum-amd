import numpy as np
from PIL import Image

from transformers import Pipeline
from transformers.image_utils import load_image


class ImageSegmentationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        if "timeout" in kwargs:
            preprocess_kwargs["timeout"] = kwargs["timeout"]

        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        inputs = self.image_processor(images=[image], return_tensors="pt")
        return inputs

    def _forward(self, model_inputs):
        target_sizes = model_inputs.pop("target_sizes")
        model_outputs = self.model(**model_inputs)
        model_outputs["target_sizes"] = target_sizes
        return model_outputs

    def postprocess(self, model_outputs):
        outputs = self.image_processor.post_process_semantic_segmentation(
            model_outputs, target_sizes=model_outputs["target_sizes"]
        )[0]

        annotation = []
        segmentation = outputs.numpy()
        labels = np.unique(segmentation)

        for label in labels:
            mask = (segmentation == label) * 255
            mask = Image.fromarray(mask.astype(np.uint8), mode="L")
            annotation.append({"score": None, "label": label, "mask": mask})
        return annotation
