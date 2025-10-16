# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


from typing import TYPE_CHECKING, Any, Optional, Union

from optimum.exporters.tasks import TasksManager
from transformers import ImageClassificationPipeline, Pipeline, PretrainedConfig
from transformers import pipeline as transformers_pipeline
from transformers.image_processing_utils import BaseImageProcessor
from transformers.onnx.utils import get_preprocessor

from ..modeling import (
    RyzenAIModel,
    RyzenAIModelForImageClassification,
    RyzenAIModelForObjectDetection,
    RyzenAIModelForSemanticSegmentation,
)
from ..models import (
    HRNetImageProcessor,
    SemanticFPNImageProcessor,
    YoloV3ImageProcessor,
    YoloV5ImageProcessor,
    YoloV8ImageProcessor,
    YoloXImageProcessor,
)
from .image_classification import TimmImageClassificationPipeline
from .image_segmentation import ImageSegmentationPipeline
from .object_detection import YoloObjectDetectionPipeline


if TYPE_CHECKING:
    from transformers.feature_extraction_utils import PreTrainedFeatureExtractor

pipeline_map = {
    "yolox": {"preprocessor": YoloXImageProcessor, "impl": YoloObjectDetectionPipeline},
    "yolov5": {"preprocessor": YoloV5ImageProcessor, "impl": YoloObjectDetectionPipeline},
    "yolov3": {"preprocessor": YoloV3ImageProcessor, "impl": YoloObjectDetectionPipeline},
    "yolov8": {"preprocessor": YoloV8ImageProcessor, "impl": YoloObjectDetectionPipeline},
    "semantic_fpn": {"preprocessor": SemanticFPNImageProcessor, "impl": ImageSegmentationPipeline},
    "hrnet": {"preprocessor": HRNetImageProcessor, "impl": ImageSegmentationPipeline},
}

RYZENAI_SUPPORTED_TASKS = {
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "class": (RyzenAIModelForImageClassification,),
        "default": "amd/resnet50",
        "type": "image",
    },
    "object-detection": {
        "impl": YoloObjectDetectionPipeline,
        "class": (RyzenAIModelForObjectDetection,),
        "default": "amd/yolox-s",
        "type": "image",
        "model_type": "yolox",
    },
    "image-segmentation": {
        "impl": ImageSegmentationPipeline,
        "class": (RyzenAIModelForSemanticSegmentation,),
        "default": "amd/SemanticFPN",
        "type": "image",
        "model_type": "semantic_fpn",
    },
}


def load_model(
    model,
    task,
    model_type,
    vaip_config,
    SUPPORTED_TASKS,
    token: Optional[Union[bool, str]] = None,
    revision: str = "main",
):
    if model is None:
        if task != "object-detection":
            raise ValueError("Creating pipeline without model for the task is not supported!")

        model_id = SUPPORTED_TASKS[task]["default"]
        if model_type is None:
            model_type = SUPPORTED_TASKS[task].get("model_type", None)
        model = SUPPORTED_TASKS[task]["class"][0].from_pretrained(
            model_id, vaip_config=vaip_config, use_auth_token=token, revision=revision
        )
    elif isinstance(model, str):
        model_id = model
        ort_model_class = SUPPORTED_TASKS[task]["class"][0]

        model = ort_model_class.from_pretrained(
            model_id, vaip_config=vaip_config, use_auth_token=token, revision=revision
        )
    elif isinstance(model, RyzenAIModel):
        model_id = None
    else:
        raise ValueError(
            f"Model {model} is not supported. Please provide a valid model either as string or RyzenAIModel."
            "You can also provide non model then a default one will be used"
        )
    return model, model_id, model_type


def pipeline(
    task,
    model: Optional[Any] = None,
    vaip_config: Optional[str] = None,
    model_type: Optional[str] = None,
    feature_extractor: Optional[Union[str, "PreTrainedFeatureExtractor"]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> Pipeline:
    """
    Utility method to build a pipeline for various RyzenAI tasks.

    This function creates a pipeline for a specified task, utilizing a given model or loading the default model for the task.
    The pipeline includes components such as a image processor and model.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Available tasks include:
            - "image-classification"
            - "object-detection"
        model (`Optional[Any]`, defaults to `None`):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model. If not provided, the default model for the specified task will be loaded.
        vaip_config (`Optional[str]`, defaults to `None`):
            Runtime configuration file for inference with Ryzen IPU. A default config file can be found in the Ryzen AI VOE package,
            extracted during installation under the name `vaip_config.json`.
        model_type (`Optional[str]`, defaults to `None`):
            Model type for the model
        feature_extractor (`Union[str, "PreTrainedFeatureExtractor"]`, defaults to `None`):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor.
        image_processor (`Union[str, BaseImageProcessor]`, defaults to `None`):
            The image processor that will be used by the pipeline for image-related tasks.
        use_fast (`bool`, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible.
        token (`Union[str, bool`], defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If True, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, defaults to `None`):
            The specific model version to use, specified as a branch name, tag name, or commit id.
        **kwargs:
            Additional keyword arguments passed to the underlying pipeline class.

    Returns:
        Pipeline:
            An instance of the specified pipeline for the given task and model.
    """
    model, model_id, model_type = load_model(
        model,
        task=task,
        model_type=model_type,
        vaip_config=vaip_config,
        SUPPORTED_TASKS=RYZENAI_SUPPORTED_TASKS,
        token=token,
        revision=revision,
    )

    if model.config is None:
        if model_type is None:
            raise ValueError(
                "Could not automatically find a `model_type` for the model, you must specify the `model_type` argument explicitly."
            )

        if model_type not in pipeline_map:
            raise ValueError(
                f"Model type: {model_type} is not supported by Ryzen pipelines. Please open an issue or submit a PR to add the support."
            )

        image_processor = pipeline_map[model_type]["preprocessor"]()
        ryzen_pipeline = pipeline_map[model_type]["impl"]

        model.config = PretrainedConfig.from_dict({})
    else:
        library_name = TasksManager._infer_library_from_model(model)

        if library_name != "timm" and image_processor is None:
            if model_id:
                if feature_extractor is None and image_processor is None:
                    image_processor = get_preprocessor(model_id)
            else:
                for preprocessor in model.preprocessors:
                    if isinstance(preprocessor, BaseImageProcessor):
                        image_processor = preprocessor
                        break
                if image_processor is None:
                    raise ValueError(
                        "Could not automatically find an image processor for the model, you must specifiy the argument `image_processor` explicitly."
                    )

        if task == "image-classification" and library_name == "timm":
            ryzen_pipeline = TimmImageClassificationPipeline
        else:
            ryzen_pipeline = transformers_pipeline

    return ryzen_pipeline(
        task=task,
        model=model,
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        use_fast=use_fast,
        **kwargs,
    )
