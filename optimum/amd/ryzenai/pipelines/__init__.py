# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from optimum.exporters import TasksManager
from transformers import (
    ImageClassificationPipeline,
    Pipeline,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextGenerationPipeline,
)
from transformers import pipeline as transformers_pipeline
from transformers.image_processing_utils import BaseImageProcessor
from transformers.onnx.utils import get_preprocessor

from ..modeling import (
    RyzenAIModel,
    RyzenAIModelForImageClassification,
    RyzenAIModelForObjectDetection,
    RyzenAIModelForSemanticSegmentation,
)
from ..modeling_decoder import RyzenAIModelForCausalLM
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
    "image-segmentation": {
        "impl": ImageSegmentationPipeline,
        "class": (RyzenAIModelForSemanticSegmentation,),
        "default": "amd/SemanticFPN",
        "type": "image",
        "model_type": "semantic_fpn",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "class": (RyzenAIModelForCausalLM,),
        "type": "text",
    },
    "object-detection": {
        "impl": YoloObjectDetectionPipeline,
        "class": (RyzenAIModelForObjectDetection,),
        "default": "amd/yolox-s",
        "type": "image",
        "model_type": "yolox",
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


def get_processor_from_model(model: RyzenAIModel, type: Tuple[Any], name: str) -> Any:
    for preprocessor in model.preprocessors:
        if isinstance(preprocessor, type):
            return preprocessor
    if preprocessor is None:
        raise ValueError(
            f"Could not automatically find a {name} for the model, you must specify the argument `{name}` explicitly."
        )
    return None


def get_processor(
    task: str,
    model: RyzenAIModel,
    model_id: Optional[str] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    feature_extractor: Optional[Union[str, "PreTrainedFeatureExtractor"]] = None,
) -> Tuple[Union[PreTrainedTokenizer, PreTrainedTokenizerFast], BaseImageProcessor]:
    supported_tasks = RYZENAI_SUPPORTED_TASKS

    no_image_processor_tasks, no_tokenizer_tasks = get_task_processor_map(supported_tasks)

    load_tokenizer = False if task in no_tokenizer_tasks else True
    load_image_processor = False if task in no_image_processor_tasks else True

    if tokenizer is None and load_tokenizer:
        tokenizer = (
            get_preprocessor(model_id)
            if model_id
            else get_processor_from_model(model, (PreTrainedTokenizer, PreTrainedTokenizerFast), "tokenizer")
        )

    if image_processor is None and feature_extractor is None and load_image_processor:
        library_name = TasksManager._infer_library_from_model(model)
        if library_name != "timm":
            image_processor = (
                get_preprocessor(model_id)
                if model_id
                else get_processor_from_model(model, BaseImageProcessor, "image_processor")
            )

    return tokenizer, image_processor


def get_task_processor_map(supported_tasks):
    no_image_processor_tasks = set()
    no_tokenizer_tasks = set()
    for _task, values in supported_tasks.items():
        if values["type"] == "text":
            no_image_processor_tasks.add(_task)
        elif values["type"] == "image":
            no_tokenizer_tasks.add(_task)
        else:
            raise ValueError(f"SUPPORTED_TASK {_task} contains invalid type {values['type']}")
    return no_image_processor_tasks, no_tokenizer_tasks


def pipeline(
    task,
    vaip_config: str,
    model: Optional[Any] = None,
    model_type: Optional[str] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
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
            - "image-segmentation"
            - "text-generation"
            - "object-detection"
        vaip_config (`str`):
            Runtime configuration file for inference with Ryzen IPU. A default config file can be found in the Ryzen AI VOE package,
            extracted during installation under the name `vaip_config.json`.
        model (`Optional[Any]`, defaults to `None`):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model. If not provided, the default model for the specified task will be loaded.
        model_type (`Optional[str]`, defaults to `None`):
            Model type for the model
        tokenizer (`Optional[Union[str, PreTrainedTokenizer]]`, defaults to `None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model identifier
            or an actual pretrained tokenizer.
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

    ryzen_pipeline = transformers_pipeline
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
        library_name = TasksManager._infer_library_from_model(model) if task == "image-classification" else None

        if library_name == "timm":
            ryzen_pipeline = TimmImageClassificationPipeline
        else:
            tokenizer, image_processor = get_processor(
                task, model, model_id, tokenizer, image_processor, feature_extractor
            )

    return ryzen_pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        use_fast=use_fast,
        **kwargs,
    )
