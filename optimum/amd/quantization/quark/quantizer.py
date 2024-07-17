# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""AMD Quark Quantizer"""

import logging
from typing import Dict, Optional, Union

import torch
from quark.torch import ModelExporter, ModelQuantizer
from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG, EMPTY_EXPORTER_CONFIG

from optimum.exporters import TasksManager
from optimum.quantization_base import OptimumQuantizer
from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


class QuarkQuantizer(OptimumQuantizer):
    """
    Handles the quantization process for models shared on huggingface.co/models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        quantization_config,
        model_name_or_path: Optional[str],
        config: Optional[PretrainedConfig] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.model = model
        self.model_type = model.config.type

        # Initialize the quantizer
        self.quantizer = ModelQuantizer(config)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        device_map: Optional[Union[Dict, str, torch.device]] = None,
        **model_kwargs,
    ):
        """
        Loads the BrevitasQuantizer and model.

        Arguments:
            model_name_or_path (`Union[str, Path]`):
                Can be either the model id of a model repo on the Hugging Face Hub, or a path to a local directory
                containing a model.
            subfolder (`str`, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*, defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            trust_remote_code (`bool`, defaults to `False`):
                Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
                you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
                model repository.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            local_files_only (`Optional[bool]`, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`Optional[str]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
        """

        # TODO: fix
        # task = TasksManager.infer_task_from_model(model_name_or_path)
        task = "text-generation"

        device = None
        if not isinstance(device_map, dict) and device_map not in ["auto", "balanced"]:
            device = device_map
            device_map = None

        model = TasksManager.get_model_from_task(
            task,
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            device=device,
            framework="pt",
            **model_kwargs,
        )

        return cls(model, model_name_or_path)

    def quantize(
        self,
        dataset: Dataset,
    ) -> torch.nn.Module:
        """
        Quantizes the model using Brevitas according to the `quantization_config`.

        Arguments:
            quantization_config (`BrevitasQuantizationConfig`):
                Quantization configuration to use to quantize the model.
            calibration_dataset (`Optional[List[Dict]]`, defaults to `None`):
                In case the quantization involves a calibration phase, this argument needs to be specified as a list of inputs to the model.
                Example: `calibration_dataset = [{"input_ids": torch.tensor([[1, 2, 3, 4]])}, {"input_ids": torch.tensor([[6, 7, 3, 4]])}]` which is a dataset for a model taking `input_ids` as an argument, and which has two samples.
        """
        calib_dataloader = get_calib_dataloader(dataset)

        self.model = self.quantizer.quantize_model(self.model, calib_dataloader)

        return self.model

    def save_pretrained(self, save_directory: str, export_config=None):
        """
        Save the quantized model to the specified directory.

        Arguments:
            save_directory (`str`):
                Directory to save the quantized model to.
        """
        model = self.quantizer.freeze(self.model)

        with torch.inference_mode():
            export_config = EMPTY_EXPORTER_CONFIG if args.no_weight_matrix_merge else DEFAULT_EXPORTER_CONFIG

            exporter = ModelExporter(config=config, export_dir=save_directory)
            exporter.export_model_info(model, model_type, model_dtype, export_type="native")


    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        seed: int = 2016,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> Dataset:
        """
        Creates the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                to load to use for the calibration step.
            num_samples (`int`, defaults to 100):
                The maximum number of samples composing the calibration dataset.
            dataset_config_name (`Optional[str]`, defaults to `None`):
                The name of the dataset configuration.
            dataset_split (`Optional[str]`, defaults to `None`):
                Which split of the dataset to use to perform the calibration step.
            preprocess_function (`Optional[Callable]`, defaults to `None`):
                Processing function to apply to each example after loading dataset.
            preprocess_batch (`bool`, defaults to `True`):
                Whether the `preprocess_function` should be batched.
            seed (`int`, defaults to 2016):
                The random seed to use when shuffling the calibration dataset.
            use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
                Deprecated. Please use the `token` argument instead.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).

        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration
            step.
        """

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if dataset_name is None:
            raise ValueError(
                "ORTQuantizer: Static quantization calibration step requires a dataset_name if no calib_dataset is "
                "provided."
            )

        calib_dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
            token=token,
        )

        if num_samples is not None:
            num_samples = min(num_samples, len(calib_dataset))
            calib_dataset = calib_dataset.shuffle(seed=seed).select(range(num_samples))

        if preprocess_function is not None:
            processed_calib_dataset = calib_dataset.map(preprocess_function, batched=preprocess_batch)
        else:
            processed_calib_dataset = calib_dataset

        return self.clean_calibration_dataset(processed_calib_dataset)

    def clean_calibration_dataset(self, dataset: Dataset) -> Dataset:
        model = onnx.load(self.onnx_model_path)
        model_inputs = {input.name for input in model.graph.input}
        ignored_columns = list(set(dataset.column_names) - model_inputs)
        return dataset.remove_columns(ignored_columns)