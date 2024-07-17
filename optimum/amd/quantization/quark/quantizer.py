# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""AMD Quark Quantizer"""

import logging
from typing import Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from quark.torch import ModelExporter, ModelQuantizer
from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG, EMPTY_EXPORTER_CONFIG
from torch.utils.data import DataLoader

from optimum.exporters import TasksManager
from optimum.quantization_base import OptimumQuantizer

from .configuration import QuarkQuantizationConfig


logger = logging.getLogger(__name__)


class QuarkQuantizer(OptimumQuantizer):
    """
    Handles the quantization process for models shared on huggingface.co/models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        quantization_config: QuarkQuantizationConfig,
        model_name_or_path: Optional[str],
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.model = model
        self.model_type = model.config.model_type
        self.model_dtype = next(model.parameters()).dtype

        # Initialize the quantizer
        self.quantizer = ModelQuantizer(quantization_config)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        quantization_config: QuarkQuantizationConfig,
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
        Loads the QuarkQuantizer and model.

        Args:
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
        ).eval()

        return cls(model, quantization_config, model_name_or_path)

    def quantize(
        self,
        dataloader: Optional[Dataset] = None,
    ) -> torch.nn.Module:
        """_summary_

        Args:
            dataloader (`Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                                  DataLoader[Dict[str, torch.Tensor]]]]`, defaults to `None`):
                The DataLoader providing data that the quantization process will use for calibration. This can be a simple DataLoader returning
                tensors, or a more complex structure returning either a list of dictionaries or a dictionary of tensors.

        Returns:
            torch.nn.Module: Quantized model
        """
        if not self.quantizer.is_all_dynamic and dataloader is None:
            raise ValueError("A calibration dataset is required for the quantization method.")

        self.model = self.quantizer.quantize_model(self.model, dataloader)

        return self.model

    def save_pretrained(self, save_directory: str, no_weight_matrix_merge=False):
        """
        Save the quantized model to the specified directory.

        Args:
            save_directory (`str`):
                Directory to save the quantized model to.
            no_weight_matrix_merge (`bool`, defaults to `False`):
                Whether to merge weight matrix when dump quantized model
        """
        if self.model_type != "llama":
            raise ValueError("Only models with dtype `llama` can be saved.")
        model = self.quantizer.freeze(self.model)

        with torch.inference_mode():
            export_config = EMPTY_EXPORTER_CONFIG if no_weight_matrix_merge else DEFAULT_EXPORTER_CONFIG

            exporter = ModelExporter(config=export_config, export_dir=save_directory)
            exporter.export_model_info(model, self.model_type, self.model_dtype, export_type="vllm-adopt")

    def get_calibration_data(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        seed: int = 2016,
        token: Optional[Union[bool, str]] = None,
        batch_size: int = 1,
    ) -> Union[
        DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]
    ]:
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

        dataloader = DataLoader(processed_calib_dataset, batch_size=batch_size, shuffle=False)

        return dataloader
