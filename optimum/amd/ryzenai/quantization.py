# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

"""Classes handling quantization with RyzenAI Quantizer"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Union

import onnx
from datasets import Dataset, load_dataset
from onnxruntime.quantization import CalibrationDataReader
from vai_q_onnx import quantize_static

from optimum.quantization_base import OptimumQuantizer
from transformers import PretrainedConfig

from .configuration import QuantizationConfig, RyzenAIConfig
from .modeling import RyzenAIModel


LOGGER = logging.getLogger(__name__)


class RyzenAICalibrationDataReader(CalibrationDataReader):
    __slots__ = ["batch_size", "dataset", "_dataset_iter"]

    def __init__(self, dataset: Dataset, batch_size: int = 1):
        if dataset is None:
            raise ValueError("Provided dataset is None.")

        if batch_size <= 0:
            raise ValueError(f"Provided batch_size should be >= 1 (got: {batch_size}).")

        self.dataset = dataset
        self.batch_size = batch_size

        self._dataset_iter = iter(self.dataset)

    def get_next(self):
        featurized_samples = None
        try:
            if self.batch_size == 1:
                featurized_samples = {key: [value] for key, value in next(self._dataset_iter).items()}
            else:
                featurized_samples = defaultdict(list)
                for _ in range(self.batch_size):
                    sample = next(self._dataset_iter)

                    for name, value in sample.items():
                        featurized_samples[name] += [value]

        except StopIteration:
            pass

        if featurized_samples is not None and len(featurized_samples) > 0:
            return featurized_samples
        return None


class RyzenAIOnnxQuantizer(OptimumQuantizer):
    """
    Handles the RyzenAI quantization process for models shared on huggingface.co/models.
    """

    def __init__(self, onnx_model_path: Path, config: Optional["PretrainedConfig"] = None):
        """
        Args:
            onnx_model_path (`Path`):
                Path to the onnx model files you want to quantize.
            config (`Optional[PretrainedConfig]`, defaults to `None`):
                The configuration of the model.
        """
        super().__init__()
        self.onnx_model_path = onnx_model_path
        self.config = config
        if self.config is None:
            try:
                self.config = PretrainedConfig.from_pretrained(self.onnx_model_path.parent)
            except OSError:
                LOGGER.warning(
                    f"Could not load the config for {self.onnx_model_path} automatically, this might make "
                    "the quantized model harder to use because it will not be able to be loaded by an RyzenAIModel without "
                    "having to specify the configuration explicitly."
                )

    @classmethod
    def from_pretrained(
        cls,
        model_or_path: Union[str, Path],
        file_name: Optional[str] = None,
    ) -> "RyzenAIOnnxQuantizer":
        """
        Instantiates a `RyzenAIOnnxQuantizer` from an ONNX model file.

        Args:
            model_or_path (`Union[str, Path]`):
                Can be either:
                    - A path to a saved exported ONNX Intermediate Representation (IR) model, e.g., `./my_model_directory/.
            file_name(`Optional[str]`, defaults to `None`):
                Overwrites the default model file name from `"model.onnx"` to `file_name`.
                This allows you to load different model files from the same repository or directory.
        Returns:
            An instance of `RyzenAIOnnxQuantizer`.
        """
        ryzenai_quantizer_error_message = "RyzenAIOnnxQuantizer does not support multi-file quantization. Please create separate RyzenAIOnnxQuantizer instances for each model/file, by passing the argument `file_name` to RyzenAIOnnxQuantizer.from_pretrained()."

        if isinstance(model_or_path, str):
            model_or_path = Path(model_or_path)

        path = None
        if isinstance(model_or_path, Path) and file_name is None:
            onnx_files = list(model_or_path.glob("*.onnx"))
            if len(onnx_files) == 0:
                raise FileNotFoundError(f"Could not find any ONNX model file in {model_or_path}")
            elif len(onnx_files) > 1:
                raise RuntimeError(
                    f"Found too many ONNX model files in {model_or_path}. {ryzenai_quantizer_error_message}"
                )
            file_name = onnx_files[0].name

        if isinstance(model_or_path, RyzenAIModel):
            if path is None:
                path = Path(model_or_path.model_path)
        elif os.path.isdir(model_or_path):
            path = Path(model_or_path) / file_name
        else:
            raise ValueError(f"Unable to load model from {model_or_path}.")
        return cls(path)

    def quantize(
        self,
        quantization_config: QuantizationConfig,
        dataset: Dataset,
        save_dir: Union[str, Path],
        batch_size: int = 1,
        file_suffix: Optional[str] = "quantized",
    ) -> Path:
        """
        Quantizes a model given the optimization specifications defined in `quantization_config`.

        Args:
            quantization_config (`QuantizationConfig`):
                The configuration containing the parameters related to quantization.
            save_dir (`Union[str, Path]`):
                The directory where the quantized model should be saved.
            file_suffix (`Optional[str]`, defaults to `"quantized"`):
                The file_suffix used to save the quantized model.
            calibration_tensors_range (`Optional[Dict[str, Tuple[float, float]]]`, defaults to `None`):
                The dictionary mapping the nodes name to their quantization ranges, used and required only when applying static quantization.

        Returns:
            The path of the resulting quantized model.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        reader = RyzenAICalibrationDataReader(dataset, batch_size)

        suffix = f"_{file_suffix}" if file_suffix else ""
        quantized_model_path = save_dir.joinpath(f"{self.onnx_model_path.stem}{suffix}").with_suffix(".onnx")

        LOGGER.info("Quantizing model...")

        format = quantization_config.map_format().value
        calibration_method = quantization_config.map_calibration_method().value
        activation_type = quantization_config.map_activations_dtype().value
        weight_type = quantization_config.map_weights_dtype().value

        quantize_static(
            model_input=Path(self.onnx_model_path).as_posix(),
            model_output=quantized_model_path.as_posix(),
            calibration_data_reader=reader,
            quant_format=format,
            calibrate_method=calibration_method,
            input_nodes=quantization_config.input_nodes,
            output_nodes=quantization_config.output_nodes,
            op_types_to_quantize=quantization_config.op_types_to_quantize,
            random_data_reader_input_shape=quantization_config.random_data_reader_input_shape,
            per_channel=quantization_config.per_channel,
            reduce_range=quantization_config.reduce_range,
            activation_type=activation_type,
            weight_type=weight_type,
            nodes_to_quantize=quantization_config.nodes_to_quantize,
            nodes_to_exclude=quantization_config.nodes_to_exclude,
            optimize_model=quantization_config.optimize_model,
            use_external_data_format=quantization_config.use_external_data_format,
            execution_providers=quantization_config.execution_providers,
            enable_ipu_cnn=quantization_config.enable_ipu_cnn,
            convert_fp16_to_fp32=quantization_config.convert_fp16_to_fp32,
            convert_nchw_to_nhwc=quantization_config.convert_nchw_to_nhwc,
            include_cle=quantization_config.include_cle,
            extra_options=(
                quantization_config.extra_options.to_diff_dict(camel_case=True)
                if quantization_config.extra_options
                else {}
            ),
        )

        LOGGER.info(f"Saved quantized model at: {save_dir}")

        # Create and save the configuration summarizing all the parameters related to quantization
        ryzenai_config = RyzenAIConfig(quantization=quantization_config)
        ryzenai_config.save_pretrained(save_dir)

        if self.config is not None:
            self.config.save_pretrained(save_dir)

        return Path(save_dir)

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        seed: Optional[bool] = 2016,
        token: bool = None,
        streaming: bool = False,
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
            token (`bool`, defaults to `False`):
                Whether to use the token generated when running `transformers-cli login` (necessary for some datasets
                like ImageNet).
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration
            step.
        """

        calib_dataset = load_dataset(
            dataset_name, name=dataset_config_name, split=dataset_split, token=token, streaming=streaming
        )

        if num_samples is not None:
            if streaming is True:
                calib_dataset = calib_dataset.shuffle(seed=seed)
                calib_dataset = calib_dataset.take(num_samples)
            else:
                num_samples = min(num_samples, len(calib_dataset))
                calib_dataset = calib_dataset.shuffle(seed=seed).select(range(num_samples))

        ignored_columns = self.identify_unused_columns(calib_dataset)

        if preprocess_function is not None:
            processed_calib_dataset = calib_dataset.map(
                preprocess_function, batched=preprocess_batch, remove_columns=ignored_columns
            )
        else:
            processed_calib_dataset = calib_dataset.remove_columns(ignored_columns)

        return processed_calib_dataset

    def identify_unused_columns(self, dataset: Dataset) -> List[str]:
        model = onnx.load(self.onnx_model_path)
        model_inputs = {input.name for input in model.graph.input}
        ignored_columns = list(set(dataset.column_names) - model_inputs)
        return ignored_columns
