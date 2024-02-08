# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import inspect
import logging
from typing import Dict, List, Optional, Union

import torch
from brevitas.graph.calibrate import calibration_mode, bias_correction_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gptq import gptq_mode
from brevitas_examples.common.generative.quantize import quantize_model
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers
from tqdm import tqdm

from optimum.exporters import TasksManager
from optimum.quantization_base import OptimumQuantizer
from transformers.utils.fx import symbolic_trace

from .accelerate_utils import offload_model, remove_hooks
from .configuration import BrevitasQuantizationConfig


logger = logging.getLogger(__name__)


class BrevitasQuantizer(OptimumQuantizer):
    """
    Handles the Runtime quantization process for models shared on huggingface.co/models.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.group_of_parallel_layers = None

    def validate_quant_config(self, quantization_config: BrevitasQuantizationConfig):
        dtype = next(iter(self.model.parameters()))
        if dtype == torch.bfloat16 and quantization_config.replace_mha_with_quantizable:
            raise RuntimeError("Scaled_dot_product does not support bfloat16 and cuda")

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
        task = TasksManager.infer_task_from_model(model_name_or_path)

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
        )

        return cls(model)

    def quantize(
        self, quantization_config: BrevitasQuantizationConfig, calibration_dataset: Optional[List[Dict]] = None
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
        self.validate_quant_config(quantization_config)

        requires_data = (
            quantization_config.activations_equalization
            or quantization_config.apply_gptq
            or quantization_config.is_static
        )
        if calibration_dataset is None and requires_data:
            raise ValueError(
                f"No calibration_dataset was passed, but a calibration dataset is required with the quantization configuration activations_equalization={quantization_config.activations_equalization}, apply_gptq={quantization_config.apply_gptq}, is_static={quantization_config.is_static}."
            )

        dtype = next(iter(self.model.parameters())).dtype

        # Insert standard MHA layers when performing fx based weight/activation equalization to avoid dealing
        # with all the variability in HF implementations.
        if quantization_config.replace_mha_with_quantizable:
            logger.info("Replace HF MHA with quantizable variants...")
            self.model = replace_mha_with_quantizable_layers(self.model, dtype)
            logger.info("Replacing done.")

        if quantization_config.requires_fx_graph():
            forward_signature = inspect.signature(self.model.forward).parameters
            if all(
                input_name in forward_signature for input_name in ["input_ids", "attention_mask", "past_key_values"]
            ):
                input_names = ["input_ids", "attention_mask", "past_key_values"]
            else:
                raise ValueError(
                    f"Quantization with an FX graph is currently only supported for models taking `input_ids`, `attention_mask` and `past_key_values` as inputs. The model only has the following inputs: {forward_signature}"
                )

            with torch.no_grad():
                model = symbolic_trace(self.model, input_names)
        else:
            model = self.model

        # Because accelerate is not compatible with FX, we keep two versions of the Model
        # one with FX-traced, the other one not.
        # Since weights are shared across the two, we can apply weight/activation equalization
        # by using one representation or the other based on needs.

        if quantization_config.apply_weight_equalization:
            logger.info("Applying weight equalization...")
            apply_weight_equalization(model)
            logger.info("Weight equalization applied.")

        if quantization_config.activations_equalization is not None:
            logger.info(
                f"Applying Activation Equalization {quantization_config.activations_equalization} (SmoothQuant)..."
            )
            apply_act_equalization(model, quantization_config.activations_equalization, calibration_dataset)
            logger.info("Activation equalization applied.")

        # We do not quantize embedding and last fully connected layer
        model = quantize_model(
            model,
            dtype=dtype,
            weight_quant_format="int",
            weight_quant_type="sym" if quantization_config.weights_symmetric else "asym",
            weight_bit_width=quantization_config.weights_bitwidth,
            weight_param_method=quantization_config.weights_param_method,
            weight_scale_precision=quantization_config.scale_precision,
            weight_quant_granularity=quantization_config.weights_quant_granularity,
            weight_group_size=quantization_config.weights_group_size,
            quantize_weight_zero_point=quantization_config.quantize_zero_point,
            input_bit_width=quantization_config.activations_bitwidth,
            input_quant_type="sym" if quantization_config.activations_symmetric else "asym",
            input_quant_format="int",
            input_param_method=quantization_config.activations_param_method,
            input_scale_precision=quantization_config.scale_precision,
            input_scale_type="static" if quantization_config.is_static else "dynamic",
            input_quant_granularity=quantization_config.activations_quant_granularity,
            input_group_size=quantization_config.activations_group_size,
            quantize_input_zero_point=quantization_config.quantize_zero_point,
        )

        # Perform a single inference pass to generate the correct state_dict
        if quantization_config.apply_gptq or quantization_config.is_static:
            with torch.no_grad():
                model(**calibration_dataset[0])

        if quantization_config.apply_gptq:
            logger.info("Applying gptq...")
            apply_gptq(
                model,
                calibration_dataset,
                act_order=quantization_config.gptq_act_oder,
                group_of_parallel_layers=self.group_of_parallel_layers,
            )
            logger.info("GPTQ applied.")

        if quantization_config.activations_bitwidth is not None and quantization_config.is_static:
            logger.info("Applying activation calibration...")
            apply_calibration(model, calibration_dataset)
            logger.info("Activation calibration applied.")

        if quantization_config.apply_gptq:
            logger.info("Applying Bias Correction...")
            apply_bias_correction(
                model,
                calibration_dataset,
            )
            logger.info("Bias Correction applied.")

        return model

    """
    TODO: test this, and maybe use it by default?
    def find_groups_of_parallel_layers(self, names_of_groups_of_parallel_layers):
        names = [name for name, _ in self.model.named_modules()]
        group_of_parallel_layers = []
        set_found_layers = set()
        for group in names_of_groups_of_parallel_layers:
            first_name = group[0]
            for name in names:
                if name.endswith(first_name) and name not in set_found_layers:
                    all_names_present = True
                    prefix = name.removesuffix(first_name)
                    for name_in_group in group:
                        if not prefix + name_in_group in names:
                            all_names_present = False
                    if all_names_present:
                        found_names = [prefix + name_in_group for name_in_group in group]
                        group_of_parallel_layers.append(found_names)
                        set_found_layers.update(found_names)
        self.group_of_parallel_layers = group_of_parallel_layers
    """


@torch.no_grad()
def apply_act_equalization(
    model: torch.nn.Module, act_equalization_type: str, dataset: List[Dict], alpha: float = 0.5
) -> None:
    model = offload_model(model)

    if act_equalization_type == "layerwise":
        with activation_equalization_mode(model, alpha, add_mul_node=True, layerwise=True):
            with torch.no_grad():
                for inps in tqdm(dataset):
                    model(**inps)

    elif act_equalization_type == "cross_layer":
        if not isinstance(model, torch.fx.GraphModule):
            raise RuntimeError(
                "An fx.GraphModule model is required to perform cross-layer SmoothQuant activation equalization."
            )

        with activation_equalization_mode(
            model, alpha, add_mul_node=False, layerwise=False, co_optimize_act_weights=True
        ):
            with torch.no_grad():
                for inps in tqdm(dataset):
                    model(**inps)

    else:
        raise ValueError(f"The activation equalization type {act_equalization_type} not supported.")

    # Remove all accelerate hooks.
    remove_hooks(model)


@torch.no_grad()
def apply_gptq(
    model: torch.nn.Module,
    dataset: List[Dict],
    act_order: bool = True,
    group_of_parallel_layers: Optional[List[List]] = None,
) -> None:
    """
    To speed up GPTQ computation, we can look through the model to find layers that can be optimized in parallel because they do not depend on each other. A typical case is the input matrices of the attention layer. We just need to specify the suffix of the layer, and they will be matched across the entire structure.
    """
    model = offload_model(model)

    with gptq_mode(
        model,
        use_quant_activations=False,
        group_of_parallel_layers=group_of_parallel_layers,
        act_order=act_order,
        create_weight_orig=False,
    ) as gptq:
        for _ in tqdm(range(gptq.num_layers)):
            for inps in dataset:
                gptq.model(**inps)
            gptq.update()

    # Remove all accelerate hooks.
    remove_hooks(model)


@torch.no_grad()
def apply_calibration(model: torch.nn.Module, dataset: List[Dict]) -> None:
    model = offload_model(model)

    with calibration_mode(model):
        with torch.no_grad():
            for inps in tqdm(dataset):
                model(**inps)

    # Remove all accelerate hooks.
    remove_hooks(model)


@torch.no_grad()
def apply_bias_correction(model, dataloader):
    model = offload_model(model)

    with bias_correction_mode(model):
        for inps in tqdm(dataloader):
            model(**inps)

    # Remove all accelerate hooks.
    remove_hooks(model)
