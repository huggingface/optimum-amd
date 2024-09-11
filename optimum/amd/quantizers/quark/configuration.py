# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""Configuration classes for quantization with AMD Quark."""

from enum import Enum
from typing import Dict, List, Optional, Union

from quark.torch.quantization.config.config import (
    AlgoConfig,
    AWQConfig,
    Config,
    GPTQConfig,
    PreQuantOptConfig,
    QuantizationConfig,
    QuantizationSpec,
    SmoothQuantConfig,
)
from quark.torch.quantization.config.type import QuantizationMode

from .algo_config_constants import ALGO_CONFIG_PARAMS
from .quantizer import QuarkConfig

from .custom_configs import (
    FLOAT16_CONFIG,
    FP8_PER_TENSOR_SPEC,
    W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG,
    W_FP8_A_FP8_PER_TENSOR_CONFIG,
    W_INT4_PER_CHANNEL_CONFIG,
    W_INT4_PER_GROUP_SYM_CONFIG,
    W_INT4_PER_TENSOR_CONFIG,
    W_INT8_A_INT8_PER_TENSOR_CONFIG,
    W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG,
    W_INT8_PER_GROUP_CONFIG,
    W_INT8_PER_TENSOR_CONFIG,
    W_MX_FP8_A_MX_FP8_CONFIG,
    W_MX_FP8_CONFIG,
    W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG,
    W_UINT4_PER_GROUP_CONFIG,
)


Config = Config

__all__ = [
    "Config",
    "QuantizationSpec",
    "SmoothQuantConfig",
    "AWQConfig",
    "GPTQConfig",
    "AlgoConfig",
]


class KVCacheDType(Enum):
    FP8 = "fp8"


class AutoQuantizationConfig:
    QUANT_CONFIG_MAP = {
        "w_fp8_a_fp8": (W_FP8_A_FP8_PER_TENSOR_CONFIG, False),
        "w_fp8_a_fp8_o_fp8": (W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG, False),
        "w_int4_per_tensor": (W_INT4_PER_TENSOR_CONFIG, False),
        "w_int4_per_channel_sym": (W_INT4_PER_CHANNEL_CONFIG, False),
        "w_int4_per_group_sym": (W_INT4_PER_GROUP_SYM_CONFIG, True),
        "w_uint4_per_group_asym": (W_UINT4_PER_GROUP_CONFIG, True),
        "w_uint4_a_bfloat16_per_group_asym": (W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG, True),
        "w_int8_per_tensor_sym": (W_INT8_PER_TENSOR_CONFIG, False),
        "w_int8_per_group_sym": (W_INT8_PER_GROUP_CONFIG, True),
        "w_int8_a_int8_per_tensor_sym": (W_INT8_A_INT8_PER_TENSOR_CONFIG, False),
        "w_int8_a_int8_per_tensor_sym_dynamic": (W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG, False),
        "w_mx_fp8": (W_MX_FP8_CONFIG, False),
        "w_mx_fp8_a_mx_fp8": (W_MX_FP8_A_MX_FP8_CONFIG, False),
        "float16": (FLOAT16_CONFIG, False),
    }

    @staticmethod
    def from_quant_scheme(
        quant_scheme: str,
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        algo_config: Optional[AlgoConfig] = None,
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[Union[PreQuantOptConfig, List[PreQuantOptConfig]]] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
        quant_algo: Optional[str] = None,
        model_type: Optional[str] = None,
        group_size: Optional[int] = None,
        quant_mode: QuantizationMode = QuantizationMode.eager_mode,  # Default value provided
    ):
        """
        Generates a quantization configuration based on the specified quantization scheme.

        Args:
            quant_scheme (str): The name of the quantization scheme to be used. Available schemes are:

                - `w_fp8_a_fp8`: Quantization with FP8 weights and activations.
                - `w_fp8_a_fp8_o_fp8`: Quantization with FP8 weights, activations, and outputs.
                - `w_int4_per_tensor`: Quantization with INT4 weights per-tensor configuration.
                - `w_int4_per_channel_sym`: Quantization with INT4 weights per-channel symmetric configuration.
                - `w_int4_per_group_sym`: Quantization with INT4 weights per-group symmetric configuration.
                - `w_uint4_per_group_asym`: Quantization with UINT4 weights per-group asymmetric configuration.
                - `w_uint4_a_bfloat16_per_group_asym`: Quantization with UINT4 weights and BFLOAT16 activations per-group asymmetric configuration.
                - `w_int8_per_tensor_sym`: Quantization with INT8 weights per-tensor symmetric configuration.
                - `w_int8_per_group_sym`: Quantization with INT8 weights per-group symmetric configuration.
                - `w_int8_a_int8_per_tensor_sym`: Quantization with INT8 weights and activations per-tensor symmetric configuration.
                - `w_int8_a_int8_per_tensor_sym_dynamic`: Dynamic quantization with INT8 weights and activations per-tensor symmetric configuration.
                - `w_mx_fp8`: Quantization with MX-FP8 configuration.
                - `w_mx_fp8_a_mx_fp8`: Quantization with MX-FP8 weights and activations configuration.
                - `float16`: Quantization with FLOAT16 configuration.
            layer_type_quant_config (Dict[str, QuantizationConfig], optional):
                A dictionary mapping from layer types (e.g., `nn.Conv2d`, `nn.Linear`) to their quantization configurations. Defaults to `{}`.
            layer_quant_config (Dict[str, QuantizationConfig], optional):
                A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization. Defaults to `{}`.
            algo_config (Optional[AlgoConfig], optional):
                Optional configuration for the quantization algorithm (e.g., GPTQ, AWQ). Defaults to `None`.
            exclude (List[str], optional):
                A list of layer names to exclude from quantization. Defaults to `[]`.
            pre_quant_opt_config (Optional[Union[PreQuantOptConfig, List[PreQuantOptConfig]]], optional):
                Optional pre-quantization optimization configurations. Defaults to `None`.
            kv_cache_dtype (Optional[KVCacheDType], optional):
                Optional data type for the key-value cache (e.g., 'fp8'). Defaults to `None`.
            quant_algo (Optional[str], optional):
                The name of the quantization algorithm (e.g., 'awq', 'gptq'). Defaults to `None`.
            model_type (Optional[str], optional):
                The type of the model (e.g., 'llama', 'opt'). Required if `quant_algo` is provided. Defaults to `None`.
            group_size (Optional[int], optional):
                Group size for the quantization scheme, if required. Defaults to `None`.
            quant_mode (QuantizationMode, optional):
                The quantization mode (e.g., EAGER_MODE or POST_TRAINING_MODE). Defaults to `QuantizationMode.eager_mode`.

        Returns:
            Config:
                The final quantization configuration for the specified quantization scheme.

        Raises:
            ValueError:
                If an invalid `quant_scheme` is provided.
                If `group_size` is required by the `quant_scheme` but not provided.
                If both `quant_algo` and `algo_config` are provided.
                If `quant_algo` is provided but `model_type` is missing.
        """
        if quant_scheme not in AutoQuantizationConfig.QUANT_CONFIG_MAP:
            raise ValueError(f"Invalid quantization scheme: {quant_scheme}")

        global_quant_config, requires_group_size = AutoQuantizationConfig.QUANT_CONFIG_MAP[quant_scheme]

        if requires_group_size and group_size is None:
            raise ValueError(f"Quantization scheme '{quant_scheme}' requires 'group_size'.")

        if requires_group_size:
            global_quant_config.weight.group_size = group_size

        # Proceed with the rest of the logic
        config = AutoQuantizationConfig._get_config(
            global_quant_config=global_quant_config,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            algo_config=algo_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
            kv_cache_dtype=kv_cache_dtype,
            quant_algo=quant_algo,
            model_type=model_type,
            quant_mode=quant_mode,
        )
        
        return  QuarkConfig(config)

    @staticmethod
    def _get_config(
        global_quant_config: QuantizationConfig,
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        algo_config: Optional[AlgoConfig] = None,
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[Union[PreQuantOptConfig, List[PreQuantOptConfig]]] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
        quant_algo: Optional[str] = None,
        model_type: Optional[str] = None,
        quant_mode: QuantizationMode = QuantizationMode.eager_mode,  # Default value provided
    ):
        if quant_algo and algo_config:
            raise ValueError("Only one of quant_algo and algo_config can be provided.")
        if quant_algo and model_type is None:
            raise ValueError("model_type must be provided when quant_algo is provided.")

        algo_config = algo_config or AutoQuantizationConfig._get_algo_config(
            quant_algo, global_quant_config, model_type
        )

        # Apply kv_cache_dtype if necessary
        if kv_cache_dtype:
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(
                global_quant_config, layer_quant_config, kv_cache_dtype
            )

        quant_config = Config(
            global_quant_config=global_quant_config,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
            algo_config=algo_config,
            quant_mode=quant_mode,
        )
        
        if algo_config and quant_algo is None:
            model_type = AutoQuantizationConfig._validate_model_type(model_type)
            algo_config_info = ALGO_CONFIG_PARAMS[model_type]
            quant_config.algo_config.inside_layer_modules = algo_config_info["inside_layer_modules"]
            quant_config.algo_config.model_decoder_layers = algo_config_info["model_decoder_layers"]
            quant_config.algo_config.embedding_layers = algo_config_info["embedding_layers"]

        return quant_config

    @staticmethod
    def _apply_fp8_config(global_quant_config, layer_quant_config, kv_cache_dtype):
        """
        Applies FP8 configuration if kv_cache_dtype is 'fp8'.

        Args:
            global_quant_config (QuantizationConfig): The global quantization configuration.
            layer_quant_config (Dict[str, QuantizationConfig]): The layer-specific quantization configuration.
            kv_cache_dtype (str): The data type for the key-value cache (e.g., 'fp8').

        Returns:
            Dict[str, QuantizationConfig]: The updated layer-specific quantization configuration.

        Raises:
            ValueError: If an invalid kv_cache_dtype is provided.
        """
        if kv_cache_dtype.lower() != "fp8":
            raise ValueError(f"Invalid value for kv_cache_dtype: {kv_cache_dtype}. Expected 'fp8'.")

        FP8_CONFIG = {
            "*.v_proj": QuantizationConfig(
                input_tensors=global_quant_config.input_tensors,
                output_tensors=FP8_PER_TENSOR_SPEC,
                weight=global_quant_config.weight,
            ),
            "*.k_proj": QuantizationConfig(
                input_tensors=global_quant_config.input_tensors,
                output_tensors=FP8_PER_TENSOR_SPEC,
                weight=global_quant_config.weight,
            ),
        }
        return {**layer_quant_config, **FP8_CONFIG}

    @staticmethod
    def _get_algo_config(
        quant_algo: Optional[str], global_quant_config: QuantizationConfig, model_type: Optional[str] = None
    ) -> Optional[AlgoConfig]:
        """
        Retrieves the appropriate algorithm configuration for the quantization algorithm.

        Args:
            quant_algo (Optional[str]): The name of the quantization algorithm (e.g., 'awq', 'gptq').
            global_quant_config (QuantizationConfig): The global quantization configuration.
            model_type (Optional[str], optional): The model type, required if using quant_algo. Defaults to None.

        Returns:
            Optional[AlgoConfig]: The algorithm configuration if a valid `quant_algo` is provided.

        Raises:
            ValueError: If an invalid model_type is provided or if unsupported configurations are used.
        """
        if quant_algo is None:
            return None

        if model_type is None:
            raise ValueError("model_type must be provided when quant_algo is specified.")

        SUPPORTED_MODEL_TYPES = ["llama", "mistral", "opt", "qwen2"]
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Invalid model_type: {model_type}. Expected one of {SUPPORTED_MODEL_TYPES}.")

        algo_config_map = {"awq": AWQConfig, "smoothquant": SmoothQuantConfig, "gptq": GPTQConfig}

        if quant_algo not in algo_config_map:
            return None

        algo_config = algo_config_map[quant_algo]()

        # Ensure compatibility of quant_algo with global_quant_config
        if quant_algo == "awq":
            assert global_quant_config in [
                W_UINT4_PER_GROUP_CONFIG,
                W_INT4_PER_GROUP_SYM_CONFIG,
                W_INT8_PER_GROUP_CONFIG,
            ]
        elif quant_algo == "gptq":
            assert global_quant_config in [W_UINT4_PER_GROUP_CONFIG]

        return algo_config

    @staticmethod
    def _validate_model_type(model_type: str):
        SUPPORTED_MODEL_TYPES = ["llama", "mistral", "opt", "qwen2"]

        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Invalid value for model_type for AutoQuantizationConfig: {model_type}. Expected one of {SUPPORTED_MODEL_TYPES}."
            )

        return model_type