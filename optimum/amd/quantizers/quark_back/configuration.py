# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""Configuration classes for quantization with AMD Quark."""

from enum import Enum
from typing import Dict, List, Optional

from quark.torch.quantization.config.config import (
    AlgoConfig,
    AWQConfig,
    Config,
    GPTQConfig,
    QuantizationConfig,
    QuantizationSpec,
    SmoothQuantConfig,
)
from quark.torch.quantization.config.custom_config import (
    DEFAULT_AWQ_CONFIG,
    DEFAULT_FLOAT16_CONFIG,
    DEFAULT_GPTQ_CONFIG,
    DEFAULT_SMOOTH_QUANT_CONFIG,
    DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG,
    DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG,
    DEFAULT_W_INT4_PER_CHANNEL_CONFIG,
    DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG,
    DEFAULT_W_INT4_PER_TENSOR_CONFIG,
    DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG,
    DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG,
    DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG,
    DEFAULT_W_UINT4_PER_GROUP_CONFIG,
    FP8_PER_TENSOR_SPEC,
)

from ..quark.algo_config_constants import ALGO_CONFIG_PARAMS


QuarkQuantizationConfig = Config

__all__ = [
    "QuarkQuantizationConfig",
    "QuantizationSpec",
    "SmoothQuantConfig",
    "AWQConfig",
    "GPTQConfig",
    "AlgoConfig",
]


class KVCacheDType(Enum):
    FP8 = "fp8"


class AutoQuantizationConfig:
    @staticmethod
    def _apply_fp8_config(layer_quant_config):
        KV_CACHE_CFG = {
            "*.v_proj": QuantizationConfig(
                input_tensors=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC
            ),
            "*.k_proj": QuantizationConfig(
                input_tensors=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC
            ),
        }
        return {**layer_quant_config, **KV_CACHE_CFG}

    @staticmethod
    def _validate_kv_cache_dtype(kv_cache_dtype: str):
        if kv_cache_dtype.lower() != "fp8":
            raise ValueError(f"Invalid value for kv_cache_dtype: {kv_cache_dtype}. Expected 'fp8'.")

    @staticmethod
    def _validate_model_type(model_type: str):
        SUPPORTED_MODEL_TYPES = ["llama", "mistral", "opt", "qwen2"]

        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Invalid value for model_type for AutoQuantizationConfig: {model_type}. Expected one of {SUPPORTED_MODEL_TYPES}."
            )

        return model_type

    @staticmethod
    def w_fp8_a_fp8(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_fp8_a_fp8_o_fp8(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_int4_per_tensor(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_INT4_PER_TENSOR_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_int4_per_channel_sym(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_INT4_PER_CHANNEL_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_int4_per_group_sym(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_uint4_per_group_asym_awq(
        model_type: str,
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        model_type = AutoQuantizationConfig._validate_model_type(model_type)
        algo_config_info = ALGO_CONFIG_PARAMS[model_type]

        quant_config = DEFAULT_AWQ_CONFIG
        quant_config.algo_config.scaling_layers = algo_config_info["scaling_layers"]
        quant_config.algo_config.model_decoder_layers = algo_config_info["model_decoder_layers"]
        quant_config.algo_config.embedding_layers = algo_config_info["embedding_layers"]
        return QuarkQuantizationConfig(
            global_quant_config=quant_config.global_quant_config,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            algo_config=quant_config.algo_config,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_uint4_per_group_asym_smoothquant(
        model_type: str,
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        model_type = AutoQuantizationConfig._validate_model_type(model_type)
        algo_config_info = ALGO_CONFIG_PARAMS[model_type]

        quant_config = DEFAULT_SMOOTH_QUANT_CONFIG
        quant_config.algo_config.scaling_layers = algo_config_info["scaling_layers"]
        quant_config.algo_config.model_decoder_layers = algo_config_info["model_decoder_layers"]
        quant_config.algo_config.embedding_layers = algo_config_info["embedding_layers"]
        return QuarkQuantizationConfig(
            global_quant_config=quant_config.global_quant_config,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            algo_config=quant_config.algo_config,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_uint4_per_group_asym_gptq(
        model_type: str,
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        model_type = AutoQuantizationConfig._validate_model_type(model_type)
        algo_config_info = ALGO_CONFIG_PARAMS[model_type]

        quant_config = DEFAULT_GPTQ_CONFIG
        quant_config.algo_config.inside_layer_modules = algo_config_info["inside_layer_modules"]
        quant_config.algo_config.model_decoder_layers = algo_config_info["model_decoder_layers"]
        quant_config.algo_config.embedding_layers = algo_config_info["embedding_layers"]
        return QuarkQuantizationConfig(
            global_quant_config=quant_config.global_quant_config,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            algo_config=quant_config.algo_config,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_uint4_per_group_asym(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_uint4_a_bfloat16_per_group_asym(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_int8_a_int8_per_tensor_sym(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def w_int8_a_int8_per_tensor_sym_dynamic(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )

    @staticmethod
    def float16(
        layer_type_quant_config: Dict[str, QuantizationConfig] = {},
        layer_quant_config: Dict[str, QuantizationConfig] = {},
        exclude: List[str] = [],
        pre_quant_opt_config: Optional[QuantizationConfig] = None,
        kv_cache_dtype: Optional[KVCacheDType] = None,
    ):
        if kv_cache_dtype:
            AutoQuantizationConfig._validate_kv_cache_dtype(kv_cache_dtype)
            layer_quant_config = AutoQuantizationConfig._apply_fp8_config(layer_quant_config)

        return QuarkQuantizationConfig(
            global_quant_config=DEFAULT_FLOAT16_CONFIG,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            exclude=exclude,
            pre_quant_opt_config=pre_quant_opt_config,
        )
