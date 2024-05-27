# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""Configuration classes for quantization with RyzenAI."""

import re
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

import vai_q_onnx

from optimum.configuration_utils import BaseConfig


class CalibrationMethod(Enum):
    """CalibrationMethod is an enumeration of the calibration methods supported by RyzenAI quantization."""

    MinMax = vai_q_onnx.CalibrationMethod.MinMax
    Entropy = vai_q_onnx.CalibrationMethod.Entropy
    Percentile = vai_q_onnx.CalibrationMethod.Percentile
    NonOverflow = vai_q_onnx.PowerOfTwoMethod.NonOverflow
    MinMSE = vai_q_onnx.PowerOfTwoMethod.MinMSE


class QuantFormat(Enum):
    """QuantFormat is an enumeration of the quantization formats supported by RyzenAI quantization."""

    QOperator = vai_q_onnx.QuantFormat.QOperator
    QDQ = vai_q_onnx.QuantFormat.QDQ
    VitisQuantFormat_QDQ = vai_q_onnx.VitisQuantFormat.QDQ
    VitisQuantFormat_FixNeuron = vai_q_onnx.VitisQuantFormat.FixNeuron


class QuantType(Enum):
    """QuantType is an enumeration of the quantization types supported by RyzenAI quantization."""

    QInt8 = vai_q_onnx.QuantType.QInt8
    QUInt8 = vai_q_onnx.QuantType.QUInt8
    QUInt16 = vai_q_onnx.VitisQuantType.QUInt16
    QInt16 = vai_q_onnx.VitisQuantType.QInt16
    QUInt32 = vai_q_onnx.VitisQuantType.QUInt32
    QInt32 = vai_q_onnx.VitisQuantType.QInt32
    QFloat16 = vai_q_onnx.VitisQuantType.QFloat16
    QBFloat16 = vai_q_onnx.VitisQuantType.QBFloat16


@dataclass
class ExtraOptions:
    """
    ExtraOptions is a dataclass handling additional options for quantization.

    Args:
        activation_symmetric (`bool`, defaults to `False`):
            If True, symmetrize calibration data for activations.
        weight_symmetric (`bool`, defaults to `True`):
            If True, symmetrize calibration data for weights.
        use_unsigned_relu (`bool`, defaults to `False`):
            If True, the output tensor of ReLU and Clip, whose min is 0, will be forced to be asymmetric.
        quantize_bias (`bool`, defaults to `True`):
            If True, quantize the Bias as normal weights.
        remove_input_init (`bool`, defaults to `True`):
            If True, initializer in graph inputs will be removed because it will not be treated as a constant value/weight.
            This may prevent some of the graph optimizations, like const folding.
        enable_subgraph (`bool`, defaults to `False`):
            If True, the subgraph will be quantized. More support for this feature is planned in the future.
        force_quantize_no_input_check (`bool`, defaults to `False`):
            If True, latent operators such as maxpool and transpose will always quantize their inputs, generating quantized
            outputs even if their inputs have not been quantized.
        matmul_const_b_only (`bool`, defaults to `False`):
            If True, only MatMul operations with a constant 'B' will be quantized.
        add_qdq_pair_to_weight (`bool`, defaults to `False`):
            If True, both QuantizeLinear and DeQuantizeLinear nodes are inserted for weight, maintaining its floating-point format.
            In the PowerOfTwoMethod calibration method, this setting will also be effective for the bias.
        op_types_to_exclude_output_quantization (`List[str] or None`, defaults to `[]`):
            If specified, the output of operators with these types will not be quantized.
        dedicated_qdq_pair (`bool`, defaults to `False`):
            If True, an identical and dedicated QDQ pair is created for each node, allowing multiple nodes to share a single QDQ pair
            as their inputs.
        qdq_op_type_per_channel_support_to_axis (`Dict`, defaults to `{}`):
            Sets the channel axis for specific operator types (e.g., {'MatMul': 1}).
        use_qdq_vitis_custom_ops (`bool`, defaults to `True`):
            If True, The UInt8 and Int8 quantization will be executed by the custom operations library, otherwise by the library
            of onnxruntime extensions. Only valid in vai_q_onnx.VitisQuantFormat.QDQ.
        calib_tensor_range_symmetric (`bool`, defaults to `False`):
            If True, the final range of the tensor during calibration will be symmetrically set around the central point "0".
            In PowerOfTwoMethod calibration method, the default is True.
        calib_moving_average (`bool`, defaults to `False`):
            If True, the moving average of the minimum and maximum values will be computed when the calibration method selected is
            MinMax. In PowerOfTwoMethod calibration method, this should be set to False.
        calib_moving_average_constant (`float`, defaults to `0.01`):
            Specifies the constant smoothing factor to use when computing the moving average of the minimum and maximum values.
            Only effective when the calibration method selected is MinMax and CalibMovingAverage is set to True.
            In PowerOfTwoMethod calibration method, this option is unsupported.
        random_data_reader_input_data_range (`Dict or None`, defaults to `None`):
            Specifies the data range for each input if used random data reader (calibration_data_reader is None).
        int16_scale (`bool`, defaults to `False`):
            If True, the float scale will be replaced by the closest value corresponding to M and 2**N, where the range of M and 2**N
            is within the representation range of int16 and uint16.
        min_mse_mode (`str`, defaults to `'All'`):
            When using vai_q_onnx.PowerOfTwoMethod.MinMSE, you can specify the method for calculating minmse.
            By default, minmse is calculated using all calibration data. Alternatively, you can set the mode to "MostCommon",
            where minmse is calculated for each batch separately and take the most common value.
        convert_bn_to_conv (`bool`, defaults to `True`):
            If True, the BatchNormalization operation will be converted to Conv operation when enable_ipu_cnn is True.
        convert_reduce_mean_to_global_avg_pool (`bool`, defaults to `True`):
            If True, the Reduce Mean operation will be converted to Global Average Pooling operation when enable_ipu_cnn is True.
        split_large_kernel_pool (`bool`, defaults to `True`):
            If True, the large kernel Global Average Pooling operation will be split into multiple Average Pooling operation when
            enable_ipu_cnn is True.
        convert_split_to_slice (`bool`, defaults to `True`):
            If True, the Split operation will be converted to Slice operation when enable_ipu_cnn is True.
        fuse_instance_norm (`bool`, defaults to `False`):
            If True, the split instance norm operation will be fused to InstanceNorm operation when enable_ipu_cnn is True.
        fuse_l2_norm (`bool`, defaults to `False`):
            If True, a set of L2norm operations will be fused to L2Norm operation when enable_ipu_cnn is True.
        convert_clip_to_relu (`bool`, defaults to `False`):
            If True, the Clip operations that have a min value of 0 will be converted to ReLU operations.
        simulate_dpu (`bool`, defaults to `True`):
            If True, a simulation transformation that replaces some operations with an approximate implementation will be applied
            for DPU when enable_ipu_cnn is True.
        convert_leaky_relu_to_dpu_version (`bool`, defaults to `True`):
            If True, the Leaky Relu operation will be converted to DPU version when SimulateDPU is True.
        convert_sigmoid_to_hard_sigmoid (`bool`, defaults to `True`):
            If True, the Sigmoid operation will be converted to Hard Sigmoid operation when SimulateDPU is True.
        convert_hard_sigmoid_to_dpu_version (`bool`, defaults to `True`):
            If True, the Hard Sigmoid operation will be converted to DPU version when SimulateDPU is True.
        convert_avg_pool_to_dpu_version (`bool`, defaults to `True`):
            If True, the global or kernel-based Average Pooling operation will be converted to DPU version when SimulateDPU is True.
        convert_reduce_mean_to_dpu_version (`bool`, defaults to `True`):
            If True, the ReduceMean operation will be converted to DPU version when SimulateDPU is True.
        convert_softmax_to_dpu_version (`bool`, defaults to `False`):
            If True, the Softmax operation will be converted to DPU version when SimulateDPU is True.
        ipu_limitation_check (`bool`, defaults to `True`):
            If True, the quantization scale will be adjusted due to the limitation of DPU/NPU.
        adjust_shift_cut (`bool`, defaults to `True`):
            If True, adjust the shift cut of nodes when ipu_limitation_check is True.
        adjust_shift_bias (`bool`, defaults to `True`):
            If True, adjust the shift bias of nodes when ipu_limitation_check is True.
        adjust_shift_read (`bool`, defaults to `True`):
            If True, adjust the shift read of nodes when ipu_limitation_check is True.
        adjust_shift_write (`bool`, defaults to `True`):
            If True, adjust the shift write of nodes when ipu_limitation_check is True.
        adjust_hard_sigmoid (`bool`, defaults to `True`):
            If True, adjust the pos of hard sigmoid nodes when ipu_limitation_check is True.
        adjust_shift_swish (`bool`, defaults to `True`):
            If True, adjust the shift swish when ipu_limitation_check is True.
        align_concat (`bool`, defaults to `True`):
            If True, adjust the quantization pos of concat when ipu_limitation_check is True.
        align_pool (`bool`, defaults to `True`):
            If True, adjust the quantization pos of pooling when ipu_limitation_check is True.
        replace_clip6_relu (`bool`, defaults to `False`):
            If True, replace Clip(0,6) with Relu in the model.
        cle_steps (`int`, defaults to `1`):
            Specifies the steps for CrossLayerEqualization execution when include_cle is set to true. When set to -1,
            an adaptive CrossLayerEqualization will be conducted.
        cle_total_layer_diff_threshold (`float`, defaults to `2e-7`):
            Specifies The threshold represents the sum of mean transformations of CrossLayerEqualization transformations across
            all layers when utilizing CrossLayerEqualization.
        cle_scale_append_bias (`bool`, defaults to `True`):
            Whether the bias be included when calculating the scale of the weights.
        remove_qdq_conv_leaky_relu (`bool`, defaults to `False`):
            If True, the QDQ between Conv and LeakyRelu will be removed for DPU when enable_ipu_cnn is True.
        remove_qdq_conv_prelu (`bool`, defaults to `False`):
            If True, the QDQ between Conv and PRelu will be removed for DPU when enable_ipu_cnn is True.
    """

    activation_symmetric: bool = False
    weight_symmetric: bool = True
    use_unsigned_relu: bool = False
    quantize_bias: bool = True
    remove_input_init: bool = True
    enable_subgraph: bool = False
    force_quantize_no_input_check: bool = False
    matmul_const_b_only: bool = False
    add_qdq_pair_to_weight: bool = False
    op_types_to_exclude_output_quantization: Union[List[str], None] = None
    dedicated_qdq_pair: bool = False
    qdq_op_type_per_channel_support_to_axis: Dict = field(default_factory=dict)
    use_qdq_vitis_custom_ops: bool = True
    calib_tensor_range_symmetric: bool = False
    calib_moving_average: bool = False
    calib_moving_average_constant: float = 0.01
    random_data_reader_input_data_range: Union[Dict, None] = None
    int16_scale: bool = False
    min_mse_mode: str = "All"
    convert_bn_to_conv: bool = True
    convert_reduce_mean_to_global_avg_pool: bool = True
    split_large_kernel_pool: bool = True
    convert_split_to_slice: bool = True
    fuse_instance_norm: bool = False
    fuse_l2_norm: bool = False
    convert_clip_to_relu: bool = False
    simulate_dpu: bool = True
    convert_leaky_relu_to_dpu_version: bool = True
    convert_sigmoid_to_hard_sigmoid: bool = True
    convert_hard_sigmoid_to_dpu_version: bool = True
    convert_avg_pool_to_dpu_version: bool = True
    convert_reduce_mean_to_dpu_version: bool = True
    convert_softmax_to_dpu_version: bool = False
    ipu_limitation_check: bool = True
    adjust_shift_cut: bool = True
    adjust_shift_bias: bool = True
    adjust_shift_read: bool = True
    adjust_shift_write: bool = True
    adjust_hard_sigmoid: bool = True
    adjust_shift_swish: bool = True
    align_concat: bool = True
    align_pool: bool = True
    replace_clip6_relu: bool = False
    cle_steps: int = 1
    cle_total_layer_diff_threshold: float = 2e-7
    cle_scale_append_bias: bool = True
    remove_qdq_conv_leaky_relu: bool = False
    remove_qdq_conv_prelu: bool = False

    @property
    def snake_to_camel(self):
        return {
            "qdq_op_type_per_channel_support_to_axis": "QDQOpTypePerChannelSupportToAxis",
            "ipu_limitation_check": "IPULimitationCheck",
            "cle_steps": "CLESteps",
            "cle_total_layer_diff_threshold": "CLETotalLayerDiffThreshold",
            "cle_scale_append_bias": "CLEScaleAppendBias",
        }

    @property
    def camel_to_snake(self):
        return {value: key for key, value in self.snake_to_camel.items()}

    def __setattr__(self, name, value):
        snake_case_name = self.camel_to_snake.get(name, re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_"))

        super().__setattr__(snake_case_name, value)

    def __getattr__(self, name):
        snake_case_name = self.camel_to_snake.get(name, re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_"))
        return getattr(self, snake_case_name)

    def to_diff_dict(self, camel_case=False) -> dict:
        """
        Returns a dictionary of non-default values in the configuration.
        """
        non_default_values = {}
        for option in fields(self):
            if camel_case:
                name = self.snake_to_camel.get(
                    option.name, "".join(word.capitalize() for word in option.name.split("_"))
                )
            else:
                name = option.name
            if getattr(self, option.name) != option.default and getattr(self, option.name) != {}:
                non_default_values[name] = getattr(self, option.name)
        return non_default_values


@dataclass
class QuantizationConfig:
    """
    QuantizationConfig is the configuration class handling all the RyzenAI quantization parameters.

    Args:
        is_static (`bool`):
            Whether to apply static quantization or dynamic quantization.
        format (Union[QuantFormat, str], defaults to `QuantFormat.QDQ`):
            This parameter is used to specify the quantization format of the model.
            Options:
            - `QuantFormat.QOperator`: Quantizes the model directly using quantized operators.
            - `QuantFormat.QDQ`: Quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor.
              Supports 8-bit quantization only.
            - `QuantFormat.VitisQuantFormat`: Quantizes the model by inserting VitisQuantizeLinear/VitisDequantizeLinear
              into the tensor. Supports a wider range of bit-widths and precisions.
            - `QuantFormat.FixNeuron` (Experimental): Quantizes the model by inserting FixNeuron (a combination of
              QuantizeLinear and DeQuantizeLinear) into the tensor. Experimental and not recommended for deployment.
        calibration_method (Union[CalibrationMethod, str], defaults to `CalibrationMethod.MinMSE`):
            The method used in calibration.
            Options (for CNNs running on NPU, power-of-two methods; for Transformers on NPU or CNNs on CPU, float scale methods):
            - `CalibrationMethod.NonOverflow`: Power-of-two method to prevent min/max values from overflowing.
            - `CalibrationMethod.MinMSE`: Power-of-two method to minimize mean-square-loss of quantized values and float values.
              Longer calibration time but usually better accuracy.
            - `CalibrationMethod.MinMax`: Obtain quantization parameters based on minimum and maximum values of each tensor.
            - `CalibrationMethod.Entropy`: Determine quantization parameters based on the entropy algorithm of each tensor's distribution.
            - `CalibrationMethod.Percentile`: Calculate quantization parameters using percentiles of tensor values.
        activations_dtype (QuantType, defaults to `QuantType.QUInt8`):
            The quantization data type to use for the activations.
        weights_dtype (QuantType, defaults to `QuantType.QInt8`):
            The quantization data type to use for the weights.
        enable_ipu_cnn (bool, defaults to `True`):
            Flag to generate a quantized model suitable for DPU/NPU computations. If True, the quantization process will
            consider specific limitations and requirements of DPU/NPU, optimizing the model accordingly.
        input_nodes (List[str], defaults to an empty list `[]`):
            List of names of starting nodes to be quantized. Nodes before these nodes will not be quantized.
        output_nodes (List[str], defaults to an empty list `[]`):
            List of names of end nodes to be quantized. Nodes after these nodes will not be quantized.
        op_types_to_quantize (List[str], defaults to an empty list `[]`):
            If specified, only operators of the given types will be quantized (e.g., ['Conv'] to quantize Convolutional layers).
        random_data_reader_input_shape (Union[List[int], Tuple[int], Dict[str, List[int]]], defaults to an empty list `[]`):
            Shapes of input nodes for internal random data reader. If dynamic axes require specific values, provide shapes.
            Format: list/tuple for single input, list of lists for multiple inputs, or dict {name: shape} for named inputs.
        per_channel (bool, defaults to `False`):
            Determines whether weights should be quantized per channel. Must be False for DPU/NPU devices.
        reduce_range (bool, defaults to `False`):
            If True, quantizes weights with 7-bits. Must be False for DPU/NPU devices.
        activation_type (QuantType, defaults to `QuantType.QInt8`):
            Specifies the quantization data type for activations.
        weight_type (QuantType, defaults to `QuantType.QInt8`):
            Specifies the quantization data type for weights. Must be `QuantType.QInt8` for NPU devices.
        nodes_to_quantize (List[str], defaults to an empty list `[]`):
            If specified, only the nodes in this list are quantized.
        nodes_to_exclude (List[str], defaults to an empty list `[]`):
            If specified, nodes in this list will be excluded from quantization.
        optimize_model (bool, defaults to `True`):
            If True, optimizes the model before quantization.
        use_external_data_format (bool, defaults to `False`):
            Flag for large size (>2GB) models. If True, model proto and data will be stored in separate files.
        execution_providers (List[str], defaults to `['CPUExecutionProvider']`):
            Defines the execution providers used by ONNX Runtime for model calibration.
        convert_fp16_to_fp32 (bool, defaults to `False`):
            Controls whether to convert the input model from float16 to float32 before quantization.
        convert_nchw_to_nhwc (bool, defaults to `False`):
            Controls whether to convert the input NCHW model to NHWC model before quantization.
        include_cle (bool, defaults to `False`):
            Flag to optimize models using CrossLayerEqualization; can improve accuracy for some models.
        extra_options (Union[Dict, None, ExtraOptions], defaults to an instance of `ExtraOptions` with default values):
            Contains key-value pairs for various options in different cases.
    """

    format: Literal["qdq", "qop", "vitisqdq"] = "qdq"
    calibration_method: Literal["nonoverflow", "mse", "minmax", "entropy", "percentile"] = "mse"
    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)
    op_types_to_quantize: List[str] = field(default_factory=list)
    random_data_reader_input_shape: Union[List[int], Tuple[int], Dict[str, List[int]]] = field(default_factory=list)
    per_channel: bool = False
    reduce_range: bool = False
    activations_dtype: Literal["uint8", "int8", "uint16", "int16", "uint32", "int32", "bfloat16", "float16"] = "uint8"
    weights_dtype: Literal["uint8", "int8", "uint16", "int16", "uint32", "int32", "bfloat16", "float16"] = "int8"
    nodes_to_quantize: List[str] = field(default_factory=list)
    nodes_to_exclude: List[str] = field(default_factory=list)
    optimize_model: bool = True
    use_external_data_format: bool = False
    execution_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    enable_ipu_cnn: bool = False
    convert_fp16_to_fp32: bool = False
    convert_nchw_to_nhwc: bool = False
    include_cle: bool = False
    extra_options: ExtraOptions = field(default_factory=ExtraOptions)

    def __post_init__(self):
        if isinstance(self.extra_options, dict):
            self.extra_options = ExtraOptions(**self.extra_options)
        self.format = self._map_format(self.format)
        self.calibration_method = self._map_calibration_method(self.calibration_method)
        self.activations_dtype, self.weights_dtype = self._map_dtypes(self.activations_dtype, self.weights_dtype)

        self.check_dtype_and_format(self.activations_dtype, "activations_dtype", self.format)
        self.check_dtype_and_format(self.weights_dtype, "weights_dtype", self.format)

        if self.enable_ipu_cnn:
            if self.format not in ["qdq"]:
                raise ValueError('ipu cnn configuration only support format "qdq".')
            if self.calibration_method not in ["nonoverflow", "mse"]:
                raise ValueError('ipu cnn configuration only support calibration_method "nonoverflow" and "mse".')
            if not (self.extra_options.activation_symmetric and self.extra_options.weight_symmetric):
                raise ValueError(
                    "ipu cnn configuration requires setting activation_symmetric and weight_symmetric to true."
                )
            if self.activations_dtype not in ["uint8", "int8"]:
                raise ValueError('ipu cnn configuration only support activations_dtype "uint8" and "int8".')
            if self.weights_dtype not in ["int8"]:
                raise ValueError('ipu cnn configuration only support weights_dtype "int8".')
            if self.per_channel:
                raise ValueError("ipu cnn configuration only supports per tensor.")

    def __setattr__(self, name, value):
        if name == "extra_options" and isinstance(value, dict):
            setattr(self, "extra_options", ExtraOptions(**value))
        else:
            super().__setattr__(name, value)

    def to_diff_dict(self) -> dict:
        """
        Returns a dictionary of non-default values in the configuration.
        """
        non_default_values = {}
        for option in fields(self):
            if option.name == "extra_options":
                extra_options_dict = getattr(self, option.name).to_diff_dict()
                if extra_options_dict:
                    non_default_values[option.name] = extra_options_dict
            else:
                value = getattr(self, option.name)

                if value != option.default and value not in ({}, []):
                    if option.name == "execution_providers" and value == ["CPUExecutionProvider"]:
                        continue

                    if isinstance(value, Enum):
                        value = value.name
                    elif isinstance(value, list):
                        value = [elem.name if isinstance(elem, Enum) else elem for elem in value]

                    non_default_values[option.name] = value
        return non_default_values

    @staticmethod
    def check_dtype_and_format(dtype, dtype_name, format):
        if dtype not in ["uint8", "int8"] and format not in ["vitisqdq"]:
            raise ValueError(f'{dtype_name} is: "{dtype}", format must be "vitisqdq".')

    @staticmethod
    def _map_format(format_str):
        mapping = {
            "qdq": QuantFormat.QDQ,
            "qop": QuantFormat.QOperator,
            "vitisqdq": QuantFormat.VitisQuantFormat_QDQ,
        }
        return QuantizationConfig._map_value(mapping, format_str, "format")

    @staticmethod
    def _map_calibration_method(method_str):
        mapping = {
            "nonoverflow": CalibrationMethod.NonOverflow,
            "mse": CalibrationMethod.MinMSE,
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }
        return QuantizationConfig._map_value(mapping, method_str, "calibration method")

    @staticmethod
    def _map_dtypes(activations_dtype_str, weights_dtype_str):
        mapping = {
            "uint8": QuantType.QUInt8,
            "int8": QuantType.QInt8,
            "uint16": QuantType.QUInt16,
            "int16": QuantType.QInt16,
            "uint32": QuantType.QUInt32,
            "int32": QuantType.QInt32,
            "float16": QuantType.QFloat16,
            "bfloat16": QuantType.QBFloat16,
        }
        activations_dtype = QuantizationConfig._map_value(mapping, activations_dtype_str, "activations dtype")
        weights_dtype = QuantizationConfig._map_value(mapping, weights_dtype_str, "weights dtype")
        return activations_dtype, weights_dtype

    @staticmethod
    def _map_value(mapping, value, name):
        try:
            return mapping[value]
        except KeyError:
            valid_values = ", ".join(f'"{v}"' for v in mapping.keys())
            raise ValueError(f'{name} only supports the following values: {valid_values}. Received "{value}".')

    @staticmethod
    def quantization_type_str(activations_dtype, weights_dtype) -> str:
        str_mapping = {
            QuantType.QUInt8: "u8",
            QuantType.QInt8: "s8",
            QuantType.QUInt16: "u16",
            QuantType.QInt16: "s16",
            QuantType.QUInt32: "u32",
            QuantType.QInt32: "s32",
            QuantType.QFloat16: "f16",
            QuantType.QBFloat16: "bf16",
        }
        activations_str = str_mapping.get(activations_dtype)
        weights_str = str_mapping.get(weights_dtype)
        if activations_str is None or weights_str is None:
            raise ValueError("Unsupported quantization type")
        return f"{activations_str}/{weights_str}"

    @property
    def use_symmetric_calibration(self) -> bool:
        if self.extra_options:
            return self.extra_options.activation_symmetric and self.extra_options.weight_symmetric

        return ExtraOptions().activation_symmetric and ExtraOptions().weight_symmetric

    def __str__(self):
        return (
            f"{self.format} ("
            f"schema: {QuantizationConfig.quantization_type_str(self.activation_type, self.weight_type)}, "
            f"enable_ipu_cnn: {self.enable_ipu_cnn})"
        )


class AutoQuantizationConfig:
    @staticmethod
    def ipu_cnn_config(
        calibrate_method: Literal["nonoverflow", "mse", "minmax", "entropy", "percentile"] = "mse",
        nodes_to_quantize: Optional[List[str]] = None,
        nodes_to_exclude: Optional[List[str]] = None,
        op_types_to_quantize: Optional[List[str]] = None,
        extra_options: Optional[Union[Dict[str, bool], ExtraOptions]] = None,
    ):
        extra_options = extra_options or {}
        if isinstance(extra_options, dict):
            extra_options = ExtraOptions(**extra_options)

        extra_options_dict = extra_options.__dict__
        extra_options_dict["activation_symmetric"] = extra_options_dict.get("activation_symmetric", True)

        return QuantizationConfig(
            format="qdq",
            calibration_method=calibrate_method,
            activations_dtype="uint8",
            weights_dtype="int8",
            enable_ipu_cnn=True,

            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            extra_options=ExtraOptions(**extra_options_dict),
        )

    @staticmethod
    def ipu_transformer_config(
        calibrate_method: Literal["nonoverflow", "mse", "minmax", "entropy", "percentile"] = "minmax",
        nodes_to_quantize: Optional[List[str]] = None,
        nodes_to_exclude: Optional[List[str]] = None,
        op_types_to_quantize: Optional[List[str]] = None,
        extra_options: Optional[Union[Dict[str, bool], ExtraOptions]] = None,
    ):
        extra_options = extra_options or {}
        if isinstance(extra_options, dict):
            extra_options = ExtraOptions(**extra_options)

        extra_options_dict = extra_options.__dict__
        extra_options_dict["activation_symmetric"] = extra_options_dict.get("activation_symmetric", True)

        return QuantizationConfig(
            format="qdq",
            calibration_method=calibrate_method,
            activations_dtype="int8",
            weights_dtype="int8",
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            extra_options=ExtraOptions(**extra_options_dict),
        )

    @staticmethod
    def cpu_cnn_config(
        calibrate_method: Literal["nonoverflow", "mse", "minmax", "entropy", "percentile"] = "minmax",
        nodes_to_quantize: Optional[List[str]] = None,
        nodes_to_exclude: Optional[List[str]] = None,
        op_types_to_quantize: Optional[List[str]] = None,
        extra_options: Optional[Union[Dict[str, bool], ExtraOptions]] = None,
    ):
        extra_options = extra_options or {}
        if isinstance(extra_options, dict):
            extra_options = ExtraOptions(**extra_options)

        return QuantizationConfig(
            format="qdq",
            calibration_method=calibrate_method,
            activations_dtype="uint8",
            weights_dtype="int8",
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            extra_options=extra_options,
        )


class RyzenAIConfig(BaseConfig):
    """
    RyzenAIConfig is the configuration class handling all the VitisAI parameters related to the ONNX IR model export,
     and quantization parameters.

    Attributes:
        opset (`Optional[int]`, defaults to `None`):
            ONNX opset version to export the model with.
        quantization (`Optional[QuantizationConfig]`, defaults to `None`):
            Specify a configuration to quantize ONNX model
    """

    CONFIG_NAME = "ryzenai_config.json"
    FULL_CONFIGURATION_FILE = "ryzenai_config.json"

    def __init__(
        self,
        opset: Optional[int] = None,
        quantization: Optional[QuantizationConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.opset = opset
        self.quantization = quantization.to_diff_dict() if quantization is not None else None
        self.optimum_version = kwargs.pop("optimum_version", None)

    @staticmethod
    def dataclass_to_dict(config) -> dict:
        new_config = {}
        if config is None:
            return new_config
        if isinstance(config, dict):
            return config
        for k, v in asdict(config).items():
            if isinstance(v, Enum):
                v = v.name
            elif isinstance(v, list):
                v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
            new_config[k] = v
        return new_config
