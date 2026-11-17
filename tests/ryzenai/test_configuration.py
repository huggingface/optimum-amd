import unittest

import vai_q_onnx
from parameterized import parameterized

from optimum.amd.ryzenai import AutoQuantizationConfig, ExtraOptions, QuantizationConfig


class TestExtraOptions(unittest.TestCase):
    def test_default_values(self):
        options = ExtraOptions()
        self.assertEqual(options.activation_symmetric, False)
        self.assertEqual(options.weight_symmetric, True)
        self.assertEqual(options.use_unsigned_relu, False)
        self.assertEqual(options.quantize_bias, True)
        self.assertEqual(options.remove_input_init, True)
        self.assertEqual(options.enable_subgraph, False)
        self.assertEqual(options.force_quantize_no_input_check, False)
        self.assertEqual(options.matmul_const_b_only, False)
        self.assertEqual(options.add_qdq_pair_to_weight, False)
        self.assertEqual(options.op_types_to_exclude_output_quantization, [])
        self.assertEqual(options.dedicated_qdq_pair, False)
        self.assertEqual(options.qdq_op_type_per_channel_support_to_axis, {})
        self.assertEqual(options.use_qdq_vitis_custom_ops, True)
        self.assertEqual(options.calib_tensor_range_symmetric, False)
        self.assertEqual(options.calib_moving_average, False)
        self.assertEqual(options.calib_moving_average_constant, 0.01)
        self.assertEqual(options.random_data_reader_input_data_range, None)
        self.assertEqual(options.int16_scale, False)
        self.assertEqual(options.min_mse_mode, "All")
        self.assertEqual(options.convert_bn_to_conv, True)
        self.assertEqual(options.convert_reduce_mean_to_global_avg_pool, True)
        self.assertEqual(options.split_large_kernel_pool, True)
        self.assertEqual(options.convert_split_to_slice, True)
        self.assertEqual(options.fuse_instance_norm, False)
        self.assertEqual(options.fuse_l2_norm, False)
        self.assertEqual(options.convert_clip_to_relu, False)
        self.assertEqual(options.simulate_dpu, True)
        self.assertEqual(options.convert_leaky_relu_to_dpu_version, True)
        self.assertEqual(options.convert_sigmoid_to_hard_sigmoid, True)
        self.assertEqual(options.convert_hard_sigmoid_to_dpu_version, True)
        self.assertEqual(options.convert_avg_pool_to_dpu_version, True)
        self.assertEqual(options.convert_reduce_mean_to_dpu_version, True)
        self.assertEqual(options.convert_softmax_to_dpu_version, False)
        self.assertEqual(options.ipu_limitation_check, True)
        self.assertEqual(options.adjust_shift_cut, True)
        self.assertEqual(options.adjust_shift_bias, True)
        self.assertEqual(options.adjust_shift_read, True)
        self.assertEqual(options.adjust_shift_write, True)
        self.assertEqual(options.adjust_hard_sigmoid, True)
        self.assertEqual(options.adjust_shift_swish, True)
        self.assertEqual(options.align_concat, True)
        self.assertEqual(options.align_pool, True)
        self.assertEqual(options.replace_clip6_relu, False)
        self.assertEqual(options.cle_steps, 1)
        self.assertEqual(options.cle_total_layer_diff_threshold, 2e-7)
        self.assertEqual(options.cle_scale_append_bias, True)
        self.assertEqual(options.remove_qdq_conv_leaky_relu, False)
        self.assertEqual(options.remove_qdq_conv_prelu, False)

    def test_snake_to_camel(self):
        options = ExtraOptions()
        camel_case_dict = options.snake_to_camel
        self.assertEqual(
            camel_case_dict["qdq_op_type_per_channel_support_to_axis"], "QDQOpTypePerChannelSupportToAxis"
        )
        self.assertEqual(camel_case_dict["ipu_limitation_check"], "IPULimitationCheck")
        self.assertEqual(camel_case_dict["cle_steps"], "CLESteps")
        self.assertEqual(camel_case_dict["cle_total_layer_diff_threshold"], "CLETotalLayerDiffThreshold")
        self.assertEqual(camel_case_dict["cle_scale_append_bias"], "CLEScaleAppendBias")

    def test_camel_to_snake_setattr(self):
        options = ExtraOptions()
        options.QDQOpTypePerChannelSupportToAxis = "some_value"
        options.IPULimitationCheck = False
        options.CLESteps = 5
        options.CLETotalLayerDiffThreshold = 1e-7
        options.CLEScaleAppendBias = False
        self.assertEqual(options.qdq_op_type_per_channel_support_to_axis, "some_value")
        self.assertFalse(options.ipu_limitation_check)
        self.assertEqual(options.cle_steps, 5)
        self.assertEqual(options.cle_total_layer_diff_threshold, 1e-7)
        self.assertFalse(options.cle_scale_append_bias)

    def test_to_diff_dict(self):
        options = ExtraOptions(activation_symmetric=True)
        diff_dict = options.to_diff_dict()
        self.assertEqual(diff_dict, {"activation_symmetric": True})

    @parameterized.expand(
        [
            ("activation_symmetric", False, True),
            ("weight_symmetric", True, False),
            ("use_unsigned_relu", False, True),
        ]
    )
    def test_parametric_setting_attributes(self, attribute, default_value, new_value):
        options = ExtraOptions()
        self.assertEqual(getattr(options, attribute), default_value)
        setattr(options, attribute, new_value)
        self.assertEqual(getattr(options, attribute), new_value)


class TestQuantizationConfig(unittest.TestCase):
    def test_default_values(self):
        config = QuantizationConfig()
        self.assertEqual(config.format, vai_q_onnx.QuantFormat.QDQ)
        self.assertEqual(config.calibration_method, vai_q_onnx.PowerOfTwoMethod.MinMSE)
        self.assertEqual(config.input_nodes, [])
        self.assertEqual(config.output_nodes, [])
        self.assertEqual(config.op_types_to_quantize, [])
        self.assertEqual(config.random_data_reader_input_shape, [])
        self.assertFalse(config.per_channel)
        self.assertFalse(config.reduce_range)
        self.assertEqual(config.activations_dtype, vai_q_onnx.QuantType.QUInt8)
        self.assertEqual(config.weights_dtype, vai_q_onnx.QuantType.QInt8)
        self.assertEqual(config.nodes_to_quantize, [])
        self.assertEqual(config.nodes_to_exclude, [])
        self.assertTrue(config.optimize_model)
        self.assertFalse(config.use_external_data_format)
        self.assertEqual(config.execution_providers, ["CPUExecutionProvider"])
        self.assertFalse(config.enable_ipu_cnn)
        self.assertFalse(config.convert_fp16_to_fp32)
        self.assertFalse(config.convert_nchw_to_nhwc)
        self.assertFalse(config.include_cle)
        self.assertIsInstance(config.extra_options, ExtraOptions)

    def test_extra_options_initialization(self):
        extra_options = ExtraOptions(activation_symmetric=True)
        config = QuantizationConfig(extra_options=extra_options)
        self.assertEqual(config.extra_options.activation_symmetric, True)

    def test_use_symmetric_calibration(self):
        config = QuantizationConfig(extra_options=ExtraOptions(activation_symmetric=True, weight_symmetric=True))
        self.assertTrue(config.use_symmetric_calibration)

    @parameterized.expand(
        [
            ("format", vai_q_onnx.QuantFormat.QDQ, vai_q_onnx.QuantFormat.QOperator),
            ("calibration_method", vai_q_onnx.PowerOfTwoMethod.MinMSE, vai_q_onnx.CalibrationMethod.Entropy),
            ("activations_dtype", vai_q_onnx.QuantType.QUInt8, vai_q_onnx.QuantType.QInt8),
            ("weights_dtype", vai_q_onnx.QuantType.QInt8, vai_q_onnx.QuantType.QUInt8),
        ]
    )
    def test_parametric_setting_attributes(self, attribute, default_value, new_value):
        config = QuantizationConfig()
        self.assertEqual(getattr(config, attribute), default_value)
        setattr(config, attribute, new_value)
        self.assertEqual(getattr(config, attribute), new_value)


class TestAutoQuantizationConfig(unittest.TestCase):
    def test_ipu_cnn_config(self):
        config = AutoQuantizationConfig.ipu_cnn_config()
        self.assertEqual(config.format, vai_q_onnx.QuantFormat.QDQ)
        self.assertEqual(config.calibration_method, vai_q_onnx.PowerOfTwoMethod.MinMSE)
        self.assertTrue(config.extra_options.activation_symmetric)

    def test_ipu_transformer_config(self):
        config = AutoQuantizationConfig.ipu_transformer_config()
        self.assertEqual(config.format, vai_q_onnx.QuantFormat.QDQ)
        self.assertEqual(config.calibration_method, vai_q_onnx.CalibrationMethod.MinMax)
        self.assertTrue(config.extra_options.activation_symmetric)

    def test_cpu_cnn_config(self):
        config = AutoQuantizationConfig.cpu_cnn_config()

        self.assertEqual(config.format, vai_q_onnx.QuantFormat.QDQ)
        self.assertEqual(config.calibration_method, vai_q_onnx.CalibrationMethod.MinMax)
