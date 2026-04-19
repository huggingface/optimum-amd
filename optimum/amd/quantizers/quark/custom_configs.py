#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


from quark.torch.quantization.config.config import (
    QuantizationConfig,
    QuantizationSpec,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerBlockMXObserver,
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
    PlaceholderObserver,
)


# Configure `QuantizationSpec` for torch.Tensors. Specify attributes such as dtype, observer_cls, etc.
FLOAT16_SPEC = QuantizationSpec(dtype=Dtype.float16, observer_cls=PlaceholderObserver)

BFLOAT16_SPEC = QuantizationSpec(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver)

FP8_PER_TENSOR_SPEC = QuantizationSpec(
    dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
)

INT4_PER_TENSER_SPEC = QuantizationSpec(
    dtype=Dtype.int4,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

INT4_PER_CHANNEL_SPEC = QuantizationSpec(
    dtype=Dtype.int4,
    observer_cls=PerChannelMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_channel,
    ch_axis=0,
    is_dynamic=False,
)

INT4_PER_GROUP_SYM_SPEC = QuantizationSpec(
    dtype=Dtype.int4,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group,
    ch_axis=1,
    is_dynamic=False,
    group_size=128,
)

UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(
    dtype=Dtype.uint4,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=False,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group,
    ch_axis=1,
    is_dynamic=False,
    group_size=128,
)

INT8_PER_TENSER_SPEC = QuantizationSpec(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

INT8_PER_TENSER_DYNAMIC_SPEC = QuantizationSpec(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=True,
)

INT8_PER_GROUP_SYM_SPEC = QuantizationSpec(
    dtype=Dtype.int8,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group,
    ch_axis=1,
    is_dynamic=False,
    group_size=128,
)

W_MX_FP8_SPEC = QuantizationSpec(
    dtype=Dtype.mx,
    observer_cls=PerBlockMXObserver,
    qscheme=QSchemeType.per_group,
    mx_element_dtype=Dtype.fp8_e4m3,
    ch_axis=-1,
    group_size=32,
    is_dynamic=False,
    round_method=RoundType.half_even,
)
A_MX_FP8_SPEC = QuantizationSpec(
    dtype=Dtype.mx,
    observer_cls=PerBlockMXObserver,
    qscheme=QSchemeType.per_group,
    mx_element_dtype=Dtype.fp8_e4m3,
    ch_axis=-1,
    group_size=32,
    is_dynamic=True,
    round_method=RoundType.half_even,
)


# Establish `QuantizationConfig` for nn.Module. Define the QuantizationSpec of input_tensors, output_tensors, weight, and bias.
# Float16 config
FLOAT16_CONFIG = QuantizationConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)

# Fp8(e4m3) config
W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)

W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QuantizationConfig(
    input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
)

# Int per tensor config
W_INT4_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT4_PER_TENSER_SPEC)

W_INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT8_PER_TENSER_SPEC)

W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSER_SPEC, weight=INT8_PER_TENSER_SPEC)

W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSER_DYNAMIC_SPEC, weight=INT8_PER_TENSER_DYNAMIC_SPEC
)

# Int per Channel Config
W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)

# Int per Group Config
W_INT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=INT4_PER_GROUP_SYM_SPEC)

W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)

W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG = QuantizationConfig(input_tensors=BFLOAT16_SPEC, weight=UINT4_PER_GROUP_ASYM_SPEC)

W_INT8_PER_GROUP_CONFIG = QuantizationConfig(weight=INT8_PER_GROUP_SYM_SPEC)

W_MX_FP8_CONFIG = QuantizationConfig(weight=W_MX_FP8_SPEC)
W_MX_FP8_A_MX_FP8_CONFIG = QuantizationConfig(weight=W_MX_FP8_SPEC, input_tensors=A_MX_FP8_SPEC)
