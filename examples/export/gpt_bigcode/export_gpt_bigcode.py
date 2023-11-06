# from optimum.exporters.onnx.convert import export_pytorch
import contextlib
import gc
import os
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import onnx
import torch
from accelerate.utils import find_tied_parameters

from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.config import TextDecoderOnnxConfig, ConfigBehavior, TextDecoderWithPositionIdsOnnxConfig
from optimum.onnx import remove_duplicate_weights_from_tied_info
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    DummyInputGenerator,
    DummyTextInputGenerator,
    DummyPastKeyValuesGenerator,
    NormalizedConfigManager,
    is_diffusers_available,
    logging,
)
from transformers import AutoConfig
from transformers.utils import is_tf_available
from modeling_gpt_bigcode import GPTBigCodeForCausalLM


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    if is_tf_available():
        from transformers import TFPreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin


logger = logging.get_logger(__name__)

MAX_SEQUENCE_LENGTH = 512  # TODO: fix to max_length for gpt bigcode


class CustomDummyTextInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids", "position_ids")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        min_value = 1
        max_value = 2 if input_name != "input_ids" else self.vocab_size

        if input_name == "attention_mask":
            shape = [self.batch_size, MAX_SEQUENCE_LENGTH]
        else:
            shape = [self.batch_size, self.sequence_length]

        if self.task == "multiple-choice":
            shape = [self.batch_size, self.num_choices, self.sequence_length]

        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class GPTBigCodeDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_value_shape = (
            self.batch_size,
            MAX_SEQUENCE_LENGTH,
            self.hidden_size // self.num_attention_heads * 2,
        )
        return [
            self.random_float_tensor(past_key_value_shape, framework=framework, dtype=float_dtype)
            for _ in range(self.num_layers)
        ]


class GPTBigCodeOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        CustomDummyTextInputGenerator,
        GPTBigCodeDummyPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = GPTBigCodeDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("gpt_bigcode")

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            # No dim for `n_head` when using multi-query attention
            inputs_or_outputs[f"{name}.{i}.key_value"] = {
                0: "batch_size",
                1: decoder_sequence_name,
            }

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key_value"] = t

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs.keys() if not key.startswith("past_key_values")]

        if self.use_past_in_inputs:
            input_names.append("past_key_values")

        for input_name in input_names:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = self.overwrite_shape_and_generate_input(
                        dummy_input_gen,
                        input_name,
                        framework,
                        input_shapes=kwargs,
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                )

        return dummy_inputs


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
    dtype: Optional["torch.dtype"] = None,
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to an ONNX Intermediate Representation.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Path to save the exported ONNX file to.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        dtype (`Optional[torch.dtype]`, defaults to `None`):
            Data type to remap the model inputs to. PyTorch-only. Only `torch.float16` is supported.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_config` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named outputs from
        the ONNX configuration.
    """
    from torch.onnx import export as onnx_export
    from torch.utils._pytree import tree_map

    logger.info(f"Using framework PyTorch: {torch.__version__}")
    FORCE_ONNX_EXTERNAL_DATA = os.getenv("FORCE_ONNX_EXTERNAL_DATA", "0") == "1"

    with torch.no_grad():
        model.config.return_dict = True
        model.eval()

        if input_shapes is None:
            input_shapes = {}  # will use the defaults from DEFAULT_DUMMY_SHAPES

        # Check that inputs match, and order them properly
        dummy_inputs = config.generate_dummy_inputs(framework="pt", **input_shapes)

        device = torch.device(device)

        def remap(value):
            if isinstance(value, torch.Tensor):
                value = value.to(device)

            return value

        if device.type == "cuda" and torch.cuda.is_available():
            model.to(device)
            dummy_inputs = tree_map(remap, dummy_inputs)

        dummy_inputs = config.rename_ambiguous_inputs(dummy_inputs)

        with config.patch_model_for_export(model, model_kwargs=model_kwargs):
            inputs = config.ordered_inputs(model)
            input_names = list(inputs.keys())
            output_names = list(config.outputs.keys())

            # Export can work with named args but the dict containing named args has to be the last element of the args
            # tuple.
            onnx_export(
                model,
                (dummy_inputs,),
                f=output.as_posix(),
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=True,
                opset_version=opset,
            )

        # check if external data was exported
        # TODO: this is quite inefficient as we load in memory if models are <2GB without external data
        onnx_model = onnx.load(str(output), load_external_data=False)
        model_uses_external_data = check_model_uses_external_data(onnx_model)

        if model_uses_external_data or FORCE_ONNX_EXTERNAL_DATA:
            tensors_paths = _get_onnx_external_data_tensors(onnx_model)
            logger.info("Saving external data to one file...")

            # try free model memory
            del model
            del onnx_model
            gc.collect()
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            onnx_model = onnx.load(
                str(output), load_external_data=True
            )  # this will probably be too memory heavy for large models
            onnx.save(
                onnx_model,
                str(output),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=output.name + "_data",
                size_threshold=1024 if not FORCE_ONNX_EXTERNAL_DATA else 0,
            )

            # delete previous external data
            for tensor in tensors_paths:
                os.remove(output.parent / tensor)

    return input_names, output_names


# model_id = "bigcode/gpt_bigcode-santacoder"
model_id = "bigcode/starcoderbase-1b"
task = "text-generation"

cfg = AutoConfig.from_pretrained(model_id)

decoder_onnx_config = GPTBigCodeOnnxConfig(cfg, task, use_past=True, use_past_in_inputs=False)
decoder_onnx_config_with_past = GPTBigCodeOnnxConfig(cfg, task, use_past=True, use_past_in_inputs=True)

custom_onnx_configs = {"decoder_model": decoder_onnx_config, "decoder_model": decoder_onnx_config}


model = GPTBigCodeForCausalLM.from_pretrained(model_id)

model = model.eval()


# save_dir = "gpt_bigcode_onnx"
save_dir = "starcoder_onnx"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

input_shapes = {"batch_size": 1, "sequence_length": MAX_SEQUENCE_LENGTH}

with torch.no_grad():
    export_pytorch(
        model,
        decoder_onnx_config,
        opset=13,
        output=Path(save_dir + "/decoder_model.onnx"),
        input_shapes=input_shapes,
    )

model.config.use_cache = True

input_shapes = {"batch_size": 1, "sequence_length": 1}

with torch.no_grad():
    export_pytorch(
        model,
        decoder_onnx_config_with_past,
        opset=13,
        output=Path(save_dir + "/decoder_model_with_past.onnx"),
        input_shapes=input_shapes,
    )

# Remove duplicate weights from the PyTorch ONNX export.
onnx_model = onnx.load(save_dir + "/decoder_model.onnx")
tied_params = find_tied_parameters(model)
_ = remove_duplicate_weights_from_tied_info(onnx_model, model, tied_params, save_path=save_dir + "/decoder_model.onnx")

onnx_model = onnx.load(save_dir + "/decoder_model_with_past.onnx")
tied_params = find_tied_parameters(model)
_ = remove_duplicate_weights_from_tied_info(
    onnx_model, model, tied_params, save_path=save_dir + "/decoder_model_with_past.onnx"
)
