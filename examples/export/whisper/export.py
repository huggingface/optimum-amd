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
from modeling_whisper import WhisperForConditionalGeneration

from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.config import AudioToTextOnnxConfig, ConfigBehavior
from optimum.onnx import remove_duplicate_weights_from_tied_info
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    DummyAudioInputGenerator,
    DummyInputGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummyTextInputGenerator,
    NormalizedSeq2SeqConfig,
    is_diffusers_available,
    logging,
)
from transformers import AutoConfig
from transformers.utils import is_tf_available

from symbolic_shape_infer import SymbolicShapeInference


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    if is_tf_available():
        from transformers import TFPreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin


logger = logging.get_logger(__name__)


class CustomDummyTextInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids", "position_ids", "decoder_attention_mask")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        min_value = 0
        max_value = 2 if input_name != "input_ids" else self.vocab_size

        print("dec att", input_name == "decoder_attention_mask")
        if input_name == "decoder_attention_mask":
            shape = [self.batch_size, 448]  # TODO: fix to max_length for whisper
        else:
            shape = [self.batch_size, self.sequence_length]

        if self.task == "multiple-choice":
            shape = [self.batch_size, self.num_choices, self.sequence_length]

        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class CustomDummySeq2SeqDecoderTextInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        print("input_name", input_name)
        if input_name in ["encoder_outputs", "encoder_hidden_states"]:
            return self.random_float_tensor(
                shape=[self.batch_size, 1500, self.hidden_size],
                min_value=0,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )

        return super().generate(input_name, framework=framework, int_dtype=int_dtype)


class CustomDummySeq2SeqPastKeyValuesGenerator(DummyInputGenerator):
    """
    Generates dummy past_key_values inputs for seq2seq architectures.
    """

    SUPPORTED_INPUT_NAMES = ("past_key_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedSeq2SeqConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        encoder_sequence_length: Optional[int] = None,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.normalized_config = normalized_config

        self.batch_size = batch_size
        self.encoder_sequence_length = 1500
        self.sequence_length = 448  # TODO: fix to max_length for whisper

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        encoder_shape = (
            self.batch_size,
            self.normalized_config.encoder_num_attention_heads,
            self.encoder_sequence_length,
            self.normalized_config.hidden_size // self.normalized_config.encoder_num_attention_heads,
        )
        decoder_shape = (
            self.batch_size,
            self.normalized_config.decoder_num_attention_heads,
            self.sequence_length,
            self.normalized_config.hidden_size // self.normalized_config.decoder_num_attention_heads,
        )
        return tuple(
            (
                self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.normalized_config.decoder_num_layers)
        )


class CustomWhisperOnnxConfig(AudioToTextOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyAudioInputGenerator,
        CustomDummyTextInputGenerator,
        CustomDummySeq2SeqDecoderTextInputGenerator,
        CustomDummySeq2SeqPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = CustomDummySeq2SeqPastKeyValuesGenerator

    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig
    ATOL_FOR_VALIDATION = 1e-3

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        behavior: ConfigBehavior = ConfigBehavior.MONOLITH,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            behavior=behavior,
            preprocessors=preprocessors,
        )
        if self._behavior is ConfigBehavior.ENCODER:
            self.use_past_in_inputs = False
        else:
            self.use_past_in_inputs = True

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}

        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_features"] = {}

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["decoder_input_ids"] = {}
            common_inputs["decoder_attention_mask"] = {}
            common_inputs["encoder_outputs"] = {}

            self.add_past_key_values(common_inputs, direction="inputs")
            common_inputs["position_ids"] = {}

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        # In the other cases, the sequence_length axis is not dynamic, always of length 1
        if self.task == "feature-extraction":
            common_outputs = OrderedDict({"last_hidden_state": {}})
        else:
            if self._behavior is ConfigBehavior.DECODER:
                common_outputs = OrderedDict({"logits": {1: "1"}})  # NOTE: for some reason dynamic shape detected
            else:
                common_outputs = OrderedDict({"logits": {}})
            # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
            self.add_past_key_values(common_outputs, direction="outputs")

        return common_outputs

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            name = "past_key_values"
        else:
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {}

            if self._behavior is ConfigBehavior.DECODER:
                # TODO: we only need to call it encoder_sequence_length_out in the merge case - but at torch.onnx.export()
                # time we have currently no case to check whether we will merge at a later step or not (self.is_merged is
                # not yet set at this time)
                inputs_or_outputs[f"{name}.{i}.encoder.key"] = {}
                inputs_or_outputs[f"{name}.{i}.encoder.value"] = {}

        return inputs_or_outputs

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        return contextlib.nullcontext()


cfg = AutoConfig.from_pretrained("openai/whisper-tiny.en")
task = "automatic-speech-recognition"

encoder_onnx_config = CustomWhisperOnnxConfig(cfg, task, behavior=ConfigBehavior.ENCODER)

decoder_onnx_config = CustomWhisperOnnxConfig(cfg, task, behavior=ConfigBehavior.DECODER, use_past=True)

custom_onnx_configs = {"encoder_model": encoder_onnx_config, "decoder_model": decoder_onnx_config}

input_shapes = {"batch_size": 1, "sequence_length": 1}

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

model = model.eval()
with torch.no_grad():
    for i in range(len(model.model.decoder.layers)):
        model.model.decoder.layers[i].encoder_attn = torch.jit.script(model.model.decoder.layers[i].encoder_attn)


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
    dtype: Optional["torch.dtype"] = None,
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    dummy_inputs: Optional[Dict] = None,
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
        # model.config.return_dict = True
        model.eval()

        # Check if we need to override certain configuration item
        # if config.values_override is not None:
        #     logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        #     for override_config_key, override_config_value in config.values_override.items():
        #         logger.info(f"\t- {override_config_key} -> {override_config_value}")
        #         setattr(model.config, override_config_key, override_config_value)

        if input_shapes is None:
            input_shapes = {}  # will use the defaults from DEFAULT_DUMMY_SHAPES

        device = torch.device(device)

        if dummy_inputs is None:
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
            dummy_inputs = (dummy_inputs,)

        with config.patch_model_for_export(model, model_kwargs=model_kwargs):
            # inputs = config.ordered_inputs(model)
            inputs = config.inputs
            print("inputs keys", inputs.keys())
            input_names = list(inputs.keys())
            output_names = list(config.outputs.keys())

            # Export can work with named args but the dict containing named args has to be the last element of the args
            # tuple.
            onnx_export(
                model,
                dummy_inputs,
                f=output.as_posix(),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dict(chain(inputs.items(), config.outputs.items())),
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


if not os.path.exists("whisper_onnx"):
    os.makedirs("whisper_onnx")

export_pytorch(
    model.get_encoder(),
    encoder_onnx_config,
    opset=13,
    output=Path("whisper_onnx/encoder_model.onnx"),
    input_shapes=input_shapes,
)

dummy_inputs = decoder_onnx_config.generate_dummy_inputs(framework="pt", **input_shapes)

dummy_inputs = (
    dummy_inputs["decoder_input_ids"],
    dummy_inputs["decoder_attention_mask"],
    dummy_inputs["encoder_outputs"],
    dummy_inputs["past_key_values"],
    dummy_inputs["position_ids"],
)

model.config.use_cache = True
model.config.return_dict = False

with torch.no_grad():
    traced_model = torch.jit.trace(model, dummy_inputs)

    export_pytorch(
        traced_model,
        decoder_onnx_config,
        opset=13,
        output=Path("whisper_onnx/decoder_model.onnx"),
        input_shapes=input_shapes,
        dummy_inputs=dummy_inputs,
    )

# Remove duplicate weights from the PyTorch ONNX export.
onnx_model = onnx.load("whisper_onnx/encoder_model.onnx")
tied_params = find_tied_parameters(model)
_ = remove_duplicate_weights_from_tied_info(
    onnx_model, model, tied_params, save_path="whisper_onnx/encoder_model.onnx"
)

onnx_model = onnx.load("whisper_onnx/decoder_model.onnx")
tied_params = find_tied_parameters(model)
_ = remove_duplicate_weights_from_tied_info(
    onnx_model, model, tied_params, save_path="whisper_onnx/decoder_model.onnx"
)

# Apply custom symbolic shape inference.
int_max = 2**31 - 1
guess_output_rank = False
auto_merge = False
verbose = 0
save_as_external_data = False
all_tensors_to_one_file = False
external_data_location = "./"
external_data_size_threshold = 1024

print("Doing symbolic shape inference...")
output = "whisper_onnx/encoder_model.onnx"
out_mp = SymbolicShapeInference.infer_shapes(
    onnx.load(output),
    int_max,
    auto_merge,
    guess_output_rank,
    verbose,
)
if out_mp:
    if save_as_external_data:
        onnx.save_model(
            out_mp,
            output,
            save_as_external_data=True,
            all_tensors_to_one_file=all_tensors_to_one_file,
            location=external_data_location,
            size_threshold=external_data_size_threshold,
            convert_attribute=False,
        )
    else:
        onnx.save(out_mp, output)

output = "whisper_onnx/decoder_model.onnx"
out_mp = SymbolicShapeInference.infer_shapes(
    onnx.load(output),
    int_max,
    auto_merge,
    guess_output_rank,
    verbose,
)
if out_mp:
    if save_as_external_data:
        onnx.save_model(
            out_mp,
            output,
            save_as_external_data=True,
            all_tensors_to_one_file=all_tensors_to_one_file,
            location=external_data_location,
            size_threshold=external_data_size_threshold,
            convert_attribute=False,
        )
    else:
        onnx.save(out_mp, output)
