# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""
RyzenAIModelForXXX classes related to seq2seq, allowing to run ONNX Models with ONNX Runtime using the same API as
Transformers.
"""

import logging
import shutil
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download

from transformers import (
    AutoModelForSpeechSeq2Seq,
    GenerationConfig,
    WhisperForConditionalGeneration,
)
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES

from ..onnx.utils import _get_external_data_paths
from ..utils import check_if_transformers_greater
from ..utils.file_utils import validate_file_exists
from ..utils.save_utils import maybe_load_preprocessors
from .base import RyzenAIDecoderForSeq2Seq, RyzenAIEncoder
from .constants import (
    DECODER_ONNX_FILE_PATTERN,
    ENCODER_ONNX_FILE_PATTERN,
)
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, RyzenAIModel
from .utils import (
    ONNX_DECODER_NAME,
    ONNX_ENCODER_NAME,
)


if check_if_transformers_greater("4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin

from huggingface_hub.utils import EntryNotFoundError


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)

SPEECH_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor`):
            Mel / fbank features extracted from the raw speech waveform. `(batch_size, feature_size, encoder_sequence_length)`.
"""

SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor`):
            Mel features extracted from the raw speech waveform.
            `(batch_size, feature_size, encoder_sequence_length)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

_PROCESSOR_FOR_DOC = "AutoProcessor"

AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor.feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> gen_tokens = model.generate(inputs=inputs.input_features)
    >>> outputs = processor.tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> speech_recognition = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> pred = speech_recognition(ds[0]["audio"]["array"])
    ```
"""


class RyzenAIEncoderForSpeech(RyzenAIEncoder):
    """
    Encoder model for ONNX Runtime inference for speech model.

    Args:
        session (`ort.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    @add_start_docstrings_to_model_forward(SPEECH_ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:
        use_torch = isinstance(input_features, torch.Tensor)
        self.parent_model.raise_on_numpy_input_io_binding(use_torch)

        if use_torch:
            onnx_inputs = {"input_features": input_features.cpu().detach().numpy()}
            if "attention_mask" in self.input_names:
                onnx_inputs["attention_mask"] = attention_mask.cpu().detach().numpy()
        else:
            onnx_inputs = {"input_features": input_features}
            if "attention_mask" in self.input_names:
                onnx_inputs["attention_mask"] = attention_mask

        outputs = self.session.run(None, onnx_inputs)

        last_hidden_state = outputs[self.output_names["last_hidden_state"]]
        if use_torch:
            last_hidden_state = torch.from_numpy(last_hidden_state).to(self.device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class RyzenAIModelForConditionalGeneration(RyzenAIModel, ABC):
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.

    Important attributes:
        config ([`PretrainedConfig`]):
            Instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
        providers (`List[str`]):
            The list of execution providers the model is running on.
        encoder (`RyzenAIEncoder`):
            The encoder model.
        decoder (`RyzenAIDecoderForSeq2Seq`):
            The decoder model.

    Other attributes:
        encoder_file_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_ENCODER_NAME`):
            The name of the ONNX file containing the encoder part of the model.
        decoder_file_name (`str`,  defaults to `optimum.onnxruntime.utils.ONNX_DECODER_NAME`):
            The name of the ONNX file containing the decoder part of the model.
        model_save_dir (`str`, defaults to `""`):
            The directory under which the model exported to ONNX was saved.

    """

    # Used in from_transformers to export model to onnxRyzenAIEncoder
    base_model_prefix = "onnx_model"

    def __init__(
        self,
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        config: "PretrainedConfig",
        onnx_paths: List[str],
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        """
        Args:
            encoder_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the encoder.
            decoder_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the decoder.
            config ([`PretrainedConfig`]):
                `config` is an instance of the configuration associated to the model. Initializing with a config file
                does not load the weights associated with the model, only the configuration.
            onnx_paths (`List[str]`):
                Path to ONNX files associated with the model.
            model_save_dir (`str`, *optional*, defaults to `""`):
                The directory under which the model exported to ONNX was saved.
            preprocessors (`Optional[List]`, defaults to `None`):
                The list of the preprocessors (tokenizer, processor, feature_extractor) to save alongside the RyzenAIModel.
            generation_config (`Optional[GenerationConfig]`, defaults to `None`):
                The generation configuration used by default when calling `generate()`.
                Refer to https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate.
        """
        ABC.__init__(self)

        self.shared_attributes_init(
            encoder_session,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
        )
        self.config = config
        self.name_or_path = config.name_or_path

        self.onnx_paths = onnx_paths

        self.encoder = self._initialize_encoder(encoder_session)
        self.encoder_model_path = Path(encoder_session._model_path)
        self.encoder_model_name = self.encoder_model_path.name

        self.decoder = RyzenAIDecoderForSeq2Seq(decoder_session, self)
        self.decoder_model_path = Path(decoder_session._model_path)
        self.decoder_model_name = self.decoder_model_path.name

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config

    @abstractmethod
    def _initialize_encoder(self, session: ort.InferenceSession) -> RyzenAIEncoder:
        pass

    @staticmethod
    def load_model(
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict] = None,
    ):
        """
        Creates an instance of [`~optimum.onnxruntime.modeling_seq2seq.RyzenAIModelForConditionalGeneration`].
        Three inference sessions will be created for respectively the encoder, decoder and decoder with past key values
        models. The default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            encoder_path (`Union[str, Path]`):
                The path of the encoder ONNX model.
            decoder_path (`Union[str, Path]`):
                The path of the decoder ONNX model.
            provider (`str`, *optional*, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[ort.SessionOptions]`, *optional*),:
                ONNX Runtime session options to use for loading the model. Defaults to `None`.
            provider_options (`Optional[Dict]`, *optional*):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html . Defaults to `None`.
        """
        encoder_session = RyzenAIModel.load_model(encoder_path, provider, session_options, provider_options)
        decoder_session = RyzenAIModel.load_model(decoder_path, provider, session_options, provider_options)

        return encoder_session, decoder_session

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model encoder, decoder and decoder with past key values as well as its configuration file to a
        directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_seq2seq.RyzenAIModelForSeq2SeqLM.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path`]):
                The directory where to save the model files.
        """
        save_directory = Path(save_directory)
        src_paths = [Path(path) for path in self.onnx_paths]
        dst_paths = [save_directory / path.name for path in src_paths]

        # add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copyfile(src_path, dst_path)

        self.generation_config.save_pretrained(save_directory)

    @classmethod
    def _retrieve_model_path(
        cls,
        model_id: Union[str, Path],
        file_name: Union[str, Path],
        argument_name: str,
        file_name_pattern: List[str],
        regular_file_name: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        subfolder: str = "",
    ):
        model_path = Path(model_id)

        if not validate_file_exists(model_id, file_name, subfolder=subfolder, revision=revision):
            path = RyzenAIModelForConditionalGeneration.infer_onnx_filename(
                model_id,
                file_name_pattern,
                argument_name,
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
            )
        else:
            path = model_path / subfolder / file_name

        decoder_regular_onnx_filenames = RyzenAIModelForConditionalGeneration._generate_regular_names_for_filename(
            regular_file_name
        )
        if path.name not in decoder_regular_onnx_filenames:
            logger.warning(
                f"The ONNX file {path.name} is not a regular name used in optimum.ryzenai that are {decoder_regular_onnx_filenames}, the "
                f"{cls.__name__} might not behave as expected."
            )

        return path

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: str = ONNX_ENCODER_NAME,
        decoder_file_name: str = ONNX_DECODER_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        model_path = Path(model_id)

        decoder_path = RyzenAIModelForConditionalGeneration._retrieve_model_path(
            model_id,
            decoder_file_name,
            "decoder_file_name",
            [DECODER_ONNX_FILE_PATTERN],
            ONNX_DECODER_NAME,
            subfolder=subfolder,
            use_auth_token=use_auth_token,
            revision=revision,
        )

        encoder_path = RyzenAIModelForConditionalGeneration._retrieve_model_path(
            model_id,
            encoder_file_name,
            encoder_file_name,
            "encoder_file_name",
            [ENCODER_ONNX_FILE_PATTERN],
            ONNX_ENCODER_NAME,
            subfolder=subfolder,
            use_auth_token=use_auth_token,
            revision=revision,
        )

        preprocessors = None
        if model_path.is_dir():
            new_model_save_dir = model_path
            preprocessors = maybe_load_preprocessors(model_id)
        else:
            attribute_name_to_filename = {
                "last_encoder_model_name": encoder_path.name,
                "last_decoder_model_name": decoder_path.name,
            }
            paths = {}
            for attr_name, filename in attribute_name_to_filename.items():
                if filename is None:
                    continue
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=filename,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                # try download external data
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        subfolder=subfolder,
                        filename=filename + "_data",
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                except EntryNotFoundError:
                    # model doesn't use external data
                    pass

                paths[attr_name] = Path(model_cache_path).name
            new_model_save_dir = Path(model_cache_path).parent
            preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

            decoder_path = new_model_save_dir / paths["last_decoder_model_name"]
            encoder_path = new_model_save_dir / paths["last_encoder_model_name"]

        ort_inference_sessions = cls.load_model(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
            )
        except OSError:
            logger.info("Generation config file not found, using a generation config created from the model config.")

        onnx_paths = [encoder_path, decoder_path]

        return cls(
            *ort_inference_sessions[:2],
            config,
            onnx_paths=onnx_paths,
            use_cache=use_cache,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
            generation_config=generation_config,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        use_cache: bool = True,
        use_merged: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> "RyzenAIModelForConditionalGeneration":
        return NotImplementedError(
            "Exporting the model from transformers is not supported. Please follow the documentation to export the model and run the model using the RyzenAIModel!"
        )

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `RyzenAIModel`: the model placed on the requested device.
        """
        # TODO: update
        return self

    def can_generate(self):
        logger.warning(
            "RyzenAIModelForConditionalGeneration is an abstract class and is not meant to be used for generation. Please use RyzenAIModelForSeq2SeqLM or RyzenAIModelForSpeechSeq2Seq."
        )
        return False


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class RyzenAIModelForSpeechSeq2Seq(RyzenAIModelForConditionalGeneration, GenerationMixin):
    """
    Speech Sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports whisper.
    """

    auto_model_class = AutoModelForSpeechSeq2Seq
    main_input_name = "input_features"

    def __init__(
        self,
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        config: "PretrainedConfig",
        onnx_paths: List[str],
        decoder_with_past_session: Optional[ort.InferenceSession] = None,
        use_cache: bool = True,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        super().__init__(
            encoder_session=encoder_session,
            decoder_session=decoder_session,
            config=config,
            onnx_paths=onnx_paths,
            decoder_with_past_session=decoder_with_past_session,
            use_cache=use_cache,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
            generation_config=generation_config,
            **kwargs,
        )
        # Following a breaking change in transformers that relies directly on the mapping name and not on the greedy model mapping (that can be extended), we need to hardcode the RyzenAIModel in this dictionary. Other pipelines do not seem to have controlflow depending on the mapping name.
        # See: https://github.com/huggingface/transformers/pull/24960/files
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES["ort_speechseq2seq"] = self.__class__.__name__

    def _initialize_encoder(self, session: ort.InferenceSession) -> RyzenAIEncoder:
        return RyzenAIEncoderForSpeech(session, self)

    @add_start_docstrings_to_model_forward(
        SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING
        + AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="RyzenAIModelForSpeechSeq2Seq",
            checkpoint="optimum/whisper-tiny.en",
        )
    )
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)

        # Decode
        decoder_outputs = self.decoder_with_past(
            input_ids=decoder_input_ids[:, -1:],
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self) -> RyzenAIEncoder:
        return self.encoder

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        **kwargs,
    ):
        if "WhisperForConditionalGeneration" in config.architectures:
            return _RyzenAIModelForWhisper._from_pretrained(model_id, config, **kwargs)
        else:
            return super()._from_pretrained(model_id, config, **kwargs)


class MetaClassRemoveParentsAndReorder(ABCMeta):
    def mro(cls):
        """
        Avoids inheritting from PreTrainedModel, nn.Module, ModuleUtilsMixin, PushToHubMixin,
        and put GenerationMixin at the end of the MRO
        """
        top_inheritance_index = RyzenAIModelForSpeechSeq2Seq.__mro__.index(GenerationMixin)
        return (
            (cls,)
            + RyzenAIModelForSpeechSeq2Seq.__mro__[:top_inheritance_index]
            + (WhisperForConditionalGeneration,)
            + RyzenAIModelForSpeechSeq2Seq.__mro__[top_inheritance_index:]
        )


class _RyzenAIModelForWhisper(
    RyzenAIModelForSpeechSeq2Seq, WhisperForConditionalGeneration, metaclass=MetaClassRemoveParentsAndReorder
):
    """
    Whisper implements its own generate() method.
    """

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        **kwargs,
    ):
        return super(RyzenAIModelForSpeechSeq2Seq, cls)._from_pretrained(model_id, config, **kwargs)
