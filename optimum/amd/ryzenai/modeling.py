# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""RyzenAIModelForXXX classes, allowing to run ONNX Models with ONNX Runtime VITIS-AI EP using the same API as Transformers."""

import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import onnxruntime as ort
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from optimum.exporters import TasksManager
from optimum.modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from optimum.onnx.utils import _get_external_data_paths
from optimum.utils.file_utils import find_files_matching_pattern
from optimum.utils.save_utils import maybe_load_preprocessors
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.file_utils import add_start_docstrings

from .utils import (
    ONNX_WEIGHTS_NAME,
    get_device_for_provider,
    validate_provider_availability,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


class classproperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


class RyzenAIModel(OptimizedModel):
    """
    Base class for implementing models using ONNX Runtime.

    The RyzenAIModel implements generic methods for interacting with the Hugging Face Hub as well as exporting vanilla
    transformers models to ONNX using `optimum.exporters.onnx` toolchain.

    Class attributes:
        - model_type (`str`, *optional*, defaults to `"onnx_model"`) -- The name of the model type to use when
        registering the RyzenAIModel classes.
        - auto_model_class (`Type`, *optional*, defaults to `AutoModel`) -- The "AutoModel" class to represented by the
        current RyzenAIModel class.

    Common attributes:
        - model (`ort.InferenceSession`) -- The ONNX Runtime InferenceSession that is running the model.
        - config ([`~transformers.PretrainedConfig`] -- The configuration of the model.
        - model_save_dir (`Path`) -- The directory where the model exported to ONNX is saved.
        By defaults, if the loaded model is local, the directory where the original model will be used. Otherwise, the
        cache directory is used.
        - providers (`List[str]) -- The list of execution providers available to ONNX Runtime.
    """

    model_type = "onnx_model"
    auto_model_class = AutoModel

    @classproperty
    def export_feature(cls):
        logger.warning(f"{cls.__name__}.export_feature is deprecated, and will be removed in optimum 2.0.")
        try:
            feature = TasksManager.infer_task_from_model(cls.auto_model_class)
        except ValueError:
            feature = None
        return feature

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return TasksManager.infer_task_from_model(auto_model_class)

    def shared_attributes_init(
        self,
        model: ort.InferenceSession,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        """
        Initializes attributes that may be shared among several ONNX Runtime inference sesssions.
        """
        # TODO: remove at version 2.0
        if kwargs.pop("latest_model_name", None) is not None:
            logger.warning(
                f"The latest_model_name argument to create an {self.__class__.__name__} is deprecated, and not used "
                "anymore."
            )
        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments."
            )

        self.providers = model.get_providers()
        self._device = get_device_for_provider(
            self.providers[0], provider_options=model.get_provider_options()[self.providers[0]]
        )

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying ONNX model.
        self._model_save_dir_tempdirectory_instance = None
        if model_save_dir is None:
            self.model_save_dir = Path(model._model_path).parent
        elif isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir

        self.preprocessors = preprocessors if preprocessors is not None else []

        if self._device is None:
            # TODO: change for the ryzenai device
            logger.warning(
                f"RyzenAIModel outputs will be sent to CPU as the device could not be inferred from the execution provider {self.providers[0]}."
                f" Use `ort_model.to()` to send the outputs to the wanted device."
            )

        # Registers the RyzenAIModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def __init__(
        self,
        model: ort.InferenceSession,
        config: "PretrainedConfig",
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(model, config)

        self.model_path = Path(model._model_path)
        self.model_name = self.model_path.name

        self.shared_attributes_init(
            model,
            model_save_dir,
            preprocessors,
            **kwargs,
        )

        self.inputs_names = {input_key.name: idx for idx, input_key in enumerate(model.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(model.get_outputs())}

    # TODO: why do we make device a property since we are only access the value, and do not do any check when setting the value?
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`torch.device` or `str` or `int`):
                TODO: write the description for the ryzen device
        Returns:
            `RyzenAIModel`: the model placed on the requested device.
        """
        # TODO: update
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_model(
        path: Union[str, Path],
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> ort.InferenceSession:
        """
        Loads an ONNX Inference session with a given provider. Default provider is `CPUExecutionProvider` to match the
        default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            path (`Union[str, Path]`):
                Path of the ONNX model.
            provider (`str`, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`):
                ONNX Runtime session options to use for loading the model.
            provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
        """
        # TODO: change the default provider to VitisAIExecutionProvider
        validate_provider_availability(provider)  # raise error if the provider is not available

        providers = [provider]

        if not isinstance(path, str):
            path = str(path)

        # `providers` and `provider_options` need to be of the same length
        if provider_options is not None:
            providers_options = [provider_options] + [{} for _ in range(len(providers) - 1)]
        else:
            providers_options = None

        return ort.InferenceSession(
            path,
            providers=providers,
            sess_options=session_options,
            provider_options=providers_options,
        )

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_ort.RyzenAIModel.from_pretrained`] class method. It will always save the
        file under model_save_dir/latest_model_name.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        src_paths = [self.model_path]
        dst_paths = [Path(save_directory) / self.model_path.name]

        # add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copyfile(src_path, dst_path)

    @staticmethod
    def _generate_regular_names_for_filename(filename: str):
        name, extension = filename.rsplit(".", maxsplit=1)
        return [filename, f"{name}_quantized.{extension}", f"{name}_optimized.{extension}"]

    @staticmethod
    def infer_onnx_filename(
        model_name_or_path: Union[str, Path],
        patterns: List[str],
        argument_name: str,
        subfolder: str = "",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        fail_if_not_found: bool = True,
    ) -> str:
        onnx_files = []
        for pattern in patterns:
            onnx_files = find_files_matching_pattern(
                model_name_or_path,
                pattern,
                glob_pattern="**/*.onnx",
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
            )
            if onnx_files:
                break

        path = model_name_or_path
        if subfolder != "":
            path = f"{path}/{subfolder}"

        if len(onnx_files) == 0:
            if fail_if_not_found:
                raise FileNotFoundError(f"Could not find any ONNX model file for the regex {patterns} in {path}.")
            return None
        elif len(onnx_files) > 1:
            if argument_name is not None:
                raise RuntimeError(
                    f"Too many ONNX model files were found in {path}, specify which one to load by using the "
                    f"{argument_name} argument."
                )
        return onnx_files[0]

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ) -> "RyzenAIModel":
        model_path = Path(model_id)
        regular_onnx_filenames = RyzenAIModel._generate_regular_names_for_filename(ONNX_WEIGHTS_NAME)

        if file_name is None:
            if model_path.is_dir():
                onnx_files = list(model_path.glob("*.onnx"))
            else:
                if isinstance(use_auth_token, bool):
                    token = HfFolder().get_token()
                else:
                    token = use_auth_token
                repo_files = map(Path, HfApi().list_repo_files(model_id, revision=revision, token=token))
                pattern = "*.onnx" if subfolder == "" else f"{subfolder}/*.onnx"
                onnx_files = [p for p in repo_files if p.match(pattern)]

            if len(onnx_files) == 0:
                raise FileNotFoundError(f"Could not find any ONNX model file in {model_path}")
            elif len(onnx_files) > 1:
                raise RuntimeError(
                    f"Too many ONNX model files were found in {model_path}, specify which one to load by using the "
                    "file_name argument."
                )
            else:
                file_name = onnx_files[0].name

        if file_name not in regular_onnx_filenames:
            logger.warning(
                f"The ONNX file {file_name} is not a regular name used in optimum.onnxruntime, the RyzenAIModel might "
                "not behave as expected."
            )

        preprocessors = None
        if model_path.is_dir():
            model = RyzenAIModel.load_model(
                model_path / file_name,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )
            new_model_save_dir = model_path
            preprocessors = maybe_load_preprocessors(model_id)
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                subfolder=subfolder,
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
                    filename=file_name + "_data",
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except EntryNotFoundError:
                # model doesn't use external data
                pass

            model = RyzenAIModel.load_model(
                model_cache_path, provider=provider, session_options=session_options, provider_options=provider_options
            )
            new_model_save_dir = Path(model_cache_path).parent
            preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance, in which case we want to keep it
        # instead of the path only.
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            model=model,
            config=config,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> "RyzenAIModel":
        return NotImplementedError(
            "Exporting the model from transformers is not supported. Please follow the documentation to export the model and run the model using the RyzenAIModel!"
        )

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        config: Optional["PretrainedConfig"] = None,
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        provider (`str`, defaults to `"CPUExecutionProvider"`):
            ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/ for
            possible providers.
        session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`),:
            ONNX Runtime session options to use for loading the model.
        provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
            Provider option dictionaries corresponding to the provider used. See available options
            for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
        kwargs (`Dict[str, Any]`):
            Will be passed to the underlying model loading methods.

        > Parameters for decoder models (RyzenAIForSpeechSeq2Seq)

        use_cache (`Optional[bool]`, defaults to `True`):
            Whether or not past key/values cache should be used. Defaults to `True`.

        Returns:
            `RyzenAIModel`: The loaded RyzenAIModel model.
        """
        return super().from_pretrained(
            model_id,
            export=export,
            force_download=force_download,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            subfolder=subfolder,
            config=config,
            local_files_only=local_files_only,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            **kwargs,
        )
