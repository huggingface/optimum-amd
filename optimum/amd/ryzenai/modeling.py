# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""RyzenAIModelForXXX classes, allowing to run ONNX Models with ONNX Runtime VITIS-AI EP using the same API as Transformers."""

import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from onnx import shape_inference
from onnx.tools import update_model_dims

from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager
from optimum.modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from optimum.onnx.utils import _get_external_data_paths
from optimum.utils.save_utils import maybe_load_preprocessors
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    PretrainedConfig,
)
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import ImageClassifierOutput, ModelOutput

from .utils import (
    DEFAULT_VAIP_CONFIG,
    ONNX_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME_STATIC,
    validate_provider_availability,
)


logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"


class RyzenAIModel(OptimizedModel):
    """
    Base class for implementing models using ONNX Runtime.

    The RyzenAIModel implements generic methods for interacting with the Hugging Face Hub as well as exporting vanilla
    transformers models to ONNX using `optimum.exporters.onnx` toolchain.

    Class attributes:
        - model_type (`str`, defaults to `"onnx_model"`) -- The name of the model type to use when
        registering the RyzenAIModel classes.
        - auto_model_class (`Type`, defaults to `AutoModel`) -- The "AutoModel" class to represented by the
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
    default_vaip_config = DEFAULT_VAIP_CONFIG

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
        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments."
            )

        self.providers = model.get_providers()

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

        # Registers the RyzenAIModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def __init__(
        self,
        model: ort.InferenceSession,
        config: PretrainedConfig,
        vaip_config: Union[str, Path] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(model, config)

        self.model_path = Path(model._model_path)
        self.model_name = self.model_path.name
        self.vaip_config = Path(vaip_config) if vaip_config else None

        self.shared_attributes_init(
            model,
            model_save_dir,
            preprocessors,
            **kwargs,
        )

        self.inputs_names = {input_key.name: idx for idx, input_key in enumerate(model.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(model.get_outputs())}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def to(self, device: Union[torch.device, str, int]):
        # Necessary for compatibility with transformer pipelines
        return self

    @staticmethod
    def load_model(
        path: Union[str, Path],
        provider: str = "VitisAIExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> ort.InferenceSession:
        """
        Loads an ONNX Inference session with a given provider. Default provider is `VitisAIExecutionProvider`.

        Args:
            path (`Union[str, Path]`):
                Path of the ONNX model.
            provider (`str`, defaults to `"VitisAIExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`):
                ONNX Runtime session options to use for loading the model.
            provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
        """
        validate_provider_availability(provider)  # raise error if the provider is not available

        providers = [provider]

        if not isinstance(path, str):
            path = str(path)

        # `providers` and `provider_options` need to be of the same length
        if provider_options is not None:
            providers_options = [provider_options] + [{} for _ in range(len(providers) - 1)]
        else:
            providers_options = None

        is_dynamic = RyzenAIModel._check_uses_static_shape(path)
        if is_dynamic and provider == "VitisAIExecutionProvider":
            raise ValueError(
                "The model provided has dynamic axes in input/output. Please provide model with static shapes for inference with RyzenAI."
            )

        return ort.InferenceSession(
            path,
            providers=providers,
            sess_options=session_options,
            provider_options=providers_options,
        )

    def _save_config(self, save_directory):
        """
        Saves a model configuration into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        if self.config is not None:
            self.config.save_pretrained(save_directory)

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

        if self.vaip_config:
            src_paths.append(self.vaip_config)
            dst_paths.append(Path(save_directory) / self.vaip_config.name)

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copyfile(src_path, dst_path)

    @staticmethod
    def _generate_regular_names_for_filename(filename: str):
        name, extension = filename.rsplit(".", maxsplit=1)
        return [filename, f"{name}_quantized.{extension}", f"{name}_optimized.{extension}"]

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        vaip_config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        provider: str = "VitisAIExecutionProvider",
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
                f"The ONNX file {file_name} is not a regular name used in optimum.amd.ryzenai, the RyzenAIModel might "
                "not behave as expected."
            )

        if provider == "VitisAIExecutionProvider":
            if vaip_config is None and "config_file" not in (provider_options or {}):
                logger.warning(
                    f"No Ryzen AI configuration file was provided. Using default: {cls.default_vaip_config}.\n"
                )
                vaip_config = cls.default_vaip_config
            elif vaip_config is not None and provider_options is not None and "config_file" in provider_options:
                raise ValueError(
                    "Configuration file paths were found in both `vaip_config` and `provider_options`."
                    "To avoid conflicts, please specify the configuration file path in either `vaip_config`"
                    "or `provider_options`"
                )

            if vaip_config:
                provider_options = provider_options if provider_options is not None else {}
                provider_options["config_file"] = vaip_config

            vaip_config = provider_options["config_file"]

            if not os.path.exists(vaip_config):
                raise ValueError(f"Config file not found: {vaip_config}")

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
                model_cache_path,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
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
            vaip_config=vaip_config,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
        )

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: PretrainedConfig,
        vaip_config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "VitisAIExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> "RyzenAIModel":
        raise NotImplementedError(
            "Exporting the model from transformers is not supported. Please follow the documentation to export the model and run the model using the RyzenAIModel!"
        )

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        vaip_config: str = None,
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        config: Optional[PretrainedConfig] = None,
        local_files_only: bool = False,
        provider: str = "VitisAIExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        library_name: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        provider (`str`, defaults to `"VitisAIExecutionProvider"`):
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
        if isinstance(model_id, Path):
            model_id = model_id.as_posix()

        if len(model_id.split("@")) == 2:
            if revision is not None:
                logger.warning(
                    f"The argument `revision` was set to {revision} but will be ignored for {model_id.split('@')[1]}"
                )
            model_id, revision = model_id.split("@")

        all_files, _ = TasksManager.get_model_files(model_id, subfolder, cache_dir=cache_dir)
        has_config = True if CONFIG_NAME in all_files else False

        if has_config is False and subfolder != "":
            all_files, _ = TasksManager.get_model_files(model_id, cache_dir=cache_dir)
            has_config = True if CONFIG_NAME in all_files else False

        if has_config:
            library_name = TasksManager.infer_library_from_model(model_id, subfolder, revision, cache_dir)

            if library_name == "timm":
                config = PretrainedConfig.from_pretrained(model_id, subfolder, revision)

            if config is None:
                if os.path.isdir(os.path.join(model_id, subfolder)) and cls.config_name == CONFIG_NAME:
                    if CONFIG_NAME in os.listdir(os.path.join(model_id, subfolder)):
                        config = AutoConfig.from_pretrained(
                            os.path.join(model_id, subfolder, CONFIG_NAME), trust_remote_code=trust_remote_code
                        )
                    elif CONFIG_NAME in os.listdir(model_id):
                        config = AutoConfig.from_pretrained(
                            os.path.join(model_id, CONFIG_NAME), trust_remote_code=trust_remote_code
                        )
                        logger.info(
                            f"config.json not found in the specified subfolder {subfolder}. Using the top level config.json."
                        )
                    else:
                        raise OSError(f"config.json not found in {model_id} local folder")
                else:
                    config = cls._load_config(
                        model_id,
                        revision=revision,
                        cache_dir=cache_dir,
                        use_auth_token=use_auth_token,
                        force_download=force_download,
                        subfolder=subfolder,
                        trust_remote_code=trust_remote_code,
                    )
            elif isinstance(config, (str, os.PathLike)):
                config = cls._load_config(
                    config,
                    revision=revision,
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                    force_download=force_download,
                    subfolder=subfolder,
                    trust_remote_code=trust_remote_code,
                )

        else:
            logger.warning("Configuration for the model not found. The RyzenAIModel might not behave as expected.")

        if not export and trust_remote_code:
            logger.warning(
                "The argument `trust_remote_code` is to be used along with export=True. It will be ignored."
            )
        elif export and trust_remote_code is None:
            trust_remote_code = False

        from_pretrained_method = cls._export if export else cls._from_pretrained

        return from_pretrained_method(
            model_id=model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            use_auth_token=use_auth_token,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            vaip_config=vaip_config,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            **kwargs,
        )

    @staticmethod
    def _check_uses_static_shape(model_path: Union[str, Path]):
        is_dynamic = False
        if Path(model_path).suffix == ".onnx":
            model = onnx.load(model_path)
            is_dynamic = any(
                any(dim.dim_param for dim_index, dim in enumerate(inp.type.tensor_type.shape.dim) if dim_index != 0)
                for inp in model.graph.input
            )

        return is_dynamic

    @staticmethod
    def _update_inputs_outputs_dims(
        model_path: Union[str, Path],
        input_shape_dict: Dict[str, Tuple[int]],
        output_shape_dict: Dict[str, Tuple[int]],
    ) -> onnx.ModelProto:
        model = onnx.load(model_path)

        updated_model = update_model_dims.update_inputs_outputs_dims(model, input_shape_dict, output_shape_dict)
        return shape_inference.infer_shapes(updated_model)

    @staticmethod
    def reshape(
        model_path: Union[str, Path],
        input_shape_dict: Dict[str, Tuple[int]],
        output_shape_dict: Dict[str, Tuple[int]],
    ) -> Union[str, Path]:
        """
        Propagates the given input shapes on the model's layers, fixing the input shapes of the model.

        Args:
            model_path (Union[str, Path]):
                Path to the model.
            input_shape_dict (Dict[str, Tuple[int]]):
                Input shapes for the model.
            output_shape_dict (Dict[str, Tuple[int]]):
                Output shapes for the model.

        Returns:
            Union[str, Path]:
                Path to the model after updating the input shapes.

        Raises:
            ValueError: If the model provided has dynamic axes in input/output and no input/output shape is provided.
        """
        if isinstance(model_path, (str, Path)) and Path(model_path).suffix == ".onnx":
            model = RyzenAIModel._update_inputs_outputs_dims(model_path, input_shape_dict, output_shape_dict)

            static_model_path = Path(model_path).parent / ONNX_WEIGHTS_NAME_STATIC
            onnx.save(model, static_model_path)

            return static_model_path

        return model_path


class RyzenAIModelForCustomTasks(RyzenAIModel):
    def forward(self, **kwargs):
        use_torch = isinstance(next(iter(kwargs.values())), torch.Tensor)
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = self._prepare_onnx_inputs(use_torch=use_torch, **kwargs)

        # run inference
        onnx_outputs = self.model.run(None, onnx_inputs)
        outputs = self._prepare_onnx_outputs(onnx_outputs, use_torch=use_torch)

        # converts output to namedtuple for pipelines post-processing
        return ModelOutput(outputs)

    def _prepare_onnx_inputs(self, use_torch: bool, **kwargs):
        onnx_inputs = {}
        # converts pytorch inputs into numpy inputs for onnx
        for input in self.inputs_names.keys():
            onnx_inputs[input] = kwargs.pop(input)

            if use_torch:
                onnx_inputs[input] = onnx_inputs[input].cpu().detach().numpy()

        return onnx_inputs

    def _prepare_onnx_outputs(self, onnx_outputs, use_torch: bool):
        outputs = {}
        # converts onnxruntime outputs into tensor for standard outputs
        for output, idx in self.output_names.items():
            outputs[output] = onnx_outputs[idx]

            if use_torch:
                outputs[output] = torch.from_numpy(outputs[output])

        return outputs


class RyzenAIModelForImageClassification(RyzenAIModelForCustomTasks):
    auto_model_class = AutoModelForImageClassification

    def forward(self, pixel_values):
        use_torch = isinstance(pixel_values, torch.Tensor)
        if use_torch:
            pixel_values = pixel_values.cpu().detach().numpy()
        onnx_inputs = {
            list(self.inputs_names.keys())[0]: pixel_values,
        }

        # run inference
        onnx_outputs = self.model.run(None, onnx_inputs)
        outputs = self._prepare_onnx_outputs(onnx_outputs, use_torch=use_torch)

        return ImageClassifierOutput(logits=next(iter(outputs.values())))

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig" = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        vaip_config: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "RyzenAIModel":
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task="image-classification",
            opset=17,
            batch_size=1,
            no_dynamic_axes=True,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        return cls._from_pretrained(
            save_dir_path,
            config,
            vaip_config=vaip_config,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            subfolder=subfolder,
            local_files_only=local_files_only,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            model_save_dir=save_dir,
            **kwargs,
        )


class RyzenAIModelForObjectDetection(RyzenAIModelForCustomTasks):
    def forward(self, pixel_values):
        use_torch = isinstance(pixel_values, torch.Tensor)
        if use_torch:
            pixel_values = pixel_values.cpu().detach().numpy()
        onnx_inputs = {
            list(self.inputs_names.keys())[0]: pixel_values,
        }

        # run inference
        onnx_outputs = self.model.run(None, onnx_inputs)
        outputs = self._prepare_onnx_outputs(onnx_outputs, use_torch=use_torch)

        return ModelOutput(outputs)


class RyzenAIModelForImageSegmentation(RyzenAIModelForObjectDetection):
    pass


class RyzenAIModelForSemanticSegmentation(RyzenAIModelForObjectDetection):
    pass


class RyzenAIModelForImageToImage(RyzenAIModelForObjectDetection):
    pass
