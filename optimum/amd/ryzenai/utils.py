# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


import builtins
import logging
import os
import shutil
import subprocess

import onnxruntime as ort


logger = logging.getLogger(__name__)

ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_WEIGHTS_NAME_STATIC = "model_static.onnx"

DEFAULT_TVM_GEMM_M = "1,8,"
DEFAULT_TVM_DLL_NUM = "2"
DEFAULT_DEVICE = "phx"
DEFAULT_DLL_FILES = ["qlinear\\libGemmQnnAie_1x2048_2048x2048.dll", "qlinear\\libGemmQnnAie_8x2048_2048x2048.dll"]

DEFAULT_BUILTIN_IMPL = "v0"
DEFAULT_BUILTIN_QUANT_MODE = "w8a8"

RYZEN_SW_COMMIT_HASH = "82c524a06693a18e167f032dbf5574a98dd24452"


def validate_provider_availability(provider: str):
    """
    Ensure the ONNX Runtime execution provider `provider` is available, and raise an error if it is not.

    Args:
        provider (str): Name of an ONNX Runtime execution provider.
    """
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise ValueError(
            f"Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}."
        )


def set_builtins():
    """Set the builtins.impl and builtins.quant_mode environment variables."""
    builtins.impl = os.getenv("BUILTINS_IMPL", DEFAULT_BUILTIN_IMPL)
    builtins.quant_mode = os.getenv("BUILTINS_IMPL", DEFAULT_BUILTIN_QUANT_MODE)


def clone_repository(repo_url: str, repo_path: str):
    try:
        if not os.path.exists(repo_path):
            subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", repo_url, repo_path], check=True)
            subprocess.run(["git", "-C", repo_path, "checkout", RYZEN_SW_COMMIT_HASH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)


def set_env_var(key, value):
    if key not in os.environ:
        os.environ[key] = value


def normalize_path(path):
    return os.path.normpath(path)


def set_environment_variables():
    ryzenai_sw_path = os.environ.get("RYZENAI_SW_PATH")
    if not ryzenai_sw_path:
        logger.warning(
            "RYZENAI_SW_PATH environment variable is not set. Attempting to clone RyzenAI-SW repository now...\n"
        )
        ryzenai_sw_path = normalize_path(os.path.join(os.getcwd(), "RyzenAI-SW"))
        clone_repository("https://github.com/amd/RyzenAI-SW/", ryzenai_sw_path)

    # Set other environment variables
    ryzenai_transformers_path = normalize_path(os.path.join(ryzenai_sw_path, "example/transformers"))
    third_party = normalize_path(os.path.join(ryzenai_transformers_path, "third_party"))

    set_env_var("THIRD_PARTY", third_party)
    set_env_var(
        "TVM_LIBRARY_PATH",
        f"{normalize_path(os.path.join(third_party, 'lib'))};{normalize_path(os.path.join(third_party, 'bin'))}",
    )
    set_env_var("DEVICE", DEFAULT_DEVICE)
    set_env_var(
        "XLNX_VART_FIRMWARE",
        normalize_path(os.path.join(ryzenai_transformers_path, "xclbin", os.environ.get("DEVICE", DEFAULT_DEVICE))),
    )

    dll_path = os.path.join(ryzenai_transformers_path, "dll", os.environ.get("DEVICE", DEFAULT_DEVICE))
    tvm_module_paths = []
    for dll_file in DEFAULT_DLL_FILES:
        tvm_module_paths.append(normalize_path(os.path.join(dll_path, dll_file)))

    set_env_var("TVM_MODULE_PATH", ",".join(tvm_module_paths) + ",")
    set_env_var("TVM_GEMM_M", DEFAULT_TVM_GEMM_M)
    set_env_var("TVM_DLL_NUM", DEFAULT_TVM_DLL_NUM)
