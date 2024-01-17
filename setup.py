# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/intel/version.py
try:
    filepath = "optimum/amd/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

# ORT 1.16 is not compatible: https://github.com/Xilinx/Vitis-AI/issues/1343
INSTALL_REQUIRE = ["optimum", "transformers", "onnx", "onnxruntime-extensions"]

TESTS_REQUIRE = ["pytest", "parameterized", "evaluate", "timm", "scikit-learn"]

QUALITY_REQUIRE = ["black~=23.1", "ruff>=0.0.241,<=0.0.259"]

EXTRAS_REQUIRE = {
    "quality": QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
}

setup(
    name="optimum-amd",
    version=__version__,
    description="Optimum Library is an extension of the Hugging Face Transformers library, providing a framework to "
    "integrate third-party libraries from Hardware Partners and interface with their specific "
    "functionality.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, amd, ryzen, ipu, quantization, on-device, instinct",
    url="https://github.com/huggingface/optimum-amd",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="MIT",
    packages=find_namespace_packages(include=["optimum*"]),
    entry_points={"console_scripts": ["amdrun=optimum.amd.cli:amdrun"]},
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)
