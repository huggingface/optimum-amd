# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

FROM rocm/dev-ubuntu-22.04:6.1

LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    unzip \
    libsndfile1-dev \
    tesseract-ocr \
    espeak-ng \
    rocthrust-dev \
    hipsparse-dev \
    hipblaslt-dev \
    hipsolver-dev \
    hipblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install -U pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 --no-cache-dir

WORKDIR /quark
RUN wget -O quark-0.2.0-23-py3-none-any.whl https://www.xilinx.com/bin/public/openDownload?filename=quark-0.2.0+6af1bac23-py3-none-any.whl && \
    pip install quark-0.2.0-23-py3-none-any.whl && \
    rm -rf quark-0.2.0-23-py3-none-any.whl

RUN python -c "import quark.torch.kernel"

RUN pip install git+https://github.com/mht-sharma/transformers.git@fc62c00e1f2a927acb354e28e43828a47fa776b6

ENTRYPOINT ["bash"]