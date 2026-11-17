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
    hipblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install -U pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0 --no-cache-dir

WORKDIR /quark
RUN wget -O quark.zip https://www.xilinx.com/bin/public/openDownload?filename=quark-0.1.0+a9827f5.zip && \
    unzip quark.zip -d quark && \
    pip install quark/quark-0.1.0+a9827f5/quark-0.1.0+a9827f5-py39-none-any.whl && \
    rm -rf quark.zip quark

RUN python -c "import quark.torch.kernel"

ENTRYPOINT ["bash"]
