# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

FROM rocm/dev-ubuntu-22.04:6.0.2

LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
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
RUN pip install -U --no-cache-dir ninja packaging git+https://github.com/facebookresearch/detectron2.git pytesseract "itsdangerous<2.1.0"

ARG FLASH_ATT_V2_COMMIT_ROCM=2554f490101742ccdc56620a938f847f61754be6

RUN git clone https://github.com/ROCm/flash-attention.git flash-attention-v2 && \
    cd flash-attention-v2 && git submodule update --init --recursive && \
    GPU_ARCHS="gfx90a;gfx942" PYTORCH_ROCM_ARCH="gfx90a;gfx942" python setup.py install && \
    cd .. && \
    rm -rf flash-attention

WORKDIR /
RUN git clone --depth 1 --branch main https://github.com/huggingface/transformers.git && cd transformers
RUN pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]
RUN pip uninstall -y tensorflow flax
