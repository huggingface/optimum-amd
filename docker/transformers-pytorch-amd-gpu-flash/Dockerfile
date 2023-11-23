# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

FROM rocm/dev-ubuntu-22.04:5.6
# We do not use rocm/dev-ubuntu-22.04:5.6.1, with which we get the following error during the compilation of Flash Attention backward kernels:
# 145.6 fatal error: error in backend: Not supported instr: <MCInst 0 <MCOperand Reg:519> <MCOperand Reg:527> <MCOperand Expr:(.LBB2_3)> <MCOperand Reg:487> <MCOperand Expr:(.LBB2_-1)>>
# 145.6 clang-16: error: clang frontend command failed with exit code 70 (use -v to see invocation)
# 145.6 AMD clang version 16.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.6.1 23332 4f9bb99d78a4d8d9770be38b91ebd004ea4d2a3a)

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

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 --no-cache-dir
RUN pip install -U --no-cache-dir ninja packaging git+https://github.com/facebookresearch/detectron2.git pytesseract "itsdangerous<2.1.0"

RUN git clone https://github.com/ROCmSoftwarePlatform/flash-attention.git && \
    cd flash-attention && \
    git submodule init && \
    git submodule update && \
    GPU_ARCHS="gfx90a" python setup.py install && \
    cd .. && \
    rm -rf flash-attention

WORKDIR /
RUN git clone --depth 1 --branch main https://github.com/huggingface/transformers.git && cd transformers
RUN pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]
RUN pip uninstall -y tensorflow flax
