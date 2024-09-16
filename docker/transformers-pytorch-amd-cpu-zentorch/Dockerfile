# Copyright 2023 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION=22.04

FROM condaforge/miniforge3:24.7.1-0

# Install python and g++ compiler
ENV DEBIAN_FRONTEND noninteractive
ENV PATH="/home/user/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    libjemalloc-dev \
    software-properties-common \
    curl \
    numactl

RUN apt-get install gnupg2  -y
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get install -y g++-11 && \
    rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.10

WORKDIR /MAMBA
ARG MAMBA_ARCH=x86_64
ARG MAMBA_VERSION=24.7.1-0

RUN /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ; \
    /opt/conda/bin/conda clean -ya

# # Install PyTorch
RUN pip install --no-cache-dir --pre torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install git+https://github.com/huggingface/optimum-benchmark.git
RUN pip install git+https://github.com/huggingface/optimum-amd.git@fbd225616ef5a16b3cb762bc762e83d30b8ee1c9
RUN pip install optimum==v1.21.4
RUN conda install -c conda-forge llvm-openmp=18.1.8=hf5423f3_1 -y

COPY zentorch-5.0.0-cp310-cp310-manylinux_2_28_x86_64.whl .
RUN pip install zentorch-5.0.0-cp310-cp310-manylinux_2_28_x86_64.whl

ENV OMP_WAIT_POLICY=ACTIVE
ENV OMP_DYNAMIC=FALSE
ENV KMP_BLOCKTIME=1
ENV KMP_TPAUSE=0
ENV KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
ENV KMP_PLAIN_BARRIER_PATTERN=dist,dist
ENV KMP_REDUCTION_BARRIER_PATTERN=dist,dist
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:$LD_PRELOAD
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV ZENDNN_WEIGHT_CACHING=1
ENV ZENDNN_MATMUL_ALGO=FP32:4,BF16:0
ENV ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
ENV HUGGINGFACE_HUB_CACHE=/data/hf_cache/

WORKDIR /workspace
