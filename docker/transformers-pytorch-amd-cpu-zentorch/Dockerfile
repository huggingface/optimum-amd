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

ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION}

ARG TORCH_VERSION=2.2.1

# Install python and g++ compiler
ENV DEBIAN_FRONTEND noninteractive
ENV PATH="/home/user/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    libjemalloc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    pip install --upgrade pip

# Create a non-root user
ARG GROUP_ID
ARG USER_ID

RUN addgroup --gid $GROUP_ID group
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user
WORKDIR /home/user

# Install PyTorch
RUN if [ "${TORCH_VERSION}" = "stable" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
elif [ "${TORCH_VERSION}" = "nighly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu ; \
else \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
fi

# Copy and install ZenTorch wheel
COPY zentorch-0.1.0-cp38-cp38-manylinux2014_x86_64.whl /home/user/zentorch-0.1.0-cp38-cp38-manylinux2014_x86_64.whl
RUN pip install --no-cache-dir /home/user/zentorch-0.1.0-cp38-cp38-manylinux2014_x86_64.whl
