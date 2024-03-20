ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION}

ARG TORCH_VERSION=2.2.1

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Install python and g++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    libjemalloc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Update pip
RUN pip install --upgrade pip

# Install PyTorch
RUN pip install --no-cache-dir torch==$TORCH_VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy and install ZenTorch wheel
COPY zentorch-0.1.0-cp38-cp38-manylinux2014_x86_64.whl /home/user/zentorch-0.1.0-cp38-cp38-manylinux2014_x86_64.whl
RUN pip install --no-cache-dir /home/user/zentorch-0.1.0-cp38-cp38-manylinux2014_x86_64.whl
