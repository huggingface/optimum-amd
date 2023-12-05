# Use rocm image
FROM rocm/pytorch:latest
CMD rocm-smi

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Versions
# available options 3.9
ARG PYTHON_VERSION=3.9

# Bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install and update tools to minimize security vulnerabilities - are all of these really necessary?
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    apt-utils \
    patchelf \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    mercurial \
    subversion \
    libopenmpi-dev \
    ffmpeg && \
    apt-get clean && \
    apt-get autoremove -y

ARG PYTHON_EXE=/opt/conda/envs/py_$PYTHON_VERSION/bin/python

RUN $PYTHON_EXE -m pip install -U pip
RUN $PYTHON_EXE -m pip install cmake onnx ninja transformers --no-cache-dir

# Install ONNXRuntime from source
RUN git clone --recursive  https://github.com/ROCmSoftwarePlatform/onnxruntime.git && cd onnxruntime && git checkout rocm5.7_internal_testing_eigen-3.4.zip_hash

RUN cd onnxruntime && ./build.sh --config Release --build_wheel --allow_running_as_root --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --use_rocm --rocm_home=/opt/rocm
RUN pip install onnxruntime/build/Linux/Release/dist/*

RUN $PYTHON_EXE -m pip install git+https://github.com/huggingface/optimum.git

WORKDIR .

CMD ["/bin/bash"]
