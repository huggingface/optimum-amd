# Use rocm image
FROM rocm/dev-ubuntu-22.04:6.0.2

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Versions
# available options 3.10
ARG PYTHON_VERSION=3.10

# Bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install and update tools to minimize security vulnerabilities - are all of these really necessary?
RUN apt-get update && apt-get install -y \
    apt-utils \
    bzip2 \
    ca-certificates \
    cmake \
    ffmpeg \
    git \
    hipsparse-dev \
    hipblas-dev \
    hipblaslt-dev \
    hipcub-dev \
    hipfft-dev \
    hiprand-dev \
    hipsolver-dev \
    libglib2.0-0 \
    libopenmpi-dev \
    libprotobuf-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    mercurial \
    miopen-hip-dev \
    patchelf \
    protobuf-compiler \
    python3 \
    python3-dev \
    python3-pip \
    rccl-dev \
    rocthrust-dev \
    rocrand-dev \
    rocblas-dev \
    software-properties-common \
    subversion \
    wget \
    && \
    apt-get clean && \
    apt-get autoremove -y

RUN python3 -m pip install -U pip
RUN python3 -m pip install cmake onnx ninja transformers --no-cache-dir
    
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0 --no-cache-dir

# Install ONNXRuntime from source
RUN git clone --single-branch --branch main --recursive https://github.com/Microsoft/onnxruntime onnxruntime

RUN cd onnxruntime && ./build.sh --config Release --build_wheel --allow_running_as_root --update --build --parallel --cmake_extra_defines CMAKE_HIP_ARCHITECTURES=gfx90a,gfx942 ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --use_rocm --rocm_home=/opt/rocm
RUN pip install onnxruntime/build/Linux/Release/dist/*

RUN python3 -m pip install git+https://github.com/huggingface/optimum.git

CMD ["/bin/bash"]
