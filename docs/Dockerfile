FROM python:3.10.12

ARG commit_sha
ARG clone_url

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    gnupg \
    ffmpeg \
    libsm6 \
    libxext6

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y npm
RUN npm install npm@9.8.1 -g && \
    npm install n -g && \
    n latest

RUN pip install --no-cache-dir --upgrade pip uv

ENV UV_SYSTEM_PYTHON=1
RUN uv pip install --no-cache-dir git+https://github.com/huggingface/doc-builder.git

RUN git clone $clone_url && cd optimum-amd && git checkout $commit_sha && uv pip install .[brevitas,tests]
RUN git clone --depth 1 --branch v3.5 https://github.com/Xilinx/Vitis-AI.git && cd Vitis-AI/src/vai_quantizer/vai_q_onnx && sh build.sh && uv pip install pkgs/*.whl
RUN uv pip uninstall torchao diffusers
RUN uv pip install "onnxruntime==1.14"