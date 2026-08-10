ARG PYTORCH_VERSION="2.6.0"
ARG CUDA_VERSION="12.4"
ARG CUDNN_VERSION="9"
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

LABEL maintainer="zhiying.zou"
LABEL org.opencontainers.image.title="sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything"
LABEL org.opencontainers.image.version="v0.1"
LABEL org.opencontainers.image.description="Personal PyTorch environment for Parcours_recherche_server_sync"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.ustc.edu.cn/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.ustc.edu.cn/ubuntu|g' \
        /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        sudo \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

COPY Orient-Anything-V2/requirements.txt /tmp/orient-anything-v2-requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/ \
    && python -m pip install -r /tmp/orient-anything-v2-requirements.txt \
    && python -m pip install debugpy \
    && rm -f /tmp/orient-anything-v2-requirements.txt

WORKDIR /workspace/zhiying.zou/Parcours_recherche_server_sync

CMD ["bash"]
