# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
        git \
        wget \
        curl \
        # OpenCV / decord system deps
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        # decord needs ffmpeg
        ffmpeg \
        # misc build tools
        build-essential \
        cmake \
        ninja-build \
        # git-lfs for HuggingFace large model downloads
        git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/python3
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Bootstrap pip for 3.11
RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
 && python /tmp/get-pip.py \
 && rm /tmp/get-pip.py

# ── PyTorch 2.3.1 + CUDA 12.1 ────────────────────────────────────────────────
# Command from https://pytorch.org/get-started/previous-versions/
RUN pip install \
        torch==2.3.1 \
        torchvision==0.18.1 \
        torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121

# ── Shared / common deps (both projects) ─────────────────────────────────────
# numpy<2 satisfies vggt (pins 1.26.1) and vjepa2 (unpinned)
RUN pip install \
        "numpy<2" \
        Pillow \
        einops \
        safetensors \
        huggingface_hub \
        pyyaml \
        opencv-python \
        pandas \
        scipy \
        matplotlib \
        requests \
        tqdm \
        psutil \
        h5py

# ── vjepa2-specific deps ──────────────────────────────────────────────────────
RUN pip install \
        tensorboard \
        wandb \
        iopath \
        submitit \
        braceexpand \
        webdataset \
        timm \
        "transformers<4.47" \
        peft \
        decord \
        beartype \
        fire \
        python-box \
        scikit-learn \
        scikit-image \
        ftfy \
        jupyter \
        hf

# ── vggt demo deps ────────────────────────────────────────────────────────────
RUN pip install \
        "pydantic==2.10.6" \
        "gradio==5.17.1" \
        "viser==0.2.23" \
        hydra-core \
        omegaconf \
        onnxruntime \
        trimesh \
        "pycolmap==3.10.0" \
        "pyceres==2.3"

# LightGlue (required by vggt colmap demo)
RUN pip install "git+https://github.com/jytime/LightGlue.git#egg=lightglue"

# Ensure pip-installed scripts (huggingface-cli, etc.) are on PATH
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /workspace

CMD ["/bin/bash"]
