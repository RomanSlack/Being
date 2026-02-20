# Being — Full stack Docker image
# Includes: CUDA, PyTorch, InsTaG, Being server
#
# Build:
#   docker build -t being -f docker/Dockerfile .
#
# Run inference server:
#   docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
#     being serve --port 8000
#
# Run data pipeline:
#   docker run --gpus all -v $(pwd)/data:/app/data \
#     being prepare /app/data/avatars/myavatar/myavatar.mp4

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /app

# Copy environment file first (cache conda env layer)
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make conda env active by default
SHELL ["conda", "run", "-n", "being", "/bin/bash", "-c"]

# Clone InsTaG
RUN mkdir -p extern && \
    git clone --recursive https://github.com/Fictionarry/InsTaG.git extern/InsTaG

# Install PyTorch3D
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Download InsTaG pre-trained weights
RUN cd extern/InsTaG && bash scripts/prepare.sh

# Install EasyPortrait
RUN pip install -U openmim && mim install mmcv-full==1.7.1 prettytable && \
    wget -q "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
    -O extern/InsTaG/data_utils/easyportrait/fpn-fp-512.pth

# Copy project files
COPY . .

# Install Being package
RUN pip install -e .

# Create data directories
RUN mkdir -p data/avatars output

EXPOSE 8000

ENTRYPOINT ["conda", "run", "-n", "being", "being"]
CMD ["serve", "--port", "8000"]
