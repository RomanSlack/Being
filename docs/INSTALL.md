# Installation Guide

## Quick Setup (Recommended)

```bash
git clone --recursive https://github.com/anthropics/Being.git
cd Being
bash scripts/setup.sh
```

## Manual Setup

### 1. Clone InsTaG

```bash
mkdir -p extern
git clone --recursive https://github.com/Fictionarry/InsTaG.git extern/InsTaG
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate being
```

### 3. Install PyTorch3D

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 4. Install Being

```bash
pip install -e .
```

### 5. Download External Models

```bash
# InsTaG face parsing + landmark weights
cd extern/InsTaG && bash scripts/prepare.sh && cd ../..

# EasyPortrait (teeth masks)
pip install -U openmim
mim install mmcv-full==1.7.1 prettytable
wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
  -O extern/InsTaG/data_utils/easyportrait/fpn-fp-512.pth
```

### 6. Basel Face Model (Manual Download Required)

1. Go to https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
2. Register (free) and download `01_MorphableModel.mat`
3. Copy to `extern/InsTaG/data_utils/face_tracking/3DMM/`
4. Convert:
```bash
cd extern/InsTaG/data_utils/face_tracking
python convert_BFM.py
cd ../../../../
```

### 7. Pre-trained Checkpoints

Download from: https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP

Or use the download script:
```bash
bash scripts/download_checkpoints.sh
```

Extract to `output/pretrained/`.

### 8. OpenFace (Optional — for blink detection)

**Option A: Docker (easiest)**
```bash
docker build -t being-openface -f docker/openface.Dockerfile .
```

**Option B: Build from source**
```bash
git clone https://github.com/TadasBaltrusaitis/OpenFace.git extern/OpenFace
cd extern/OpenFace
bash download_models.sh
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(nproc)
sudo make install
```

### 9. Sapiens (Optional — for geometry priors in few-shot adaptation)

This requires a **separate** conda environment:
```bash
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks

# Download models
cd extern/InsTaG
bash scripts/prepare_sapiens.sh
```

## Verify Installation

```bash
conda activate being
being check
```

## Hardware Requirements

| Task | GPU | VRAM | RAM |
|------|-----|------|-----|
| Data preprocessing | Any NVIDIA | 4GB+ | 16GB |
| Pre-training | A100/H100 | 40-80GB | 64GB+ |
| Adaptation | RTX 4070 Super+ | 12GB+ | 32GB |
| Inference | RTX 4070 Super | 12GB | 16GB |

## Troubleshooting

### CUDA extension compilation fails
Make sure your CUDA toolkit version matches your PyTorch CUDA version:
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### PyTorch3D won't install
Try installing from a pre-built wheel:
```bash
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
```

### mmcv-full fails to install
```bash
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```
