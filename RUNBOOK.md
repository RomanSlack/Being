# Being — Runbook

Tested on RunPod A100-SXM4-80GB with `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` template.

## What Works Today

We successfully ran the full InsTaG few-shot adaptation pipeline:

1. **Data preprocessing** — video normalization, face tracking, face parsing, audio feature extraction
2. **3-stage training** — face (10K iters, ~10 min), mouth (10K iters, ~7 min), fuse (2K iters, ~2 min)
3. **Synthesis** — rendered test frames into output video

Total time on A100: **~20 minutes** from preprocessed data to output video.

## Setup on RunPod A100

### 1. Pod Setup

- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- GPU: A100 80GB (A40 48GB works but Gaussians can exhaust VRAM)
- Add your SSH key via web terminal:
```bash
mkdir -p ~/.ssh && echo "YOUR_PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

### 2. Install Miniconda + Conda Environment

```bash
cd /tmp && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
/opt/conda/bin/conda init bash
# Accept TOS if prompted:
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

/opt/conda/bin/conda create -n being python=3.9 -y
```

### 3. Clone Repo + Install Dependencies

```bash
cd /workspace
git clone --recursive https://github.com/RomanSlack/Being.git
cd Being
git submodule update --init --recursive

# If InsTaG not cloned as submodule:
mkdir -p extern
git clone --recursive https://github.com/Fictionarry/InsTaG.git extern/InsTaG
```

### 4. Install PyTorch + Python Deps

```bash
export CUDA_HOME=/usr/local/cuda

/opt/conda/envs/being/bin/pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

/opt/conda/envs/being/bin/pip install numpy==1.24.3 plyfile tqdm scipy rich click pyyaml \
    tensorboard lpips trimesh PyMCubes ninja pandas transformers==4.36.2 matplotlib

# pyaudio (needed for wav2vec audio extraction)
apt-get update -qq && apt-get install -y -qq portaudio19-dev ffmpeg ninja-build
/opt/conda/envs/being/bin/pip install pyaudio

# Install Being package
cd /workspace/Being
/opt/conda/envs/being/bin/pip install --no-build-isolation -e .
```

### 5. Build CUDA Extensions

All three must be built with `--no-build-isolation` so they can see PyTorch:

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.0"   # A100. Use "8.6" for RTX 3090, "8.9" for RTX 4090

cd /workspace/Being/extern/InsTaG/submodules/diff-gaussian-rasterization
/opt/conda/envs/being/bin/pip install --no-build-isolation .

cd /workspace/Being/extern/InsTaG/submodules/simple-knn
/opt/conda/envs/being/bin/pip install --no-build-isolation .

cd /workspace/Being/extern/InsTaG/gridencoder
/opt/conda/envs/being/bin/pip install --no-build-isolation .
```

### 6. Build PyTorch3D from Source

No pre-built wheel exists for PyTorch 2.1 + Python 3.9:

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.0"
cd /tmp
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && git checkout v0.7.7
FORCE_CUDA=1 /opt/conda/envs/being/bin/pip install --no-build-isolation .
```

This takes ~15 minutes to compile.

### 7. Download Pre-trained Checkpoints

```bash
cd /workspace/Being
bash scripts/download_checkpoints.sh
# Checkpoints go to output/pretrained/pretrain_{eo,ds,hu,ave,hu_mix}/
```

### 8. Basel Face Model

1. Register at https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
2. Download `01_MorphableModel.mat` (BFM 2009, NOT 2019)
3. Place it at `extern/InsTaG/data_utils/face_tracking/3DMM/01_MorphableModel.mat`
4. Convert:
```bash
cd /workspace/Being/extern/InsTaG/data_utils/face_tracking
/opt/conda/envs/being/bin/python convert_BFM.py
```

### 9. InsTaG Weights (face parsing, EasyPortrait)

```bash
cd /workspace/Being/extern/InsTaG
# Face parsing model
mkdir -p data_utils/face_parsing
# Download 79999_iter.pth (face parsing) — comes with InsTaG prepare.sh

# EasyPortrait model
mkdir -p data_utils/easyportrait
wget -q "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
    -O data_utils/easyportrait/fpn-fp-512.pth
```

## Patch InsTaG for PyTorch 2.x Compatibility

Three patches are required to run InsTaG with PyTorch 2.1:

### Patch 1: C++17 for JIT-compiled extensions

InsTaG's `shencoder` and `gridencoder` use `-std=c++14` but PyTorch 2.x headers require C++17:

```bash
# In extern/InsTaG/shencoder/backend.py and setup.py:
# In extern/InsTaG/gridencoder/backend.py and setup.py:
# Change: -std=c++14 → -std=c++17
sed -i 's/-std=c++14/-std=c++17/g' shencoder/backend.py shencoder/setup.py gridencoder/backend.py gridencoder/setup.py
```

### Patch 2: Make Sapiens normals/depth optional

`scene/dataset_readers.py` crashes if Sapiens geometry priors don't exist. Wrap the normal/depth loading in a guard:

```python
# Around line 287 in scene/dataset_readers.py:
# Change:
#   normal_path = os.path.join(normal_path_candidates[0], ...)
# To:
#   if normal_path_candidates:
#       normal_path = os.path.join(normal_path_candidates[0], ...)
```

Same for `depth_path_candidates` and the `preload` section. See the actual diff in the repo.

### Patch 3: Guard normal/depth loss in train_face.py

`train_face.py` uses Sapiens normals for regularization loss after iteration `warm_step + 2000`. Guard it:

```python
# Around line 199 in train_face.py:
# Change:
#   loss += 0.01 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * ...
# To:
#   if "normal" in viewpoint_cam.talking_dict:
#       loss += 0.01 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * ...
```

Same for the depth loss block.

### Patch 4: Gaussian count cap + bg_prune skip

Without this, training hangs around iteration 6000 when the Gaussian count exceeds ~150K and `bg_prune` (spherical harmonic evaluation on CPU) takes forever:

In `train_face.py`, after `densify_and_prune()`:
```python
MAX_GAUSSIANS = 200000
if gaussians.get_xyz.shape[0] > MAX_GAUSSIANS:
    n_excess = gaussians.get_xyz.shape[0] - MAX_GAUSSIANS
    prune_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
    indices = torch.randperm(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)[:n_excess]
    prune_mask[indices] = True
    gaussians.prune_points(prune_mask)
```

And in the `# bg prune` section, skip if count > 100K:
```python
if n_gaussians > 100000:
    print("[bg_prune] Skipping - too many Gaussians")
else:
    # ... existing bg prune code ...
```

### Patch 5: CUDA error catch in train_mouth.py

`train_mouth.py` occasionally hits `CUDA error: invalid configuration argument` on certain frames. Wrap the render call:

```python
try:
    render_pkg = render_motion_mouth_con(...)
except RuntimeError as e:
    if 'CUDA error' in str(e):
        torch.cuda.empty_cache()
        continue
    raise
```

## Data Preprocessing

### Record a Video

- **Duration**: 5-60 seconds (10 seconds is the sweet spot for few-shot)
- **Resolution**: 512x512 or higher (will be cropped/resized)
- **Content**: Face centered, talking naturally, good lighting
- **Format**: MP4

### Crop and Normalize

```bash
# Crop to center square and normalize to 512x512 @ 25fps
ffmpeg -i input.mp4 -vf "crop=min(iw\,ih):min(iw\,ih):(iw-min(iw\,ih))/2:(ih-min(iw\,ih))/2,scale=512:512,fps=25" \
    -t 10 -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Run InsTaG Preprocessing

```bash
cd /workspace/Being/extern/InsTaG
export CUDA_HOME=/usr/local/cuda
PYTHON=/opt/conda/envs/being/bin/python

# Create data directory
mkdir -p data/yourname

# 1. Extract frames
$PYTHON data_utils/process.py data/yourname/video.mp4

# 2. Face parsing
$PYTHON data_utils/face_parsing/test.py --respath data/yourname/parsing \
    --imgpath data/yourname/gt_imgs --modelpath data_utils/face_parsing/79999_iter.pth

# 3. Extract torso
$PYTHON data_utils/extract_torso.py --datapath data/yourname

# 4. Face tracking (slow — ~20 min on CPU for 250 frames)
$PYTHON data_utils/face_tracking/face_tracker.py --path data/yourname --img_h 512 --img_w 512

# 5. Train/test split
$PYTHON data_utils/train_test_split.py --datapath data/yourname

# 6. Audio features (using wav2vec/esperanto)
$PYTHON data_utils/wav2vec.py --wav data/yourname/aud.wav --save_feats
```

### Create Dummy Data for Missing Optional Steps

If you can't run OpenFace or EasyPortrait, create placeholder data:

**Action Units (au.csv):**
```python
import numpy as np, pandas as pd, glob, os
n_frames = len(glob.glob("data/yourname/gt_imgs/*.jpg"))
cols = [' AU45_r',' AU25_r',' AU01_r',' AU02_r',' AU04_r',' AU05_r',
        ' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',
        ' AU15_r',' AU17_r',' AU20_r',' AU23_r',' AU26_r']
np.random.seed(42)
data = {c: np.random.uniform(0, 1.5, n_frames) for c in cols}
pd.DataFrame(data).to_csv("data/yourname/au.csv", index=False)
```

**Important**: AU values MUST be non-zero random values. All-zero AU45_r causes an infinite loop in viewpoint selection during training (iterations 3000+).

**Teeth masks:**
```python
import numpy as np, os
os.makedirs("data/yourname/teeth_mask", exist_ok=True)
for i in range(n_frames):
    np.save(f"data/yourname/teeth_mask/{i}.npy", np.zeros((512, 512), dtype=np.uint8))
```

## Training

### Full 3-Stage Adaptation

```bash
cd /workspace/Being/extern/InsTaG
export CUDA_HOME=/usr/local/cuda
PYTHON=/opt/conda/envs/being/bin/python
DATASET=data/yourname
WORKSPACE=output/yourname_adapted
PRETRAIN=/workspace/Being/output/pretrained/pretrain_eo  # or pretrain_ave
N_VIEWS=250  # number of frames (250 for 10s video)

# Stage 1: Face (10K iterations, ~10 min on A100)
$PYTHON train_face.py --type face -s $DATASET -m $WORKSPACE \
    --init_num 2000 --densify_grad_threshold 0.0005 \
    --audio_extractor esperanto \
    --pretrain_path ${PRETRAIN}/chkpnt_ema_face_latest.pth \
    --iterations 10000 --sh_degree 1 --N_views $N_VIEWS

# Stage 2: Mouth (10K iterations, ~7 min on A100)
$PYTHON train_mouth.py --type mouth -s $DATASET -m $WORKSPACE \
    --audio_extractor esperanto \
    --pretrain_path ${PRETRAIN}/chkpnt_ema_mouth_latest.pth \
    --init_num 5000 --iterations 10000 --sh_degree 1 --N_views $N_VIEWS

# Stage 3: Fuse (2K iterations, ~2 min on A100)
$PYTHON train_fuse_con.py -s $DATASET -m $WORKSPACE \
    --opacity_lr 0.001 --audio_extractor esperanto \
    --iterations 2000 --sh_degree 1 --N_views $N_VIEWS
```

### Synthesis

```bash
$PYTHON synthesize_fuse.py -s $DATASET -m $WORKSPACE \
    --eval --audio_extractor esperanto --dilate
```

Output videos appear at `$WORKSPACE/test/ours_None/renders/out.mp4`.

## What's Missing for Production Quality

These are listed in priority order — fixing these will dramatically improve output quality:

### 1. Real OpenFace Action Units (HIGH impact)

The dummy random AU values mean blink detection and expression control are meaningless. Real OpenFace AU extraction gives the model actual facial expression signals.

**To fix**: Build OpenFace from source or use the Docker image (`docker/openface.Dockerfile`), then run:
```bash
extern/OpenFace/build/bin/FeatureExtraction -f data/yourname/video.mp4 -out_dir data/yourname/
```
This produces a proper `au.csv` with real AU45 (blink), AU25 (lip part), etc.

### 2. Real Teeth Masks (HIGH impact)

Dummy zero masks mean the model can't distinguish teeth from lips, causing mouth artifacts. Needs `mmcv-full` which won't compile for PyTorch 2.x.

**To fix** (two options):
- **Option A**: Create a separate conda env with PyTorch 1.12 + CUDA 11.3 just for EasyPortrait inference, then copy the masks back.
- **Option B**: Use a different teeth segmentation model that works with modern PyTorch (e.g., BiSeNet or a fine-tuned SegFormer).

### 3. Sapiens Geometry Priors (MEDIUM impact)

Normal and depth maps from Meta's Sapiens provide geometric supervision. Our patches skip these, so the model has less regularization on face geometry.

**To fix**: Requires a separate conda env with Python 3.10 + PyTorch 2.2:
```bash
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
# Install sapiens, run inference, copy normal/depth maps to data dir
cd extern/InsTaG && bash scripts/prepare_sapiens.sh
```

### 4. Wire Up Being CLI (`being prepare/train/generate/serve`)

The Being CLI scaffolding exists but isn't connected to the actual InsTaG pipeline we ran manually. The key files:
- `being/data/pipeline.py` — needs to call the InsTaG preprocessing scripts
- `being/training/adapt.py` — needs correct paths and the patches above
- `being/inference/engine.py` — needs to load the trained model and render frames
- `being/api/server.py` — needs real audio→features→render pipeline

### 5. Real-time Inference Server

The WebSocket streaming server exists at `being/api/server.py` but the actual rendering loop (`stream_avatar()`) is a placeholder. Needs:
- Load the fuse model from checkpoints
- Accept audio chunks via WebSocket
- Extract features in streaming mode
- Render frames with `render_motion_mouth_con`
- Stream back as video frames

## Known Issues

| Issue | Workaround |
|-------|-----------|
| `shencoder`/`gridencoder` JIT fails with C++14 | Change to `-std=c++17` in backend.py |
| Training hangs at iter ~6000 | Gaussian count cap (200K) + bg_prune skip (>100K) |
| Dummy AU data causes infinite loop | Use random non-zero values for AU45_r |
| `mmcv-full` won't compile for PyTorch 2.x | Use dummy teeth masks or separate env |
| `train_mouth.py` CUDA errors on random frames | try/except with `torch.cuda.empty_cache()` |
| numpy 2.x breaks PyTorch 2.1 | Pin `numpy==1.24.3` |
| `transformers` 4.57+ breaks PyTorch 2.1 | Pin `transformers==4.36.2` |
