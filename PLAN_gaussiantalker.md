# Plan: GaussianTalker on A100

**Date**: 2026-02-21
**Goal**: Get GaussianTalker trained on our existing roman dataset and compare output quality vs InsTaG.

---

## Prerequisites

- A100 RunPod instance running (same one or new pod with same template)
- SSH access: `ssh root@<IP> -p <PORT> -i ~/.ssh/runpod_key`
- Existing data at `/workspace/Being/extern/InsTaG/data/roman/` (from previous session)
- Existing conda env `being` with Python 3.9 + PyTorch 2.1 + CUDA 12.1

---

## Step 1: Clone GaussianTalker

```bash
cd /workspace/Being/extern
git clone --recursive https://github.com/cvlab-kaist/GaussianTalker.git
cd GaussianTalker
git submodule update --init --recursive
```

---

## Step 2: Install Dependencies

Most deps already installed from InsTaG work. Need to check what's missing.

```bash
export CUDA_HOME=/usr/local/cuda
PYTHON=/opt/conda/envs/being/bin/python
PIP=/opt/conda/envs/being/bin/pip

# Check what we already have
$PIP list | grep -i "torch\|pytorch3d\|tensorflow\|numpy\|scipy\|tqdm\|rich"

# GaussianTalker-specific deps we may need:
$PIP install tensorflow-gpu==2.8.0 protobuf==3.20.1   # only if using DeepSpeech audio features
$PIP install configargparse                             # GaussianTalker uses this for config

# CUDA extensions — we already built these for InsTaG, but GaussianTalker
# has its own copies in submodules/. Need to build from GaussianTalker's versions:
cd /workspace/Being/extern/GaussianTalker/submodules/diff-gaussian-rasterization
$PIP install --no-build-isolation .

cd /workspace/Being/extern/GaussianTalker/submodules/simple-knn
$PIP install --no-build-isolation .
```

**Note**: GaussianTalker's diff-gaussian-rasterization may be a different fork than InsTaG's (GaussianTalker extends the rasterizer for audio-conditioned deformation). Must build GaussianTalker's version, not reuse InsTaG's.

**Risk**: TensorFlow 2.8.0 + PyTorch 2.1 in the same env could conflict. If so, either:
- Use wav2vec features instead of DeepSpeech (skip TF entirely)
- Install TF in a separate env just for feature extraction

---

## Step 3: Symlink BFM Model

GaussianTalker expects the same Basel Face Model in the same relative path:

```bash
cd /workspace/Being/extern/GaussianTalker
# Check if 3DMM dir exists in data_utils/face_tracking/
ls data_utils/face_tracking/3DMM/

# Symlink from InsTaG (already converted)
ln -s /workspace/Being/extern/InsTaG/data_utils/face_tracking/3DMM/01_MorphableModel.mat \
      data_utils/face_tracking/3DMM/01_MorphableModel.mat
ln -s /workspace/Being/extern/InsTaG/data_utils/face_tracking/3DMM/3DMM_info.npy \
      data_utils/face_tracking/3DMM/3DMM_info.npy

# Also symlink the supporting files if not already in repo
# (keys_info.npy, exp_info.npy, topology_info.npy, sub_mesh.obj — these ship with the repo)
```

---

## Step 4: Prepare Data

### 4a: Copy/symlink dataset

```bash
mkdir -p /workspace/Being/extern/GaussianTalker/data
ln -s /workspace/Being/extern/InsTaG/data/roman \
      /workspace/Being/extern/GaussianTalker/data/roman
```

### 4b: Add vertices to track_params.pt

Our InsTaG track_params.pt has `{id, exp, euler, trans, focal}` but GaussianTalker also needs `vertices`.

```python
# /tmp/add_vertices.py
import torch, sys
sys.path.insert(0, "/workspace/Being/extern/GaussianTalker")
from data_utils.face_tracking.facemodel import Face_3DMM

data_dir = "/workspace/Being/extern/GaussianTalker/data/roman"
params = torch.load(f"{data_dir}/track_params.pt")

if "vertices" in params:
    print("vertices already present, skipping")
else:
    model = Face_3DMM(
        "/workspace/Being/extern/GaussianTalker/data_utils/face_tracking/3DMM",
        100, 79, 100, 34650
    )
    id_expanded = params["id"].expand(params["exp"].shape[0], -1)
    vertices = model.forward_geo(id_expanded, params["exp"])
    params["vertices"] = vertices.detach().cpu()
    torch.save(params, f"{data_dir}/track_params.pt")
    print(f"Added vertices: {vertices.shape}")
```

```bash
$PYTHON /tmp/add_vertices.py
```

### 4c: Handle audio features

**Decision needed**: wav2vec (44-dim, what we have) vs DeepSpeech (29-dim, GaussianTalker default).

**Option A — Use our existing wav2vec features (faster, no TF needed):**
```bash
# Symlink our wav2vec features as the audio file GaussianTalker expects
cd /workspace/Being/extern/GaussianTalker/data/roman
cp aud_eo.npy aud_ds.npy   # or ln -s aud_eo.npy aud_ds.npy
```
Then patch `audio_in_dim` in `scene/deformation.py`:
```python
# Change: audio_in_dim = 29
# To:     audio_in_dim = 44
```

**Option B — Extract DeepSpeech features (cleaner, model was designed for this):**
```bash
# Download DeepSpeech model
cd /workspace/Being/extern/GaussianTalker
mkdir -p data_utils/deepspeech_features
# Download deepspeech-0_1_0-b90017e8.pb (frozen TF graph)
# Run extraction:
$PYTHON data_utils/deepspeech_features/extract_ds_features.py \
    --input data/roman/aud.wav
```
This requires TensorFlow 2.8.0.

**Recommendation**: Try Option A first (fastest). If quality is bad, try Option B.

### 4d: Verify face parsing model

```bash
# GaussianTalker uses the same 79999_iter.pth face parsing model
ls /workspace/Being/extern/GaussianTalker/data_utils/face_parsing/
# If missing, symlink:
ln -s /workspace/Being/extern/InsTaG/data_utils/face_parsing/79999_iter.pth \
      /workspace/Being/extern/GaussianTalker/data_utils/face_parsing/79999_iter.pth
```

---

## Step 5: Train

```bash
cd /workspace/Being/extern/GaussianTalker
export CUDA_HOME=/usr/local/cuda
PYTHON=/opt/conda/envs/being/bin/python

$PYTHON train.py \
    -s data/roman \
    --model_path output/roman_gt \
    --configs arguments/64_dim_1_transformer.py
```

**Expected**: Single-stage training. Time TBD (estimate: 15-30 min on A100 based on InsTaG times).

Monitor for:
- CUDA extension compilation errors (C++14 vs C++17 issue — may need same patch as InsTaG)
- Audio feature dimension mismatches (if using wav2vec with Option A)
- Missing files or wrong paths
- OOM errors (unlikely on A100 80GB)

---

## Step 6: Render / Test

```bash
$PYTHON render.py \
    -s data/roman \
    --model_path output/roman_gt \
    --configs arguments/64_dim_1_transformer.py \
    --iteration 10000 \
    --batch 128
```

Output should appear in `output/roman_gt/`.

### Test with novel audio (if we have a separate wav file):
```bash
# First extract audio features for the novel audio
# Then render with --custom_aud and --custom_wav flags
$PYTHON render.py \
    -s data/roman \
    --model_path output/roman_gt \
    --configs arguments/64_dim_1_transformer.py \
    --custom_aud novel_audio.npy \
    --custom_wav novel_audio.wav \
    --skip_train --skip_test
```

---

## Step 7: Download & Compare

```bash
# From local machine:
scp -P <PORT> -i ~/.ssh/runpod_key \
    "root@<IP>:/workspace/Being/extern/GaussianTalker/output/roman_gt/renders/*.mp4" \
    ~/Being/output/results/gaussiantalker/
```

Compare side-by-side with InsTaG output at `~/Being/output/results/roman_synthesis.mp4`.

---

## Potential Issues & Fallbacks

| Issue | Fix |
|-------|-----|
| CUDA extensions fail to build | Apply same C++17 patch as InsTaG (`-std=c++14` → `-std=c++17`) |
| Audio dim mismatch crash | Check error message, fix `audio_in_dim` in deformation.py |
| TF 2.8 conflicts with PyTorch 2.1 | Skip TF, use wav2vec features (Option A) |
| GaussianTalker's diff-gaussian-rasterization is incompatible | Check if it's a custom fork; may need to build from their submodule specifically |
| track_params.pt vertex shape wrong | Verify vertex count matches what GaussianTalker expects (34,650) |
| Python 3.9 vs 3.7 issues | GaussianTalker says Python 3.7 but should work on 3.9 — watch for deprecation warnings |
| transforms_*.json format mismatch | Compare InsTaG's and GaussianTalker's format; may need a conversion script |
| Training hangs / slow iterations | Check Gaussian count, apply similar cap as InsTaG if needed |

---

## Success Criteria

1. Training completes without errors
2. Rendered output video exists and shows recognizable face
3. Lip sync is at least as good as InsTaG output
4. Rendering speed confirmed >60fps

---

## Time Estimate

| Step | Time |
|------|------|
| Clone + install deps | 15 min |
| Symlink BFM + face parsing | 5 min |
| Prepare data (vertices + audio) | 10 min |
| Training | 15-30 min (estimate) |
| Rendering | 5 min |
| Download + compare | 5 min |
| **Total (no blockers)** | **~1 hour** |
| **Total (with debugging)** | **~2-3 hours** |

---

## Step 8 (Stretch Goal): Diffusion Refinement Pass

If GaussianTalker training works, try adding a 1-step diffusion refinement to boost photorealism.

### The idea

```
Audio → GaussianTalker (130fps, 512x512) → raw 3DGS frame
                                                 ↓
                              StreamDiffusion img2img (1 step, SD-turbo)
                              denoise_strength ~0.2-0.3 (preserve structure, refine texture)
                                                 ↓
                                          Refined photorealistic frame
```

GaussianTalker handles the hard part (3D structure, lip sync, head pose).
The diffusion pass just makes it look **real** — skin texture, lighting subtlety, artifact removal.

### Why it works at real-time speeds

- StreamDiffusion + SD-turbo: **93fps img2img on RTX 4090** (1 denoising step)
- Low denoise strength (0.2-0.3) = minimal structural change, just texture refinement
- Pipeline parallelism: GaussianTalker renders frame N+1 while diffusion refines frame N
- Combined throughput: **~60fps** (bottlenecked by diffusion step)

### Install StreamDiffusion

```bash
PIP=/opt/conda/envs/being/bin/pip

# StreamDiffusion
$PIP install streamdiffusion

# Or from source for latest:
cd /workspace/Being/extern
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
$PIP install -e .

# SD-turbo weights (auto-downloaded from HuggingFace on first use)
# Model: stabilityai/sd-turbo (~3.4GB)
```

### Quick test script

```python
# /tmp/test_diffusion_refine.py
"""Test diffusion refinement on a single GaussianTalker output frame."""
import torch
from PIL import Image
from streamdiffusion import StreamDiffusion

# Init StreamDiffusion with SD-turbo (1-step)
stream = StreamDiffusion(
    "stabilityai/sd-turbo",
    t_index_list=[0],         # single step
    torch_dtype=torch.float16,
)
stream.prepare(
    prompt="photorealistic portrait, high quality skin texture, natural lighting",
    num_inference_steps=1,
    guidance_scale=0.0,        # no CFG needed for turbo
)

# Load a GaussianTalker output frame
frame = Image.open("/path/to/gaussiantalker/output/frame_0000.png")

# Refine
# denoise_strength controls how much the diffusion changes the image
# 0.0 = no change, 1.0 = full regeneration
# Sweet spot for refinement: 0.2-0.3
refined = stream.img2img(frame, strength=0.25)
refined.save("/tmp/refined_frame.png")
print("Done — compare original vs refined")
```

### Risks for diffusion refinement

| Risk | Mitigation |
|------|-----------|
| Identity drift (face changes) | Low denoise strength (0.2-0.3), ControlNet face landmarks, IP-Adapter identity lock |
| Lip sync lost | Very low strength preserves geometry; ControlNet openpose/face forces lip positions |
| Temporal flickering between frames | StreamDiffusion's stochastic similarity filter; consistent seed; temporal smoothing |
| VRAM: GaussianTalker + SD-turbo together | SD-turbo is ~3.4GB fp16; A100 80GB has plenty of room |
| Latency added per frame | At 93fps the overhead is ~11ms/frame — acceptable |

### Quality impact estimate

| Setup | Quality (Tavus=100) |
|-------|-------------------|
| GaussianTalker alone (real data) | 40-50 |
| GaussianTalker + diffusion refinement | **65-80** |
| Tavus | 100 |

The diffusion pass bridges "3D render" → "photorealistic" — that's a 15-25 point jump.

### References

- StreamDiffusion: https://github.com/cumulo-autumn/StreamDiffusion (MIT license, 93fps img2img)
- SD-turbo: https://huggingface.co/stabilityai/sd-turbo (1-step distilled SD)
- ControlNet: https://github.com/lllyasviel/ControlNet (structure preservation)
- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter (identity preservation)
