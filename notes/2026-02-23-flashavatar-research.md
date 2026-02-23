# FlashAvatar Setup Research — 2026-02-23

## Executive Summary

FlashAvatar (CVPR 2024) uses FLAME-driven 3D Gaussian Splatting for head avatars. It achieves 300+ FPS at 512x512. The pipeline is: **monocular video -> metrical-tracker (FLAME fitting) -> FlashAvatar training -> real-time rendering**. The main challenge for us is that FlashAvatar expects tracking data from a specific tracker called **metrical-tracker** (by Zielon/MICA team), which outputs `.frame` files in a particular format. We have two options for the tracking stage: use metrical-tracker directly, or use **flame-head-tracker** (Peizhi Yan) and write a conversion script.

---

## 1. FlashAvatar (USTC3DV/FlashAvatar-code)

### 1.1 Dependencies

**environment.yml specifies:**
- Python 3.7.13
- PyTorch 1.12.1
- CUDA Toolkit 11.6
- torchvision 0.13.1, torchaudio 0.12.1
- plyfile 0.8.1

**pip packages:**
- `submodules/diff-gaussian-rasterization` (custom 3DGS CUDA extension)
- `submodules/simple-knn` (KNN CUDA extension)
- scipy, chumpy, scikit-image, opencv-python, ninja, lpips, loguru

**Also requires:**
- PyTorch3D (installed via conda)
- yacs (for config parsing, comes from metrical-tracker's config)

**CRITICAL COMPATIBILITY NOTE:** The environment specifies Python 3.7 + PyTorch 1.12.1 + CUDA 11.6. Our pod runs Python 3.11 + PyTorch 2.4.1 + CUDA 12.4. We will need to either:
1. Create a separate conda env matching their spec, OR
2. Port forward to modern PyTorch (likely feasible since the code is straightforward 3DGS)

### 1.2 Required Model Files

FlashAvatar needs these files in its `flame/` directory (most come from the repo itself):

```
flame/
  FlameMesh.obj                         # INCLUDED in repo
  head_template_mesh.obj                # INCLUDED in repo
  landmark_embedding.npy                # INCLUDED in repo
  __init__.py                           # INCLUDED in repo
  flame_mica.py                         # INCLUDED in repo
  lbs.py                                # INCLUDED in repo
  mica_flame_config.py                  # INCLUDED in repo
  blendshapes/
    l_eyelid.npy                        # INCLUDED in repo
    r_eyelid.npy                        # INCLUDED in repo
  mediapipe/
    landmarks.py                        # INCLUDED in repo
    mediapipe_landmark_embedding.npz    # INCLUDED in repo
  FLAME_masks/
    FLAME_masks.pkl                     # MUST DOWNLOAD from FLAME website
```

**Additionally needs (referenced in flame_mica.py via config):**
```
flame/generic_model.pkl                 # MUST DOWNLOAD from FLAME website (FLAME 2020)
```

The `generic_model.pkl` is the core FLAME 2020 model. Download from https://flame.is.tue.mpg.de/ after registration.

The `FLAME_masks.pkl` is downloaded from https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip

### 1.3 Data Directory Structure

FlashAvatar expects two parallel directory trees:

```
FlashAvatar-code/
  dataset/
    <idname>/
      imgs/
        00001.jpg          # Video frames, 5-digit zero-padded, starting from 00001
        00002.jpg          # (NOTE: frame_delta=1, so frame 0 in tracker = image 00001)
        ...
      alpha/
        00001.jpg          # Alpha/matting masks (same naming as imgs)
        00002.jpg
        ...
      parsing/
        00001_neckhead.png # Head+neck segmentation mask
        00001_mouth.png    # Mouth region mask
        00002_neckhead.png
        00002_mouth.png
        ...
      log/                 # Created during training
        train/             # Training visualizations
        ckpt/              # Checkpoints saved here

  metrical-tracker/
    output/
      <idname>/
        checkpoint/
          00000.frame      # Per-frame FLAME tracking results (torch.save pickle)
          00001.frame
          ...
```

### 1.4 Metrical-Tracker `.frame` File Format

Each `.frame` file is a `torch.save()` dictionary with this exact structure:

```python
{
    'flame': {
        'exp':     np.ndarray shape (1, 100),  # FLAME expression coefficients
        'shape':   np.ndarray shape (1, 300),  # FLAME shape coefficients (canonical, same across frames)
        'tex':     np.ndarray shape (1, 140),  # FLAME texture coefficients
        'sh':      np.ndarray shape (...),     # Spherical harmonics lighting
        'eyes':    np.ndarray shape (1, 12),   # Eye pose (6D rotation x2, left+right)
        'eyelids': np.ndarray shape (1, 2),    # Eyelid parameters (left, right)
        'jaw':     np.ndarray shape (1, 6),    # Jaw pose (6D rotation)
    },
    'camera': {
        'R':  np.ndarray,  # Camera rotation (PyTorch3D format)
        't':  np.ndarray,  # Camera translation
        'fl': np.ndarray,  # Focal length
        'pp': np.ndarray,  # Principal point
    },
    'opencv': {
        'R': np.ndarray shape (1, 3, 3),  # OpenCV rotation matrix
        't': np.ndarray shape (1, 3),     # OpenCV translation vector
        'K': np.ndarray shape (1, 3, 3),  # OpenCV intrinsic matrix
    },
    'img_size': np.ndarray shape (2,),    # [height, width]
    'frame_id': int,
    'global_step': int,
}
```

**CRITICAL: FLAME params use 6D rotation representation (rotation_6d from PyTorch3D), NOT axis-angle.** This is key for conversion.

### 1.5 What FlashAvatar Reads from `.frame` Files

From `scene/__init__.py` (Scene_mica class), for each frame it reads:

```python
payload = torch.load(ckpt_path)
flame_params = payload['flame']

# Shape (read once from frame 0, shared across all frames)
shape_param = torch.as_tensor(flame_params['shape'])      # (1, 300)

# Per-frame parameters
exp_param = torch.as_tensor(flame_params['exp'])           # (1, 100)
eyes_pose = torch.as_tensor(flame_params['eyes'])          # (1, 12) -- 6D rotation
eyelids   = torch.as_tensor(flame_params['eyelids'])       # (1, 2)
jaw_pose  = torch.as_tensor(flame_params['jaw'])           # (1, 6)  -- 6D rotation

# Camera (OpenCV format)
opencv = payload['opencv']
w2cR = opencv['R'][0]     # (3, 3)
w2cT = opencv['t'][0]     # (3,)

# Image size + intrinsics (from frame 0 only)
orig_w, orig_h = payload['img_size']
K = payload['opencv']['K'][0]
fl_x = K[0, 0]
fl_y = K[1, 1]
```

**The frame numbering has a delta of 1:** metrical-tracker frame `00000.frame` corresponds to image `00001.jpg`. This is because metrical-tracker skips the first frame.

### 1.6 The Deformation Model (MLP on FLAME mesh)

The `Deform_Model` in `src/deform_model.py`:
- Loads FLAME via `flame/generic_model.pkl`
- Uses FLAME mesh UV coordinates from `flame/FlameMesh.obj`
- Uses FLAME masks from `flame/FLAME_masks/FLAME_masks.pkl` (to exclude boundary vertices)
- Rasterizes FLAME vertices to a 128x128 UV space
- An MLP takes positional-encoded UV coordinates + condition vector as input
- Condition vector = `cat(exp[100], jaw_pose[6], eyes_pose[12], eyelids[2])` = **120 dims**
- MLP input: positional_encoding(49 dims for freq=8) + 120 condition = 169 dims
- MLP output: 10 dims (3 position offset + 4 rotation quaternion delta + 3 scale coefficient)
- MLP architecture: 6 hidden layers, 256 hidden dim, ReLU activations
- Learning rate: 1e-4 (Adam)

### 1.7 Training

```bash
python train.py --idname <id_name>
```

Key training parameters:
- **Default iterations:** 150,000 (from `OptimizationParams`)
- **LPIPS perceptual loss** kicks in after iteration 15,000
- **Mouth region loss** has 40x weight multiplier
- **Checkpoints** saved every 5,000 iterations to `dataset/<idname>/log/ckpt/chkpntXXXXX.pth`
- **SH degree** increased every 500 iterations up to max degree 3
- **Max training frames:** 2,000 (randomly sampled from training set)
- **Train/test split:** last 500 frames reserved for test, last 50 for eval

### 1.8 Testing/Rendering

```bash
python test.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt150000.pth
```

Output: AVI video at 25fps, side-by-side GT vs rendered, saved to `dataset/<idname>/log/test.avi`

### 1.9 Performance Claims
- Training: "minutes" (not specified exactly, but the paper claims ~5 minutes on RTX 3090)
- Rendering: 300+ FPS at 512x512

---

## 2. Flame Head Tracker (PeizhiYan/flame-head-tracker)

### 2.1 Dependencies

**Requires Python 3.10 specifically** (higher numpy versions cause problems).

```bash
conda create --name tracker -y python=3.10
conda activate tracker
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Key requirements.txt packages:
- pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8
- mediapipe==0.10.15
- open3d==0.17.0
- face-alignment==1.4.1
- kornia==0.7.3
- yacs==0.1.8
- roma==1.5.0
- numpy==1.23.4

**COMPATIBILITY NOTE:** Requires CUDA 11.7 compiler even for PyTorch 2.0.1. Our pod has CUDA 12.4, so we'd need a separate conda env or adjust installation.

### 2.2 Required Model Downloads

```
models/
  FLAME2020/
    generic_model.pkl        # From FLAME website (same as FlashAvatar needs)
  landmark_embedding.npy     # FLAME landmark embedding
  FLAME_albedo_from_BFM.npz  # BFM-to-FLAME texture
  deca_model.tar             # DECA pretrained model
  mica.tar                   # MICA pretrained model
  face_landmarker.task        # MediaPipe face landmarker (renamed to face_landmarker_v2_with_blendshapes.task)
  79999_iter.pth             # Face parsing model (BiSeNet, already in repo)
  head_template.obj          # Head template mesh
  ear_landmarker.pth         # Optional: ear landmark detector
```

Download FLAME 2020 via:
```bash
./download_FLAME.sh  # prompts for FLAME website credentials
```

### 2.3 Running on a Monocular Video

From `Example_2_video_tracking.ipynb`:

```python
from tracker_video import track_video

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth',       # optional
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './models/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'use_matting': True,
    'optimize_fov': False,
    'device': 'cuda:0',

    # Video settings
    'original_fps': 25,          # our video is 25fps
    'subsample_fps': 25,         # keep all frames
    'photometric_fitting': True, # better quality, slower
    'video_path': './path/to/video.mp4',
    'save_path': './output',
    'batch_size': 16,            # A100 can probably do 32
    'realign': True,
}

track_video(tracker_cfg)
```

**Processing speed:** ~1.9s/frame in photometric mode on 4090. For our 6:43 video at 25fps = 10,075 frames, this would take ~5.3 hours on a 4090. Possibly faster on A100.

### 2.4 Output Format

Per-frame `.npz` files saved to `output/<video_name>/<frame_id>.npz`:

```python
{
    'shape':    np.ndarray (1, 300),   # FLAME shape coefficients (canonical, same all frames)
    'exp':      np.ndarray (1, 100),   # FLAME expression coefficients
    'head_pose': np.ndarray (1, 3),    # Head pose (axis-angle, zeroed out -- camera used instead)
    'jaw_pose': np.ndarray (1, 3),     # Jaw pose (axis-angle)
    'tex':      np.ndarray (1, 50),    # Texture coefficients (first 50 only)
    'light':    np.ndarray (1, 9, 3),  # SH lighting
    'neck_pose': np.ndarray (1, 3),    # Neck pose (unused)
    'eye_pose': np.ndarray (1, 6),     # Eye pose (axis-angle, 3 left + 3 right)
    'cam':      np.ndarray (1, 6),     # Camera pose (yaw, pitch, roll, x, y, z)
    'fov':      np.ndarray (1,),       # Field of view

    # Additional data
    'img':              np.ndarray,    # Original image
    'img_aligned':      np.ndarray,    # Aligned/cropped image
    'parsing':          np.ndarray,    # Face parsing map
    'parsing_aligned':  np.ndarray,    # Aligned parsing
    'lmks_68':          np.ndarray,    # 68 facial landmarks
    'lmks_eyes':        np.ndarray,    # 10 eye landmarks
    'lmks_ears':        np.ndarray,    # 20 ear landmarks (if ear_landmarker used)
    'blendshape_scores': np.ndarray,   # 52 MediaPipe blendshape scores
    'img_rendered':     np.ndarray,    # Rendered visualization
    'mesh_rendered':    np.ndarray,    # Mesh rendering
}
```

### 2.5 Conversion to FlashAvatar's Metrical-Tracker Format

**The formats are NOT directly compatible.** Key differences:

| Parameter | flame-head-tracker | metrical-tracker (FlashAvatar) |
|-----------|-------------------|-------------------------------|
| shape | (1, 300) | (1, 300) -- SAME |
| expression | (1, 100) | (1, 100) -- SAME |
| jaw_pose | (1, 3) axis-angle | (1, 6) 6D rotation |
| eye_pose | (1, 6) axis-angle (3+3) | (1, 12) 6D rotation (6+6) |
| eyelids | NOT OUTPUT | (1, 2) -- MISSING |
| camera | (1, 6) custom + fov | opencv R(3x3), t(3), K(3x3) |
| texture | (1, 50) | (1, 140) |
| lighting | (1, 9, 3) | spherical harmonics |
| file format | .npz (numpy) | .frame (torch.save pickle) |

**Conversion needed:**
1. **Jaw pose:** axis-angle (3) -> 6D rotation (6) via `pytorch3d.transforms.axis_angle_to_matrix` then `matrix_to_rotation_6d`
2. **Eye pose:** axis-angle (3+3) -> 6D rotation (6+6), same conversion
3. **Eyelids:** Need to extract from blendshape_scores or set to zero
4. **Camera:** Reconstruct OpenCV R, t, K from the tracker's cam parameters and fov
5. **Texture:** Pad from 50 to 140 coefficients (or FlashAvatar only uses what it uses)
6. **File format:** Re-save as torch pickle `.frame` files
7. **Frame numbering:** Align 0-indexed `.npz` to 00000.frame convention

**Eyelid issue:** flame-head-tracker does not directly output `eyelids` in the same format as metrical-tracker. The metrical-tracker uses custom eyelid blendshapes (`l_eyelid.npy`, `r_eyelid.npy`) that are vertex offsets applied after FLAME. The flame-head-tracker does have MediaPipe blendshape_scores that include eye blink values, but these need conversion. Alternatively, we could set eyelids to zero and accept slightly worse eye region.

---

## 3. Preprocessing Pipeline for FlashAvatar (What We Need)

### 3.1 Image Extraction
Extract video frames to `dataset/<idname>/imgs/`:
```bash
ffmpeg -i video.mp4 -qscale:v 2 dataset/<idname>/imgs/%05d.jpg
```
Note: FlashAvatar expects 1-indexed frames (00001.jpg, 00002.jpg, ...) because of frame_delta=1.

### 3.2 Alpha/Matting
Need background matting for each frame saved to `dataset/<idname>/alpha/`:
- Same filenames as imgs (00001.jpg, etc.)
- Grayscale or single-channel mask
- Options: RobustVideoMatting (flame-head-tracker already uses this), MODNet, BackgroundMattingV2
- The flame-head-tracker includes matting via `matting_video_frames()` using RobustVideoMatting

### 3.3 Semantic Parsing
Need face parsing masks saved to `dataset/<idname>/parsing/`:
- `<frame>_neckhead.png` -- binary mask of head+neck region
- `<frame>_mouth.png` -- binary mask of mouth region
- Generated from BiSeNet face parsing model (flame-head-tracker includes `79999_iter.pth`)
- The flame-head-tracker outputs `parsing` and `parsing_aligned` per frame which could be converted

### 3.4 FLAME Tracking
Run either metrical-tracker or flame-head-tracker, output to `metrical-tracker/output/<idname>/checkpoint/`

---

## 4. DiffPoseTalk (DiffPoseTalk/DiffPoseTalk)

### 4.1 What It Does
DiffPoseTalk generates audio-driven FLAME animation sequences. It takes audio + shape coefficient + style feature and outputs per-frame FLAME expression and pose coefficients. This is the audio-to-animation stage -- it does NOT render anything itself for 3DGS; it generates FLAME params that can drive FlashAvatar.

### 4.2 Pretrained Models

Two model sets available on Google Drive (https://drive.google.com/drive/folders/1pOwtK95u8O1qG_CiRdD8YcvuKSlFEk-b):

| Set | Style Encoder | Denoising Network | Head Motion |
|-----|--------------|-------------------|-------------|
| 1 | head-L4H4-T0.1-BS32 (@26k) | head-SA-hubert-WM (@110k) | YES |
| 2 | L4H4-T0.1-BS32 (@34k) | SA-hubert-WM (@100k) | NO |

Models go in: `experiments/DPT/<exp_name>/checkpoints/iter_XXXXXXX.pt`
Style encoder goes in: `experiments/SE/<exp_name>/checkpoints/iter_XXXXXXX.pt`

### 4.3 FLAME Param Format

DiffPoseTalk uses **FLAME 2020** with:
- **n_shape:** 100 (NOT 300 like metrical-tracker)
- **n_exp:** 50 (NOT 100 like metrical-tracker)
- **Rotation:** axis-angle representation
- **Pose:** 6 values = global_rot(3) + jaw(3), with y,z of jaw dropped in practice -> 4 dims used

Output `coef_dict`:
```python
{
    'shape': (n_rep, L, 100),   # FLAME shape, 100 coeffs
    'exp':   (n_rep, L, 50),    # FLAME expression, 50 coeffs
    'pose':  (n_rep, L, 6),     # global_rot(3) + jaw_rot(3), axis-angle
}
```

### 4.4 Running Inference

**Step 1: Extract style from a reference motion sequence**
```bash
python extract_style.py \
    --exp_name head-L4H4-T0.1-BS32 --iter 26000 \
    -c <motion_seq.npz>  \  # needs 'exp' and 'pose' keys
    -o <output_name.npy> \
    -s 0                     # starting frame
```

The `.npz` for style extraction needs `exp` (N, 50) and `pose` (N, 6) in axis-angle format.

**Step 2: Generate animation**
```bash
python demo.py \
    --exp_name head-SA-hubert-WM --iter 110000 \
    -a <audio.wav> \
    -c <shape_template.npy> \  # shape coefficients (100,)
    -s <style_feature.npy> \
    -o <output.mp4> \
    -n 1 \                     # number of repetitions
    --dtr 0.99                 # dynamic thresholding for quality
```

### 4.5 Compatibility with FlashAvatar

**NOT directly compatible.** Key differences:

| | DiffPoseTalk | FlashAvatar |
|--|-------------|-------------|
| shape params | 100 | 300 |
| expression params | 50 | 100 |
| jaw rotation | axis-angle (3) | 6D rotation (6) |
| eye pose | NOT output | 6D rotation (12) |
| eyelids | NOT output | scalar (2) |
| head pose | axis-angle global (3) | camera R,t instead |

**To drive FlashAvatar with DiffPoseTalk output, we need:**
1. Pad shape from 100 to 300 (zero-pad)
2. Pad expression from 50 to 100 (zero-pad)
3. Convert jaw pose from axis-angle to 6D rotation
4. Generate camera matrices (use fixed camera from training)
5. Set eye_pose to default (identity 6D rotation for both eyes)
6. Set eyelids to 0
7. Construct `.frame` files in metrical-tracker format

This is feasible but will give limited expressiveness (no eye movement or blinks from audio).

---

## 5. Practical Plan: Training FlashAvatar on Our 6:43 Video

### 5.1 Option A: Use metrical-tracker directly (recommended for best compatibility)

**Steps:**
1. Clone metrical-tracker on the pod
2. Install dependencies (needs MICA model, FLAME 2020, etc.)
3. Run MICA to get `identity.npy` shape code
4. Run metrical-tracker on our video
5. Extract frames with ffmpeg
6. Generate alpha masks (RobustVideoMatting or similar)
7. Generate parsing masks (BiSeNet)
8. Clone FlashAvatar, install deps
9. Organize directory structure
10. Train

**Pros:** Direct compatibility, no conversion needed
**Cons:** Metrical-tracker setup is complex (MICA dependency, specific FLAME files), older Python/PyTorch

### 5.2 Option B: Use flame-head-tracker + conversion script

**Steps:**
1. Install flame-head-tracker on the pod
2. Run video tracking (outputs .npz per frame)
3. Write conversion script: .npz -> .frame format
4. Extract frames, alpha, parsing from tracker output
5. Clone FlashAvatar, install deps
6. Train

**Conversion script pseudocode:**
```python
import torch
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

for frame_id in range(num_frames):
    npz = np.load(f'output/{video_name}/{frame_id}.npz')

    # Convert jaw_pose: AA(3) -> 6D(6)
    jaw_aa = torch.tensor(npz['jaw_pose'])  # (1, 3)
    jaw_mat = axis_angle_to_matrix(jaw_aa)  # (1, 3, 3)
    jaw_6d = matrix_to_rotation_6d(jaw_mat) # (1, 6)

    # Convert eye_pose: AA(3+3) -> 6D(6+6)
    eye_aa = torch.tensor(npz['eye_pose'])  # (1, 6)
    left_eye = axis_angle_to_matrix(eye_aa[:, :3])
    right_eye = axis_angle_to_matrix(eye_aa[:, 3:])
    eye_6d = torch.cat([matrix_to_rotation_6d(left_eye),
                        matrix_to_rotation_6d(right_eye)], dim=1)  # (1, 12)

    # Eyelids: extract from blendshapes or zero
    eyelids = np.zeros((1, 2), dtype=np.float32)

    # Camera: reconstruct OpenCV format from tracker output
    # flame-head-tracker uses cam=(yaw,pitch,roll,x,y,z) + fov
    # Need to construct R(3x3), t(3), K(3x3)
    cam = npz['cam'][0]  # (6,)
    fov = npz['fov'][0]  # scalar
    # ... camera conversion logic ...

    # Texture: pad from 50 to 140
    tex_50 = npz['tex']  # (1, 50)
    tex_140 = np.zeros((1, 140), dtype=np.float32)
    tex_140[:, :50] = tex_50

    frame_dict = {
        'flame': {
            'exp': npz['exp'],
            'shape': npz['shape'],
            'tex': tex_140,
            'sh': npz['light'].reshape(-1),  # match format
            'eyes': eye_6d.numpy(),
            'eyelids': eyelids,
            'jaw': jaw_6d.numpy(),
        },
        'opencv': {
            'R': R_opencv,
            't': t_opencv,
            'K': K_opencv,
        },
        'img_size': np.array([512, 512]),
        'frame_id': frame_id,
        'global_step': 0,
    }

    torch.save(frame_dict, f'checkpoint/{frame_id:05d}.frame')
```

**Pros:** flame-head-tracker is newer, better documented, includes matting/parsing
**Cons:** Camera conversion is tricky, eyelid data may be missing, potential alignment issues

### 5.3 Option C: Use existing GaussianTalker preprocessing and convert

We already have preprocessed data for GaussianTalker. If GaussianTalker used FLAME tracking (e.g., via 3DDFA or similar), we might be able to convert. However, GaussianTalker uses BFM 2009 face model, NOT FLAME, so this data is NOT reusable for FlashAvatar.

### 5.4 Recommended Approach

**Option B (flame-head-tracker)** is likely the most practical because:
- It's the most modern and well-documented
- It already includes video matting and face parsing
- The conversion to metrical-tracker format is straightforward (mainly rotation representation conversion)
- The camera conversion is the trickiest part but solvable

### 5.5 Estimated Timeline on A100

| Step | Time Estimate |
|------|--------------|
| Install flame-head-tracker deps | 30 min |
| Download FLAME/DECA/MICA models | 15 min |
| Run video tracking (10,075 frames, photometric) | 3-5 hours |
| Write + run conversion script | 1 hour |
| Extract frames + organize dirs | 15 min |
| Install FlashAvatar deps | 30 min |
| Train FlashAvatar (150k iterations) | ~30-60 min |
| **Total** | **~6-8 hours** |

---

## 6. Key Files Reference

### FlashAvatar repo structure (what matters):
- `/train.py` -- training entry point
- `/test.py` -- evaluation entry point
- `/scene/__init__.py` -- data loading (Scene_mica class)
- `/scene/cameras.py` -- Camera class
- `/scene/gaussian_model.py` -- GaussianModel (3DGS)
- `/src/deform_model.py` -- Deform_Model (FLAME-conditioned MLP)
- `/flame/flame_mica.py` -- FLAME model wrapper
- `/flame/mica_flame_config.py` -- FLAME config (paths, param counts)
- `/gaussian_renderer/__init__.py` -- diff-gaussian-rasterization render()
- `/arguments/__init__.py` -- ModelParams, OptimizationParams, PipelineParams

### flame-head-tracker repo structure (what matters):
- `/Example_2_video_tracking.ipynb` -- main entry for video tracking
- `/tracker_video.py` -- video tracking pipeline
- `/tracker_base.py` -- core Tracker class
- `/requirements.txt` -- dependencies
- `/download_FLAME.sh` -- download FLAME 2020 model

### DiffPoseTalk repo structure (what matters):
- `/demo.py` -- inference entry point
- `/extract_style.py` -- style feature extraction
- `/models/flame.py` -- FLAME config (n_shape=100, n_exp=50)
- `/utils/common.py` -- coef_dict format (get_coef_dict, coef_dict_to_vertices)
- `/setup/fetch_data.sh` -- download FLAME + masks
