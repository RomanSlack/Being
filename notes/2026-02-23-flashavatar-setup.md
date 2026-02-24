# FlashAvatar Setup on A100 Pod — 2026-02-23

## Pod Access
```
ssh -i ~/.ssh/runpod_key -p 19870 root@154.54.102.23
```
Pod may change between sessions — verify connection. System Python 3.11, PyTorch 2.4.1, CUDA 12.4, 80GB A100.

## What's Installed

### Conda env: `tracker` (Python 3.9, PyTorch 1.13, CUDA 11.6)
```bash
source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate tracker
```
- pytorch3d 0.7.2, mediapipe 0.10.5, face-alignment, chumpy, insightface
- **numpy pinned to 1.23.5** (chumpy breaks on numpy 2.x — if anything reinstalls numpy, run `pip install numpy==1.23.5`)

### Repos on pod
| Repo | Path | Purpose |
|------|------|---------|
| metrical-tracker | `/workspace/metrical-tracker/` | FLAME per-frame fitting |
| MICA | `/workspace/MICA/` | FLAME identity shape extraction |
| FlashAvatar | `/workspace/FlashAvatar/` | 3DGS avatar training |
| face-parsing.PyTorch | `/workspace/face-parsing/` | BiSeNet face segmentation |
| RobustVideoMatting | `/workspace/RobustVideoMatting/` | Alpha matting |

### FLAME model files
- `generic_model.pkl` → in metrical-tracker/data/FLAME2020/, MICA/data/FLAME2020/, FlashAvatar/flame/
- `FLAME_masks.pkl` → same locations
- `FLAME_texture.npz` → metrical-tracker/data/FLAME2020/
- Source downloads on local machine: `/home/roman/Downloads/FLAME2020/`, `/home/roman/Downloads/FLAME_masks/`, `/home/roman/Downloads/TextureSpace/`

### FlashAvatar CUDA extensions
Built against system Python 3.11 + PyTorch 2.4.1 + CUDA 12.4:
- diff-gaussian-rasterization (from submodules/)
- simple-knn (from submodules/)
- lpips, loguru, scipy installed via pip

### FlashAvatar directory structure
```
/workspace/FlashAvatar/
  metrical-tracker -> /workspace/metrical-tracker  (symlink)
  dataset/roman/
    imgs/      # 512x512 cropped face images (XXXXX.jpg)
    alpha/     # grayscale alpha masks (XXXXX.jpg)
    parsing/   # XXXXX_neckhead.png + XXXXX_mouth.png
  flame/
    generic_model.pkl
    FLAME_masks/FLAME_masks.pkl
```

## What's Running (background on pod)

### 1. metrical-tracker FLAME fitting (PID 36834)
```bash
# Launched as:
cd /workspace/metrical-tracker
nohup python tracker.py --cfg ./configs/actors/roman.yml > /workspace/tracker.log 2>&1 &
```
- Config: `configs/actors/roman.yml` — crop_image=true, 512x512, 25fps
- Input: `/workspace/metrical-tracker/input/roman/video.mp4` (6:43, 1920x1080)
- Identity: `/workspace/metrical-tracker/input/roman/identity.npy` (from MICA)
- Output: `/workspace/metrical-tracker/output/roman/checkpoint/XXXXX.frame`
- **Status at ~00:30 UTC**: Frame 9 of 10,081 (still in keyframe calibration)
- **Expected total time: 4-6 hours**
- Monitor: `tail -1 /workspace/tracker.log | tr '\r' '\n' | tail -1`

### 2. Mask generation (PID 38792)
```bash
nohup python3 /workspace/generate_masks.py > /workspace/masks.log 2>&1 &
```
- Generates alpha (RVM) + parsing (BiSeNet neckhead + mouth) for all 10,081 cropped frames
- Output: `/workspace/FlashAvatar/dataset/roman/{alpha,parsing,imgs}/`
- **Status at ~00:30 UTC**: ~1000/10081, ~1.5 it/s
- **Expected: ~2 hours**
- Monitor: `tail -1 /workspace/masks.log | tr '\r' '\n' | tail -1`

## What's Left To Do

### After tracker finishes:
1. Verify checkpoint count matches frame count:
   ```bash
   ls /workspace/metrical-tracker/output/roman/checkpoint/ | wc -l  # should be ~10080
   ```

2. Verify dataset alignment (frame_delta=1: tracker 00000.frame → image 00001.jpg):
   ```bash
   ls /workspace/FlashAvatar/dataset/roman/imgs/ | wc -l
   ls /workspace/FlashAvatar/dataset/roman/alpha/ | wc -l
   ls /workspace/FlashAvatar/dataset/roman/parsing/ | wc -l  # should be 2x (neckhead + mouth)
   ```

3. Train FlashAvatar (~5 min on A100):
   ```bash
   cd /workspace/FlashAvatar
   python train.py --idname roman
   ```
   Saves checkpoints every 5K iters to `dataset/roman/log/ckpt/`

4. Test render:
   ```bash
   python test.py --idname roman --checkpoint dataset/roman/log/ckpt/chkpnt150000.pth
   ```

## Lessons / Notes for Next Time

### GPU sizing for metrical-tracker
The tracker does **per-frame sequential optimization** (~310 gradient steps × 4 pyramid levels per frame). GPU utilization was ~1% on the A100 — each step is a single 512x512 differentiable render. An RTX 3090 or even 4070 would be nearly the same speed. **Use the cheapest GPU pod for tracker runs** (~$0.50/hr vs $2-3/hr for A100). Only need A100 for the final FlashAvatar training, and even that's only 5 min.

### Practical timeline
- Tracker dataset generation (face crop + landmarks): ~30 min (CPU-bound, ~5 it/s)
- Tracker FLAME fitting: **4-6 hours** (this is the bottleneck, inherently serial)
- Mask generation (alpha + parsing): ~2 hours (GPU inference, parallelizable with tracker)
- FlashAvatar training: ~5 min
- **Strategy**: kick off tracker, go do something else, come back in 5-6h

### Dependency pain points
- **numpy version**: chumpy needs <2.0, insightface installs 2.x. Always re-pin after installing insightface.
- **mediapipe version**: metrical-tracker needs <=0.10.5 (the `mediapipe.python.solutions` import path changed in 0.10.7+)
- **module name conflicts**: face-parsing and RVM both have `model` module. Script at `/workspace/generate_masks.py` handles this by clearing `sys.modules` between loads.
- **conda TOS**: new conda (25.x) requires accepting TOS before creating envs: `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`

### Data flow
```
video.mp4 (1920x1080, 60fps)
  → ffmpeg 25fps → source/XXXXX.png (10081 frames, 1-indexed)
  → face crop 512x512 → images/XXXXX.png
  → BiSeNet → parsing/XXXXX_neckhead.png, XXXXX_mouth.png
  → RVM → alpha/XXXXX.jpg
  → metrical-tracker → checkpoint/XXXXX.frame (0-indexed)
  → FlashAvatar reads: imgs/XXXXX.jpg + alpha + parsing + checkpoint (frame_delta=1)
```
