# Flame Tracker Switch — 2026-02-24

## Why: metrical-tracker was too slow
- metrical-tracker: 310 optimization iters × 4 pyramid levels per frame = ~1.4 min/frame
- 10,081 frames × 1.4 min = ~235 hours (10 days). Killed it at frame 88.

## Replacement: PeizhiYan/flame-head-tracker
- Repo: https://github.com/PeizhiYan/flame-head-tracker
- Installed at `/workspace/flame-head-tracker/`
- Landmark mode: ~1.15 s/frame (with our optimizations), ~3.2 hours total
- Outputs same 100 FLAME expression params + jaw + eyes + camera
- MIT license

## Patches applied to flame-head-tracker
1. **video_utils.py**: Subsample during reading (not after) + resize to 720px max → saves ~50GB RAM
2. **matting_utils.py**: In-place matting (modify frames[] in-place instead of building second list) → fixed OOM at 74%
3. **image_utils.py + fitting_util.py**: `PIL.Image.ANTIALIAS` → `PIL.Image.LANCZOS` (removed in Pillow 10+)
4. **tracker_base.py**:
   - `tex_type: 'BFM'` → `'FLAME'` (avoids needing FLAME_albedo_from_BFM.npz / BFM 2017 model)
   - Skip face parsing in `prepare_intermediate_data_from_image` (not used in landmark-only mode)
   - Reduce optimization iterations 200 → 100
   - Remove visualization rendering block (skip prepare_batch_visualized_results)
   - Remove img/img_aligned/parsing/parsing_aligned from ret_dict
5. **DECA config** (`submodules/decalib/deca_utils/config.py`): tex_type BFM → FLAME
6. **chumpy** (system package):
   - `inspect.getargspec` → `inspect.getfullargspec` (Python 3.11)
   - `from numpy import bool, int, float...` → `from numpy import nan, inf` (numpy 1.26.4)
7. **head_template.obj**: Replaced MICA's (no UVs) with metrical-tracker's `head_template_mesh.obj`

## Container memory limit
- Pod has 2TB physical RAM but **233GB cgroup limit**
- Original: 58GB frames + growing matted copy → OOM at ~110GB
- Fix: in-place matting keeps memory flat at ~15GB (720px) or ~64GB (1080p)

## Output format conversion
- flame-head-tracker: `{frame_id}.npz` (axis-angle rotations, Euler camera, 256px K matrix)
- FlashAvatar: `{frame_id:05d}.frame` (6D rotations, rotation matrix camera, 512px K matrix)
- Script: `/workspace/convert_to_frame.py`
- Key conversions:
  - Jaw: axis-angle (3) → 6D rotation (6)
  - Eyes: axis-angle (6) → 6D rotation (12)
  - Eyelids: from MediaPipe blendshape scores [9]=eyeBlinkLeft, [10]=eyeBlinkRight
  - Camera: Euler (yaw,pitch,roll) → rotation matrix (3×3)
  - K: scale from 256→512 (multiply fx,fy,cx,cy by 2)

## Mouth mask bug fix
- See `notes/2026-02-24-mouth-mask-bug.md`
- BiSeNet label 10 (nose) → 11 (mouth)
- Regenerated via `/workspace/fix_mouth_masks.py` — completed, all 10081 masks

## Current status
- Tracker running in tmux session on pod: `tmux attach -t tracker`
- Speed: ~1.15 s/frame, ~3.2h total for 10080 frames
- Output: `/workspace/flame-head-tracker/output/roman/video/{0..10079}.npz`
- After tracker finishes:
  1. `python3 /workspace/convert_to_frame.py` → writes to `/workspace/metrical-tracker/output/roman/checkpoint/`
  2. `cd /workspace/FlashAvatar && python train.py --idname roman` (~5 min)
  3. `python test.py --idname roman --checkpoint dataset/roman/log/ckpt/chkpnt150000.pth`
