# FlashAvatar Training Log — 2026-02-24

## Goal
Train a FlashAvatar (FLAME-based 3DGS) model on a reference video of Roman, then render a test video to verify quality.

## What is FlashAvatar's actual pipeline?
FlashAvatar expects data from **metrical-tracker** — a FLAME-based face tracker that outputs `.frame` files containing:
- FLAME params: expression (100), shape (300), jaw (6D rot), eyes (6D rot), eyelids (2)
- Camera: R (3x3 rotation matrix), t (3 translation), K (3x3 intrinsics)
- All in OpenCV convention (Y-down, Z-forward)

The intended pipeline is:
```
Video → metrical-tracker → .frame files → FlashAvatar train.py
```
No conversion scripts needed. metrical-tracker's output is FlashAvatar's native input format.

## What we did (and why it failed)

### Mistake: Bypassing metrical-tracker with flame-head-tracker
- metrical-tracker was too slow: ~1.4 min/frame × 10,080 frames = 10 days
- Switched to PeizhiYan/flame-head-tracker: ~1.15 s/frame, 3.2 hours total
- But flame-head-tracker outputs `.npz` with different conventions:
  - Camera as Euler angles (not rotation matrices)
  - Different coordinate system (needed flip + invert)
  - Axis-angle rotations (not 6D)
- Wrote `convert_to_frame.py` to bridge the formats
- **This custom conversion was the source of all our camera problems**

### Camera bugs from the conversion
1. **Blank render** (Issue 6): Wrong sign convention in R matrix. R diagonal was [+,+,+] instead of [+,-,-]. Fixed by replicating tracker's projection pipeline (flip Y,Z + invert).
2. **Camera jitter** (Issue 7): Even after fixing signs, the render showed the camera "spazzing". Added Gaussian temporal smoothing (sigma=3.0) — still not enough.
3. **Still shaking at 55K** (Issue 8): Smoothed camera STILL produced visible shaking. 270 out of 500 consecutive frames had translation jumps > 0.02. The conversion is fundamentally unreliable.

### Other issues encountered along the way
- **OOM at 74%** during matting: Container cgroup limit = 233GB. Fixed with in-place matting.
- **PIL.Image.ANTIALIAS**: Removed in Pillow 10+. Replaced with LANCZOS in 4 files.
- **Mouth mask label**: BiSeNet label 10 (nose) → 11 (mouth). Regenerated all masks.
- **Video too long**: 7 minutes is massive overkill. FlashAvatar only needs 300-500 frames (12-20 seconds).

### Training runs (all abandoned)
1. **Run 1-2**: Camera convention wrong → blank render
2. **Run 3**: Camera fixed but jittery → visible shaking
3. **Run 4**: Temporal smoothing applied → still shaking. Killed.

## Lesson learned
**Don't bypass the intended pipeline.** The custom conversion between flame-head-tracker and FlashAvatar introduced subtle bugs that were extremely hard to debug. Two different tracker codebases with different coordinate conventions, rotation representations, and camera models made reliable conversion nearly impossible.

## Fresh start — SUCCESS (2026-02-25)

### What we did
1. **New 30s recording** at 1080p/60fps (recording script: `notes/recording-script-30s.md`)
2. **metrical-tracker** on the new video: 838 frames at 25fps, completed in ~2h50m (~12s/frame on A100, NOT 1.4min as originally estimated)
3. **MICA identity** regenerated for the new video
4. **Masks** generated (RVM alpha + BiSeNet parsing) — 9 minutes for 839 images
5. **FlashAvatar training** with test_num=100 (738 train, 100 test): 150K iterations, ~45 min total

### Results
- **Camera: FIXED.** No blank render, no conversion bugs. Native metrical-tracker output works perfectly.
- **Face quality: Good.** Recognizable, expressions tracked well, smooth between frames.
- **Minor jitter:** metrical-tracker estimates per-frame camera pose (doesn't know camera was static), causes mild whole-bust shifting. Not a showstopper but noticeable.
- **No neck/shoulders:** This is a fundamental FlashAvatar limitation — FLAME mesh only covers face + scalp. The rendered avatar is a floating head.

### Conclusion
FlashAvatar works as a research demo but isn't practical for a real avatar product:
- No neck, shoulders, or body — just a floating head on the FLAME mesh
- No built-in audio pipeline — it's a renderer only, needs a separate audio→FLAME model
- The camera jitter from per-frame estimation is cosmetically annoying

**Decision: Return to GaussianTalker** which renders the full frame (head + shoulders + hair + background) and has an end-to-end audio→video pipeline. See `notes/2026-02-25-next-steps.md`.

### What we keep from this work
- 30s recording and recording script
- 838 frames of FLAME tracking data (backed up locally: `data/metrical-tracker-backup/roman/checkpoint/`)
- Knowledge of metrical-tracker pipeline (much faster than estimated on A100)
- FlashAvatar model as a comparison baseline

## Key files (on pod)
- `/workspace/flame-head-tracker/` — tracker with all patches (no longer using)
- `/workspace/metrical-tracker/` — original tracker, used for fresh start
- `/workspace/FlashAvatar/` — training code, model at `dataset/roman/log/ckpt/chkpnt150000.pth`
- `/workspace/convert_to_frame.py` — custom conversion (source of bugs, no longer using)

## Key files (local)
- `assets/test_audio.wav` — 19s benchmark audio (16kHz mono)
- `assets/flashavatar_test_15k.avi` — test render from 15K (camera bugs, pre-smoothing)
- `assets/flashavatar_test_55k_smoothed.avi` — test render from 55K (smoothed but still shaking)
- `assets/flashavatar_test_35k_fresh.avi` — fresh start, 35K (first clean render)
- `assets/flashavatar_test_85k_fresh.avi` — fresh start, 85K
- `assets/flashavatar_test_150k_fresh.avi` — fresh start, 150K (final)

## Related notes
- `notes/2026-02-24-flame-tracker-switch.md` — details on flame-head-tracker patches
- `notes/2026-02-24-mouth-mask-bug.md` — BiSeNet label fix
- `notes/2026-02-23-flashavatar-setup.md` — initial FlashAvatar setup
- `notes/2026-02-25-next-steps.md` — plan for GaussianTalker round 2
- `notes/recording-script-30s.md` — condensed recording script
