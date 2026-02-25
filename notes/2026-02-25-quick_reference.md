# Metrical-Tracker Quick Reference Card

## TL;DR

| Question | Answer |
|----------|--------|
| Does metrical-tracker crop? | **Yes** - face detection → bbox → crop → resize to 512x512 |
| Where's the crop box stored? | `{actor}/bbox.pt` (filesystem, not in .frame) |
| What's in .frame files? | FLAME params, K/R/t, mesh, depth - all for 512x512 space |
| Can I recover crop from K? | **Only if you have bbox.pt** - needs arithmetic transformation |
| Input format expected? | MP4 video (any resolution) or pre-extracted images |
| Output format? | 512x512 PNG images + .frame checkpoints + landmarks |
| Is it lossy? | **Partially** - resampling is lossy, but crop box is recoverable |

## One-Minute Recovery Process

```python
import torch
import numpy as np

# Load bbox (CRITICAL!)
bbox = torch.load('bbox.pt')
xb_min, xb_max, yb_min, yb_max = bbox

# Load .frame file
frame = torch.load('checkpoint/00000.frame')
K_512 = frame['opencv']['K'][0]

# Transform K for original 1920x1080 image
K_orig = K_512.copy()
K_orig[0, 2] += xb_min  # Shift principal point x
K_orig[1, 2] += yb_min  # Shift principal point y
# Focal length stays same!

print(f"Original crop: x=[{xb_min}, {xb_max}], y=[{yb_min}, {yb_max}]")
print(f"K for 1920x1080:\n{K_orig}")
```

## Key Numbers

For your data with `K = [[2339, 0, 255], [0, 2339, 250], [0, 0, 1]]`:

```
Image size: 512x512
Focal length: 2339 pixels
Principal point: (255, 250) - perfectly centered!
FOV: 12.5 degrees - VERY narrow (telephoto)
Interpretation: Face occupies most of the frame, zoomed in

Face size in original 1920x1080: ~256 pixels wide
Suggests original crop was ~800x800 or similar
```

## Critical Files

| File | Location | Purpose | Status |
|------|----------|---------|--------|
| bbox.pt | `{actor}/bbox.pt` | Crop box [xb_min, xb_max, yb_min, yb_max] | **FIND THIS!** |
| *.frame | `{actor}/checkpoint/*.frame` | FLAME params, K/R/t, metadata | Always present |
| images/*.png | `{actor}/images/*.png` | Processed 512x512 images | Always present |
| source/*.png | `{actor}/source/*.png` | Original resolution (if kept) | Optional |
| kpt/*.npy | `{actor}/kpt/*.npy` | 68-point landmarks | Always present |
| kpt_dense/*.npy | `{actor}/kpt_dense/*.npy` | 478-point landmarks (MediaPipe) | Always present |

## Transformation Formulas

### Crop Box Recovery
```
If bbox.pt exists:
  [xb_min, xb_max, yb_min, yb_max] = torch.load('bbox.pt')
  crop_w = xb_max - xb_min
  crop_h = yb_max - yb_min

If bbox.pt lost:
  - Can estimate from source/images comparison (lossy)
  - Or re-run face detection on source frames
```

### K Matrix Transformation
```
From 512x512 to original 1920x1080:
  K_orig[0, 0] = K_512[0, 0]  # Focal X unchanged
  K_orig[1, 1] = K_512[1, 1]  # Focal Y unchanged
  K_orig[0, 2] = K_512[0, 2] + xb_min  # Shift principal X
  K_orig[1, 2] = K_512[1, 2] + yb_min  # Shift principal Y
  K_orig[2, 2] = 1.0  # Always 1

From 512x512 to different size (e.g., 768x768):
  scale = 768 / 512 = 1.5
  K_768[0, 0] = K_512[0, 0] * scale
  K_768[1, 1] = K_512[1, 1] * scale
  K_768[0, 2] = K_512[0, 2] * scale
  K_768[1, 2] = K_512[1, 2] * scale
```

### Landmark Transformation
```
From 512x512 to original (if you have crop box):
  lmk_orig_x = (lmk_512_x / 0.64) + xb_min
  lmk_orig_y = (lmk_512_y / 0.64) + yb_min
  # 0.64 = 512 / 800 (assuming 800x800 original crop)
  # Adjust if your crop size differs
```

## Pipeline Flow

```
1. Video Input (any resolution)
2. Frame Extraction @ 25fps
3. Face Detection (frame 0) → landmarks
4. Bbox Calculation (bbox_scale=2.5) → bbox.pt
5. Crop All Frames (same bbox)
6. Pad to Square (if needed)
7. Resize to 512x512
8. Track FLAME + Optimize K/R/t
9. Save:
   - 512x512 images
   - Landmarks (in 512x512 space)
   - .frame files with K (for 512x512)
   - FLAME meshes + depth
```

## Common Gotchas

1. **K is always for the final 512x512 size**, not original video
2. **Crop box in bbox.pt is in original image coordinates**, not 512x512
3. **Focal length doesn't change when shifting principal point**
4. **Resampling is lossy**, can't perfectly invert 512x512 → original
5. **Padding constants not stored** (assumed black borders)
6. **All landmarks in .npy are in 512x512 space**, need transformation

## For GaussianTalker Round 2

**Current state:** 512x512, mouth-only tracking
**Problem:** Neck/shoulders cropped, limited context

**Solutions:**
- **Minimal:** Use 512x512 as-is (fastest, limited quality)
- **Recommended:** Re-run with bbox_scale=3.5, image_size=[768,768], optimize_jaw=true
- **Maximum:** bbox_scale=4.0, image_size=[1024,1024], full-face optimization

## Helper Tools

```bash
# Analyze a .frame file
python scripts/recover_metrical_tracker_crop.py \
  --frame checkpoint/00000.frame \
  --original-size 1080 1920 \
  --analyze

# Outputs: intrinsics, available data, recovery instructions
```

## Documentation Files

In `/home/roman/Being/notes/`:
- `2026-02-25-SUMMARY.md` - Executive summary
- `2026-02-25-metrical_tracker_preprocessing.md` - Full technical analysis
- `2026-02-25-crop_recovery_guide.md` - Step-by-step recovery for your data
- `2026-02-25-crop_transformation_diagram.md` - Visual diagrams and math
- `2026-02-25-insta_integration.md` - How INSTA uses metrical-tracker
- `2026-02-25-quick_reference.md` - This file!

## Key Takeaway

**Metrical-tracker's preprocessing is sound. The key is finding bbox.pt and using simple arithmetic to recover the original crop box and transform camera intrinsics. The transformation is reversible (except for padding and resampling), so you can use tracking data with original resolution rendering.**
