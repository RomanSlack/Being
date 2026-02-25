# Crop Recovery Guide for Your Metrical-Tracker Data

## Your Specific Situation

You have metrical-tracker output with:
- Original video: 1920x1080
- K matrix: `[[2339, 0, 255], [0, 2339, 250], [0, 0, 1]]`
- Output image size: 512x512
- Backup location: `data/metrical-tracker-backup/roman/checkpoint/`

## Step 1: Check if bbox.pt Still Exists

The most critical file for crop recovery is `bbox.pt`.

```bash
# On your local machine
ls -la ~/Being/data/metrical-tracker-backup/roman/

# On the A100 pod
ssh root@154.54.102.23 -p 19870 "ls -la /workspace/Being/data/metrical-tracker-backup/roman/"
```

If it exists, you're in luck:

```python
import torch

bbox = torch.load('data/metrical-tracker-backup/roman/bbox.pt')
print(bbox)  # Should print [xb_min, xb_max, yb_min, yb_max]

xb_min, xb_max, yb_min, yb_max = bbox
crop_w = xb_max - xb_min
crop_h = yb_max - yb_min
print(f"Crop region: ({xb_min}, {yb_min}) to ({xb_max}, {yb_max})")
print(f"Crop size: {crop_w} x {crop_h}")
```

## Step 2: Understand Your K Matrix

Your stored K matrix is for the **512x512 cropped image**:

```
K_cropped = [[2339,    0,  255],
             [   0, 2339,  250],
             [   0,    0,    1]]

Interpretation:
- Focal length: 2339 pixels (very zoomed in)
- Principal point: (255, 250) - center of 512x512 is (256, 256), very close!
- This means the tracker centered the face in the crop region
```

## Step 3: Transform K Back to Original Image (If You Have bbox.pt)

```python
import torch
import numpy as np

# Load your .frame file
frame = torch.load('data/metrical-tracker-backup/roman/checkpoint/00000.frame')
K_cropped = frame['opencv']['K'][0]

# Load the bbox
bbox = torch.load('data/metrical-tracker-backup/roman/bbox.pt')
xb_min, xb_max, yb_min, yb_max = bbox

# Transform K to original image coordinates
K_orig = K_cropped.copy()
K_orig[0, 2] += xb_min  # Shift cx by crop left offset
K_orig[1, 2] += yb_min  # Shift cy by crop top offset

print(f"Original K matrix (for 1920x1080):")
print(K_orig)

# Now you can use K_orig with your 1920x1080 video!
```

**Important:** The focal length (2339 pixels) stays the same because it's not affected by the offset.

## Step 4: Analyze What Was Cropped

With the bbox, you can understand what happened:

```python
original_size = (1080, 1920)  # H, W
bbox = torch.load('data/metrical-tracker-backup/roman/bbox.pt')
xb_min, xb_max, yb_min, yb_max = bbox

crop_w = xb_max - xb_min
crop_h = yb_max - yb_min

print(f"Original image: {original_size}")
print(f"Crop region: x=[{xb_min}, {xb_max}], y=[{yb_min}, {yb_max}]")
print(f"Crop dimensions: {crop_w}x{crop_h}")
print(f"Padding before 512x512 resize:")

# The crop is square, then padded and resized to 512x512
if crop_w == crop_h:
    print(f"  Square crop, no padding needed before resize")
    print(f"  Scale factor for resize: 512 / {crop_w} = {512 / crop_w:.2f}x")
else:
    max_dim = max(crop_w, crop_h)
    print(f"  Padded to {max_dim}x{max_dim}")
    print(f"  Scale factor for resize: 512 / {max_dim} = {512 / max_dim:.2f}x")
```

## Step 5: Estimate Crop if bbox.pt is Lost

If `bbox.pt` is missing, you can attempt recovery from landmarks:

```python
import cv2
import numpy as np
from pathlib import Path

# Reconstruct using stored landmarks and the .frame file
frame = torch.load('checkpoint/00000.frame')

# The landmarks are stored relative to the 512x512 image
# But the original 68-point landmarks can help

# For metrical-tracker, landmarks are the face_alignment 68-point set
# These are relative to the CROPPED image

# If you had the source video, you could:
# 1. Re-run face detection on source frames
# 2. Compute bbox from those landmarks
# But this requires re-detection and may differ from original

print("Note: Exact crop recovery without bbox.pt requires re-detection")
print("which may produce slightly different results (different face detector)")
```

## Step 6: Use the Data with GaussianTalker Round 2

For your GaussianTalker Round 2 work:

1. **If you have bbox.pt:** Use the transformed K matrix above
2. **If you don't:** You have two options:
   - Accept the 512x512 resolution (may limit quality)
   - Re-extract frames and run metrical-tracker again with different settings

### Using 512x512 Directly

Many projects use the 512x512 tracker output directly:

```python
# Your current setup works fine for:
# - FLAME mesh tracking (doesn't care about absolute resolution)
# - Audio-driven animation (works at any resolution)
# - Synthesis (more pixels = more detail, but not critical)

# The focal length of 2339 for 512x512 is very high
# This means: face occupies ~256 pixels in the 512x512 crop
# When you upscale to higher resolution, you'll lose face detail

# Better option: Get larger bbox to capture more context
# Edit config for re-run:
#   bbox_scale: 2.5 -> 3.5 (larger crop)
#   image_size: [512, 512] -> [768, 768] (larger output)
```

## Step 7: Compare with FlashAvatar Data

From your notes, you had:

```
metrical-tracker-backup/roman/checkpoint/  (838 .frame files, 6.6MB)
```

This is from your FlashAvatar attempt, which used metrical-tracker's mouth-only tracking.

**For GaussianTalker Round 2, you need:**
- Full-face tracking (metrical-tracker with `optimize_shape=true`)
- All 478 MediaPipe landmarks, not just mouth
- Eyelids, eyes, eyebrows tracking enabled

Compare your settings with FlashAvatar run if available.

## Practical Script Usage

```bash
# Use the helper script
cd ~/Being

# Analyze a single frame
python scripts/recover_metrical_tracker_crop.py \
  --frame data/metrical-tracker-backup/roman/checkpoint/00000.frame \
  --original-size 1080 1920 \
  --analyze

# Output will show:
# - Intrinsics (K, principal point, focal length)
# - Available data in the .frame file
# - Instructions for crop recovery
```

## Summary Table

| Item | Your Data | Action |
|------|-----------|--------|
| Original video | 1920x1080 | Reference only |
| Stored K matrix | For 512x512 | Transform using bbox |
| Focal length | 2339 px | Very high zoom |
| Principal point | (255, 250) | Nearly centered in crop |
| Crop coordinates | In bbox.pt | CRITICAL - find this! |
| Lossy transform | Padding + resize | Accept or re-run |

## Next Steps for GaussianTalker Round 2

1. **Find bbox.pt** - Check the A100 pod backup location
2. **Transform K matrix** - Use script above to get K for 1920x1080
3. **Decide on resolution:**
   - Keep 512x512: Works but may limit quality
   - Re-run with larger output: 768x768 or 1024x1024
   - Use original video: Best quality, but need to re-track
4. **Enable full-face tracking:**
   - `optimize_shape: true`
   - `optimize_jaw: true`
   - Use all 478 MediaPipe landmarks
   - Increase iterations if needed

## Files Referenced

- Analysis: `/home/roman/Being/notes/2026-02-25-metrical_tracker_preprocessing.md`
- Recovery script: `/home/roman/Being/scripts/recover_metrical_tracker_crop.py`
- Backup data: `/home/roman/Being/data/metrical-tracker-backup/roman/checkpoint/`
- Source: `/tmp/metrical-tracker/` (cloned for reference)
