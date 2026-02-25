# Metrical-Tracker Crop Transformation Visual Guide

## Overview: How Images Flow Through metrical-tracker

```
Original Video (1920x1080)
        |
        v
[Frame Extraction] ffmpeg @ 25fps
        |
        v
Source Images (1920x1080)
        |
        v
[Face Detection] MediaPipe + face_alignment on frame 0
        |
        v
Landmarks (68 points) in 1920x1080 space
        |
        v
[Bbox Calculation] bbox_scale=2.5
        |
        v
Crop Box (e.g., x=[400, 1200], y=[200, 1000])
        | bbox.pt saved here!
        v
[Crop All Frames] Apply same bbox to all frames
        |
        v
Cropped Images (800x800 - square)
        |
        v
[Pad + Resize]
  - Pad to square (if not already)
  - Resize to 512x512 with INTER_CUBIC
        |
        v
Final Images (512x512)
Landmarks rescaled to 512x512 space
        |
        v
[Tracker Optimization]
  - Calibrate K matrix for 512x512
  - K stored in .frame files
        |
        v
Output: 512x512 images + K matrix (for 512x512)
        + .frame checkpoints with FLAME params
        + depth maps
        + meshes
```

## Coordinate System Transformations

### Original Image Space (1920x1080)

```
(0,0) -------- 1920 -------- (1920,0)
|                              |
|                              |
1080                           1080
|                              |
|                              |
(0,1080) --------------- (1920,1080)

Face somewhere in here, detected by landmarks
```

### After Bbox Extraction (e.g., x=[400,1200], y=[200,1000])

```
Original: 1920x1080
Crop box: x_min=400, x_max=1200, y_min=200, y_max=1000

(400,200) ---------- 800 px ----------- (1200,200)
|                                         |
|                                         |
800 px                                    800 px
|                                         |
|                                         |
(400,1000) ----- 800x800 square ----- (1200,1000)

Cropped region has size 800x800
Landmarks now relative to (400, 200) origin
```

### After Padding and Resize (800x800 -> 512x512)

```
800x800 cropped image
  |
  +-- Already square, so padding = 0
  |
  +-- Resize: 800x800 -> 512x512
        Scale factor: 512/800 = 0.64x
  |
  v
512x512 final image

All coordinates scaled by 0.64x:
  Original coords in crop space * 0.64x -> 512x512 space
```

## Camera Matrix Transformation Chain

### Focal Length Interpretation

```
Camera matrix K for 512x512 image:
K = [[2339,    0,  255],
     [   0, 2339,  250],
     [   0,    0,    1]]

Focal length in pixels: 2339

This is very high! What does it mean?

FOV angle = 2 * arctan(w / (2 * focal_length))
          = 2 * arctan(512 / (2 * 2339))
          = 2 * arctan(512 / 4678)
          = 2 * arctan(0.1094)
          = 2 * 6.25 degrees
          = 12.5 degrees (very narrow FOV)

Interpretation:
  - Face takes up most of the 512x512 image
  - FOV is ~12.5 degrees (telephoto lens equivalent)
  - Not a wide-angle lens
```

### Principal Point Analysis

```
K[0, 2] = 255 (x coordinate)
K[1, 2] = 250 (y coordinate)

In 512x512 image:
- Center would be (256, 256)
- Principal point is (255, 250)
- Offset: (-1, -6) pixels

Conclusion: Face is almost perfectly centered in the crop!
This makes sense because:
  1. Face detection finds face center
  2. Bbox is built around that center
  3. Face is nearly centered in bbox
  4. After resize to 512x512, still centered
```

## Recovering Original K Matrix

### The Transformation

```
Original image space (1920x1080):
  K_original[0, 2] = K_cropped[0, 2] + xb_min
                   = 255 + 400
                   = 655

  K_original[1, 2] = K_cropped[1, 2] + yb_min
                   = 250 + 200
                   = 450

Focal length stays the same:
  K_original[0, 0] = K_cropped[0, 0] = 2339
  K_original[1, 1] = K_cropped[1, 1] = 2339

K_original = [[2339,    0,  655],
              [   0, 2339,  450],
              [   0,    0,    1]]

This K matrix now works with 1920x1080 images!
```

### Why Focal Length Doesn't Change

```
Focal length is defined as:
  f_pixels = f_mm * (sensor_width_px / sensor_width_mm)

It's invariant to translation. Shifting the principal point
doesn't change the focal length.

However, if you want to use K_original with a DIFFERENT
image size (e.g., 1024x1024), you need to scale focal length:

  f_1024 = f_2339 * (1024 / 512)
         = 2339 * 2
         = 4678
```

## The Padding Mystery

### Why We Can't Recover Exact Padding

```
Step 1: Crop 800x800 from 1920x1080 image
         -> 800x800 image

Step 2: Is 800x800 already square?
         -> Yes! No padding needed

Step 3: Resize 800x800 -> 512x512
         -> Done

BUT if the crop wasn't square (e.g., 800x600):

Step 1: Crop 800x600
Step 2: Squarefy:
         max(800, 600) = 800
         Need to pad 600 to 800
         Padding: top=100, bottom=100
         -> np.pad(image, [(100, 100), (0, 0), (0, 0)])

Step 3: Resize 800x800 -> 512x512

Problem: Padding constant is NOT stored!
We don't know what color was padded (usually black=0)
```

## Landmark Coordinate Transformation

### From Original Space to 512x512 Space

```
Landmarks in original image (1920x1080):
  lmk_orig = [x1, y1, x2, y2, ..., x68, y68]
  All coordinates in [0, 1920] x [0, 1080] range

After face detection in original space:
  (These are detected by face_alignment on original)

Stored landmarks in .npy files (for 512x512):
  lmk_512 = ?

Transformation chain:
  1. Original landmark: (x_orig, y_orig)
  2. Relative to crop origin: (x_crop, y_crop) = (x_orig - xb_min, y_orig - yb_min)
  3. In 800x800 space: (x_800, y_800) = (x_crop, y_crop)
  4. In 512x512 space: (x_512, y_512) = (x_800 * 512/800, y_800 * 512/800)
                                      = (x_800 * 0.64, y_800 * 0.64)

Reverse:
  (x_orig, y_orig) = ((x_512 / 0.64) + xb_min, (y_512 / 0.64) + yb_min)
                   = ((x_512 * 1.5625) + xb_min, (y_512 * 1.5625) + yb_min)
```

## Summary: Three Key Files for Recovery

```
File 1: bbox.pt
  Contains: [xb_min, xb_max, yb_min, yb_max]
  Location: {actor}/bbox.pt
  Status: CRITICAL - only way to recover crop
  Size: < 1KB

File 2: .frame checkpoints
  Contains: K, R, t, FLAME params, img_size
  Location: {actor}/checkpoint/*.frame
  Status: Always available
  Content: All relative to 512x512 image

File 3: Original images (if kept)
  Contains: Source frames before cropping
  Location: {actor}/source/*
  Status: Optional but helpful
  Keeps: Original resolution reference

You have File 2, find File 1!
```

## Your Specific Case Walkthrough

```
Step 0: You have .frame files with K_512x512
  K_512x512 = [[2339,    0,  255],
               [   0, 2339,  250],
               [   0,    0,    1]]

Step 1: Find bbox.pt in backup
  bbox = torch.load('data/metrical-tracker-backup/roman/bbox.pt')
  -> [xb_min, xb_max, yb_min, yb_max] = ?

Step 2: Assuming you find bbox = [400, 1200, 200, 1000]
  (This is just an example)

Step 3: Transform K for original 1920x1080 image:
  K_orig[0, 2] = 255 + 400 = 655
  K_orig[1, 2] = 250 + 200 = 450
  K_orig[0, 0] = 2339 (unchanged)
  K_orig[1, 1] = 2339 (unchanged)

Step 4: Now use K_orig with your 1920x1080 video!
  K_orig works with any video at original resolution
  Independent of 512x512 training resolution
```

## Code Reference: The Actual Transformation

From `/tmp/metrical-tracker/datasets/generate_dataset.py`:

```python
# Line 66-68: Get bbox from first frame
lmk, _ = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
bbox = get_bbox(image, lmk, bb_scale=self.config.bbox_scale)
torch.save(bbox, bbox_path)  # SAVE IT!

# Line 70-73: Apply crop to all frames
if self.config.crop_image:
    image = crop_image_bbox(image, bbox)
    if self.config.image_size[0] == self.config.image_size[1]:
        image = squarefiy(image, size=self.config.image_size[0])
```

## Loss of Information During Crop

```
LOSSY OPERATIONS:
1. Cropping: Removes context around face
   - Hair, shoulders, neck context is lost
   - Can't be recovered

2. Padding (if needed): Constants not stored
   - We don't know what was padded with
   - Usually black, but could be other colors

3. Resizing: INTER_CUBIC resampling
   - Information loss due to downsampling
   - Can't perfectly invert

RECOVERABLE:
1. Crop box: Stored in bbox.pt
   - Exact pixel coordinates in original space
   - Allows K matrix transformation

2. Focal length and intrinsics: In .frame files
   - Can be transformed to original image
   - Focal length itself unchanged

CONSEQUENCE:
  You can use tracking with original images,
  but you've lost peripheral detail (shoulders, hair).

  For rendering: Use cropped/tracked region,
  composite back to original if needed.
```
