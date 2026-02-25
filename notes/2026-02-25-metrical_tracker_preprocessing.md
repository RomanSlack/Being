# Metrical-Tracker Preprocessing and Cropping Analysis

## Overview
Metrical-Tracker is a monocular face tracker that optimizes FLAME mesh parameters. It includes an automatic cropping pipeline that processes video frames to extract face regions before tracking.

## Input Processing Pipeline

### 1. Video Frame Extraction
**File:** `datasets/generate_dataset.py` - `GeneratorDataset.initialize()`

- **Input:** Video file (`video.mp4`) or pre-extracted image sequence
- **Process:**
  - If no images exist in `source/` directory, extracts frames from video using ffmpeg
  - Command: `ffmpeg -i {video_file} -vf fps={config.fps} -q:v 1 {source}/%05d.png`
  - Default FPS: 25 (set in config)
  - Quality: q:v 1 (highest quality)

**Key insight:** The original video resolution is used at this stage - NO cropping yet.

### 2. Cropping Decision
**File:** `datasets/generate_dataset.py` - `GeneratorDataset.run()`

Config parameter: `crop_image = True` (default)

If enabled, the pipeline:
1. Detects faces in the first frame using face_alignment library
2. Extracts landmarks
3. Computes bounding box from landmarks
4. Applies scale factor (default `bbox_scale = 2.5`)
5. Saves bbox to `{actor}/bbox.pt` for reuse across all frames

### 3. The Bounding Box Calculation
**File:** `image.py` - `get_bbox(image, lmks, bb_scale=2.0)`

```python
def get_bbox(image, lmks, bb_scale=2.0):
    h, w, c = image.shape
    lmks = lmks.astype(np.int32)

    # Find extent of landmarks
    x_min, x_max = np.min(lmks[:, 0]), np.max(lmks[:, 0])
    y_min, y_max = np.min(lmks[:, 1]), np.max(lmks[:, 1])

    # Find center and compute size with scale
    x_center = int((x_max + x_min) / 2.0)
    y_center = int((y_max + y_min) / 2.0)
    size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))

    # Compute bounds with clamping to image
    xb_min = max(x_center - size // 2, 0)
    xb_max = min(x_center + size // 2, w - 1)
    yb_min = max(y_center - size // 2, 0)
    yb_max = min(y_center + size // 2, h - 1)

    # Ensure even dimensions
    if (xb_max - xb_min) % 2 != 0: xb_min += 1
    if (yb_max - yb_min) % 2 != 0: yb_min += 1

    return np.array([xb_min, xb_max, yb_min, yb_max])
```

**Key points:**
- Bbox is a square (centered on face, sides are max(dx, dy))
- Scale factor (bb_scale) controls padding around face
- Result is clamped to image bounds
- Dimensions forced to be even

### 4. Cropping and Resizing
**File:** `datasets/generate_dataset.py` - `GeneratorDataset.run()`

```python
if self.config.crop_image:
    image = crop_image_bbox(image, bbox)
    if self.config.image_size[0] == self.config.image_size[1]:
        image = squarefiy(image, size=self.config.image_size[0])
else:
    image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]))
```

**Default config:** `image_size = [512, 512]` (height, width)

The `squarefiy()` function:
```python
def squarefiy(image, size=512):
    h, w, c = image.shape
    if w != h:
        max_wh = max(w, h)
        # Pad to make square
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        image = np.pad(image, [(vp, vp), (hp, hp), (0, 0)], mode='constant')
    # Resize to final size
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
```

**Important:** Padding is applied BEFORE resizing. This preserves aspect ratio.

## Intrinsics and Camera Parameters

### 5. Landmark Extraction on Cropped Image
**File:** `datasets/generate_dataset.py` - `GeneratorDataset.process_face()`

Two landmark sets are extracted on the CROPPED image:
- **68-point landmarks** via face_alignment (sparse)
- **478-point landmarks** via MediaPipe (dense)

These are saved as numpy arrays with coordinates in the cropped image coordinate system.

### 6. Camera Calibration
**File:** `tracker.py` - `Tracker.optimize_camera()`

During initial keyframe optimization, camera intrinsics are optimized:
- `focal_length` - single parameter per batch, calibrated
- `principal_point` - 2D point, calibrated
- Camera extrinsics (R, T) are also optimized

### 7. Camera Conversion to OpenCV Format
**File:** `tracker.py` - `Tracker.save_checkpoint()`

```python
def save_checkpoint(self, frame_id):
    opencv = opencv_from_cameras_projection(self.cameras, self.image_size)

    frame = {
        'opencv': {
            'R': opencv[0].cpu().numpy(),      # 3x3 rotation
            't': opencv[1].cpu().numpy(),      # 3x1 translation
            'K': opencv[2].cpu().numpy(),      # 3x3 intrinsic matrix
        },
        'img_size': self.image_size.cpu().numpy()[0],  # [height, width]
        ...
    }
    torch.save(frame, f'{checkpoint_folder}/{frame_id}.frame')
```

**Critical:** The K matrix, R, t, and img_size stored in .frame files are all relative to the **CROPPED IMAGE SPACE**, not the original video.

## Reconstructing the Original Crop

### How to Reverse the Crop

Given:
- Original video dimensions: `(H_orig, W_orig)` = (1080, 1920)
- Stored image size in .frame: `(h, w)` = (512, 512)
- Stored K matrix: e.g., `[[2339, 0, 255], [0, 2339, 250], [0, 0, 1]]`

The crop was applied in this order:
1. **Face region detection** (first frame) → bbox `[xb_min, xb_max, yb_min, yb_max]`
2. **Crop** → `cropped_image = img[yb_min:yb_max, xb_min:xb_max, :]`
3. **Pad to square** → dimensions become square with padding
4. **Resize to 512x512**

To INVERT this transformation:

#### Step 1: Find the saved bbox (not stored!)
Unfortunately, metrical-tracker **does not save the crop coordinates** in the .frame files.

**The only artifacts are:**
- `{actor}/bbox.pt` - contains the crop bbox (if still exists)
- Original images in `{actor}/source/`
- Cropped images in `{actor}/images/`

#### Step 2: Invert from image files
If you still have both source and images directories:

```python
import cv2
import numpy as np

source_img = cv2.imread('input/source/00000.png')  # Original
cropped_img = cv2.imread('input/images/00000.png')  # Stored in tracker

# Compare to find shift
# But this only works if images are identical (they're not - resampling)
```

The transformation is lossy because:
- Padding constants are not stored
- Resize is lossy (INTER_CUBIC)

#### Step 3: Use bbox.pt if available
```python
bbox = torch.load('{actor}/bbox.pt')
xb_min, xb_max, yb_min, yb_max = bbox
crop_h = yb_max - yb_min
crop_w = xb_max - xb_min
```

This gives you the exact crop box in original image coordinates.

## Using K Matrix with Different Image Sizes

The K matrix in the .frame file is computed for the **512x512 cropped image**.

To use it with the original 1920x1080 video:

### Option A: Unapply the crop from K
The K matrix encodes:
- Focal length (diagonal elements)
- Principal point (K[0,2], K[1,2])

The principal point is relative to the **cropped image center (256, 256)**.

To transform K for the original image:

```python
bbox = torch.load('bbox.pt')
xb_min, xb_max, yb_min, yb_max = bbox
crop_w = xb_max - xb_min
crop_h = yb_max - yb_min

# K is for 512x512 cropped image
K_cropped = payload['opencv']['K']

# Unapply crop: shift principal point back to original image
K_orig = K_cropped.copy()
K_orig[0, 2] += xb_min  # cx offset
K_orig[1, 2] += yb_min  # cy offset

# The focal length stays the same (it's in pixels, independent of image size)
```

### Option B: Use K for 512x512 images
If you want to re-render or use the tracking with the 512x512 cropped images, use K as-is.

## Your Data: Analysis

Given:
```
opencv.K = [[2339, 0, 255], [0, 2339, 250], [0, 0, 1]]
img_size = [512, 512]
original_video = 1920x1080
```

**Interpretation:**
- Focal length: 2339 pixels (very high - face is zoomed in)
- Principal point: (255, 250) in the 512x512 image (roughly centered)
- The focal length suggests original face was roughly 256 pixels tall in the original image

**Reverse calculation:**
```
focal_length_cropped = 2339
img_size_cropped = 512
img_size_original = ? (depends on pre-crop size)

# From FOV perspective:
# K[0,0] = focal_length = f * (image_width / sensor_width)
# For a given face size, f increases as image size decreases
```

## Summary

| Aspect | Details |
|--------|---------|
| **Does it crop?** | Yes, if `crop_image=True` (default) |
| **How?** | Face detection → landmark → bbox → crop → pad → resize to 512x512 |
| **Where stored?** | `bbox.pt` for bbox; K/img_size in .frame files (for 512x512) |
| **Input format** | MP4 video or image sequence |
| **Output format** | 512x512 PNG images + .frame checkpoints + landmarks |
| **Lossy?** | Yes - resampling, padding constants not stored |
| **Recoverable?** | Only if `bbox.pt` still exists |

## Key Code References

- **Bbox calculation:** `/tmp/metrical-tracker/image.py` lines 22-42
- **Cropping pipeline:** `/tmp/metrical-tracker/datasets/generate_dataset.py` lines 49-93
- **Checkpoint saving:** `/tmp/metrical-tracker/tracker.py` lines 183-225
- **INSTA data conversion:** `/tmp/INSTA/scripts/transforms.py` lines 53-87 (for how INSTA uses this data)
