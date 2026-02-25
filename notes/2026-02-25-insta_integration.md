# How INSTA Uses metrical-tracker Output

This document explains how INSTA (Instant Volumetric Head Avatars) uses metrical-tracker's preprocessing and camera data.

## Background: INSTA Project

INSTA is a volumetric head avatar synthesis method from CVPR 2023. It takes:
- Video of a person's head
- Audio (optional)

And produces:
- Real-time volumetric rendering at 60+ FPS
- Can be driven by new audio in real-time

## Integration with metrical-tracker

INSTA uses metrical-tracker to:
1. Extract per-frame FLAME mesh and parameters
2. Optimize camera intrinsics and extrinsics
3. Get depth maps for geometry
4. Extract per-frame landmark tracking

Then INSTA uses this tracking data for:
1. Training volumetric NeRF model
2. Binding tracking parameters to the model
3. Audio-driven animation synthesis

## Data Format: INSTA's transforms.json

INSTA converts metrical-tracker output to a standard format:

```json
{
  "w": 512,
  "h": 512,
  "cx": 255,
  "cy": 250,
  "fl_x": 2339,
  "fl_y": 2339,
  "camera_angle_x": 0.10939559204530716,
  "camera_angle_y": 0.10939559204530716,
  "x_fov": 12.501057290184295,
  "y_fov": 12.501057290184295,
  "integer_depth_scale": 0.001,
  "frames": [
    {
      "transform_matrix": [[...4x4 matrix...]],
      "file_path": "images/000000.png",
      "mesh_path": "meshes/000000.obj",
      "exp_path": "flame/exp/000000.txt",
      "depth_path": "depth/000000.png",
      "seg_mask_path": "seg_mask/000000.png"
    },
    // ... more frames
  ]
}
```

## How INSTA Extracts Data from .frame Files

From `/tmp/INSTA/scripts/transforms.py`:

### 1. Loading .frame Files
```python
def dump_intrinsics(frame):
    payload = torch.load(frame)

    # Extract dimensions
    h = payload['img_size'][0]  # Height: 512
    w = payload['img_size'][1]  # Width: 512

    # Extract camera matrix
    opencv = payload['opencv']
    K = opencv['K'][0]  # 3x3 intrinsic matrix

    # Extract intrinsics
    cx = K[0, 2]  # Principal point x
    cy = K[1, 2]  # Principal point y
    fl_x = K[0, 0]  # Focal length x
    fl_y = K[1, 1]  # Focal length y

    return {
        'w': w,
        'h': h,
        'cx': cx,
        'cy': cy,
        'fl_x': fl_x,
        'fl_y': fl_y,
    }
```

### 2. Computing Field of View
```python
# From focal length to FOV angle
angle_x = math.atan(w / (fl_x * 2)) * 2  # Radians
angle_y = math.atan(h / (fl_y * 2)) * 2
fovx = angle_x * 180 / math.pi            # Degrees
fovy = angle_y * 180 / math.pi
```

For your data:
```
w = 512, h = 512
fl_x = 2339, fl_y = 2339

angle_x = atan(512 / 4678) * 2
        = atan(0.1094) * 2
        = 0.1094 radians
        = 6.27 degrees (half-angle)

fovx = 0.1094 * 180 / pi = 12.5 degrees (full angle)
```

This is a **very narrow FOV** - the face takes up most of the image!

### 3. Extrinsics and Mesh Data
```python
def dump_frame(payload):
    frame, src, output = payload
    payload = torch.load(frame)

    # Get FLAME mesh
    mesh_path = frame.replace('.frame', '.ply').replace('checkpoint', 'mesh')
    trimesh.load(mesh_path, process=False).export(f'{output}/meshes/{frame_id}.obj')

    # Get depth map
    depth_path = frame.replace('checkpoint', 'depth').replace('.frame', '.png')
    if os.path.exists(depth_path):
        os.system(f'cp {depth_path} {output}/depth/{frame_id}.png')

    # Extract camera pose
    opencv = payload['opencv']
    R = opencv['R'][0]  # 3x3 rotation
    t = opencv['t'][0]  # 3x1 translation

    # Convert to world-to-camera matrix
    w2c = np.eye(4)
    w2c[0:3, 0:3] = R
    w2c[0:3, 3] = t

    # Convert to camera-to-world (inverse)
    c2w = np.linalg.inv(w2c)

    # Package for NeRF format
    data_frame = {
        'transform_matrix': c2w,
        'file_path': f'images/{frame_id}.png',
        'mesh_path': f'meshes/{frame_id}.obj',
        'exp_path': f'flame/exp/{frame_id}.txt',
        'depth_path': f'depth/{frame_id}.png',
    }

    return data_frame
```

### 4. FLAME Parameters
```python
def dump_flame(flame, frame_id, output):
    # Extract expression parameters
    exp = flame['exp']  # Expression coefficients
    exp = exp[0].flatten('F')  # Flatten to 1D

    # Save to text file for NeRF integration
    np.savetxt(f'{output}/flame/exp/{frame_id}.txt', exp, fmt='%.8f')
```

## Key Insights for Your Work

### 1. Resolution is Locked to Cropped Space

INSTA assumes:
- All images are 512x512
- All camera intrinsics are for 512x512
- If you want to use original resolution, you need to:
  - Transform K matrix using bbox.pt
  - Resample all images
  - Create new transforms.json

### 2. FOV Is Very Narrow

The 2339 pixel focal length for 512x512 gives ~12.5 degree FOV. This means:
- Face is zoomed in (good for rendering details)
- Neck/shoulders/hair are cropped out (bad for full-body avatars)
- This is why GaussianTalker Round 2 needs larger crop!

### 3. Camera Pose Format

INSTA uses **camera-to-world (c2w)** matrix format:
```
[R00 R01 R02 | tx]
[R10 R11 R12 | ty]
[R20 R21 R22 | tz]
[  0   0   0 |  1]

Where [R | t] is the camera extrinsic (world-to-camera)
c2w = inverse of world-to-camera matrix
```

This is the standard NeRF format, used by many volumetric rendering methods.

### 4. Mesh and Depth Alignment

INSTA expects:
- FLAME mesh (.obj) for each frame
- Depth map (.png) for geometric constraints
- Landmarks for photometric refinement

All aligned to the same coordinate frame (512x512 image space).

## Integration with GaussianTalker

GaussianTalker uses a similar but simpler pipeline:
1. Takes FLAME parameters from tracker
2. Learns Gaussian deformation networks
3. Conditions on audio

Unlike INSTA (which builds a volumetric NeRF):
- GaussianTalker doesn't need depth maps
- GaussianTalker doesn't do volumetric rendering
- GaussianTalker is much faster (130 FPS vs 60 FPS)

## What You Should Do

For GaussianTalker Round 2:

### Option A: Keep 512x512
```
Pros:
  - No changes needed
  - Exactly same as your FlashAvatar run
  - Fast training

Cons:
  - Neck/shoulders cropped
  - Limited facial context
  - Same quality issues as before
```

### Option B: Expand Crop (Recommended)
```
Steps:
1. Re-run metrical-tracker with:
   - bbox_scale: 2.5 -> 3.5 (larger crop)
   - image_size: [512, 512] -> [768, 768]
   - optimize_jaw: true
   - All 478 landmarks tracked

2. This will give you:
   - Larger field of view (more context)
   - Higher resolution (more detail)
   - Full-face tracking (not just mouth)

3. Then proceed with GaussianTalker training
   - Config will auto-adjust to 768x768
   - Training will be slightly slower
   - Quality will be significantly better
```

### Option C: Use Original Resolution
```
Steps:
1. Find bbox.pt
2. Transform K matrix to 1920x1080
3. Upsample images (or re-extract at higher quality)
4. Create new transforms_original.json
5. Use with GaussianTalker

Pros:
  - Full resolution (best quality)
  - Neck/shoulders/hair preserved
  - Maximum detail

Cons:
  - Requires significant work
  - Memory overhead during training
  - Slower convergence
```

## Code Reference

INSTA's data conversion: `/tmp/INSTA/scripts/transforms.py`
- Lines 44-50: FLAME parameter extraction
- Lines 53-87: Camera intrinsics extraction
- Lines 90-92: Image file handling
- Lines 95-135: Per-frame data conversion

metrical-tracker output: `/tmp/metrical-tracker/tracker.py`
- Lines 183-225: Checkpoint saving (.frame format)

## Summary

INSTA treats metrical-tracker output as a complete tracking solution and focuses on learning volumetric radiance fields. It:

1. **Accepts**: 512x512 images + .frame checkpoints
2. **Extracts**: Camera poses, FLAME parameters, meshes, depth
3. **Produces**: transforms.json for volumetric training
4. **Assumes**: All data is in cropped 512x512 space

For your GaussianTalker work, you should:
- **Understand** that crop box is in bbox.pt
- **Know** that K matrix is for cropped space only
- **Decide** whether to keep 512x512 or expand
- **Plan** accordingly for full-face tracking (not just mouth)

The metrical-tracker preprocessing is solid; your previous issue was tracking scope (mouth-only), not the preprocessing itself.
