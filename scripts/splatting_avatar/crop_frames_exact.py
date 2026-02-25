"""Reproduce metrical-tracker's exact face cropping pipeline.

Uses the same logic as metrical-tracker/datasets/generate_dataset.py:
1. face_alignment landmarks on frame 0 → compute bbox (bb_scale=2.5)
2. crop_image_bbox (may be non-square due to image boundary clipping)
3. squarefiy (pad to square + resize to 512x512)

This ensures images perfectly match the K matrices in .frame files.
"""
import cv2
import numpy as np
import face_alignment
import os
import sys
from pathlib import Path


def get_bbox(image, lmks, bb_scale=2.5):
    """Exact copy of metrical-tracker/image.py get_bbox"""
    h, w, c = image.shape
    lmks = lmks.astype(np.int32)
    x_min, x_max = np.min(lmks[:, 0]), np.max(lmks[:, 0])
    y_min, y_max = np.min(lmks[:, 1]), np.max(lmks[:, 1])
    x_center = int((x_max + x_min) / 2.0)
    y_center = int((y_max + y_min) / 2.0)
    size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))
    xb_min = max(x_center - size // 2, 0)
    xb_max = min(x_center + size // 2, w - 1)
    yb_min = max(y_center - size // 2, 0)
    yb_max = min(y_center + size // 2, h - 1)

    yb_max = min(yb_max, h - 1)
    xb_max = min(xb_max, w - 1)
    yb_min = max(yb_min, 0)
    xb_min = max(xb_min, 0)

    if (xb_max - xb_min) % 2 != 0:
        xb_min += 1
    if (yb_max - yb_min) % 2 != 0:
        yb_min += 1

    return np.array([xb_min, xb_max, yb_min, yb_max])


def crop_image_bbox(image, bbox):
    """Exact copy of metrical-tracker/image.py crop_image_bbox"""
    xb_min, xb_max, yb_min, yb_max = bbox
    return image[max(yb_min, 0):min(yb_max, image.shape[0] - 1),
                 max(xb_min, 0):min(xb_max, image.shape[1] - 1), :]


def squarefiy(image, size=512):
    """Exact copy of metrical-tracker/image.py squarefiy"""
    h, w, c = image.shape
    if w != h:
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        image = np.pad(image, [(vp, vp), (hp, hp), (0, 0)], mode='constant')
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def main():
    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    out_path = Path(output_dir) / 'image'
    out_path.mkdir(parents=True, exist_ok=True)

    # Extract frames from video
    print(f"Extracting frames at {target_fps}fps...")
    tmp_dir = '/tmp/source_frames'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f'ffmpeg -y -loglevel warning -i "{video_path}" -vf fps={target_fps} -q:v 1 {tmp_dir}/%05d.png')

    frames = sorted(Path(tmp_dir).glob('*.png'))
    print(f"Extracted {len(frames)} frames")

    # Detect face on frame 0 using face_alignment (same as tracker)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

    frame0 = cv2.imread(str(frames[0]))
    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

    lmks_list = fa.get_landmarks_from_image(frame0_rgb)
    if lmks_list is None or len(lmks_list) == 0:
        print("ERROR: No face detected on frame 0!")
        return
    lmks = lmks_list[0]  # First face

    # Compute bbox using tracker's exact logic
    bbox = get_bbox(frame0, lmks, bb_scale=2.5)
    print(f"Bbox: xmin={bbox[0]}, xmax={bbox[1]}, ymin={bbox[2]}, ymax={bbox[3]}")
    print(f"Crop size: {bbox[1]-bbox[0]} x {bbox[3]-bbox[2]}")

    # Process all frames with the same bbox
    for i, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path))

        # Crop + squarefiy (exact tracker pipeline)
        cropped = crop_image_bbox(image, bbox)
        squared = squarefiy(cropped, size=512)

        out_file = out_path / f'{i:05d}.png'
        cv2.imwrite(str(out_file), squared)

        if i % 100 == 0:
            print(f"  {i}/{len(frames)}")

    print(f"Done! {len(frames)} cropped frames in {out_path}")
    print(f"Bbox saved for reference: {bbox}")

    # Also print what the crop dimensions were (for debugging)
    test_crop = crop_image_bbox(frame0, bbox)
    print(f"Cropped shape before squarefiy: {test_crop.shape}")
    print(f"After squarefiy: 512x512")


if __name__ == '__main__':
    main()
