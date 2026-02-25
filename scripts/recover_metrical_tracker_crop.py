#!/usr/bin/env python3
"""
Recover metrical-tracker crop box information.

Metrical-tracker crops faces before processing but doesn't store the crop
coordinates in the .frame files. This script recovers them using the landmarks
and stored images.
"""

import numpy as np
import torch
from pathlib import Path
import cv2
from typing import Optional, Tuple


def load_frame_file(frame_path: str) -> dict:
    """Load a metrical-tracker .frame checkpoint file."""
    return torch.load(frame_path, map_location='cpu')


def get_bbox_from_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    bb_scale: float = 2.5
) -> np.ndarray:
    """
    Reconstruct bounding box from landmarks (same algorithm as metrical-tracker).

    Args:
        image: Input image (H, W, C)
        landmarks: Landmark points (N, 2) in image coordinates
        bb_scale: Scale factor (default 2.5 like metrical-tracker)

    Returns:
        bbox as [xb_min, xb_max, yb_min, yb_max]
    """
    h, w = image.shape[:2]
    lmks = landmarks.astype(np.int32)

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
    if (xb_max - xb_min) % 2 != 0:
        xb_min += 1
    if (yb_max - yb_min) % 2 != 0:
        yb_min += 1

    return np.array([xb_min, xb_max, yb_min, yb_max])


def analyze_frame_intrinsics(
    frame_path: str,
    verbose: bool = True
) -> dict:
    """
    Analyze intrinsics in a .frame file.

    Returns:
        Dictionary with focal length, principal point, and img_size
    """
    frame = load_frame_file(frame_path)

    opencv = frame.get('opencv', {})
    K = opencv.get('K', np.eye(3))[0]  # Get first batch element
    img_size = frame.get('img_size', [512, 512])

    focal_x = K[0, 0]
    focal_y = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    info = {
        'img_size': img_size.tolist(),
        'focal_length_x': float(focal_x),
        'focal_length_y': float(focal_y),
        'principal_point_x': float(cx),
        'principal_point_y': float(cy),
        'K_matrix': K.tolist(),
    }

    if verbose:
        print(f"\nFrame: {frame_path}")
        print(f"  Image size: {info['img_size']}")
        print(f"  Focal length: ({focal_x:.1f}, {focal_y:.1f})")
        print(f"  Principal point: ({cx:.1f}, {cy:.1f})")

    return info


def unapply_crop_from_K(
    K: np.ndarray,
    crop_bbox: np.ndarray,
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Transform K matrix from cropped image to original image coordinates.

    Args:
        K: 3x3 intrinsic matrix from .frame file (for cropped image)
        crop_bbox: [xb_min, xb_max, yb_min, yb_max] in original image
        original_size: (H, W) of original image

    Returns:
        K matrix adjusted for original image coordinates
    """
    K_orig = K.copy()
    xb_min, xb_max, yb_min, yb_max = crop_bbox

    # Shift principal point back to original coordinates
    K_orig[0, 2] += xb_min  # cx offset
    K_orig[1, 2] += yb_min  # cy offset

    # Focal length stays the same (it's in pixels, invariant to image offset)

    return K_orig


def estimate_original_crop_from_files(
    source_dir: str,
    images_dir: str,
    first_frame: int = 0
) -> Optional[np.ndarray]:
    """
    Try to estimate the crop box by comparing source and images directories.

    This is lossy (resampling) but gives approximate results if bbox.pt is lost.

    Args:
        source_dir: Path to source images (original resolution)
        images_dir: Path to cropped images (512x512)
        first_frame: Frame index to analyze

    Returns:
        Estimated bbox or None if estimation fails
    """
    source_files = sorted(Path(source_dir).glob('*.png')) + sorted(
        Path(source_dir).glob('*.jpg')
    )
    images_files = sorted(Path(images_dir).glob('*.png')) + sorted(
        Path(images_dir).glob('*.jpg')
    )

    if not source_files or not images_files:
        print("Could not find source or images files")
        return None

    source_img = cv2.imread(str(source_files[first_frame]))
    images_img = cv2.imread(str(images_files[first_frame]))

    if source_img is None or images_img is None:
        print("Could not load images for comparison")
        return None

    print(
        f"Source: {source_img.shape}, Images: {images_img.shape}"
    )
    print(
        "Exact crop recovery from images is not possible due to "
        "resampling and padding. Use bbox.pt if available."
    )

    return None


def print_summary(
    frame_path: str,
    original_video_size: Optional[Tuple[int, int]] = None
):
    """
    Print a comprehensive summary of crop and camera info.

    Args:
        frame_path: Path to .frame file
        original_video_size: (H, W) of original video, if known
    """
    frame = load_frame_file(frame_path)

    print("=" * 70)
    print("METRICAL-TRACKER CAMERA AND CROP ANALYSIS")
    print("=" * 70)

    # Intrinsics
    opencv = frame.get('opencv', {})
    K = opencv.get('K', np.eye(3))[0]
    R = opencv.get('R', np.eye(3))[0]
    t = opencv.get('t', np.zeros(3))[0]
    img_size = frame.get('img_size', [512, 512])

    print("\nCROPPED IMAGE (what .frame file is based on):")
    print(f"  Image size: {img_size}")
    print(f"  K matrix (intrinsics):")
    print(f"    {K}")
    print(f"  Principal point: ({K[0, 2]:.1f}, {K[1, 2]:.1f})")
    print(f"  Focal length: ({K[0, 0]:.1f}, {K[1, 1]:.1f})")

    if original_video_size:
        print(f"\nORIGINAL VIDEO:")
        print(f"  Size: {original_video_size}")
        print(f"  To adjust K for original image, you need bbox.pt")
        print(f"  K_orig[0, 2] = K[0, 2] + xb_min")
        print(f"  K_orig[1, 2] = K[1, 2] + yb_min")

    print("\nAVAILABLE DATA IN .FRAME:")
    print(f"  Frame ID: {frame.get('frame_id', 'unknown')}")
    print(f"  FLAME params: {list(frame.get('flame', {}).keys())}")
    print(f"  Camera params: {list(frame.get('camera', {}).keys())}")
    print(f"  Has extrinsics (R, t): {bool(R is not None and t is not None)}")

    print("\nCROP RECOVERY:")
    print(f"  Stored in .frame: NO - not saved")
    print(f"  Stored in bbox.pt: YES (if exists in tracker output)")
    print(f"  Recoverable from images: PARTIAL - very lossy")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Recover metrical-tracker crop information'
    )
    parser.add_argument(
        '--frame',
        type=str,
        help='Path to .frame checkpoint file'
    )
    parser.add_argument(
        '--original-size',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        help='Original video size (H W) for reference'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze and print detailed information'
    )

    args = parser.parse_args()

    if args.frame:
        if args.analyze:
            print_summary(
                args.frame,
                original_video_size=tuple(args.original_size)
                if args.original_size
                else None
            )
        else:
            analyze_frame_intrinsics(args.frame, verbose=True)
    else:
        parser.print_help()
