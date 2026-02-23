"""
Gaze correction for rendered avatar frames.
Uses MediaPipe Face Landmarker to detect iris positions, then warps irises
toward the center of each eye to simulate forward-looking gaze.

Usage:
    python fix_gaze.py input.mp4 output.mp4 [--strength 0.7]
"""
import cv2
import numpy as np
import mediapipe as mp
import argparse
import sys
import subprocess
from pathlib import Path

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")

# MediaPipe face mesh landmark indices
# Iris landmarks (478-point mesh with refine_landmarks)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Eye contour landmarks
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def get_iris_shift(landmarks, h, w, strength=0.7):
    """Calculate how much to shift each iris to look at camera."""
    shifts = []

    for iris_idx, inner, outer, top, bottom in [
        (LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM),
        (RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
    ]:
        if iris_idx >= len(landmarks):
            continue  # No iris landmarks

        iris = landmarks[iris_idx]
        iris_x, iris_y = iris.x * w, iris.y * h

        eye_inner = landmarks[inner]
        eye_outer = landmarks[outer]
        eye_top = landmarks[top]
        eye_bot = landmarks[bottom]

        eye_center_x = (eye_inner.x + eye_outer.x) / 2 * w
        eye_center_y = (eye_top.y + eye_bot.y) / 2 * h

        dx = eye_center_x - iris_x
        dy = eye_center_y - iris_y

        eye_w = abs(eye_outer.x - eye_inner.x) * w
        eye_h = abs(eye_bot.y - eye_top.y) * h

        shifts.append({
            'iris_x': iris_x,
            'iris_y': iris_y,
            'dx': dx * strength,
            'dy': dy * strength * 0.5,
            'eye_w': eye_w,
            'eye_h': eye_h,
        })

    return shifts


def warp_eye_region(frame, shift_info, radius_factor=1.8):
    """Apply a smooth radial warp to shift the iris toward eye center."""
    h, w = frame.shape[:2]
    result = frame.copy()

    cx = shift_info['iris_x']
    cy = shift_info['iris_y']
    dx = shift_info['dx']
    dy = shift_info['dy']

    radius = max(shift_info['eye_w'], shift_info['eye_h']) * radius_factor

    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return result

    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius))
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius))

    if y_max <= y_min or x_max <= x_min:
        return result

    ys = np.arange(y_min, y_max)
    xs = np.arange(x_min, x_max)
    grid_x, grid_y = np.meshgrid(xs, ys)

    dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
    weight = np.clip(1.0 - (dist / radius) ** 2, 0, 1) ** 2

    src_x = (grid_x - dx * weight).astype(np.float32)
    src_y = (grid_y - dy * weight).astype(np.float32)

    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    region = cv2.remap(frame, src_x, src_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)

    result[y_min:y_max, x_min:x_max] = region
    return result


def process_video(input_path, output_path, strength=0.7):
    """Process video frame-by-frame with gaze correction."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {w}x{h} @ {fps}fps, {total} frames")
    print(f"Gaze correction strength: {strength}")
    print(f"Model: {MODEL_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_path = str(output_path) + '.tmp.mp4'
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    # Create FaceLandmarker
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    BaseOptions = mp.tasks.BaseOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    frame_count = 0
    corrected = 0
    ts_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, ts_ms)
        ts_ms += int(1000 / fps)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]

            if len(landmarks) > RIGHT_IRIS_CENTER:
                shifts = get_iris_shift(landmarks, h, w, strength)
                for shift in shifts:
                    frame = warp_eye_region(frame, shift)
                corrected += 1

        out.write(frame)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"  {frame_count}/{total} frames ({corrected} corrected)")

    cap.release()
    out.release()
    landmarker.close()

    print(f"Processed {frame_count} frames, {corrected} with gaze correction")

    # Mux with original audio
    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'a',
         '-show_entries', 'stream=codec_type', str(input_path)],
        capture_output=True, text=True
    )

    if 'audio' in probe.stdout:
        print("Muxing with original audio...")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', tmp_path,
            '-i', str(input_path),
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',
            str(output_path)
        ], capture_output=True)
    else:
        print("Encoding video...")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', tmp_path,
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            str(output_path)
        ], capture_output=True)

    Path(tmp_path).unlink(missing_ok=True)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Done! Output: {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix avatar eye gaze')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--strength', type=float, default=0.7,
                        help='Gaze correction strength (0=none, 1=full)')
    args = parser.parse_args()
    process_video(args.input, args.output, args.strength)
