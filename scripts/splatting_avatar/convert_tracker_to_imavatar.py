"""Convert metrical-tracker .frame files to SplattingAvatar's flame_params.json format.
V2: Fixed world_mat to store R.T (transposed) as expected by SplattingAvatar's loader."""
import torch
import numpy as np
import json
import os
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

def rotation_6d_to_matrix(d6):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1 = d6[:3]
    a2 = d6[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)

def matrix_to_axis_angle(R):
    """Convert 3x3 rotation matrix to 3D axis-angle."""
    return Rotation.from_matrix(R).as_rotvec()

def convert(checkpoint_dir, output_dir, img_size=(512, 512)):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(checkpoint_dir.glob('*.frame'))
    print(f"Found {len(frame_files)} .frame files")

    frames = []
    shape_params = None
    intrinsics = None

    for ff in frame_files:
        data = torch.load(ff, map_location='cpu', weights_only=False)
        flame = data['flame']
        cam = data['opencv']
        frame_id = data['frame_id']
        h, w = int(data['img_size'][0]), int(data['img_size'][1])

        # Shape params (identity, shared)
        if shape_params is None:
            shape_params = flame['shape'][0].flatten()[:100].tolist()

        # Intrinsics from K matrix
        if intrinsics is None:
            K = cam['K'][0]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            intrinsics = [fx / w, fy / h, cx / w, cy / h]

        # Expression: first 50 of 100
        expression = flame['exp'][0].flatten()[:50].tolist()

        # Pose: [3 global_orient, 3 neck, 3 jaw, 6 eye] = 15
        global_orient = [0.0, 0.0, 0.0]
        neck = [0.0, 0.0, 0.0]

        # Jaw: 6D rotation -> axis-angle
        jaw_6d = flame['jaw'][0].flatten()
        jaw_R = rotation_6d_to_matrix(jaw_6d[:6])
        jaw_aa = matrix_to_axis_angle(jaw_R).tolist()

        # Eyes: 2 * 6D rotation -> 2 * axis-angle
        eyes_6d = flame['eyes'][0].flatten()
        eye_L_R = rotation_6d_to_matrix(eyes_6d[:6])
        eye_R_R = rotation_6d_to_matrix(eyes_6d[6:12])
        eye_L_aa = matrix_to_axis_angle(eye_L_R).tolist()
        eye_R_aa = matrix_to_axis_angle(eye_R_R).tolist()
        eye_pose = eye_L_aa + eye_R_aa

        pose = global_orient + neck + jaw_aa + eye_pose

        # World matrix construction:
        # SplattingAvatar's loader does:
        #   R = world_mat[:3,:3]
        #   T = world_mat[:3,3]
        #   R[1:,:] *= -1   (OpenGL→OpenCV row flip)
        #   T[1:] *= -1
        # Then getWorld2View2 does: Rt[:3,:3] = R.T  (transpose!)
        # So the final W2C rotation = R.T
        # We need R.T = R_opencv, therefore R = R_opencv.T
        # Before the row flip: R_stored = R_opencv.T with rows 1,2 negated
        R_opencv = cam['R'][0]  # (3, 3)
        t_opencv = cam['t'][0].flatten()  # (3,)

        # IMavatar's FLAME has factor=4 (4x scale on all geometry).
        # To keep same projection: scale camera translation by 4.
        t_opencv = t_opencv * 4.0

        # Store R_opencv.T with rows 1,2 negated (OpenGL convention)
        R_stored = R_opencv.T.copy()
        R_stored[1:, :] *= -1
        t_stored = t_opencv.copy()
        t_stored[1:] *= -1

        world_mat = np.zeros((3, 4))
        world_mat[:3, :3] = R_stored
        world_mat[:3, 3] = t_stored

        frames.append({
            'file_path': f'image/{frame_id}',
            'world_mat': world_mat.tolist(),
            'pose': pose,
            'expression': expression,
        })

    result = {
        'intrinsics': intrinsics,
        'shape_params': shape_params,
        'frames': frames,
    }

    out_json = output_dir / 'flame_params.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {out_json} with {len(frames)} frames")
    print(f"Intrinsics: {intrinsics}")

if __name__ == '__main__':
    checkpoint_dir = sys.argv[1]
    output_dir = sys.argv[2]
    convert(checkpoint_dir, output_dir)
