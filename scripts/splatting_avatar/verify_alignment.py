"""Verify FLAME mesh projection aligns with face in images."""
import torch
import numpy as np
import cv2
import json
import sys
sys.path.insert(0, "/workspace/SplattingAvatar")
from model.imavatar.flame import FLAME

def main():
    dat_dir = "/workspace/SplattingAvatar/data/roman"

    with open(f"{dat_dir}/flame_params.json") as f:
        data = json.load(f)

    intrinsics = data['intrinsics']
    shape_params = torch.tensor(data['shape_params']).unsqueeze(0).float()

    # Load FLAME
    flame = FLAME('model/imavatar/FLAME2020/generic_model.pkl',
                   'model/imavatar/FLAME2020/landmark_embedding.npy',
                   n_shape=100, n_exp=50,
                   shape_params=shape_params,
                   canonical_expression=None,
                   canonical_pose=None)

    # Check a few frames
    for idx in [0, 100, 400]:
        frame = data['frames'][idx]

        # Get FLAME mesh
        full_pose = torch.tensor(frame['pose']).unsqueeze(0).float()
        expression = torch.tensor(frame['expression']).unsqueeze(0).float()
        vertices, _, _ = flame(expression, full_pose)
        verts = vertices[0].detach().numpy()  # (5023, 3)

        # Get camera from world_mat
        w2c = np.array(frame['world_mat'])
        w2c = np.vstack([w2c, [0, 0, 0, 1]])
        R = w2c[:3, :3].copy()
        T = w2c[:3, 3].copy()
        # Apply the same transform as the data loader
        R[1:, :] *= -1
        T[1:] *= -1

        # Project vertices: p_cam = R.T @ p + T (as per getWorld2View2)
        # Actually the loader stores R and getWorld2View2 transposes it
        # So W2C rotation = R.T, W2C translation = T
        p_cam = (R.T @ verts.T).T + T  # (5023, 3)

        # Camera intrinsics
        img = cv2.imread(f"{dat_dir}/{frame['file_path']}.png")
        h, w = img.shape[:2]
        fx = abs(w * intrinsics[0])
        fy = abs(h * intrinsics[1])
        cx = abs(w * intrinsics[2])
        cy = abs(h * intrinsics[3])

        # Project to image
        x = fx * p_cam[:, 0] / p_cam[:, 2] + cx
        y = fy * p_cam[:, 1] / p_cam[:, 2] + cy

        # Draw on image
        for i in range(0, len(x), 5):  # every 5th vertex
            px, py = int(x[i]), int(y[i])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 1, (0, 255, 0), -1)

        out_path = f"/tmp/alignment_frame{idx:03d}.jpg"
        cv2.imwrite(out_path, img)
        print(f"Frame {idx}: verts z-range [{p_cam[:,2].min():.2f}, {p_cam[:,2].max():.2f}], saved {out_path}")
        print(f"  x-range [{x.min():.0f}, {x.max():.0f}], y-range [{y.min():.0f}, {y.max():.0f}]")

if __name__ == '__main__':
    main()
