"""Drive SplattingAvatar with ARTalk motion output.
Maps ARTalk's 106-dim FLAME params → SplattingAvatar rendering."""
import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, "/workspace/SplattingAvatar")

from model.splatting_avatar_model import SplattingAvatarModel
from dataset.imavatar_data import IMavatarDataset
from scene.dataset_readers import convert_to_scene_cameras
from model import libcore


def main():
    parser = ArgumentParser()
    parser.add_argument('--motion', type=str, required=True, help='ARTalk .pt motion file')
    parser.add_argument('--audio', type=str, default=None, help='Audio file for muxing')
    parser.add_argument('--output', type=str, default='artalk_driven.mp4')
    parser.add_argument('--dat_dir', type=str, default='data/roman')
    parser.add_argument('--pc_dir', type=str, default=None, help='Point cloud dir (auto-detected if not set)')
    parser.add_argument('--configs', type=str, default='configs/splatting_avatar.yaml')
    parser.add_argument('--no_head_motion', action='store_true', help='Disable head rotation')
    parser.add_argument('--exp_scale', type=float, default=1.0, help='Scale expression coefficients')
    args = parser.parse_args()

    device = 'cuda'

    # Load ARTalk motion
    motion_data = torch.load(args.motion, map_location='cpu', weights_only=False)
    pred = motion_data['pred_motions']  # (T, 106)
    fps = motion_data['fps']
    T = pred.shape[0]
    print(f"ARTalk motion: {T} frames at {fps}fps")

    # Load config + dataset (for FLAME model, mesh template, camera)
    config = libcore.load_from_config([args.configs])
    config.dataset.dat_dir = args.dat_dir
    frameset = IMavatarDataset(config.dataset, split='train')

    # Get reference camera from frame 0
    batch0 = frameset[0]
    ref_cam = batch0['scene_cameras'][0].cuda()
    print(f"Reference camera loaded (frame 0)")

    # FLAME model and mesh template from dataset
    flame = frameset.flame
    mesh_py3d = frameset.mesh_py3d

    # Auto-detect point cloud dir
    if args.pc_dir is None:
        output_dirs = sorted(Path(args.dat_dir).glob('output-splatting/*/point_cloud/iteration_*'))
        if output_dirs:
            args.pc_dir = str(output_dirs[-1])
            print(f"Auto-detected pc_dir: {args.pc_dir}")
        else:
            print("ERROR: No point cloud found. Specify --pc_dir")
            return

    # Load trained Gaussian model
    pipe = config.pipe
    gs_model = SplattingAvatarModel(config.model, verbose=True)
    gs_model.load_ply(os.path.join(args.pc_dir, 'point_cloud.ply'))
    gs_model.load_from_embedding(os.path.join(args.pc_dir, 'embedding.json'))
    print(f"Loaded {gs_model._xyz.shape[0]} Gaussians from {args.pc_dir}")

    # Render
    out_dir = '/tmp/artalk_frames'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {T} frames...")

    for i in range(T):
        # Map ARTalk output → FLAME params
        # ARTalk: [100 exp, 3 head_rot, 3 jaw_rot]
        exp_100 = pred[i, :100]
        head_rot = pred[i, 100:103]  # axis-angle global orientation
        jaw_rot = pred[i, 103:106]   # axis-angle jaw

        # SplattingAvatar FLAME expects:
        # expression: 50 dims (first 50 of ARTalk's 100)
        # full_pose: 15 dims [3 global_orient, 3 neck, 3 jaw, 6 eye]
        expression = (exp_100[:50] * args.exp_scale).unsqueeze(0).float()

        # Head rotation: apply via FLAME global_orient
        # Training had global_orient=[0,0,0] with rotation in camera,
        # but ARTalk head rotations are tiny (±3 deg) so direct application works
        if args.no_head_motion:
            global_orient = torch.zeros(3)
        else:
            global_orient = head_rot

        neck = torch.zeros(3)
        eye_pose = torch.zeros(6)
        full_pose = torch.cat([global_orient, neck, jaw_rot, eye_pose]).unsqueeze(0).float()

        # FLAME forward → mesh
        with torch.no_grad():
            vertices, _, _ = flame(expression, full_pose)
            frame_mesh = mesh_py3d.update_padded(vertices)

        mesh_info = {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }

        # Update Gaussians and render
        gs_model.update_to_posed_mesh(mesh_info)
        render_pkg = gs_model.render_to_camera(ref_cam, pipe, background='white')
        image = render_pkg['render']

        # Save frame
        # image is (3, H, W) tensor
        img_np = (image.permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
        img_bgr = img_np[:, :, ::-1]  # RGB → BGR
        cv2.imwrite(os.path.join(out_dir, f'{i:05d}.jpg'), img_bgr)

        if i % 50 == 0:
            print(f"  {i}/{T}")

    print(f"Done rendering. Composing video...")

    # Compose video with ffmpeg (better than OpenCV VideoWriter for codec support)
    tmp_video = '/tmp/artalk_tmp.mp4'
    os.system(f'ffmpeg -y -loglevel warning -framerate {fps} -i {out_dir}/%05d.jpg '
              f'-c:v libx264 -pix_fmt yuv420p -crf 18 {tmp_video}')

    # Mux audio if provided
    if args.audio and os.path.exists(args.audio):
        os.system(f'ffmpeg -y -loglevel warning -i {tmp_video} -i "{args.audio}" '
                  f'-c:v copy -c:a aac -shortest "{args.output}"')
        os.remove(tmp_video)
    else:
        os.rename(tmp_video, args.output)

    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
