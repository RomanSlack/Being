#!/usr/bin/env bash
# Being — ARTalk (motion-only) setup
#
# Goal: get ARTalk exporting per-frame motion codes (106 dims) without installing PyTorch3D/FLAME rendering.
# This avoids the heavy PyTorch3D build and avoids downloading FLAME-derived assets.
#
# Usage (on a fresh GPU box):
#   bash scripts/setup_artalk_motion_only.sh
#   cd /workspace/ARTalk
#   python3 export_motion.py -a demo/eng1.wav -s natural_0
#
set -euo pipefail

echo "=== ARTalk motion-only setup ==="

WORKDIR="${WORKDIR:-/workspace}"
ARTALK_DIR="${ARTALK_DIR:-$WORKDIR/ARTalk}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ ! -d "$ARTALK_DIR" ]]; then
  git clone --recurse-submodules https://github.com/xg-chu/ARTalk.git
fi

cd "$ARTALK_DIR"

python3 -m pip install -q -U pip

# Note: ARTalk's code imports MimiModel from transformers unconditionally (even if you use wav2vec),
# so we install a recent transformers version where MimiModel exists.
python3 -m pip install -q "transformers>=5.0.0" einops soundfile tqdm scipy

mkdir -p assets/style_motion

# Download only the ARTalk checkpoint + config + a style motion.
# (Do NOT download FLAME_with_eye.pt / GAGAvatar unless you've accepted the FLAME license.)
wget -q https://huggingface.co/xg-chu/ARTalk/resolve/main/ARTalk_wav2vec.pt -O assets/ARTalk_wav2vec.pt
wget -q https://huggingface.co/xg-chu/ARTalk/resolve/main/config.json -O assets/config.json
wget -q https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_0.pt -O assets/style_motion/natural_0.pt

cat > export_motion.py <<'PY'
import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio

from app import BitwiseARModel


def load_audio_mono_16k(audio_path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0)  # mono
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav


def maybe_savgol_smooth(motion: torch.Tensor) -> torch.Tensor:
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return motion
    motion_np = motion.detach().cpu().numpy()
    if motion_np.shape[0] < 5:
        return motion
    motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
    if motion_np.shape[0] >= 9:
        motion_np_smoothed[..., 100:103] = savgol_filter(
            motion_np[..., 100:103], window_length=9, polyorder=3, axis=0
        )
    return torch.from_numpy(motion_np_smoothed).to(motion.device).type_as(motion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", "-a", required=True)
    parser.add_argument("--style", "-s", default=None, help="Style id (e.g. natural_0) or path to .pt")
    parser.add_argument("--out", "-o", default=None, help="Output .pt path (default: outputs/<audio>_<style>.pt)")
    parser.add_argument("--clip_length", "-l", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_smooth", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    assets = root / "assets"

    cfg_path = assets / "config.json"
    ckpt_path = assets / "ARTalk_wav2vec.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    configs = json.loads(cfg_path.read_text())
    configs["AR_CONFIG"]["AUDIO_ENCODER"] = "wav2vec"

    model = BitwiseARModel(configs).eval().to(args.device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)

    style_motion = None
    style_name = "default"
    if args.style:
        style_name = Path(args.style).stem
        style_path = Path(args.style)
        if not style_path.exists():
            style_path = assets / "style_motion" / f"{args.style}.pt"
        style_motion = torch.load(str(style_path), map_location="cpu", weights_only=True)
        if style_motion.shape != (50, 106):
            raise ValueError(f"Unexpected style motion shape: {tuple(style_motion.shape)}")
        style_motion = style_motion[None].to(args.device)

    audio = load_audio_mono_16k(args.audio).to(args.device)
    batch = {"audio": audio[None], "style_motion": style_motion}

    with torch.no_grad():
        pred = model.inference(batch, with_gtmotion=False)[0]
    if args.clip_length is not None:
        pred = pred[: args.clip_length]
    if not args.no_smooth:
        pred = maybe_savgol_smooth(pred)

    out_path = Path(args.out) if args.out else (root / "outputs" / f"{Path(args.audio).stem}_{style_name}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_motions": pred.detach().cpu(),
            "fps": 25,
            "audio_hz": 16000,
            "style": style_name,
            "audio_path": os.path.abspath(args.audio),
        },
        str(out_path),
    )
    print(f"saved: {out_path} shape={tuple(pred.shape)}")


if __name__ == "__main__":
    main()
PY

chmod +x export_motion.py || true

echo ""
echo "Done."
echo "Next:"
echo "  cd $ARTALK_DIR"
echo "  python3 export_motion.py -a demo/eng1.wav -s natural_0"

