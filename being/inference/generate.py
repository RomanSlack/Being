"""Offline video generation — drive a trained avatar with an audio file."""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
INSTAG_ROOT = PROJECT_ROOT / "extern" / "InsTaG"


def generate_video(
    data_dir: Path,
    model_dir: Path,
    audio_path: Path,
    output_path: Path = Path("output.mp4"),
    gpu_id: int = 0,
):
    """Generate a video from an audio file using a trained avatar.

    Args:
        data_dir: Preprocessed avatar data directory.
        model_dir: Trained model directory (output of `being train`).
        audio_path: Audio file (.wav) to drive the avatar.
        output_path: Where to save the output video.
        gpu_id: CUDA device ID.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    data_dir = Path(data_dir).resolve()
    model_dir = Path(model_dir).resolve()
    audio_path = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()

    console.print(f"[bold green]Being — Video Generation[/]")
    console.print(f"Avatar: {data_dir.name}")
    console.print(f"Model: {model_dir}")
    console.print(f"Audio: {audio_path}")
    console.print(f"Output: {output_path}")
    console.print()

    # First, extract audio features for the driving audio
    audio_feat_path = _extract_audio_features(audio_path, data_dir)

    # Run InsTaG synthesis
    cmd = [
        sys.executable, str(INSTAG_ROOT / "synthesize_fuse.py"),
        "-S", str(data_dir),
        "-M", str(model_dir),
        "--dilate",
        "--use_train",
    ]

    if audio_feat_path:
        cmd.extend(["--audio", str(audio_feat_path)])

    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, cwd=INSTAG_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Synthesis failed with exit code {result.returncode}")

    # Find the generated video and copy to output path
    # InsTaG outputs to the model directory
    generated = _find_generated_video(model_dir)
    if generated and generated != output_path:
        import shutil
        shutil.copy2(generated, output_path)
        console.print(f"[green]Video saved to: {output_path}[/]")
    else:
        console.print(f"[yellow]Check {model_dir} for generated output.[/]")


def _extract_audio_features(audio_path: Path, data_dir: Path) -> Path | None:
    """Extract audio features from a driving audio file."""
    # Detect which extractor was used by checking existing features
    if (data_dir / "aud_eo.npy").exists():
        extractor = "wav2vec"
        script = INSTAG_ROOT / "data_utils" / "wav2vec.py"
        cmd = [sys.executable, str(script), "--wav", str(audio_path), "--save_feats"]
        suffix = "_eo.npy"
    elif (data_dir / "aud_ds.npy").exists():
        extractor = "deepspeech"
        script = INSTAG_ROOT / "data_utils" / "deepspeech_features" / "extract_ds_features.py"
        cmd = [sys.executable, str(script), "--input", str(audio_path)]
        suffix = "_ds.npy"
    else:
        console.print("[yellow]Could not detect audio extractor, using wav2vec.[/]")
        script = INSTAG_ROOT / "data_utils" / "wav2vec.py"
        cmd = [sys.executable, str(script), "--wav", str(audio_path), "--save_feats"]
        suffix = "_eo.npy"

    console.print(f"[bold]Extracting audio features ({extractor})...[/]")
    result = subprocess.run(cmd, cwd=INSTAG_ROOT)
    if result.returncode != 0:
        console.print("[yellow]Audio feature extraction failed, trying without custom audio.[/]")
        return None

    # Feature file should be alongside the audio with the suffix
    feat_path = audio_path.with_suffix("").parent / (audio_path.stem + suffix)
    if feat_path.exists():
        return feat_path

    # Also check without prefix
    feat_path = audio_path.parent / (audio_path.stem + suffix)
    return feat_path if feat_path.exists() else None


def _find_generated_video(model_dir: Path) -> Path | None:
    """Find the generated video in the model output directory."""
    # InsTaG typically outputs to a results/ or renders/ subdirectory
    for pattern in ["**/*.mp4", "**/*.avi"]:
        videos = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if videos:
            return videos[0]
    return None
