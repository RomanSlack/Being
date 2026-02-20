"""Avatar adaptation — fine-tune a pre-trained model to a new person.

Wraps InsTaG's few-shot training scripts with sensible defaults
and checkpoint management.
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
INSTAG_ROOT = PROJECT_ROOT / "extern" / "InsTaG"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "output" / "pretrained"

# Pre-trained checkpoint URLs from InsTaG Google Drive
PRETRAINED_CHECKPOINTS = {
    "deepspeech": "https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP",
    "wav2vec": "https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP",
    "hubert": "https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP",
    "ave": "https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP",
}

AUDIO_EXTRACTOR_FLAGS = {
    "deepspeech": "deepspeech",
    "wav2vec": "esperanto",  # InsTaG uses "esperanto" for wav2vec internally
    "hubert": "hubert",
    "ave": "ave",
}


def _ensure_checkpoint(checkpoint_dir: Path | None, audio_extractor: str) -> Path:
    """Ensure pre-trained checkpoint is available."""
    if checkpoint_dir and checkpoint_dir.exists():
        return checkpoint_dir

    # Check default location
    default_dir = DEFAULT_CHECKPOINT_DIR / audio_extractor
    if default_dir.exists() and any(default_dir.iterdir()):
        return default_dir

    console.print("[bold yellow]Pre-trained checkpoint not found.[/]")
    console.print(f"Expected at: {default_dir}")
    console.print(f"\nDownload from: {PRETRAINED_CHECKPOINTS.get(audio_extractor, 'N/A')}")
    console.print(f"Extract to: {DEFAULT_CHECKPOINT_DIR}/")
    console.print("\nOr use: being download-checkpoints")
    raise FileNotFoundError(f"Pre-trained checkpoint not found for {audio_extractor}")


def run_adaptation(
    data_dir: Path,
    checkpoint_dir: Path | None = None,
    audio_extractor: str = "wav2vec",
    num_frames: int = 250,
    gpu_id: int = 0,
    output_dir: Path | None = None,
):
    """Run few-shot adaptation training.

    Args:
        data_dir: Preprocessed avatar data directory.
        checkpoint_dir: Pre-trained checkpoint directory.
        audio_extractor: Audio feature extractor used during pre-training.
        num_frames: Number of training frames (250 = 10s at 25fps).
        gpu_id: CUDA device ID.
        output_dir: Where to save the adapted model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    data_dir = Path(data_dir).resolve()
    avatar_name = data_dir.name

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / f"{avatar_name}_adapted"

    checkpoint_dir = _ensure_checkpoint(checkpoint_dir, audio_extractor)
    extractor_flag = AUDIO_EXTRACTOR_FLAGS.get(audio_extractor, audio_extractor)

    console.print(f"[bold green]Being — Avatar Adaptation[/]")
    console.print(f"Avatar: {avatar_name}")
    console.print(f"Data: {data_dir}")
    console.print(f"Checkpoint: {checkpoint_dir}")
    console.print(f"Audio extractor: {audio_extractor} ({extractor_flag})")
    console.print(f"Training frames: {num_frames}")
    console.print(f"Output: {output_dir}")
    console.print()

    # InsTaG's adaptation has 3 stages: face, mouth, fuse
    # We run them sequentially via the provided training scripts.

    # Ensure InsTaG can find the data
    instag_data_link = INSTAG_ROOT / "data" / avatar_name
    if not instag_data_link.exists():
        instag_data_link.parent.mkdir(parents=True, exist_ok=True)
        instag_data_link.symlink_to(data_dir)

    # Stage 1: Face adaptation
    console.print("[bold cyan]━━━ Stage 1/3: Face adaptation ━━━[/]")
    face_output = output_dir / "face"
    _run_training(
        script=INSTAG_ROOT / "train_face.py",
        data_dir=data_dir,
        output_dir=face_output,
        checkpoint_dir=checkpoint_dir / "face" if (checkpoint_dir / "face").exists() else checkpoint_dir,
        audio_extractor=extractor_flag,
        num_frames=num_frames,
        stage="face",
    )

    # Stage 2: Mouth adaptation
    console.print("[bold cyan]━━━ Stage 2/3: Mouth adaptation ━━━[/]")
    mouth_output = output_dir / "mouth"
    _run_training(
        script=INSTAG_ROOT / "train_mouth.py",
        data_dir=data_dir,
        output_dir=mouth_output,
        checkpoint_dir=checkpoint_dir / "mouth" if (checkpoint_dir / "mouth").exists() else checkpoint_dir,
        audio_extractor=extractor_flag,
        num_frames=num_frames,
        stage="mouth",
    )

    # Stage 3: Fuse (combine face + mouth)
    console.print("[bold cyan]━━━ Stage 3/3: Fuse adaptation ━━━[/]")
    _run_fuse_training(
        data_dir=data_dir,
        output_dir=output_dir,
        face_model=face_output,
        mouth_model=mouth_output,
        audio_extractor=extractor_flag,
        num_frames=num_frames,
    )

    console.print(f"\n[bold green]Adaptation complete![/]")
    console.print(f"Model saved to: {output_dir}")
    console.print(f"\nGenerate video: being generate {data_dir} --model-dir {output_dir} --audio <audio.wav>")
    console.print(f"Start server:   being serve --avatar {data_dir} --model-dir {output_dir}")


def _run_training(
    script: Path,
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    audio_extractor: str,
    num_frames: int,
    stage: str,
):
    """Run a single training stage."""
    cmd = [
        sys.executable, str(script),
        "-S", str(data_dir),
        "-M", str(output_dir),
        "--audio_extractor", audio_extractor,
        "--N_views", str(num_frames),
    ]

    if checkpoint_dir.exists():
        cmd.extend(["--pretrained", str(checkpoint_dir)])

    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, cwd=INSTAG_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} training failed with exit code {result.returncode}")


def _run_fuse_training(
    data_dir: Path,
    output_dir: Path,
    face_model: Path,
    mouth_model: Path,
    audio_extractor: str,
    num_frames: int,
):
    """Run the fuse training stage."""
    fuse_output = output_dir / "fuse"
    cmd = [
        sys.executable, str(INSTAG_ROOT / "train_fuse_con.py"),
        "-S", str(data_dir),
        "-M", str(fuse_output),
        "--face_model", str(face_model),
        "--mouth_model", str(mouth_model),
        "--audio_extractor", audio_extractor,
        "--N_views", str(num_frames),
    ]

    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, cwd=INSTAG_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Fuse training failed with exit code {result.returncode}")
