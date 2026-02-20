"""Unified data preprocessing pipeline.

Orchestrates all the InsTaG preprocessing steps into a single command:
1. Video normalization (25fps, 512x512)
2. Face tracking + 3DMM parameter extraction
3. Face parsing + segmentation
4. Train/test split
5. OpenFace Action Unit extraction
6. Teeth mask generation (EasyPortrait)
7. Geometry priors (Sapiens) — for few-shot adaptation
8. Audio feature extraction (DeepSpeech/wav2vec/HuBERT)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Root of the Being project
PROJECT_ROOT = Path(__file__).parent.parent.parent
INSTAG_ROOT = PROJECT_ROOT / "extern" / "InsTaG"


def _run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command with real-time output."""
    merged_env = {**os.environ, **(env or {})}
    console.print(f"[dim]$ {' '.join(str(c) for c in cmd)}[/]")
    result = subprocess.run(cmd, cwd=cwd, env=merged_env, capture_output=False)
    if result.returncode != 0:
        console.print(f"[bold red]Command failed with exit code {result.returncode}[/]")
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result


def _ensure_instag():
    """Verify InsTaG submodule is cloned and available."""
    if not (INSTAG_ROOT / "synthesize_fuse.py").exists():
        console.print("[bold red]InsTaG not found at extern/InsTaG/[/]")
        console.print("Run: git submodule update --init --recursive")
        raise FileNotFoundError("InsTaG submodule not initialized")


def normalize_video(video_path: Path, output_dir: Path) -> Path:
    """Normalize video to 25fps, 512x512, and copy to data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    avatar_name = output_dir.name
    output_video = output_dir / f"{avatar_name}.mp4"

    if output_video.exists():
        console.print(f"[yellow]Normalized video already exists: {output_video}[/]")
        return output_video

    console.print("[bold]Step 0: Normalizing video (25fps, 512x512)...[/]")
    _run_cmd([
        "ffmpeg", "-i", str(video_path),
        "-r", "25",
        "-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        str(output_video),
    ])
    return output_video


def run_face_tracking(video_path: Path, data_dir: Path):
    """Step 1: Process video — face tracking, landmarks, parsing, background."""
    console.print("[bold]Step 1: Face tracking + parsing...[/]")
    _run_cmd(
        [sys.executable, str(INSTAG_ROOT / "data_utils" / "process.py"), str(video_path)],
        cwd=INSTAG_ROOT,
    )


def run_train_test_split(video_path: Path):
    """Step 2: Split into train/test sets."""
    console.print("[bold]Step 2: Train/test split...[/]")
    _run_cmd(
        [sys.executable, str(INSTAG_ROOT / "data_utils" / "split.py"), str(video_path)],
        cwd=INSTAG_ROOT,
    )


def run_openface(data_dir: Path):
    """Step 3: Extract Action Units with OpenFace."""
    console.print("[bold]Step 3: OpenFace Action Unit extraction...[/]")

    au_csv = data_dir / "au.csv"
    if au_csv.exists():
        console.print("[yellow]au.csv already exists, skipping OpenFace.[/]")
        return

    # Try to find OpenFace FeatureExtraction binary
    openface_bin = shutil.which("FeatureExtraction")
    if openface_bin is None:
        # Check common install locations
        for path in [
            "/usr/local/bin/FeatureExtraction",
            "/opt/OpenFace/build/bin/FeatureExtraction",
            str(PROJECT_ROOT / "extern" / "OpenFace" / "build" / "bin" / "FeatureExtraction"),
        ]:
            if Path(path).exists():
                openface_bin = path
                break

    if openface_bin is None:
        console.print("[bold yellow]OpenFace not found. Skipping AU extraction.[/]")
        console.print("Install OpenFace or use Docker: docker/openface.Dockerfile")
        console.print("Then run: FeatureExtraction -f <video> -out_dir <data_dir>")
        return

    avatar_name = data_dir.name
    video_path = data_dir / f"{avatar_name}.mp4"

    _run_cmd([
        openface_bin,
        "-f", str(video_path),
        "-out_dir", str(data_dir),
    ])

    # OpenFace outputs as <video_name>.csv, rename to au.csv
    openface_csv = data_dir / f"{avatar_name}.csv"
    if openface_csv.exists():
        openface_csv.rename(au_csv)
        console.print(f"[green]Action units saved to {au_csv}[/]")


def run_teeth_masks(data_dir: Path):
    """Step 4: Generate teeth masks with EasyPortrait."""
    console.print("[bold]Step 4: Teeth mask generation...[/]")
    teeth_dir = data_dir / "teeth"
    if teeth_dir.exists() and any(teeth_dir.iterdir()):
        console.print("[yellow]Teeth masks already exist, skipping.[/]")
        return

    easyportrait_script = INSTAG_ROOT / "data_utils" / "easyportrait" / "create_teeth_mask.py"
    env = {"PYTHONPATH": str(INSTAG_ROOT / "data_utils" / "easyportrait")}
    _run_cmd(
        [sys.executable, str(easyportrait_script), str(data_dir)],
        cwd=INSTAG_ROOT,
        env=env,
    )


def run_sapiens(data_dir: Path):
    """Step 5: Generate geometry priors with Sapiens (for few-shot adaptation)."""
    console.print("[bold]Step 5: Sapiens geometry priors...[/]")
    sapiens_script = INSTAG_ROOT / "data_utils" / "sapiens" / "run.sh"
    if not sapiens_script.exists():
        console.print("[yellow]Sapiens script not found, skipping.[/]")
        return
    _run_cmd(["bash", str(sapiens_script), str(data_dir)], cwd=INSTAG_ROOT)


def run_audio_features(data_dir: Path, extractor: str = "wav2vec"):
    """Step 6: Extract audio features."""
    console.print(f"[bold]Step 6: Audio feature extraction ({extractor})...[/]")

    audio_file = data_dir / "aud.wav"
    if not audio_file.exists():
        # Try aud_train.wav
        audio_file = data_dir / "aud_train.wav"
    if not audio_file.exists():
        console.print("[bold red]No audio file found in data directory.[/]")
        raise FileNotFoundError(f"No aud.wav or aud_train.wav in {data_dir}")

    if extractor == "deepspeech":
        script = INSTAG_ROOT / "data_utils" / "deepspeech_features" / "extract_ds_features.py"
        _run_cmd([sys.executable, str(script), "--input", str(audio_file)], cwd=INSTAG_ROOT)
    elif extractor == "wav2vec":
        script = INSTAG_ROOT / "data_utils" / "wav2vec.py"
        _run_cmd([sys.executable, str(script), "--wav", str(audio_file), "--save_feats"], cwd=INSTAG_ROOT)
    elif extractor == "hubert":
        script = INSTAG_ROOT / "data_utils" / "hubert.py"
        _run_cmd([sys.executable, str(script), "--wav", str(audio_file)], cwd=INSTAG_ROOT)
    elif extractor == "ave":
        console.print("[yellow]AVE features are extracted at training time, no preprocessing needed.[/]")
    else:
        raise ValueError(f"Unknown audio extractor: {extractor}")


def run_pipeline(
    video_path: Path,
    audio_extractor: str = "wav2vec",
    skip_openface: bool = False,
    skip_sapiens: bool = False,
    gpu_id: int = 0,
):
    """Run the complete data preprocessing pipeline."""
    _ensure_instag()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Determine data directory — use the video's parent if it's already in data/,
    # otherwise create a new directory based on the video name
    video_path = Path(video_path).resolve()
    video_stem = video_path.stem

    # Check if video is already in a proper data directory
    if video_path.parent.name == video_stem:
        data_dir = video_path.parent
    else:
        data_dir = PROJECT_ROOT / "data" / "avatars" / video_stem
        data_dir.mkdir(parents=True, exist_ok=True)
        # Copy video to data dir if not already there
        dest = data_dir / f"{video_stem}.mp4"
        if not dest.exists():
            shutil.copy2(video_path, dest)
        video_path = dest

    console.print(f"[bold green]Being — Data Pipeline[/]")
    console.print(f"Video: {video_path}")
    console.print(f"Data dir: {data_dir}")
    console.print(f"Audio extractor: {audio_extractor}")
    console.print()

    # Need to ensure InsTaG's data path points to our data directory.
    # InsTaG expects data to be in its own data/ directory, so we symlink.
    instag_data_link = INSTAG_ROOT / "data" / video_stem
    if not instag_data_link.exists():
        instag_data_link.parent.mkdir(parents=True, exist_ok=True)
        instag_data_link.symlink_to(data_dir)

    steps = [
        ("Normalizing video", lambda: normalize_video(video_path, data_dir)),
        ("Face tracking + parsing", lambda: run_face_tracking(video_path, data_dir)),
        ("Train/test split", lambda: run_train_test_split(video_path)),
    ]

    if not skip_openface:
        steps.append(("OpenFace Action Units", lambda: run_openface(data_dir)))

    steps.append(("Teeth masks", lambda: run_teeth_masks(data_dir)))

    if not skip_sapiens:
        steps.append(("Sapiens geometry priors", lambda: run_sapiens(data_dir)))

    steps.append(("Audio features", lambda: run_audio_features(data_dir, audio_extractor)))

    for i, (name, fn) in enumerate(steps, 1):
        console.print(f"\n[bold cyan]━━━ [{i}/{len(steps)}] {name} ━━━[/]")
        try:
            fn()
            console.print(f"[green]✓ {name} complete[/]")
        except Exception as e:
            console.print(f"[bold red]✗ {name} failed: {e}[/]")
            if "OpenFace" in name or "Sapiens" in name:
                console.print("[yellow]This step is optional, continuing...[/]")
                continue
            raise

    console.print(f"\n[bold green]Pipeline complete![/]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"\nNext: being train {data_dir}")
