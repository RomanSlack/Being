"""Being CLI — unified interface for data prep, training, inference, and serving."""

import click
from pathlib import Path
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Being — real-time talking head generation engine."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--audio-extractor", type=click.Choice(["deepspeech", "wav2vec", "hubert", "ave"]), default="wav2vec",
              help="Audio feature extractor to use.")
@click.option("--skip-openface", is_flag=True, help="Skip OpenFace AU extraction (will lack blink control).")
@click.option("--skip-sapiens", is_flag=True, help="Skip Sapiens geometry priors (needed for few-shot only).")
@click.option("--gpu", type=int, default=0, help="GPU device ID.")
def prepare(video_path: str, audio_extractor: str, skip_openface: bool, skip_sapiens: bool, gpu: int):
    """Run the full data preprocessing pipeline on a video.

    Takes a raw video file and produces all the data InsTaG needs:
    face tracking, parsing, teeth masks, audio features, etc.
    """
    from being.data.pipeline import run_pipeline
    run_pipeline(
        video_path=Path(video_path),
        audio_extractor=audio_extractor,
        skip_openface=skip_openface,
        skip_sapiens=skip_sapiens,
        gpu_id=gpu,
    )


@main.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Pre-trained checkpoint directory. Downloads default if not specified.")
@click.option("--audio-extractor", type=click.Choice(["deepspeech", "wav2vec", "hubert", "ave"]), default="wav2vec")
@click.option("--num-frames", type=int, default=250, help="Number of training frames (250 = 10s at 25fps).")
@click.option("--gpu", type=int, default=0, help="GPU device ID.")
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory for trained model.")
def train(data_dir: str, checkpoint: str, audio_extractor: str, num_frames: int, gpu: int, output_dir: str):
    """Adapt a pre-trained model to a new person.

    DATA_DIR should be a preprocessed avatar directory (output of `being prepare`).
    """
    from being.training.adapt import run_adaptation
    run_adaptation(
        data_dir=Path(data_dir),
        checkpoint_dir=Path(checkpoint) if checkpoint else None,
        audio_extractor=audio_extractor,
        num_frames=num_frames,
        gpu_id=gpu,
        output_dir=Path(output_dir) if output_dir else None,
    )


@main.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--model-dir", type=click.Path(exists=True), required=True, help="Trained model directory.")
@click.option("--audio", type=click.Path(exists=True), required=True, help="Audio file to drive the avatar.")
@click.option("--output", type=click.Path(), default="output.mp4", help="Output video path.")
@click.option("--gpu", type=int, default=0, help="GPU device ID.")
def generate(data_dir: str, model_dir: str, audio: str, output: str, gpu: int):
    """Generate a video from an audio file using a trained avatar."""
    from being.inference.generate import generate_video
    generate_video(
        data_dir=Path(data_dir),
        model_dir=Path(model_dir),
        audio_path=Path(audio),
        output_path=Path(output),
        gpu_id=gpu,
    )


@main.command()
@click.option("--avatar", type=click.Path(exists=True), default=None,
              help="Default avatar data directory to load on startup.")
@click.option("--model-dir", type=click.Path(exists=True), default=None,
              help="Trained model directory for the default avatar.")
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", type=int, default=8000, help="Server port.")
@click.option("--gpu", type=int, default=0, help="GPU device ID.")
def serve(avatar: str, model_dir: str, host: str, port: int, gpu: int):
    """Start the real-time streaming server."""
    import uvicorn
    from being.api.server import create_app

    app = create_app(
        default_avatar=Path(avatar) if avatar else None,
        default_model=Path(model_dir) if model_dir else None,
        gpu_id=gpu,
    )
    console.print(f"[bold green]Being server starting on {host}:{port}[/]")
    uvicorn.run(app, host=host, port=port)


@main.command()
def check():
    """Check that all dependencies are installed and working."""
    from being.utils.checks import run_checks
    run_checks()


if __name__ == "__main__":
    main()
