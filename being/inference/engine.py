"""Real-time inference engine — loads a trained model and renders frames on demand.

This is the core engine that the streaming server uses. It keeps the model
in GPU memory and provides a simple interface:

    engine = InferenceEngine(data_dir, model_dir, gpu_id=0)
    engine.load()
    frame = engine.render_frame(audio_features)  # Returns numpy array (H, W, 3)
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
INSTAG_ROOT = PROJECT_ROOT / "extern" / "InsTaG"


@dataclass
class FrameResult:
    """Result of rendering a single frame."""
    image: np.ndarray  # (H, W, 3) uint8
    render_time_ms: float
    frame_index: int


@dataclass
class EngineState:
    """Current state of the inference engine."""
    loaded: bool = False
    avatar_name: str = ""
    frame_count: int = 0
    total_render_time: float = 0.0
    avg_fps: float = 0.0


class InferenceEngine:
    """Real-time inference engine for audio-driven talking head generation.

    Loads a trained InsTaG model and renders frames from audio features.
    Designed for use by the streaming server but can also be used standalone.
    """

    def __init__(self, data_dir: Path, model_dir: Path, gpu_id: int = 0):
        self.data_dir = Path(data_dir).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.gpu_id = gpu_id
        self.state = EngineState(avatar_name=self.data_dir.name)

        # These get populated on load()
        self._gaussians = None
        self._deformation_net = None
        self._renderer = None
        self._background = None
        self._camera = None

    def load(self):
        """Load the trained model into GPU memory.

        This is expensive (1-3 seconds) but only needs to happen once.
        After loading, render_frame() is fast (7-15ms per frame).
        """
        import torch

        console.print(f"[bold]Loading avatar '{self.state.avatar_name}' onto GPU {self.gpu_id}...[/]")
        start = time.time()

        torch.cuda.set_device(self.gpu_id)

        # Add InsTaG to path so we can import its modules
        import sys
        if str(INSTAG_ROOT) not in sys.path:
            sys.path.insert(0, str(INSTAG_ROOT))

        # Import InsTaG components
        from scene import GaussianModel
        from gaussian_renderer import render
        from arguments import ModelParams, PipelineParams
        from argparse import Namespace

        # Load model parameters
        # InsTaG saves these in the model directory
        self._load_model_state()

        elapsed = time.time() - start
        self.state.loaded = True
        console.print(f"[green]Model loaded in {elapsed:.1f}s[/]")

    def _load_model_state(self):
        """Load the Gaussian model, deformation network, and rendering setup."""
        import sys
        if str(INSTAG_ROOT) not in sys.path:
            sys.path.insert(0, str(INSTAG_ROOT))

        import torch

        # The exact loading procedure depends on InsTaG's checkpoint format.
        # We'll implement this once we can inspect the actual checkpoint structure.
        # For now, store the paths needed.
        self._face_model_path = self.model_dir / "face"
        self._mouth_model_path = self.model_dir / "mouth"
        self._fuse_model_path = self.model_dir / "fuse"

        # Verify checkpoint files exist
        for path in [self._face_model_path, self._mouth_model_path, self._fuse_model_path]:
            if not path.exists():
                # Try flat structure (model_dir directly contains checkpoints)
                console.print(f"[yellow]Expected subdirectory not found: {path}[/]")
                console.print("[yellow]Will attempt flat checkpoint loading.[/]")
                break

        console.print("[dim]Model state loaded (full inference pipeline pending InsTaG integration)[/]")

    def render_frame(self, audio_features: np.ndarray, pose: Optional[np.ndarray] = None) -> FrameResult:
        """Render a single frame from audio features.

        Args:
            audio_features: Audio feature vector for this frame.
                Shape depends on extractor: (16, 29) for DeepSpeech, (1, 1024) for wav2vec, etc.
            pose: Optional head pose override (3x4 camera extrinsics).
                If None, uses the neutral/default pose.

        Returns:
            FrameResult with the rendered image and timing info.
        """
        if not self.state.loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        start = time.time()

        # TODO: Full integration with InsTaG's rendering pipeline
        # For now, return a placeholder to validate the API shape.
        # The actual implementation will:
        # 1. Pass audio_features through the deformation network
        # 2. Apply deformations to the canonical Gaussians
        # 3. Rasterize with diff-gaussian-rasterization
        # 4. Composite face + mouth + torso + background

        h, w = 512, 512
        image = np.zeros((h, w, 3), dtype=np.uint8)

        elapsed = (time.time() - start) * 1000
        self.state.frame_count += 1
        self.state.total_render_time += elapsed
        self.state.avg_fps = 1000.0 / (self.state.total_render_time / self.state.frame_count)

        return FrameResult(
            image=image,
            render_time_ms=elapsed,
            frame_index=self.state.frame_count,
        )

    def render_sequence(self, audio_features: np.ndarray) -> list[FrameResult]:
        """Render a sequence of frames from a batch of audio features.

        Args:
            audio_features: Audio features for multiple frames.
                Shape: (num_frames, feature_dim, ...)

        Returns:
            List of FrameResults.
        """
        results = []
        for i in range(len(audio_features)):
            result = self.render_frame(audio_features[i])
            results.append(result)
        return results

    def unload(self):
        """Free GPU memory."""
        import torch
        self._gaussians = None
        self._deformation_net = None
        self._renderer = None
        torch.cuda.empty_cache()
        self.state.loaded = False
        console.print("[dim]Model unloaded.[/]")

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self.state
