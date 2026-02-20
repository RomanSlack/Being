"""FastAPI server for real-time talking head streaming.

Endpoints:
    POST /api/avatars              - Create avatar from video upload
    GET  /api/avatars              - List available avatars
    GET  /api/avatars/{id}         - Get avatar status/info
    POST /api/avatars/{id}/generate - Generate video from audio file
    WS   /api/avatars/{id}/stream  - Real-time audio→video streaming
"""

import asyncio
import io
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from being.inference.engine import InferenceEngine, EngineState


class AvatarInfo(BaseModel):
    id: str
    name: str
    status: str  # "ready", "processing", "training", "error"
    data_dir: str
    model_dir: Optional[str] = None


class GenerateRequest(BaseModel):
    audio_url: Optional[str] = None


# Global state — in production you'd use a proper store
_engines: dict[str, InferenceEngine] = {}
_avatars: dict[str, AvatarInfo] = {}

PROJECT_ROOT = Path(__file__).parent.parent.parent


def create_app(
    default_avatar: Path | None = None,
    default_model: Path | None = None,
    gpu_id: int = 0,
) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="Being",
        description="Real-time talking head generation engine",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup():
        """Load default avatar on startup if specified."""
        if default_avatar and default_model:
            avatar_id = default_avatar.name
            engine = InferenceEngine(default_avatar, default_model, gpu_id)
            engine.load()
            _engines[avatar_id] = engine
            _avatars[avatar_id] = AvatarInfo(
                id=avatar_id,
                name=avatar_id,
                status="ready",
                data_dir=str(default_avatar),
                model_dir=str(default_model),
            )

    @app.on_event("shutdown")
    async def shutdown():
        """Unload all models."""
        for engine in _engines.values():
            engine.unload()

    @app.get("/")
    async def root():
        return {"service": "Being", "version": "0.1.0", "status": "running"}

    @app.get("/api/avatars")
    async def list_avatars():
        """List all available avatars."""
        return list(_avatars.values())

    @app.get("/api/avatars/{avatar_id}")
    async def get_avatar(avatar_id: str):
        """Get avatar info and status."""
        if avatar_id not in _avatars:
            raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")
        info = _avatars[avatar_id]
        engine_state = None
        if avatar_id in _engines:
            engine_state = _engines[avatar_id].get_state()
        return {"avatar": info, "engine": engine_state}

    @app.post("/api/avatars")
    async def create_avatar(video: UploadFile = File(...)):
        """Create a new avatar from a video upload.

        This triggers the full data pipeline + adaptation training.
        Returns immediately with an avatar ID — poll GET /api/avatars/{id} for status.
        """
        avatar_id = str(uuid.uuid4())[:8]
        avatar_name = Path(video.filename).stem if video.filename else avatar_id

        # Save uploaded video
        data_dir = PROJECT_ROOT / "data" / "avatars" / avatar_name
        data_dir.mkdir(parents=True, exist_ok=True)
        video_path = data_dir / f"{avatar_name}.mp4"

        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)

        _avatars[avatar_id] = AvatarInfo(
            id=avatar_id,
            name=avatar_name,
            status="processing",
            data_dir=str(data_dir),
        )

        # TODO: Launch pipeline + training as background task
        # For now, just register the avatar
        return {"avatar_id": avatar_id, "status": "processing", "message": "Data pipeline started"}

    @app.post("/api/avatars/{avatar_id}/generate")
    async def generate_video(avatar_id: str, audio: UploadFile = File(...)):
        """Generate a video from an uploaded audio file."""
        if avatar_id not in _avatars:
            raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")
        if avatar_id not in _engines:
            raise HTTPException(status_code=400, detail=f"Avatar '{avatar_id}' not loaded")

        # Save audio to temp file
        audio_content = await audio.read()
        audio_path = PROJECT_ROOT / "output" / "temp" / f"{avatar_id}_{int(time.time())}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio_content)

        # TODO: Run generation and return video
        # For now, return a placeholder response
        return {"status": "generating", "audio_path": str(audio_path)}

    @app.websocket("/api/avatars/{avatar_id}/stream")
    async def stream_avatar(websocket: WebSocket, avatar_id: str):
        """Real-time audio→video streaming via WebSocket.

        Protocol:
            Client sends: Raw PCM audio chunks (16kHz, 16-bit, mono)
            Server sends: JPEG-encoded frames (512x512)

        The server maintains a buffer of audio and generates frames
        at ~30fps, interpolating between keyframes as needed.
        """
        await websocket.accept()

        if avatar_id not in _engines:
            await websocket.send_json({"error": f"Avatar '{avatar_id}' not loaded"})
            await websocket.close()
            return

        engine = _engines[avatar_id]
        audio_buffer = AudioBuffer(sample_rate=16000, frame_rate=25)

        try:
            await websocket.send_json({
                "status": "connected",
                "avatar": avatar_id,
                "protocol": {
                    "input": "PCM 16kHz 16-bit mono",
                    "output": "JPEG 512x512",
                    "frame_rate": 25,
                },
            })

            while True:
                # Receive audio chunk
                data = await websocket.receive_bytes()
                audio_buffer.add_chunk(data)

                # Check if we have enough audio for the next frame
                features = audio_buffer.get_frame_features()
                if features is not None:
                    # Render frame
                    result = engine.render_frame(features)

                    # Encode as JPEG
                    jpeg_bytes = _encode_jpeg(result.image)

                    # Send frame
                    await websocket.send_bytes(jpeg_bytes)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"error": str(e)})
            except Exception:
                pass

    return app


class AudioBuffer:
    """Buffer for accumulating audio chunks and extracting per-frame features.

    Manages the audio windowing needed to generate features for each video frame.
    At 25fps with 16kHz audio, each frame corresponds to 640 audio samples.
    """

    def __init__(self, sample_rate: int = 16000, frame_rate: int = 25):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.samples_per_frame = sample_rate // frame_rate  # 640
        self._buffer = np.array([], dtype=np.int16)
        self._frame_index = 0

    def add_chunk(self, pcm_bytes: bytes):
        """Add a chunk of PCM audio (16-bit signed, mono)."""
        chunk = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, chunk])

    def get_frame_features(self) -> np.ndarray | None:
        """Extract audio features for the next frame, if enough audio is buffered.

        Returns None if not enough audio is available yet.
        """
        needed = self.samples_per_frame * (self._frame_index + 1)
        if len(self._buffer) < needed:
            return None

        # Extract the audio window for this frame
        start = self._frame_index * self.samples_per_frame
        end = start + self.samples_per_frame
        frame_audio = self._buffer[start:end]
        self._frame_index += 1

        # Convert to float32 [-1, 1]
        audio_float = frame_audio.astype(np.float32) / 32768.0

        # TODO: Run through actual audio feature extractor (wav2vec/HuBERT/DeepSpeech)
        # For now, return raw audio as placeholder features
        return audio_float

    def reset(self):
        """Clear the buffer."""
        self._buffer = np.array([], dtype=np.int16)
        self._frame_index = 0


def _encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode a numpy image as JPEG bytes."""
    try:
        # Try turbojpeg for speed (10x faster than PIL)
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        return jpeg.encode(image, quality=quality)
    except ImportError:
        pass

    # Fallback to PIL
    from PIL import Image
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
