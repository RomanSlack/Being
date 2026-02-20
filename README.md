# Being

Open-source real-time talking head generation engine powered by 3D Gaussian Splatting.

Train a personalized avatar from a short video clip (5-60 seconds), then drive it with arbitrary audio in real-time via WebSocket streaming.

Built on top of [InsTaG](https://github.com/Fictionarry/InsTaG) (CVPR 2025) — the current state-of-the-art for few-shot personalized talking head synthesis.

## How It Works

```
Audio Input → Feature Extraction → Deformation Network → 3DGS Renderer → Video Frame
                (wav2vec/HuBERT)    (Transformer/MLP)     (60-130 FPS)     (512x512)
```

1. **Train**: Record 5-60s of yourself talking → Being learns your appearance as 3D Gaussians
2. **Drive**: Send any audio → Being deforms the Gaussians to match speech → renders photorealistic frames
3. **Stream**: Real-time WebSocket server delivers frames at <200ms latency

## Quick Start

### Prerequisites
- NVIDIA GPU with 12GB+ VRAM (RTX 4070 Super or better for training)
- CUDA 11.3-11.7 (11.3 recommended)
- Conda/Mamba
- ~20GB disk space for models and dependencies

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/anthropics/Being.git
cd Being

# Automated setup (recommended)
bash scripts/setup.sh

# OR manual setup — see docs/INSTALL.md
```

### Create Your First Avatar

```bash
# 1. Prepare a video (25fps, 512x512, talking head, 5-60 seconds)
ffmpeg -i your_video.mp4 -r 25 -vf "scale=512:512" data/avatars/myavatar/myavatar.mp4

# 2. Run the data pipeline
being prepare data/avatars/myavatar/myavatar.mp4

# 3. Adapt the model (uses pre-trained checkpoint)
being train data/avatars/myavatar --checkpoint output/pretrained

# 4. Generate video from audio
being generate data/avatars/myavatar --audio speech.wav --output result.mp4

# 5. Or start the real-time streaming server
being serve --avatar data/avatars/myavatar --port 8000
```

### Real-Time Streaming

```bash
# Start server
being serve --avatar data/avatars/myavatar --port 8000

# Connect via WebSocket
# Send: audio chunks (PCM 16kHz)
# Receive: JPEG frames (512x512)
ws://localhost:8000/api/avatars/myavatar/stream
```

## API

```
POST /api/avatars                    # Create avatar from video upload
GET  /api/avatars                    # List avatars
GET  /api/avatars/{id}               # Get avatar status
POST /api/avatars/{id}/generate      # Generate video from audio file
WS   /api/avatars/{id}/stream        # Real-time audio→video streaming
```

## Project Structure

```
Being/
├── being/                  # Core Python package
│   ├── api/                # FastAPI server + WebSocket streaming
│   ├── core/               # 3DGS model loading, Gaussian operations
│   ├── data/               # Data pipeline orchestration
│   ├── inference/          # Real-time inference engine
│   ├── training/           # Training orchestration
│   └── utils/              # Audio features, video processing, etc.
├── configs/                # Model and training configurations
├── data/                   # Avatar data directory
│   └── avatars/            # Per-avatar data
├── docker/                 # Docker configs for OpenFace, full stack
├── extern/                 # External dependencies (InsTaG submodule)
├── output/                 # Model checkpoints
├── scripts/                # Setup and utility scripts
└── tests/                  # Test suite
```

## Architecture

Being wraps InsTaG's academic code with production infrastructure:

- **Data Pipeline** (`being/data/`): Unified CLI that orchestrates InsTaG's multi-step preprocessing (face tracking, parsing, teeth masks, audio features, geometry priors) into a single `being prepare` command.
- **Training** (`being/training/`): Manages pre-training and per-avatar adaptation with progress tracking, checkpoint management, and sensible defaults.
- **Inference** (`being/inference/`): Loads a trained avatar into GPU memory and runs the audio→deformation→render pipeline at 60-130 FPS.
- **Streaming Server** (`being/api/`): FastAPI + WebSocket server that accepts audio chunks and returns rendered frames in real-time.

## Hardware Requirements

| Task | GPU | VRAM | RAM | Time |
|------|-----|------|-----|------|
| Data preprocessing | Any NVIDIA | 4GB+ | 16GB | ~5-10 min/video |
| Pre-training | A100/H100 | 40-80GB | 64GB+ | Hours |
| Adaptation | RTX 4070 Super+ | 12GB+ | 32GB | ~10-30 min |
| Inference | RTX 4070 Super | 12GB | 16GB | 60-130 FPS |

## Acknowledgments

- [InsTaG](https://github.com/Fictionarry/InsTaG) — Core talking head synthesis (CVPR 2025)
- [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian) — Predecessor (ECCV 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Rendering backbone
- [GaussianTalker](https://github.com/cvlab-kaist/GaussianTalker) — Audio-driven 3DGS (ACM MM 2024)

## License

MIT — see [LICENSE](LICENSE) for details.
