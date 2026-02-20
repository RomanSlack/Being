# Being — Agent Guide

## What is this?

Being is an open-source real-time talking head generation engine. It wraps [InsTaG](https://github.com/Fictionarry/InsTaG) (CVPR 2025) with production infrastructure — a CLI, data pipeline orchestration, training management, and a real-time WebSocket streaming server.

The goal: record 5-60 seconds of yourself talking → get a photorealistic avatar you can drive with any audio in real-time.

## Architecture

```
being/
├── api/server.py       # FastAPI + WebSocket streaming server
├── cli.py              # Click CLI (being prepare/train/generate/serve/check)
├── core/               # 3DGS model operations (TBD — needs InsTaG integration)
├── data/pipeline.py    # Data preprocessing pipeline orchestration
├── inference/
│   ├── engine.py       # Real-time inference engine (loads model, renders frames)
│   └── generate.py     # Offline video generation
├── training/adapt.py   # Few-shot adaptation (fine-tuning)
└── utils/
    ├── audio.py        # Audio feature extraction (wav2vec/HuBERT/DeepSpeech)
    └── checks.py       # Dependency checker
```

## Key Dependencies

- **InsTaG** lives at `extern/InsTaG/` — clone with `git clone --recursive https://github.com/Fictionarry/InsTaG.git extern/InsTaG`
- **Python 3.9**, PyTorch 1.12.1, CUDA 11.3
- **diff-gaussian-rasterization** — custom CUDA extension for 3DGS rendering
- **Basel Face Model** — requires manual download + registration

## Current State

The scaffolding is complete. Key areas that need work:

1. **`being/inference/engine.py`** — The `_load_model_state()` and `render_frame()` methods need to be wired up to InsTaG's actual model loading and rendering code. Look at `extern/InsTaG/synthesize_fuse.py` for how they do it.

2. **`being/core/`** — Empty. Needs modules for Gaussian model management, deformation network wrapper, and renderer abstraction.

3. **`being/api/server.py`** — The WebSocket streaming works at the protocol level but the actual audio→features→render pipeline in `stream_avatar()` needs the real feature extractor and engine integration.

4. **`being/training/adapt.py`** — The training orchestration calls InsTaG scripts via subprocess. This works but could be tighter (import and call directly).

5. **Real-time audio feature extraction** — `being/utils/audio.py` has a `AudioFeatureExtractor` class but the streaming `extract_chunk()` method needs proper windowed extraction.

## Running

```bash
conda activate being
being check          # Verify dependencies
being prepare <video>  # Run data pipeline
being train <data_dir> --checkpoint output/pretrained
being generate <data_dir> --model-dir <model> --audio <wav>
being serve --avatar <data_dir> --model-dir <model>
```

## Conventions

- Use `rich` for CLI output (console.print with markup)
- Use `click` for CLI argument parsing
- Use `Path` objects, not strings, for file paths internally
- InsTaG integration: prefer symlinks over copying data
- Keep Being's code separate from InsTaG's — we wrap, not fork
