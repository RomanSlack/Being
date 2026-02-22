# Being — Research & Quality Roadmap

## Decision: Switch from InsTaG to GaussianTalker

**Date**: 2026-02-21

Per-person training is fine for our use case — we want to clone people from short videos and drive them in real-time. We don't need a foundation model. GaussianTalker is the better base for this.

### Why GaussianTalker over InsTaG

| | InsTaG (old) | GaussianTalker (new) |
|---|---|---|
| **Render speed** | ~30fps | **130fps** |
| **Audio decoder** | MLP (direct mapping) | **Transformer + spatial-audio attention** |
| **Face representation** | Implicit deformation | BFM vertex init → learned deformation |
| **Training** | 3 stages (face→mouth→fuse) | **1 stage** |
| **3D face model** | BFM 2009 | **BFM 2009** (same) |
| **Preprocessing** | BFM, face parsing, OpenFace, wav2vec | **Nearly identical** pipeline |
| **CUDA extensions** | diff-gaussian-rasterization, simple-knn | **Same** |
| **Venue** | CVPR 2025 | ACM MM 2024 |
| **Code** | Works but needed 5 patches for PyTorch 2.x | Released, tested on RTX 3090 / A6000 |

Key advantages:
1. **130fps** makes real-time trivial (vs struggling to hit 30fps)
2. **Transformer audio decoder** with spatial-audio attention is a richer audio→motion mapping than InsTaG's MLP
3. **Single training stage** instead of face→mouth→fuse dance
4. **Multi-resolution triplane** encodes canonical head at different spatial scales
5. **Same face model (BFM)** — our preprocessed data is reusable

### GaussianTalker Architecture

```
Audio (wav2vec 44-dim or DeepSpeech 29-dim)
  → AudioNet (Conv1d stack → 32-dim embedding)
  → AudioAttNet (8-frame attention window → weighted 32-dim)
                                      ↓
            Spatial-Audio Attention Module (transformer cross-attention)
                                      ↓
Canonical 3DGS (initialized from BFM vertices) → Per-Gaussian deformation → Rasterization → Frame
                                      ↑
                   Multi-resolution triplane features + eye AU45 + camera params
```

- Gaussians are **initialized** from BFM 3DMM vertex positions (34,650 vertices)
- At runtime, deformation is **purely audio-driven** — no face model params used at inference
- Audio features attend to spatial triplane features via cross-attention
- Outputs per-Gaussian: position offset, scale, rotation, opacity, SH color
- Eye blink comes from AU45 (OpenFace), positionally encoded to 32-dim

**Important**: BFM expression/identity parameters are NOT used at inference time. The deformation network is conditioned only on audio + eye blink + camera. Emotion control cannot be done via face model parameters — it would need to go through the audio conditioning or by modifying the deformation network.

Source: https://github.com/cvlab-kaist/GaussianTalker
Paper: https://arxiv.org/abs/2404.16012

---

## Data Compatibility: InsTaG → GaussianTalker

**Confirmed: Our InsTaG preprocessed data is almost fully reusable.**

Both repos use the same BFM 2009 face model, same face tracking code (virtually identical `data_utils/face_tracking/`), same face parsing model, same data directory structure.

### Compatibility matrix

| File | Compatible? | Notes |
|------|------------|-------|
| `gt_imgs/` | Yes | Identical format |
| `ori_imgs/` + `.lms` | Yes | Identical format |
| `parsing/` | Yes | Identical format |
| `torso_imgs/` | Yes | Identical format |
| `au.csv` | Yes | Same OpenFace AU format |
| `bc.jpg` | Yes | Same background image |
| `aud.wav` | Yes | Same audio |
| `transforms_*.json` | Yes | Same format |
| `track_params.pt` | **Needs 1 field added** | InsTaG's is missing `"vertices"` key |
| Audio features | **Rename + config tweak** | Our `aud_eo.npy` (44-dim wav2vec) works with `audio_in_dim` change |

### Fix 1: Add vertices to track_params.pt

InsTaG saves: `{id, exp, euler, trans, focal}`
GaussianTalker needs: `{id, exp, euler, trans, focal, vertices}`

The vertices are just reconstructed from the BFM coefficients we already have:

```python
import torch
from data_utils.face_tracking.facemodel import Face_3DMM

params = torch.load("track_params.pt")
model = Face_3DMM("data_utils/face_tracking/3DMM", 100, 79, 100, 34650)

id_expanded = params["id"].expand(params["exp"].shape[0], -1)
vertices = model.forward_geo(id_expanded, params["exp"])

params["vertices"] = vertices.detach().cpu()
torch.save(params, "track_params.pt")
```

### Fix 2: Audio features (wav2vec 44-dim → DeepSpeech 29-dim)

GaussianTalker defaults to DeepSpeech features (`aud_ds.npy`, 29-dim). Our InsTaG data uses wav2vec/esperanto (`aud_eo.npy`, 44-dim).

**Both use the same windowing**: `(N, 16, dim)` with kernel_size=16, stride=2.

Options:
- **Option A (easy)**: Symlink `aud_eo.npy` → `aud_ds.npy`, change `audio_in_dim` from 29 to 44 in `scene/deformation.py`
- **Option B (clean)**: Re-extract DeepSpeech features using GaussianTalker's built-in extractor (needs TF 2.8.0)
- **Option C**: GaussianTalker's `process.py` supports `--asr wav2vec` natively — use that path

Option A is fastest. Option B might give better results since the model architecture was designed around 29-dim DeepSpeech features.

---

## Competitive Landscape: Tavus (Phoenix-4)

Tavus ships a three-model architecture:

| Component | Role | Spec |
|-----------|------|------|
| **Phoenix-4** (render) | Gaussian-diffusion hybrid renderer | 1080p @ 40fps, sub-600ms latency |
| **Raven-1** (perception) | Reads user's face/voice for emotion | Real-time emotion detection |
| **Sparrow-1** (dialogue) | Conversational timing & turn-taking | Natural floor transfer |

Phoenix-4 technical details:
- **Gaussian-diffusion hybrid** — 3DGS base + diffusion model for coherent facial motion
- **Trained on thousands of hours** of conversational video data
- **10+ controllable emotion states** with smooth transitions
- **Full head+shoulders** rendering at 1080p
- **Streaming architecture** — causal transformer for audio, distilled diffusion, WebRTC delivery

Key difference from us: They use a foundation model so per-person training is just identity injection. We do full per-person training. This means we need longer input video (3-5 min vs 2 min) and longer training time, but we avoid needing a massive pretrained model.

Sources:
- https://www.tavus.io/research
- https://www.tavus.io/post/phoenix-4-real-time-human-rendering-with-emotional-intelligence
- https://www.tavus.io/post/advanced-techniques-in-talking-head-generation-3d-gaussian-splatting
- https://docs.tavus.io/sections/replica/replica-training

---

## Current State vs Target

| Dimension | Being now (InsTaG) | Being next (GaussianTalker) | Tavus |
|-----------|--------------------|-----------------------------|-------|
| Resolution | 512x512 | 512→1080p | 1080p |
| Render FPS | ~30 (offline) | **130fps** | 40fps real-time |
| Lip sync | wav2vec → MLP | wav2vec → **transformer** | Diffusion-guided |
| Emotion | None (dummy AU) | Audio-driven (no direct control yet) | 10+ states |
| Training time | 20 min (3 stages) | **TBD (1 stage)** | 4-5 hours |
| Input video | 10s | 3-5 min | 2 min |
| Real-time serving | No | **Yes (130fps headroom)** | Yes (sub-600ms) |
| Teeth/mouth | Dummy masks | Real teeth masks | Pixel-perfect |
| 3D face model | BFM 2009 | BFM 2009 | Unknown (likely custom) |

---

## Roadmap (revised)

### Phase 1: GaussianTalker Setup & First Clone (days)

Get GaussianTalker running on A100 with our existing data:

- [ ] Clone GaussianTalker repo into `extern/GaussianTalker`
- [ ] Install deps (mostly already have them — add TF 2.8.0 if using DeepSpeech)
- [ ] Add `"vertices"` key to existing `track_params.pt` (10-line script)
- [ ] Handle audio features: either symlink aud_eo.npy + change audio_in_dim, or re-extract DeepSpeech
- [ ] Train first GaussianTalker model on roman dataset
- [ ] Compare quality vs InsTaG output
- [ ] Build OpenFace for real AU extraction (no more dummy data)

### Phase 2: Quality Improvements (days-weeks)

- [ ] Real OpenFace AU extraction for proper blink/expression
- [ ] Real teeth masks (EasyPortrait or alternative segmentation)
- [ ] Higher resolution training
- [ ] Longer input video (3-5 min instead of 10s for more training data)
- [ ] Experiment with training iterations / hyperparameters
- [ ] Torso generation (GaussianTalker has a branch for this)

### Phase 2.5: Diffusion Refinement Pass (days)

**The big quality multiplier.** Use GaussianTalker for 3D structure + lip sync, then run a 1-step diffusion refinement for photorealism. This is a lightweight version of what Tavus does with their "gaussian-diffusion hybrid."

```
Audio → GaussianTalker (130fps, 512x512) → raw 3DGS frame
                                                 ↓
                              StreamDiffusion + SD-turbo (1 step, ~93fps)
                              denoise_strength 0.2-0.3 (preserve structure, refine texture)
                                                 ↓
                                          Photorealistic frame (~60fps combined)
```

- [ ] Install StreamDiffusion (`pip install streamdiffusion`, MIT license)
- [ ] Download SD-turbo weights (~3.4GB, auto from HuggingFace)
- [ ] Test on single GaussianTalker output frame — tune denoise strength
- [ ] Test temporal consistency across frame sequence
- [ ] If identity drift: add ControlNet (face landmarks) or IP-Adapter (identity lock)
- [ ] Benchmark: combined FPS, VRAM usage, latency per frame
- [ ] Compare quality: GaussianTalker alone vs GaussianTalker + diffusion

**Why this works**: Low denoise strength barely changes geometry (lip positions, head angle stay intact) but adds skin pores, lighting subtlety, micro-texture. Like an AI filter that makes 3D renders look like real footage.

**Performance**: StreamDiffusion + SD-turbo does 93fps img2img on RTX 4090. On A100, similar or better. Combined with GaussianTalker's 130fps, pipeline bottlenecks at ~60fps — still faster than Tavus's 40fps.

**VRAM**: SD-turbo is ~3.4GB in fp16. GaussianTalker uses ~4-8GB. A100 80GB has plenty of room for both.

**Risk: identity drift** — diffusion changing the person's face. Mitigations:
1. Low denoise strength (0.2-0.3)
2. ControlNet face landmarks (forces structure preservation)
3. IP-Adapter (conditions on reference photo to lock identity)

**Quality estimate**:
| Setup | Quality (Tavus=100) |
|-------|-------------------|
| GaussianTalker alone (good data) | 40-50 |
| **GaussianTalker + diffusion refinement** | **65-80** |
| Tavus | 100 |

References:
- StreamDiffusion: https://github.com/cumulo-autumn/StreamDiffusion
- SD-turbo: https://huggingface.co/stabilityai/sd-turbo
- StreamDiffusionV2: https://github.com/chenfengxu714/StreamDiffusionV2
- OSA-LCM (related work, standalone 1-step avatar): https://arxiv.org/abs/2412.13479

### Phase 3: Real-Time Pipeline (weeks)

Wire up Being's server with GaussianTalker + diffusion as the rendering backend:

- [ ] Wire `being/inference/engine.py` → GaussianTalker model loading + rendering
- [ ] Integrate StreamDiffusion refinement into render pipeline
- [ ] Streaming audio feature extraction (chunked, causal)
- [ ] Real-time render loop in `being/api/server.py`
- [ ] WebSocket video streaming
- [ ] Benchmark end-to-end latency (target: sub-100ms)

### Phase 4: Emotion & Expression Control (weeks-months)

Since GaussianTalker is purely audio-driven at inference (no face model params), emotion control needs a different approach:

- [ ] **Option A**: Modify deformation network to accept emotion conditioning vector alongside audio
- [ ] **Option B**: Manipulate audio features to encode emotional tone (emotion-conditioned audio embeddings)
- [ ] **Option C**: Add a secondary expression offset network conditioned on emotion labels
- [ ] Perception: read user's webcam with MediaPipe/InsightFace for emotion detection
- [ ] Active listening behavior (nods, micro-expressions while user speaks)

### Phase 5: Production Polish (months)

- [ ] 1080p rendering
- [ ] Background compositing
- [ ] WebRTC delivery for lowest latency
- [ ] Multi-avatar serving
- [ ] Clone creation CLI: `being clone <video>` → trained model in N minutes

---

## Other Models Evaluated

| Model | Why not |
|-------|---------|
| **InsTaG** | 30fps, 3-stage training, MLP audio decoder, needed 5 patches. Superseded by GaussianTalker. |
| **MuseTalk** | 2D only (no 3D head rotation), 256x256 face region. Good for quick demos but not our target. |
| **LivePortrait** | 2D portrait animation. High quality but no 3D, no audio-driven generation. |
| **Hallo3** | Diffusion transformer, great quality, but NOT real-time. |
| **FIAG** | Best architecture (shared Gaussian field) but code not released. Monitor for future adoption. |
| **Splat-Portrait** | Empty repo. |
| **DreamTalk** | Weights access ceased. |

---

## Answered Questions

- **Does GaussianTalker use FLAME or BFM?** → BFM 2009, same as InsTaG. No FLAME anywhere in the codebase.
- **Can we reuse InsTaG's preprocessed data?** → Yes, with 2 minor fixes (add vertices to track_params.pt, handle audio feature dim).
- **What audio features does it expect?** → Default: DeepSpeech 29-dim. Also supports wav2vec 44-dim (same as our InsTaG features). Need config change for audio_in_dim.
- **How does face model relate to runtime?** → BFM vertices initialize Gaussians. At inference, deformation is purely audio+AU45+camera driven. No face model params used.

## Open Questions

- Training time on A100 for GaussianTalker? (Paper tested on RTX 3090 / A6000)
- Can we get 1080p with GaussianTalker or does it cap at a lower resolution?
- Is wav2vec or DeepSpeech better for audio features in practice?
- Best approach for adding emotion conditioning to a purely audio-driven deformation network?
- How does the torso generation branch work and how mature is it?

---

## References

Papers:
- GaussianTalker: https://arxiv.org/abs/2404.16012
- InsTaG: https://arxiv.org/abs/2404.00369
- FIAG: https://arxiv.org/abs/2506.22044
- Hallo3: https://github.com/fudan-generative-vision/hallo3
- DreamTalk: https://arxiv.org/abs/2312.09767

Code:
- GaussianTalker: https://github.com/cvlab-kaist/GaussianTalker
- InsTaG: https://github.com/Fictionarry/InsTaG
- MuseTalk: https://github.com/TMElyralab/MuseTalk
- Awesome Talking Head Synthesis: https://github.com/Kedreamix/Awesome-Talking-Head-Synthesis
