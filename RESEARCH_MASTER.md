# Being — Master Research Document: Next-Generation Talking Head Avatar

**Date:** 2026-02-25
**Goal:** Best possible open-source audio-driven talking head with full frame (head + shoulders), full facial expressions, and real-time rendering.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Proposed Architecture](#2-proposed-architecture)
3. [Avatar Rendering Methods (3DGS)](#3-avatar-rendering-methods)
4. [Audio-to-Expression Models](#4-audio-to-expression-models)
5. [Torso & Shoulders Solutions](#5-torso--shoulders-solutions)
6. [Key Methods Deep Dive](#6-key-methods-deep-dive)
7. [Comparison Tables](#7-comparison-tables)
8. [Recommended Build Plan](#8-recommended-build-plan)
9. [Sources](#9-sources)

---

## 1. Executive Summary

After running GaussianTalker (Rounds 1 & 2) and FlashAvatar, the core lessons are:

| Experiment | Result | Limitation |
|-----------|--------|------------|
| GaussianTalker R1 (10s, 7min video) | Great lips, 130fps, full frame | Upper face frozen (BFM only tracks mouth/jaw) |
| GaussianTalker R2 (30s video) | More expression, PSNR 35.2 | Worse lips (less data), calibration head motion in custom audio |
| FlashAvatar (30s, 150K iter) | 300fps, 5min training | Floating head — no neck/shoulders/torso |

**Root cause:** GaussianTalker uses BFM 2009 (a 14-year-old face model with ~64 expression dims focused on mouth/jaw). The entire field has moved to FLAME (100 expression dims, full-face coverage). Every strong audio-to-expression model outputs FLAME parameters.

**The path forward:** Decouple the pipeline into independent layers, each using the best available method:

```
Audio → ARTalk → FLAME params (100 exp + 6 pose)
                        ↓
FLAME params → Gaussian Avatar → Head render (face + hair)
                        ↓
Head render + Static Torso Layer → Full frame composite
```

---

## 2. Proposed Architecture

### Layer 1: Audio → Expression (ARTalk)
- **Input:** Audio waveform
- **Output:** 100 FLAME expression dims + 6 pose dims per frame
- **Speed:** Real-time
- **Why:** Only method that jointly predicts expression + head pose + blinks in real-time, with style adaptation

### Layer 2: Expression → Gaussian Avatar
- **Input:** FLAME expression/pose parameters
- **Output:** Rendered head (face, hair, ears)
- **Options:** GaussianSpeech pipeline, SplattingAvatar, or custom FLAME-rigged Gaussians
- **Speed:** 100-300+ fps depending on method

### Layer 3: Torso Compositing
- **Input:** Rendered head + static/semi-static torso
- **Output:** Full frame (head + shoulders + neck + background)
- **Options:**
  - GaussianTalker-style unified scene (torso as static Gaussians)
  - Gaussian Head & Shoulders neural texture warping
  - SyncTalk++ Torso Restorer (U-Net inpainting at neck boundary)
- **Speed:** Minimal overhead

### Why Decouple?
- Each component can be trained/tested/swapped independently
- Audio model can be pretrained (ARTalk has weights on HuggingFace)
- Avatar quality isn't bottlenecked by audio model's ability to learn deformation
- Can upgrade any layer without retraining the whole pipeline

---

## 3. Avatar Rendering Methods

### FLAME-Driven Gaussian Splatting (Head Only)

| Method | Venue | Input | PSNR | FPS | Training | Key Feature | Open Source |
|--------|-------|-------|------|-----|----------|-------------|-------------|
| **GaussianAvatars** | CVPR24 H | Multi-view | 31.6 | RT | ~7h | Gaussians rigged to FLAME triangles | [Yes](https://github.com/ShenhanQian/GaussianAvatars) |
| **SplattingAvatar** | CVPR24 | Mono | — | 300+ | — | Mesh-embedded, no MLP needed, also supports SMPL body | [Yes](https://github.com/initialneil/SplattingAvatar) |
| **FlashAvatar** | CVPR24 | Mono | — | 300+ | ~5min | UV-space Gaussians + offset network | [Yes](https://github.com/USTC3DV/FlashAvatar-code) |
| **MonoGaussianAvatar** | SIG24 | Mono | — | RT | — | Adaptive point insertion/deletion | [Yes](https://github.com/yufan1012/MonoGaussianAvatar) |
| **Gaussian Head Avatar** | CVPR24 | Multi-view | 33.9 | RT | ~30h | Ultra high-fidelity, 2K resolution, SDF init | [Yes](https://github.com/YuelangX/Gaussian-Head-Avatar) |
| **GaussianHead** | TVCG25 | Multi/Mono | — | RT | — | Learnable Gaussian derivation, 12MB model | [Yes](https://github.com/chiehwangs/gaussian-head) |
| **NPGA** | SIGA24 | Multi-view | **37.7** | 31 | — | NPHM (richer than FLAME), highest PSNR | Unclear |
| **GAGAvatar** | NeurIPS24 | 1 image | 21.8 | 67 | Forward | One-shot, feed-forward | [Yes](https://github.com/xg-chu/GAGAvatar) |
| **MeGA** | CVPR25 | Multi-view | — | — | — | Hybrid: FLAME mesh face + Gaussian hair | [Yes](https://github.com/conallwang/MeGA) |
| **FATE** | CVPR25 | Mono | — | — | — | 360° head from monocular, editable textures | [Yes](https://github.com/zjwfufu/FateAvatar) |
| **GeoAvatar** | ICCV25 | Mono | — | — | — | Adaptive rigid/flexible regions, strong mouth | TBD |
| **ELITE** | arXiv Jan26 | Mono | — | — | — | Learned init + generative test-time adaptation | Not yet |

### Audio-Driven Full-Frame Methods

| Method | Venue | Face Model | Coverage | FPS | Audio | Open Source |
|--------|-------|-----------|----------|-----|-------|-------------|
| **GaussianTalker** | MM24 | BFM | Head+Torso | 130 | Yes | [Yes](https://github.com/cvlab-kaist/GaussianTalker) |
| **THGS** | CGF25 | Custom | Full Body | 150+ | Yes | Unclear |
| **TaoAvatar** | CVPR25 | SMPLX | Full Body | 90 | Yes | Partial |
| **SyncTalk++** | 2025 | 3DGS | Full Frame | RT | Yes | [Yes](https://github.com/ZiqiaoPeng/SyncTalk) |
| **GaussianSpeech** | ICCV25 | FLAME→3DGS | Head | RT | Yes | [Yes](https://github.com/shivangi-aneja/GaussianSpeech) |
| **InsTaG** | CVPR25 | ER-NeRF | Head+Torso | RT | Yes | [Yes](https://github.com/Fictionarry/InsTaG) |

---

## 4. Audio-to-Expression Models

### Methods That Output FLAME Parameters (Most Relevant)

| Method | Venue | Output | Head Pose? | Real-time? | Open Source |
|--------|-------|--------|-----------|------------|-------------|
| **ARTalk** | SIGA25 | 100 exp + 6 pose | **Yes** | **Yes** | [Yes](https://github.com/xg-chu/ARTalk) + [HF weights](https://huggingface.co/xg-chu/ARTalk) |
| **EMOTE** | SIGA23 | 50 exp + 3 jaw | No | Fast | [Yes](https://github.com/radekd91/inferno) |
| **DiffPoseTalk** | TOG/SIG24 | exp + pose | **Yes** | No (diffusion) | [Yes](https://github.com/DiffPoseTalk/DiffPoseTalk) |
| **EmoTalk** | ICCV23 | Blendshape coeffs | No | Fast | [Yes](https://github.com/psyai-net/EmoTalk_release) |
| **DEEPTalk** | AAAI25 | FLAME VQ codes | No | Maybe | [Yes](https://github.com/whwjdqls/DEEPTalk) |

### End-to-End Audio→Pixels (Less Relevant for Our Architecture)

| Method | Venue | Type | Quality | Open Source |
|--------|-------|------|---------|-------------|
| **VASA-1** | Microsoft 2024 | Disentangled latent + diffusion | Best looking | **NO** (closed) |
| **Hallo / Hallo2 / Hallo3** | 2024-25 | SD 1.5 + audio cross-attention | Good, 4K capable | [Yes](https://github.com/fudan-generative-vision/hallo2) |
| **EchoMimicV2** | CVPR25 | Diffusion, half-body + hands | Good | [Yes](https://github.com/antgroup/echomimic_v2) |
| **OmniAvatar** | 2025 | Wan2.1-T2V-14B + LoRA | Full body | [Yes](https://github.com/Omni-Avatar/OmniAvatar) |

### Top Pick: ARTalk (SIGGRAPH Asia 2025)

ARTalk is the clear winner for audio→expression:
- **Outputs:** Exactly FLAME params (100 expression + 6 pose) per frame
- **Architecture:** Multi-scale VQ autoencoder + autoregressive transformer
- **Real-time:** Yes
- **Style adaptation:** Matches speaker style without retraining
- **Handles jointly:** Lip sync + head pose + blinks (all three in one model)
- **Open source:** Code + pretrained weights on HuggingFace
- **Paper:** [arxiv.org/abs/2502.20323](https://arxiv.org/abs/2502.20323)

### Backup: EMOTE + SadTalker PoseVAE

If ARTalk doesn't work well for our use case:
- EMOTE for expression (50 exp + 3 jaw FLAME params)
- SadTalker's PoseVAE architecture for head motion
- More mature codebases, but separate models = less natural coordination

---

## 5. Torso & Shoulders Solutions

### The Problem
FLAME covers face + scalp + minimal neck. Most avatar methods stop there. For a usable talking head, we need head + neck + shoulders + background.

### Five Strategies in the Literature

| Strategy | Used By | Pros | Cons |
|----------|---------|------|------|
| **Unified Gaussian scene** | GaussianTalker | No seam, simple | Entire scene deforms with audio (less control) |
| **Two separate NeRFs** (head + torso) | AD-NeRF, GeneFace++ | Modular | Visible neck seam |
| **2D torso field** | RAD-NeRF, ER-NeRF | Efficient | Fake torso geometry |
| **Neural inpainting at boundary** | SyncTalk++ (Torso Restorer) | Explicit fix | Extra network, may blur |
| **Unified body model** (SMPL-X + 3DGS) | TaoAvatar, THGS, GUAVA | No seam, physically grounded | Needs body model fitting |

### Most Promising for Us

1. **Gaussian Head & Shoulders** (ICLR 2025) — Head Gaussians + neural texture warping for torso. 130fps. Monocular. Not yet open source but the paper describes the approach clearly.

2. **THGS** (CGF 2025) — Full body from monocular video, audio-driven, 150fps. Learnable Expression Blendshapes + Spatial Audio Attention. Most direct alternative to GaussianTalker.

3. **GUAVA** (ICCV 2025) — Upper body from single image, sub-second reconstruction. Uses Enhanced Human Model (EHM) extending SMPL-X. Open source: [github.com/Pixel-Talk/GUAVA](https://github.com/Pixel-Talk/GUAVA)

4. **SyncTalk++ Torso Restorer** — Bolt-on U-Net inpainting to fix any neck boundary artifacts. Can be added to any head method.

### Practical Approach for Being

For the torso, we don't need it to be dynamic — in a talking head scenario, shoulders barely move. The simplest approach:
- Render the head with a FLAME-driven Gaussian avatar
- Keep the torso/shoulders/background as a static or minimally-animated layer
- Composite at the neck with either:
  - Direct blending (if the boundary is clean)
  - SyncTalk++-style inpainting (if there are artifacts)
  - Gaussian Head & Shoulders neural texture warping (highest quality)

---

## 6. Key Methods Deep Dive

### GaussianSpeech (ICCV 2025) — The Blueprint

This is the closest existing system to what we want to build. From TU Munich (same group as FaceTalk, NPGA).

**Pipeline:**
1. Audio (wav2vec 2.0) → audio-conditioned transformer → lip features + expression features
2. Expression features → FLAME expression params
3. FLAME expression → Expression2Latent MLP (compact encoding)
4. Latent + lip features → motion decoder → per-vertex offsets on FLAME mesh
5. Offset mesh + bound 3D Gaussians → differentiable rendering

**Key insights:**
- Expression-dependent color (wrinkles change with expression)
- Wrinkle-aware + perceptual losses
- Gaussians bound to FLAME mesh triangles
- Open source: [github.com/shivangi-aneja/GaussianSpeech](https://github.com/shivangi-aneja/GaussianSpeech)

**Limitation:** Requires multi-view capture for training. But the architecture is instructive.

### ARTalk (SIGGRAPH Asia 2025) — The Audio Engine

**Architecture:**
1. Multi-scale VQ autoencoder encodes motion at multiple temporal resolutions into shared codebook
2. Autoregressive transformer generates motion codes from audio
3. Decoder produces FLAME expression (100) + pose (6) parameters

**Key insights:**
- VQ tokenization prevents over-smoothing (vs regression-to-mean)
- Multi-scale temporal modeling captures both fast lip movements and slow head gestures
- Style adaptation via style reference (short video clip)
- Pretrained weights available on HuggingFace

### Gaussian Head & Shoulders (ICLR 2025) — The Torso Solution

**Architecture:**
- Head: 3DMM + 3D Gaussians (standard approach)
- Torso: Sparse anchor Gaussians + neural texture warping field
  - Key insight: Naive Gaussians on torso are blurry because each has single directional radiance
  - Neural texture warping maps image coordinates to texture space, guided by sparse Gaussian anchors
  - Coarse color + pose-dependent fine color for clothing/body detail

**130fps from monocular video.** Supports cross-reenactment.

---

## 7. Comparison Tables

### Face Models

| Model | Year | Coverage | Expression Dims | Notes |
|-------|------|----------|----------------|-------|
| **BFM 2009** | 2009 | Face only | 29-64 | GaussianTalker uses this. Weak upper face. |
| **FLAME** | 2017/2020 | Head + neck | 100 expression, 300 shape | Industry standard. All top methods use this. |
| **NPHM** | 2023 | Full head | Learned latent | Richer than FLAME. Used by NPGA (37.7 PSNR). |
| **SMPL-X** | 2019 | Full body + face + hands | 10 expression (FLAME) | Gold standard for full-body. |

### Complete Method Ranking (Our Criteria)

Weighted by: full-frame rendering, expression quality, audio driving, monocular input, open source, real-time.

| Rank | Method | Score | Why |
|------|--------|-------|-----|
| 1 | **GaussianSpeech** | Architecture blueprint | Audio→FLAME→Gaussians pipeline, exactly our target |
| 2 | **THGS** | Best alternative | Monocular + audio + full body + 150fps |
| 3 | **SplattingAvatar** | Best head renderer | 300fps, FLAME-driven, also supports SMPL body, open source |
| 4 | **ARTalk** | Best audio model | Real-time FLAME output, style adaptation, open source |
| 5 | **GH&S** | Best torso solution | Neural texture warping for body, 130fps |
| 6 | **SyncTalk++** | Practical integration | Torso Restorer bolt-on, open source |
| 7 | **GUAVA** | Upper body from 1 image | SMPL-X, sub-second, open source |

---

## 8. Recommended Build Plan

### Phase 1: Validate ARTalk (1-2 days)
- Clone ARTalk, load pretrained weights
- Run inference on our 30s recording audio
- Verify output quality: are FLAME params reasonable? Do blinks/eyebrows come through?
- Test with custom audio (the test_audio.wav we already have)

### Phase 2: Set Up FLAME-Driven Gaussian Avatar (3-5 days)
- Option A: Adapt GaussianSpeech pipeline (if their training works with monocular)
- Option B: Use SplattingAvatar with FLAME (monocular, 300fps, open source)
- Option C: Modify GaussianTalker to use FLAME instead of BFM (most invasive)
- Train on our 30s video with metrical-tracker FLAME params (already have 838 frames!)

### Phase 3: Add Torso Layer (2-3 days)
- Extract static torso/background from training video
- Implement compositing (start simple: direct blend at neck)
- If artifacts: add SyncTalk++ Torso Restorer
- If quality insufficient: implement GH&S neural texture warping

### Phase 4: Integration & Audio Pipeline (2-3 days)
- Wire ARTalk output → avatar deformation input
- End-to-end: audio file → rendered video
- Real-time streaming: audio chunk → FLAME params → render → frame

### Phase 5: Quality Polish (ongoing)
- Longer training, more data
- Expression fine-tuning on our speaker
- Lip sync quality optimization
- Jitter/stability improvements

### Compute Recommendations
- **For GPU training:** A100 80GB is fine. The bottleneck hasn't been GPU.
- **For CPU preprocessing:** RunPod A100 pods have weak CPUs. Options:
  - Lambda Labs (better CPU/GPU balance)
  - CoreWeave
  - Or split: local CPU preprocessing + cloud GPU training
- **For iteration speed:** The decoupled architecture means we can test audio model and avatar independently, cutting iteration time dramatically.

---

## 9. Sources

### Avatar Rendering
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) — CVPR 2024 Highlight
- [SplattingAvatar](https://github.com/initialneil/SplattingAvatar) — CVPR 2024
- [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) — CVPR 2024
- [MonoGaussianAvatar](https://github.com/yufan1012/MonoGaussianAvatar) — SIGGRAPH 2024
- [Gaussian Head Avatar](https://github.com/YuelangX/Gaussian-Head-Avatar) — CVPR 2024
- [GaussianHead](https://github.com/chiehwangs/gaussian-head) — TVCG 2025
- [GAGAvatar](https://github.com/xg-chu/GAGAvatar) — NeurIPS 2024
- [GaussianHeads](https://github.com/Kartik-Teotia/GaussianHeads) — SIGGRAPH Asia 2024
- [MeGA](https://github.com/conallwang/MeGA) — CVPR 2025
- [FATE](https://github.com/zjwfufu/FateAvatar) — CVPR 2025
- [GeoAvatar](https://openaccess.thecvf.com/content/ICCV2025/papers/Moon_GeoAvatar_Adaptive_Geometrical_Gaussian_Splatting_for_3D_Head_Avatar_ICCV_2025_paper.pdf) — ICCV 2025
- [NPGA](https://simongiebenhain.github.io/NPGA/) — SIGGRAPH Asia 2024
- [GaussianSpeech](https://github.com/shivangi-aneja/GaussianSpeech) — ICCV 2025

### Audio-to-Expression
- [ARTalk](https://github.com/xg-chu/ARTalk) — SIGGRAPH Asia 2025 | [HuggingFace](https://huggingface.co/xg-chu/ARTalk)
- [EMOTE](https://github.com/radekd91/inferno) — SIGGRAPH Asia 2023
- [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk) — TOG/SIGGRAPH 2024
- [EmoTalk](https://github.com/psyai-net/EmoTalk_release) — ICCV 2023
- [SadTalker](https://github.com/OpenTalker/SadTalker) — CVPR 2023
- [CodeTalker](https://github.com/Doubiiu/CodeTalker) — CVPR 2023
- [FaceFormer](https://github.com/EvelynFan/FaceFormer) — CVPR 2022
- [DEEPTalk](https://github.com/whwjdqls/DEEPTalk) — AAAI 2025

### Torso & Full-Body
- [GaussianTalker](https://github.com/cvlab-kaist/GaussianTalker) — ACM MM 2024
- [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk) — CVPR 2024
- [Gaussian Head & Shoulders](https://arxiv.org/abs/2405.12069) — ICLR 2025
- [TaoAvatar](https://pixelai-team.github.io/TaoAvatar/) — CVPR 2025
- [THGS](https://sora158.github.io/THGS.github.io/) — CGF 2025
- [GUAVA](https://github.com/Pixel-Talk/GUAVA) — ICCV 2025
- [InsTaG](https://github.com/Fictionarry/InsTaG) — CVPR 2025
- [ER-NeRF](https://github.com/Fictionarry/ER-NeRF) — ICCV 2023
- [EchoMimicV2](https://github.com/antgroup/echomimic_v2) — CVPR 2025
- [OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) — 2025

### Body Models
- [FLAME](https://github.com/TimoBolkart/FLAME-Universe)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/)
- [FLAME2SMPLX](https://github.com/CvHadesSun/FLame2SMPLX)

### Commercial / Production
- [Tavus Phoenix](https://www.tavus.io/post/advanced-techniques-in-talking-head-generation-3d-gaussian-splatting) — 3DGS-based, 60+ FPS
- [HeyGen Avatar IV](https://www.heygen.com/avatars) — Full-body motion, micro-expressions
- [Apple Vision Pro Personas](https://www.roadtovr.com/vision-pro-persona-avatar-upgrade-visionos-26/) — Gaussian splatting, visionOS 26
- [Meta Codec Avatars](https://www.meta.com/emerging-tech/codec-avatars/) — Multi-view capture, highest fidelity
- [VASA-1](https://arxiv.org/abs/2404.10667) — Microsoft, best visual quality, completely closed

### Curated Lists
- [Awesome Talking Head Synthesis](https://github.com/Kedreamix/Awesome-Talking-Head-Synthesis)
