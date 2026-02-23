# FLAME-Driven 3DGS Avatar Models: Research & Comparison

**Date:** 2026-02-23
**Goal:** Replace GaussianTalker (audio + AU45 blink only) with a FLAME-driven 3DGS avatar that gives full expression control (eyebrows, eyes, mouth, cheeks) while maintaining real-time rendering.

**Our Setup:**
- Training: A100 80GB on RunPod
- Inference: RTX 4070 Super (local)
- Training video: 6:43 at 1080p, monocular
- Target: 25+ FPS real-time, full expression control + audio lip sync

---

## Table of Contents

1. [FLAME Primer](#flame-primer)
2. [Avatar Models Comparison](#avatar-models-comparison)
3. [Audio-to-FLAME Models](#audio-to-flame-models)
4. [Proposed Pipeline Architecture](#proposed-pipeline-architecture)
5. [Recommendation](#recommendation)
6. [Implementation Plan](#implementation-plan)

---

## FLAME Primer

FLAME (Faces Learned with an Articulated Model and Expressions) is a parametric 3D head model with:
- **300 shape parameters** (identity — fixed per person)
- **100 expression parameters** (facial expressions — eyebrows, cheeks, mouth shape, nose wrinkle, etc.)
- **3 jaw pose parameters** (jaw open/close/lateral)
- **6 eye pose parameters** (eye gaze direction, 3 per eye)
- **6 neck/head pose parameters** (global head rotation + translation)

Total controllable dimensions per frame: **100 expression + 3 jaw + 6 eyes + 6 pose = 115 dimensions**

This is vastly richer than GaussianTalker's input (audio features + 1 blink AU). With FLAME, every facial muscle group is independently controllable.

---

## Avatar Models Comparison

### 1. FlashAvatar

| Property | Value |
|---|---|
| **Paper** | CVPR 2024 |
| **GitHub** | https://github.com/USTC3DV/FlashAvatar-code |
| **Stars** | ~226 |
| **License** | MIT |
| **Code available** | Yes, full code |
| **Pretrained weights** | Example data + pretrained available |
| **Input type** | Monocular video |
| **FLAME params** | 100 expression + jaw pose (uses FLAME tracked expression codes) |
| **Render FPS** | **300+ FPS** at 512x512 on RTX 3090 |
| **Number of Gaussians** | ~10K (very compact) |
| **Training time** | **Minutes** (couple of minutes for photo-realistic quality) |
| **GPU requirement** | RTX 3090 (24GB) — tested |
| **Preprocessing** | FLAME tracking (DECA/MICA-based), alpha matting, semantic parsing |
| **Audio lip sync** | No — expression-parameter driven only (needs external audio→FLAME) |

**Strengths:**
- Blazing fast render (300+ FPS) — great for our RTX 4070 Super
- Very fast training (minutes, not hours)
- Extremely compact model (~10K Gaussians, ~12MB)
- Full FLAME expression control
- Monocular video input (exactly what we have)
- MIT license
- Clean, well-maintained codebase

**Weaknesses:**
- Back-of-head and hair can be poor from monocular data (common to all monocular methods)
- Struggles with out-of-distribution expressions (overfitting on training expressions)
- Cannot model dynamic hair well (hair is conditioned on FLAME mesh surface)
- Needs external audio→FLAME pipeline for speech

**Architecture:** Maintains 3D Gaussians in 2D UV space embedded on FLAME mesh surface. An offset MLP (depth 5, hidden dim 256) takes expression codes + canonical position → outputs position/rotation/scale deformation. Proper UV-based initialization keeps Gaussian count low.

---

### 2. GaussianAvatars

| Property | Value |
|---|---|
| **Paper** | CVPR 2024 Highlight |
| **GitHub** | https://github.com/ShenhanQian/GaussianAvatars |
| **Stars** | ~949 |
| **License** | CC-BY-NC-SA-4.0 (non-commercial) |
| **Code available** | Yes, full code + training |
| **Pretrained weights** | Demo model included |
| **Input type** | **Multi-view** (NeRSemble 16-camera) — monocular via VHAP possible but degraded |
| **FLAME params** | Full FLAME (expression + jaw + pose), one Gaussian per FLAME triangle (9,976) |
| **Render FPS** | ~50-100 FPS (estimated, no MLP evaluation needed at inference) |
| **Training time** | 30K iterations default |
| **GPU requirement** | Not specified (likely 24GB+) |
| **Preprocessing** | VHAP pipeline (supports monocular), FLAME tracking |
| **Audio lip sync** | No — FLAME parameter driven |

**Strengths:**
- CVPR Highlight — very well-regarded
- Elegant design: one Gaussian per FLAME triangle, moves with mesh
- No neural network at inference (just mesh deformation) — very efficient
- VHAP preprocessing pipeline supports monocular video (same author)
- Full FLAME controllability
- Most-starred repo in this category

**Weaknesses:**
- Originally designed for multi-view — quality degrades significantly with monocular input
- "Drops dramatically in performance with sparse input"
- Non-commercial license (CC-BY-NC-SA)
- Fixed topology (one Gaussian per triangle) may limit detail
- No hair/non-face modeling beyond FLAME mesh extent

**Architecture:** One 3D Gaussian initialized at center of each FLAME triangle. Gaussian is static in local triangle space but moves globally as triangle deforms. Additional learnable offsets per Gaussian. Binding is key innovation — no MLP needed at inference time.

---

### 3. SplattingAvatar

| Property | Value |
|---|---|
| **Paper** | CVPR 2024 |
| **GitHub** | https://github.com/initialneil/SplattingAvatar |
| **Stars** | ~540 |
| **License** | CC-BY-NC-SA-4.0 (non-commercial) |
| **Code available** | Yes, full code |
| **Pretrained weights** | Checkpoints via INSTA dataset |
| **Input type** | Monocular video (IMavatar pipeline) |
| **FLAME params** | FLAME 2020 (100 expression + jaw) |
| **Render FPS** | **300+ FPS** on RTX 3090, **30 FPS on iPhone 13** |
| **Training time** | Not specified (likely ~1 hour range) |
| **GPU requirement** | RTX 3090 (24GB), adjustable via max_n_gauss |
| **Preprocessing** | IMavatar pipeline, RobustVideoMatting segmentation |
| **Audio lip sync** | No — FLAME parameter driven |

**Strengths:**
- 300+ FPS rendering — matches FlashAvatar
- 30 FPS on mobile — impressive cross-platform
- "Walking on triangles" optimization for Gaussian-mesh binding
- Monocular video input
- Full FLAME expression control
- Well-documented, multiple dataset support

**Weaknesses:**
- Non-commercial license
- IMavatar preprocessing pipeline (somewhat complex)
- Less community momentum than FlashAvatar/GaussianAvatars
- Limited to FLAME mesh topology for deformation

**Architecture:** Gaussians embedded on FLAME mesh via barycentric coordinates + displacement. Uses Phong surface interpolation for smooth deformation. Novel "walking on triangle" scheme allows Gaussians to slide across mesh surface during optimization.

---

### 4. MonoGaussianAvatar

| Property | Value |
|---|---|
| **Paper** | ACM SIGGRAPH 2024 |
| **GitHub** | https://github.com/yufan1012/MonoGaussianAvatar |
| **Stars** | ~145 |
| **License** | MIT |
| **Code available** | Yes |
| **Pretrained weights** | Available (Google Drive) |
| **Input type** | Monocular video |
| **FLAME params** | FLAME 2020 (full expression + pose) |
| **Render FPS** | ~300 FPS (reported) |
| **Training time** | Minutes (comparable to FlashAvatar) |
| **GPU requirement** | RTX 3090 (24GB) |
| **Preprocessing** | IMavatar pipeline |
| **Audio lip sync** | No — FLAME parameter driven |

**Strengths:**
- SIGGRAPH venue (top quality)
- Monocular video input
- MIT license
- Fast training and rendering
- Continuous deformation field extends FLAME deformation beyond mesh surface
- Can model teeth, eyeglasses, hair to some extent

**Weaknesses:**
- Smallest community (145 stars)
- IMavatar preprocessing dependency
- Fewer issues/discussions for troubleshooting
- Less battle-tested than FlashAvatar

**Architecture:** Gaussian deformation field maps canonical Gaussians to target pose/expression using learned blendshapes + skinning weights (similar to IMavatar). Extends FLAME deformations into continuous space, allowing Gaussians beyond mesh surface.

---

### 5. Gaussian Head Avatar

| Property | Value |
|---|---|
| **Paper** | CVPR 2024 |
| **GitHub** | https://github.com/YuelangX/Gaussian-Head-Avatar |
| **Stars** | ~856 |
| **License** | Not specified |
| **Code available** | Yes |
| **Pretrained weights** | Mini demo dataset |
| **Input type** | **Multi-view** (NeRSemble) |
| **Expression model** | **BFM (Basel Face Model)** — NOT FLAME |
| **Render FPS** | Not specified (likely moderate due to MLP) |
| **Training time** | **1-2 days** (600K iterations) |
| **GPU requirement** | CUDA 11.3 capable GPU |
| **Preprocessing** | Multi-view BFM fitting to 2D landmarks |
| **Audio lip sync** | No |

**Strengths:**
- Ultra high-fidelity results
- Well-cited, second-most starred
- Two-stage pipeline (geometry guidance + Gaussian refinement)

**Weaknesses:**
- **Multi-view only** — unusable for our monocular setup
- Uses BFM, not FLAME (less expression control, different ecosystem)
- Very long training (1-2 days)
- Would need complete pipeline rework

**Verdict:** NOT SUITABLE — multi-view requirement and BFM (not FLAME) make this incompatible.

---

### 6. GaussianBlendshapes

| Property | Value |
|---|---|
| **Paper** | SIGGRAPH 2024 |
| **GitHub** | https://github.com/zjumsj/GaussianBlendshapes |
| **Stars** | ~188 |
| **License** | GPL-3.0 |
| **Code available** | Yes (Python + CUDA + C++) |
| **Pretrained weights** | Some available (SharePoint) |
| **Input type** | Monocular video |
| **FLAME params** | FLAME-based blendshapes (expression coefficients via linear blending) |
| **Render FPS** | Real-time (Gaussian splatting) |
| **Training time** | Not specified |
| **GPU requirement** | RTX 3090 / A800 tested |
| **Preprocessing** | Metrical Photometric Tracker + INSTA pipeline |
| **Audio lip sync** | No — expression coefficient driven |

**Strengths:**
- Elegant blendshape representation (neutral + expression bases = linear combo)
- SIGGRAPH quality
- Monocular input
- Captures high-frequency details well
- Natural expression interpolation via linear blending

**Weaknesses:**
- GPL-3.0 license (copyleft)
- Complex preprocessing (Metrical Tracker)
- Smaller community
- More complex architecture (need to learn basis blendshapes)

**Architecture:** Learns a neutral Gaussian model + N expression blendshapes (each a set of Gaussians). Final avatar = neutral + sum(coefficient_i * blendshape_i). Elegant but requires learning all basis shapes.

---

### 7. GaussianSpeech (BONUS — Audio + FLAME + 3DGS)

| Property | Value |
|---|---|
| **Paper** | ICCV 2025 |
| **GitHub** | https://github.com/shivangi-aneja/GaussianSpeech |
| **Stars** | ~182 |
| **Code available** | **NO — "Coming Soon"** |
| **Input type** | Multi-view (16 cameras, 3.5h recording per person) |
| **Audio lip sync** | **Yes — built-in audio→expression transformer** |

**This is exactly what we want** (audio-driven + FLAME + 3DGS) but:
- Code not released
- Requires multi-view capture (16 cameras!)
- Massive training data (3.5 hours per person)
- **Not usable currently**

---

### 8. FastTalker / EGSTalker (Newer audio-driven 3DGS)

**FastTalker** (2025): Audio-driven, uses FLAME mesh + Dynamic Neural Skinning. >100 FPS. **No public code found.**

**EGSTalker** (IEEE SMC 2025): Audio-driven, 3-5 min video input, KAN-based deformation. **No public code found.**

Both are promising but lack public implementations.

---

## Summary Table

| Model | Monocular | FLAME | Render FPS | Train Time | Audio Built-in | Code Ready | License |
|---|---|---|---|---|---|---|---|
| **FlashAvatar** | Yes | Yes (100 exp) | 300+ | Minutes | No | Yes | MIT |
| **GaussianAvatars** | Degraded | Yes (full) | 50-100 | Hours | No | Yes | CC-NC-SA |
| **SplattingAvatar** | Yes | Yes (100 exp) | 300+ | ~1h | No | Yes | CC-NC-SA |
| **MonoGaussianAvatar** | Yes | Yes (full) | ~300 | Minutes | No | Yes | MIT |
| **Gaussian Head Avatar** | No | No (BFM) | Moderate | 1-2 days | No | Yes | Unclear |
| **GaussianBlendshapes** | Yes | Yes (blendshape) | Real-time | Hours | No | Yes | GPL-3.0 |
| **GaussianSpeech** | No | Yes | Unknown | Unknown | Yes | **No** | Unknown |

---

## Audio-to-FLAME Models

Since all viable avatar models are expression-parameter driven (not audio-driven), we need a separate audio→FLAME pipeline.

### 1. DiffPoseTalk (RECOMMENDED)

| Property | Value |
|---|---|
| **Paper** | SIGGRAPH 2024 (ACM TOG) |
| **GitHub** | https://github.com/DiffPoseTalk/DiffPoseTalk |
| **Stars** | ~343 |
| **Code + pretrained** | **Yes — both available** |
| **Output** | FLAME expression + jaw pose (+ optional head pose) |
| **Audio encoder** | HuBERT |
| **Method** | Diffusion model with style encoder |
| **Style transfer** | Yes — extract style from reference video |
| **Head motion** | Optional (can enable/disable) |
| **Training data** | HDTF dataset |

**Why this is the best choice:**
- Most complete pipeline: audio → FLAME expression + jaw + head pose
- Diffusion-based = higher diversity and naturalness than regression
- Style control lets you match the speaking style of your training video
- Both pretrained models available (with/without head motion)
- Good lip sync quality
- Active maintainance
- SIGGRAPH venue

---

### 2. EMOTE (via Inferno)

| Property | Value |
|---|---|
| **Paper** | SIGGRAPH Asia 2023 |
| **GitHub** | https://github.com/radekd91/inferno |
| **Output** | FLAME 50 expression + 3 jaw pose per frame |
| **Audio encoder** | Wav2Vec 2.0 |
| **Emotion control** | Yes (intensity: mild/medium/high) |
| **Style** | Subject-specific speaking style |
| **Code + pretrained** | Code yes, models via TalkingHead subfolder |

**Pros:** Emotion control is unique — can specify emotion + intensity alongside audio.
**Cons:** Only 50 expression dims (not full 100), complex installation via Inferno framework, training code still "coming soon" for some components.

---

### 3. CodeTalker

| Property | Value |
|---|---|
| **Paper** | CVPR 2023 |
| **GitHub** | https://github.com/Doubiiu/CodeTalker |
| **Stars** | ~600+ |
| **Output** | FLAME expression parameters (or mesh vertices) |
| **Method** | VQ-VAE with discrete motion codebook |
| **Pretrained** | Yes (BIWI + VOCASET) |
| **Code** | Yes |

**Pros:** Well-established, discrete motion prior captures natural movements.
**Cons:** Older (2023), less diverse outputs than diffusion-based methods, lip sync occasionally struggles.

---

### 4. FaceFormer

| Property | Value |
|---|---|
| **Paper** | CVPR 2022 |
| **GitHub** | https://github.com/EvelynFan/FaceFormer |
| **Stars** | ~800+ |
| **Output** | FLAME mesh vertices (can extract params) |
| **Method** | Transformer, autoregressive |
| **Pretrained** | Yes (BIWI) |

**Pros:** Pioneer work, well-tested, simple architecture.
**Cons:** Oldest method, outputs vertices not FLAME params directly, less expressive than newer methods.

---

### Audio-to-FLAME Recommendation: **DiffPoseTalk**

DiffPoseTalk wins because:
1. Outputs FLAME expression + jaw + head pose directly
2. Diffusion-based = best diversity and naturalness
3. Style transfer from reference video
4. Both code and pretrained models available
5. Most recent and highest-quality results

---

## Proposed Pipeline Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Audio WAV   │ ──> │   DiffPoseTalk   │ ──> │  FLAME Params    │
│  (real-time) │     │  (HuBERT + Diff) │     │  (exp + jaw +    │
│              │     │                  │     │   head pose)     │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                    ┌──────────────────┐               │
                    │  Expression      │               │
                    │  Override/Blend  │ <─── Manual control (eyebrows,
                    │  (optional)      │      eyes, emotions via UI)
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐     ┌──────────────────┐
                    │  FlashAvatar     │ ──> │  Rendered Frame   │
                    │  (3DGS Render)   │     │  512x512, 300FPS │
                    │  ~10K Gaussians  │     │  on RTX 4070 S   │
                    └──────────────────┘     └──────────────────┘
```

**Key advantage of this two-stage approach:**
- **Full expression control**: DiffPoseTalk provides lip sync from audio, but you can override/blend ANY of the 100 expression parameters
- **Eyebrow control**: Override expression dims for brow raise/furrow
- **Eye control**: Override gaze direction (6 eye params) + blink (expression dims for eyelid)
- **Emotion overlay**: Blend in pre-defined emotion vectors on top of speech output
- **Manual puppeteering**: Drive any parameter from a UI slider in real-time

---

## Recommendation

### Primary Choice: FlashAvatar + DiffPoseTalk

**Why FlashAvatar over alternatives:**

1. **MIT License** — no commercial restrictions (vs CC-NC-SA for GaussianAvatars/SplattingAvatar)
2. **300+ FPS** at 512x512 — will easily hit 25+ FPS even at higher res on RTX 4070 Super
3. **Minutes to train** — vs hours for others. Iteration speed is crucial.
4. **Monocular native** — designed for exactly our data type
5. **~10K Gaussians** — tiny model, fast to load, low VRAM
6. **Full FLAME expression control** — 100 expression dims + jaw + eyes
7. **Clean codebase** — 100% Python, well-structured

**Why not GaussianAvatars (despite more stars):**
- Originally multi-view → quality degrades badly from monocular
- Non-commercial license

**Why not SplattingAvatar:**
- Non-commercial license
- Comparable performance to FlashAvatar but less community adoption

**Why not MonoGaussianAvatar:**
- Also good (MIT, monocular, fast), but smaller community and less documentation
- Could be a backup option

### Fallback: MonoGaussianAvatar + DiffPoseTalk

Same pipeline but swap in MonoGaussianAvatar if FlashAvatar has issues with our specific video (also MIT license, monocular, FLAME-driven, fast training).

---

## Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Back-of-head/hair quality from monocular | We have 6:43 of video — ensure diverse head angles in training frames |
| FlashAvatar overfitting on training expressions | Include diverse expressions in training video; test with held-out expressions |
| DiffPoseTalk lip sync quality | Can fine-tune on our FLAME-tracked video for person-specific style |
| FLAME tracking quality on our video | Use PeizhiYan/flame-head-tracker (robust, well-maintained) or VHAP |
| Expression dimension mismatch between DiffPoseTalk and FlashAvatar | Both use standard FLAME — dims should match; verify at integration |
| Latency of DiffPoseTalk for real-time | DiffPoseTalk uses diffusion (may be slow); can cache/batch or use lighter model |

---

## Implementation Plan

### Phase 1: FLAME Preprocessing (Day 1)

1. **Install FLAME head tracker**
   ```bash
   git clone https://github.com/PeizhiYan/flame-head-tracker
   # OR
   git clone https://github.com/ShenhanQian/VHAP
   ```

2. **Register and download FLAME 2020 model** from https://flame.is.tue.mpg.de/

3. **Run FLAME tracking on our 6:43 video**
   - Output: per-frame FLAME parameters (shape 300, expression 100, jaw 3, eyes 6, pose 6)
   - Processing time: ~1-2s/frame on GPU → ~20-40 min for our video at 25fps

4. **Extract frames + alpha mattes + semantic parsing** (FlashAvatar preprocessing)

### Phase 2: FlashAvatar Training (Day 1-2)

5. **Clone and set up FlashAvatar**
   ```bash
   git clone https://github.com/USTC3DV/FlashAvatar-code extern/FlashAvatar
   cd extern/FlashAvatar
   # conda env, dependencies, FLAME model placement
   ```

6. **Train FlashAvatar on our data**
   - Expected: couple of minutes on A100
   - Output: trained Gaussian avatar model (~12MB)

7. **Test rendering with FLAME parameter replay**
   - Replay tracked FLAME params → verify visual quality
   - Test novel expressions (smile wider, raise eyebrows, etc.)

### Phase 3: Audio-to-FLAME Pipeline (Day 2-3)

8. **Set up DiffPoseTalk**
   ```bash
   git clone https://github.com/DiffPoseTalk/DiffPoseTalk extern/DiffPoseTalk
   # Download pretrained models
   ```

9. **Test DiffPoseTalk with sample audio**
   - Verify FLAME parameter output format
   - Check lip sync quality

10. **Wire DiffPoseTalk output → FlashAvatar input**
    - Map FLAME expression + jaw → FlashAvatar expression codes
    - Test end-to-end: audio WAV → FLAME params → rendered video

### Phase 4: Real-Time Integration (Day 3-4)

11. **Build real-time pipeline in Being**
    - Audio streaming → DiffPoseTalk chunked inference → FlashAvatar render
    - Target: 25+ FPS end-to-end

12. **Add expression override API**
    - Allow blending manual expression params on top of audio-driven params
    - Expose eyebrow/eye/emotion controls

13. **WebSocket streaming** (already scaffolded in Being)
    - Plug in new render pipeline to existing `being/api/server.py`

### Phase 5: Refinement (Day 4+)

14. **Fine-tune DiffPoseTalk** on our person's speaking style (optional)
15. **Optimize inference** — profile and optimize for RTX 4070 Super
16. **A/B compare** with GaussianTalker output quality

---

## FLAME Tracking Tool Comparison

| Tool | Speed | Quality | Monocular | Ease of Use |
|---|---|---|---|---|
| **PeizhiYan/flame-head-tracker** | ~1-2s/frame | High (photometric + landmark) | Yes | Good (notebook examples) |
| **VHAP** (GaussianAvatars author) | ~batch accelerated | High | Yes (added support) | Good (NeRF/3DGS-ready output) |
| **Metrical Tracker** | Moderate | High | Yes | More complex setup |
| **DECA** | Fast | Moderate | Yes | Easy but less accurate |

**Recommendation:** Start with PeizhiYan/flame-head-tracker (100 expression dims, well-documented, outputs exactly what FlashAvatar needs). Fall back to VHAP if issues.

---

## Sources

- [FlashAvatar Project](https://ustc3dv.github.io/FlashAvatar/) | [GitHub](https://github.com/USTC3DV/FlashAvatar-code) | [Paper](https://arxiv.org/abs/2312.02214)
- [GaussianAvatars Project](https://shenhanqian.github.io/gaussian-avatars) | [GitHub](https://github.com/ShenhanQian/GaussianAvatars) | [Paper](https://arxiv.org/abs/2312.02069)
- [SplattingAvatar Project](https://initialneil.github.io/SplattingAvatar) | [GitHub](https://github.com/initialneil/SplattingAvatar)
- [MonoGaussianAvatar GitHub](https://github.com/yufan1012/MonoGaussianAvatar) | [Paper](https://arxiv.org/abs/2312.04558)
- [Gaussian Head Avatar GitHub](https://github.com/YuelangX/Gaussian-Head-Avatar)
- [GaussianBlendshapes GitHub](https://github.com/zjumsj/GaussianBlendshapes)
- [GaussianSpeech Project](https://shivangi-aneja.github.io/projects/gaussianspeech/) | [GitHub](https://github.com/shivangi-aneja/GaussianSpeech)
- [DiffPoseTalk Project](https://diffposetalk.github.io/) | [GitHub](https://github.com/DiffPoseTalk/DiffPoseTalk)
- [EMOTE/Inferno GitHub](https://github.com/radekd91/inferno)
- [CodeTalker GitHub](https://github.com/Doubiiu/CodeTalker)
- [FaceFormer GitHub](https://github.com/EvelynFan/FaceFormer)
- [FLAME Head Tracker](https://github.com/PeizhiYan/flame-head-tracker)
- [VHAP Pipeline](https://github.com/ShenhanQian/VHAP)
- [FLAME Model](https://flame.is.tue.mpg.de/)
- [FLAME-Universe (resource list)](https://github.com/TimoBolkart/FLAME-Universe)
