# Modernizing GaussianTalker — A Vision for Being v2

**Date:** 2026-02-25

## The Thesis

Every component of GaussianTalker is replaceable with something faster and better. Nobody has assembled the modern versions into a clean open-source pipeline. That's what Being becomes.

GaussianTalker proves the core idea works: 3D Gaussian Splatting can render a photorealistic talking head with shoulders, hair, neck, and background at 130fps. The architecture is sound. The individual parts are outdated.

## Can We Hit 1080p?

**Yes.** Here's why it's achievable:

1. **3DGS is resolution-independent.** Gaussians are continuous 3D primitives — you rasterize them at whatever resolution you want. The quality ceiling is determined by how many Gaussians you have and how well they're placed, not by a fixed texture size. GaussianTalker already renders full-frame — going from 512x512 to 1080p is a rasterization setting, not an architecture change.

2. **The bottleneck is training data quality, not the renderer.** If you track a 1080p video with accurate face parameters and train the Gaussian field on 1080p frames, the splats learn 1080p-level detail. Current GaussianTalker downscales to 512x512 during training — that's a choice, not a limit.

3. **VRAM is the real constraint.** More Gaussians at higher resolution = more memory. An RTX 4090 (24GB) comfortably handles ~500K Gaussians at 512x512. For 1080p you'd need ~1-2M Gaussians and ~16-20GB just for rendering. Tight but doable on current hardware. Future GPUs make this trivial.

4. **Super-resolution as a bridge.** Render at 512x512 or 768x768, then use a lightweight neural upscaler (Real-ESRGAN, SwinIR) to hit 1080p. GAGAvatar already does this — renders at 256 internally, upscales to 512. Adding a 2x upscale pass at ~5ms per frame keeps you well above real-time.

## The Component Swap — What a Modern Version Looks Like

### 1. Face Tracking: BFM Iterative → EMOCA/FLAME Neural

**Current (GaussianTalker):** Basel Face Model 2009, iterative optimization. 200 3D face scans from 1999. Per-frame gradient descent with 300+ parameters, 50-200 iterations per frame. 2+ hours for 839 frames.

**How BFM tracking actually works:** PCA was run on ~200 laser-scanned faces to extract 199 shape modes and 79 expression modes. Your face = average_face + weighted_sum(shape_modes) + weighted_sum(expression_modes). For each video frame:
- Detect 68 2D landmarks
- Construct 3D face from current parameters
- Project to 2D via camera matrix
- Compare projected vs detected landmarks
- Compute gradients (how each parameter affects the error)
- Update parameters, repeat 50-200 times
- Each gradient step knows which direction to push, but the 3D→2D projection is nonlinear so it can't solve in one step

**Modern replacement:** EMOCA or DECA — a neural network trained on millions of solved BFM/FLAME fits. Looks at the image, predicts all coefficients in one forward pass. Same accuracy, ~60x faster. 839 frames in ~2 minutes instead of 2 hours.

**Even better:** Use FLAME 2020 (100 expression dims vs BFM's 79, plus separate jaw articulation) via metrical-tracker or EMOCA. We already have this data — 838 frames of FLAME tracking from the FlashAvatar work.

**Estimated speedup:** 2 hours → 2 minutes

### 2. Face Model: BFM 2009 → FLAME 2020

| | BFM 2009 | FLAME 2020 |
|---|---|---|
| Shape dims | 199 | 300 |
| Expression dims | 79 | 100 |
| Jaw articulation | Baked into expressions | Separate 3-DOF joint |
| Eye gaze | Not modeled | 6 params (3 per eye) |
| Neck | Rigid | Articulated joint |
| Topology | ~53K vertices | ~5K vertices (cleaner) |
| Ecosystem | Legacy | Active (ARTalk, EMOCA, DECA, metrical-tracker all use it) |

FLAME is strictly better. The entire modern face reconstruction ecosystem runs on it. Switching to FLAME means every upstream tool (audio→motion, face tracking, expression transfer) just works without format conversion hacks.

### 3. Audio Features: DeepSpeech → wav2vec 2.0 / HuBERT / Whisper

Already done in Round 1 — we patched GaussianTalker to use wav2vec (44-dim) instead of DeepSpeech (29-dim). wav2vec captures more phonetic nuance. HuBERT or Whisper encoder features could be even better — this is a drop-in swap.

### 4. Audio → Motion: Custom Transformer → ARTalk

GaussianTalker trains its own audio→deformation network end-to-end. This works but means every new identity needs full retraining of the audio mapping.

ARTalk (SIGGRAPH Asia 2025) is a pretrained audio→FLAME model that generalizes across identities. It outputs 100 expression dims + jaw rotation + head rotation at 25fps from any audio. We already validated it — it produces natural blinks, lip sync, and head motion.

With ARTalk driving FLAME params, you don't need to retrain the audio decoder per identity. Train the Gaussian avatar once, drive it with any audio forever.

### 5. 3DGS Renderer: Original 3DGS → 2DGS / GOF / Modern Variants

The original 3D Gaussian Splatting (Kerbl et al. 2023) works but has known issues:
- Floaters (stray Gaussians in empty space)
- Popping artifacts during head rotation
- Suboptimal surface reconstruction

Newer variants:
- **2DGS** (Huang et al. 2024): Flat disc Gaussians, better surfaces, fewer floaters
- **GOF** (Gaussian Opacity Fields): Better geometry, cleaner edges
- **3DGS with depth regularization**: What GaussianTalker already uses, but could be tighter

The deformation network (which bends Gaussians based on expression) is independent of which Gaussian variant you use. Swappable.

### 6. Training: Per-Identity from Scratch → Few-Shot Adaptation

Current: every new person = full preprocessing + full training (~3 hours total).

Modern approach:
- Train a **universal head prior** on a large dataset (once, expensive)
- For a new identity: 30s video → FLAME tracking (2 min) → fine-tune prior (10-15 min)
- This is what GaussianTalker's "coarse + fine" stages are reaching toward, but without a shared prior

Papers doing this: GPS-Gaussian, GaussianHead, HeadGAP. The idea is proven.

## The Pipeline — Being v2

```
Input: 30s selfie video (1080p, any phone)

PREPROCESSING (3-5 min, was 2+ hours):
  Video → EMOCA/DECA → FLAME params per frame (2 min)
  Video → wav2vec/HuBERT → audio features (30s)
  Video → face parsing + background extraction (1 min)

TRAINING (15-30 min, was 1-3 hours):
  FLAME params + frames → 3DGS avatar (fine-tune from universal prior)
  1080p training, 1-2M Gaussians
  Deformation network learns expression→Gaussian displacement

INFERENCE (real-time, 60+ fps):
  Audio → ARTalk → FLAME params (pretrained, instant)
  FLAME params → deformation network → displaced Gaussians
  Gaussians → rasterize at 1080p → output frame
  Optional: render at 768p + neural upscale to 1080p

Total: 20-35 min from selfie video to real-time avatar
```

## What's Novel

Nobody has published this exact stack:
- FLAME-native throughout (no BFM↔FLAME conversion)
- Pretrained audio→motion (ARTalk) + per-identity Gaussian avatar
- 1080p rendering via high-res 3DGS training or neural upscale
- Universal head prior for fast adaptation
- Fully open-source, reproducible pipeline

Each piece is SOTA independently. The contribution is the integration + the engineering to make it actually work end-to-end.

## Feasibility Check

**What exists today and is proven:**
- EMOCA: open-source, real-time FLAME tracking
- ARTalk: open-source, audio→FLAME, validated by us
- 3DGS: open-source, rendering proven by GaussianTalker
- Deformation networks: proven by GaussianTalker, FlashAvatar, SplattingAvatar
- Neural upscaling: Real-ESRGAN, open-source, 5ms per frame

**What needs to be built:**
- FLAME-native deformation network (replace BFM assumption in GaussianTalker)
- Integration glue (format converters, pipeline orchestration)
- Universal head prior training (needs a multi-identity dataset)
- 1080p training pipeline (memory optimization, gradient checkpointing)

**What's hard but solvable:**
- Temporal consistency across the audio→motion→render chain
- Hair and shoulders deforming naturally with head motion
- Real-time on consumer GPUs (RTX 3060/4060 tier)

## Immediate Path

1. **Finish GaussianTalker Round 2** — prove the rendering quality with proper tracking
2. **Swap BFM tracking for EMOCA** — drop preprocessing from 2h to 2min
3. **Wire ARTalk as the audio driver** — eliminate per-identity audio retraining
4. **Push to 1080p** — train on full-res frames, benchmark VRAM
5. **Build the universal prior** — train on HDTF or similar multi-identity dataset

Each step is independently useful. Each one makes the pipeline better. No step requires throwing away previous work.
