# Lip Sync Post-Processing Research for GAGAvatar

**Date:** 2026-02-25
**Goal:** Find a lightweight lip-sync method that can fix JUST the mouth region on top of GAGAvatar's 512x512 renders at real-time speeds (>25fps).

**Context:** GAGAvatar produces clean face renders at ~78fps with full expressions (blinks, eyebrows, head motion), but the mouth looks unnatural/hallucinated. We need audio-driven mouth correction as a post-processing step.

---

## TL;DR Recommendation

**MuseTalk 1.5** is the clear winner for our use case. It is purpose-built for exactly what we need: real-time lower-face inpainting driven by audio. It runs at 30fps+ on a V100, operates on the mouth region only (256x256 face crop, lower-half masked), is zero-shot (no per-subject training), and produces significantly better mouth quality than Wav2Lip. The main limitation is a known blur in the tooth region on closeups and the 256x256 face crop resolution.

**Second choice: Wav2Lip** with TensorRT optimization. It is the simplest and fastest option, but the 96x96 resolution produces notably blurry mouths. Good for a quick prototype, but MuseTalk is worth the extra integration effort.

**Dark horse: LatentSync 1.6** from ByteDance -- higher quality than MuseTalk, trains at 512x512, but currently runs at ~0.1x real-time (way too slow). Watch for speed optimizations in future releases.

---

## Method-by-Method Analysis

### 1. Wav2Lip (ACM MM 2020)

**What it does:** Takes existing video + audio, replaces the lower face to match the audio. The classic lip-sync model.

| Property | Details |
|----------|---------|
| Mouth-only editing? | YES -- crops face, generates mouth at 96x96, pastes back |
| Real-time capable? | YES -- with TensorRT optimization: <1ms/frame on RTX 4070Ti. Standard PyTorch: ~30fps on A100 |
| Quality | LOW -- 96x96 resolution causes blurry mouths. Teeth are mushy. Texture details lost. |
| Per-subject training? | NO -- fully zero-shot |
| Architecture | GAN-based encoder-decoder + pre-trained SyncNet discriminator |
| VRAM | ~2-3 GB |

**Strengths:**
- Battle-tested, huge community, extremely well understood
- Fastest inference of all methods
- Excellent lip-audio sync accuracy (directly optimizes SyncNet loss)
- Dead simple to integrate -- crop face, run model, paste back

**Weaknesses:**
- 96x96 face resolution produces visibly blurry mouth region
- Teeth look unnatural/smeared
- No expression preservation in the generated region
- Paste-back boundary can be visible without blending

**Variants:**
- **Wav2Lip-HD**: Adds Real-ESRGAN super-resolution on top. Helps with resolution but adds latency.
- **Wav2Lip-HR** (2024): Improved architecture for higher resolution. Better quality but less community support.
- **Wav2Lip-ONNX-256**: Community ONNX model trained at 256x256. Faster, better quality.
- **Wav2Lip-fast**: Optimized fork for faster processing.

**Paper:** https://arxiv.org/abs/2008.10010
**Repo:** https://github.com/Rudrabha/Wav2Lip

---

### 2. VideoRetalking (SIGGRAPH Asia 2022)

**What it does:** Three-stage pipeline: (1) normalize expression to canonical, (2) audio-driven lip sync, (3) face enhancement. Produces high-quality talking head edits.

| Property | Details |
|----------|---------|
| Mouth-only editing? | PARTIAL -- edits the full face region but lip sync is targeted at mouth |
| Real-time capable? | MAYBE -- claims 30fps+ but the 3-stage pipeline adds latency |
| Quality | HIGH -- includes built-in face enhancement stage |
| Per-subject training? | NO -- zero-shot, generic pipeline |
| Architecture | Three sequential networks: expression editor + lip sync + face enhancer |

**Strengths:**
- Higher visual quality than Wav2Lip due to the face enhancement stage
- Handles expression normalization (removes conflicting expressions before lip sync)
- Good identity preservation

**Weaknesses:**
- Three-stage pipeline means higher total latency
- More complex to integrate than Wav2Lip
- Full face editing rather than strict mouth-only (overkill for our case -- we already have good expressions from GAGAvatar)
- Expression normalization stage might FIGHT with GAGAvatar's expressions

**Key concern for our use case:** The expression normalization stage (stage 1) would zero out GAGAvatar's blink/eyebrow expressions. We would need to skip stage 1 or only use stages 2+3.

**Paper:** https://arxiv.org/abs/2211.14758
**Repo:** https://github.com/OpenTalker/video-retalking

---

### 3. DINet (AAAI 2023)

**What it does:** Deformation-based inpainting for mouth region. Instead of generating mouth pixels from scratch, it spatially deforms reference frame features to create the mouth shape matching the audio.

| Property | Details |
|----------|---------|
| Mouth-only editing? | YES -- explicitly targets mouth region (64/128/256px mouth crops) |
| Real-time capable? | UNKNOWN -- no published inference speed metrics |
| Quality | HIGH -- preserves texture details because deformation moves pixels rather than generating new ones |
| Per-subject training? | SEMI -- pretrained on HDTF dataset, but best results need per-subject fine-tuning |
| Architecture | Deformation network + inpainting network, coarse-to-fine training |
| Resolution | Up to 416x320 full face, 256x256 mouth region |

**Strengths:**
- Deformation approach preserves high-frequency texture details (teeth, lip texture)
- Supports multiple resolutions via coarse-to-fine training
- Mouth-region focused by design

**Weaknesses:**
- Best quality requires per-subject fine-tuning (deal-breaker for zero-shot use)
- No published speed benchmarks -- likely not optimized for real-time
- Smaller community, less maintained
- Needs reference frames from the same subject

**Paper:** https://arxiv.org/abs/2303.03988
**Repo:** https://github.com/MRzzm/DINet

---

### 4. IP_LAP (CVPR 2023)

**What it does:** Landmark-guided talking face generation using landmark prediction + appearance-based rendering.

| Property | Details |
|----------|---------|
| Mouth-only editing? | NO -- generates full talking face from landmarks |
| Real-time capable? | UNLIKELY -- two-stage pipeline with landmark prediction then rendering |
| Quality | GOOD -- landmark guidance provides structural accuracy |
| Per-subject training? | NO -- zero-shot with pretrained models |
| Architecture | Landmark generator (transformer) + appearance-based renderer |

**Strengths:**
- Landmark guidance provides structurally accurate mouth shapes
- Good identity preservation through appearance priors

**Weaknesses:**
- Full face generation, not mouth-region editing
- Two-stage architecture adds latency
- Would need modification to work as mouth-only post-processing
- Limited speed optimization work

**Not recommended for our use case** -- designed for full face generation, not region-specific editing.

**Paper:** https://openaccess.thecvf.com/content/CVPR2023/papers/Zhong_Identity-Preserving_Talking_Face_Generation_With_Landmark_and_Appearance_Priors_CVPR_2023_paper.pdf
**Repo:** https://github.com/Weizhi-Zhong/IP_LAP

---

### 5. SadTalker (CVPR 2023)

**What it does:** Audio-driven talking head from a single image. Predicts 3DMM coefficients (head pose + expression) from audio, then renders via face renderer.

| Property | Details |
|----------|---------|
| Mouth-only editing? | NO -- generates full head motion (pose + expression + mouth) |
| Real-time capable? | NO -- generates offline video |
| Quality | GOOD for single-image animation |
| Per-subject training? | NO -- zero-shot from any photo |
| Architecture | Audio -> 3DMM coefficients -> face render |

**Strengths:**
- Works from a single image
- Natural head motion + expression generation

**Weaknesses:**
- NOT a post-processing tool -- it generates the entire talking head
- Cannot selectively fix just the mouth on an existing render
- Would conflict with GAGAvatar's expression/pose control
- SadTalker-Video variant exists that chains Wav2Lip for video lip sync, but that is just Wav2Lip with extra steps

**Not recommended for our use case** -- wrong tool for the job. This is a full talking head generator, not a lip-sync fixer.

**Paper:** https://arxiv.org/abs/2211.12194
**Repo:** https://github.com/OpenTalker/SadTalker

---

### 6. AniPortrait (Tencent, March 2024)

**What it does:** Audio-driven photorealistic portrait animation using landmark extraction + diffusion-based rendering.

| Property | Details |
|----------|---------|
| Mouth-only editing? | NO -- generates full animated portrait |
| Real-time capable? | NO -- diffusion-based, slow inference |
| Quality | HIGH -- photorealistic output |
| Per-subject training? | NO -- zero-shot from reference image |
| Architecture | Transformer (audio->landmarks) + diffusion model (landmarks->frames) |

**Strengths:**
- High visual quality from the diffusion model
- Good lip sync accuracy
- Frame interpolation module for speed

**Weaknesses:**
- Diffusion model = slow inference (tens of seconds per frame without heavy optimization)
- Full portrait animation, not mouth-region editing
- Way too slow for real-time post-processing

**Not recommended for our use case** -- too slow and not designed for region-specific editing.

**Paper:** https://arxiv.org/abs/2403.17694
**Repo:** https://github.com/Zejun-Yang/AniPortrait

---

### 7. MuseTalk (Tencent Music, Oct 2024) -- RECOMMENDED

**What it does:** Real-time audio-driven lip sync via latent space inpainting. Masks the lower half of the face and inpaints it conditioned on audio. Exactly our use case.

| Property | Details |
|----------|---------|
| Mouth-only editing? | YES -- masks and inpaints the lower face region |
| Real-time capable? | YES -- 30fps+ on V100 |
| Quality | GOOD -- significantly better than Wav2Lip, some tooth blur on closeups |
| Per-subject training? | NO -- fully zero-shot |
| Architecture | Frozen ft-mse-vae encoder + multi-scale U-Net (borrowed from SD v1.4) + frozen Whisper-tiny audio encoder |
| Resolution | 256x256 face region |
| VRAM | Moderate (V100 class GPU) |

**Strengths:**
- Purpose-built for real-time lip sync as post-processing
- Single-step inpainting (NOT a diffusion model -- no iterative denoising)
- Audio features via Whisper (good cross-lingual support)
- `bbox_shift` parameter controls mouth openness (tunable)
- v1.5 (March 2025) adds perceptual loss, GAN loss, sync loss for better quality
- Multilingual: Chinese, English, Japanese
- Active development, growing community

**Weaknesses:**
- 256x256 face region (needs upscaling for our 512x512 output)
- Known tooth blur on closeup faces
- Lower-face mask boundary can sometimes be visible
- Temporal consistency could be better (addressed in v1.5 with spatio-temporal sampling)

**Integration plan for GAGAvatar:**
1. GAGAvatar renders 512x512 frame at 78fps
2. Detect face region, crop to 256x256 lower-face area
3. Run MuseTalk inpainting conditioned on current audio chunk
4. Blend inpainted mouth back into the full frame
5. Combined pipeline: ~30fps (bottleneck is MuseTalk, not GAGAvatar)

**Paper:** https://arxiv.org/abs/2410.10122
**Repo:** https://github.com/TMElyralab/MuseTalk

---

### 8. LivePortrait (Kuaishou, July 2024)

**What it does:** Efficient portrait animation with stitching and retargeting control. Primarily video-driven (drives a portrait from a driver video), with lip retargeting capability.

| Property | Details |
|----------|---------|
| Mouth-only editing? | PARTIAL -- has dedicated lip retargeting module (MLP-based) |
| Real-time capable? | YES -- 12.8ms/frame on RTX 4090 (~78fps) |
| Quality | HIGH -- photorealistic, trained on 69M frames |
| Per-subject training? | NO -- zero-shot from single image |
| Architecture | Implicit keypoint framework + stitching/retargeting modules |

**Strengths:**
- Extremely fast (78fps on 4090)
- Dedicated lip retargeting module
- High visual quality with strong identity preservation
- Production-quality code from Kuaishou

**Weaknesses:**
- NOT natively audio-driven -- requires a driver video or integration with audio models
- Audio-driven lip sync requires chaining with FaceFormer+Whisper or Wav2Lip
- The lip retargeting is video-to-video (copies lip motion from source), not audio-to-lip
- Adding audio pipeline adds complexity and latency

**Could work for our use case** but requires building an audio->motion pipeline on top of it. The lip retargeting module is interesting but needs an audio-to-viseme converter to drive it. More complex than MuseTalk for audio-driven use.

**Paper:** https://arxiv.org/abs/2407.03168
**Repo:** https://github.com/KwaiVGI/LivePortrait

---

## Additional Methods Discovered

### 9. LatentSync (ByteDance, Dec 2024)

**What it does:** End-to-end latent diffusion model for lip sync. No intermediate motion representation -- directly operates in latent space with SyncNet supervision.

| Property | Details |
|----------|---------|
| Mouth-only editing? | YES -- face detection + affine normalization + mouth focus |
| Real-time capable? | NO -- 10s video takes ~100s on RTX 4090 (0.1x real-time) |
| Quality | VERY HIGH -- best-in-class visual quality, especially v1.6 at 512x512 |
| Per-subject training? | NO -- zero-shot |
| Architecture | Latent diffusion U-Net + Whisper audio encoder + SyncNet supervision |
| Resolution | v1.5: 256x256, v1.6: 512x512 |
| VRAM | 8GB (v1.5), 18GB (v1.6) |

**Strengths:**
- Highest visual quality of any open-source lip sync method
- v1.6 supports 512x512 (matches our GAGAvatar output exactly)
- Whisper-based audio understanding
- Active development from ByteDance

**Weaknesses:**
- WAY too slow for real-time (20 DDIM steps per frame)
- 10x slower than real-time even on top hardware
- VRAM hungry for v1.6

**Watch this one** -- if they can distill the diffusion model or add consistency models, it could become real-time. Currently offline-only.

**Repo:** https://github.com/bytedance/LatentSync

### 10. Diff2Lip (WACV 2024)

Audio-conditioned diffusion for mouth inpainting. Better quality than Wav2Lip (lower FID), but diffusion-based = slow. Not real-time capable.

**Repo:** https://github.com/soumik-kanad/diff2lip

### 11. OmniSync (NeurIPS 2025 Spotlight)

Mask-free lip sync using Diffusion Transformers. Eliminates the need for explicit face masks. Very new (May 2025). Excellent quality on both real and AI-generated faces. But DiT-based = slow inference. Research-stage, not practical for real-time yet.

### 12. SayAnything (Feb 2025)

Conditional video diffusion for lip sync. Zero-shot, works across styles. But takes ~7 seconds per 1 second of video on RTX 4090 (0.14x real-time). Not viable for real-time.

### 13. LipGAN

Predecessor to Wav2Lip. Smaller model, edge-deployable, real-time capable. But lower quality than Wav2Lip. Only interesting for mobile/embedded use cases.

---

## Comparison Matrix

| Method | Mouth-Only? | Real-Time? | Quality | Zero-Shot? | Best For |
|--------|------------|------------|---------|------------|----------|
| **MuseTalk 1.5** | YES | YES (30fps) | Good | YES | **OUR USE CASE** |
| **Wav2Lip** | YES | YES (>30fps) | Low (96px) | YES | Quick prototype |
| **Wav2Lip-ONNX-256** | YES | YES | Medium | YES | Better prototype |
| LivePortrait | Partial | YES (78fps) | High | YES | Needs audio adapter |
| VideoRetalking | Partial | Maybe (30fps) | High | YES | Would fight GAGAvatar expressions |
| LatentSync 1.6 | YES | NO (0.1x) | Very High | YES | Offline/future |
| DINet | YES | Unknown | High | Semi (needs fine-tune) | Per-subject scenarios |
| OmniSync | YES (mask-free) | NO | Very High | YES | Future/research |
| Diff2Lip | YES | NO | High | YES | Offline |
| SayAnything | YES | NO (0.14x) | High | YES | Offline |
| IP_LAP | NO | NO | Good | YES | Wrong tool |
| SadTalker | NO | NO | Good | YES | Wrong tool |
| AniPortrait | NO | NO | High | YES | Wrong tool |

---

## Hybrid Architecture: GAGAvatar + MuseTalk Pipeline

```
Audio Stream ──► Whisper-tiny ──► Audio Embeddings ──┐
                                                      │
GAGAvatar ──► 512x512 frame ──► Face Detect ──► Crop 256x256 ──► MuseTalk U-Net ──► Inpainted Mouth ──► Blend Back ──► 512x512 Output
     (78fps)                      (face bbox)    (lower half      (single-step        (256x256)         (Poisson or     (30fps final)
                                                  masked)          inpainting)                           alpha blend)
```

**Expected combined framerate:** ~30fps (MuseTalk is the bottleneck)
**VRAM budget:** GAGAvatar (~4-6GB) + MuseTalk (~4-6GB) = ~10-12GB total on a single GPU

**Key integration decisions:**
1. Face detection: Use InsightFace (same as MuseTalk uses internally) -- run once, cache bbox
2. Mask boundary: Use Gaussian-blurred alpha mask for smooth blending at the jaw line
3. Audio latency: MuseTalk uses Whisper-tiny for audio encoding. Need to align audio chunks with frames.
4. Temporal smoothing: Apply EMA or optical-flow-based blending between consecutive MuseTalk outputs to reduce flicker

---

## Emerging Approaches to Watch

1. **GaussianFlameTalk** -- Transformer-based audio->FLAME params + Gaussian avatar rendering. Could eventually replace the entire GAGAvatar+MuseTalk stack with a single pipeline.

2. **SyncTalk++** -- Gaussian splatting talking head with built-in Face-Sync Controller. If it matures, could make the post-processing step unnecessary.

3. **GenSync (May 2025)** -- Multi-subject lip sync via 3D Gaussian Splatting. Audio-driven, generalized.

4. **One-step multi-frame inpainting** -- VAE + multi-scale U-Net for direct lip region synthesis. Research-stage but targeting real-time.

5. **Distilled diffusion models** -- If LatentSync or OmniSync get consistency-model distillation, they could become real-time with much higher quality than MuseTalk.

---

## Next Steps

1. **Prototype with Wav2Lip** -- 30 minutes to integrate, get baseline quality
2. **Set up MuseTalk 1.5** -- Main target, expect 2-4 hours for integration
3. **Test blend quality** -- The GAGAvatar-to-MuseTalk handoff at the mask boundary is the critical quality factor
4. **Benchmark combined pipeline** -- Measure actual fps with both models on same GPU
5. **If quality insufficient** -- Try LatentSync offline to establish quality ceiling, then wait for speed improvements
