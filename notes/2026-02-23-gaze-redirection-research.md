# Gaze Redirection / Eye Contact Correction Research

**Date:** 2026-02-23
**Use case:** Post-process GaussianTalker rendered avatar frames to redirect static/staring eyes toward camera. Must be fast enough for real-time (25+ fps), work frame-by-frame on rendered images, and preserve the rest of the face.

---

## Summary of Best Options

### Tier 1: Best Bets for Our Use Case

#### 1. NVIDIA Maxine Eye Contact (RECOMMENDED FIRST TRY)
- **What:** Encoder-decoder CNN that crops a 256x64 eye patch, redirects gaze, blends back
- **Speed:** <5ms per frame on RTX GPUs (200+ fps) -- by far the fastest
- **Availability:** Free SDK download (AR SDK for Windows/Linux), also available as Docker NIM microservice
- **GPU req:** Turing/Ampere/Ada/Blackwell with Tensor Cores (NOT A100/H100 -- no NVENC). Works on RTX 4090, RTX 3090, T4, L4, A10, etc.
- **Open source?** No -- proprietary SDK, but free to download and self-host
- **Approach:** Disentangled encoder estimates gaze angle + embeddings, decoder redirects to frontal, inverse-transform blends back into frame. Only touches eye region.
- **Pros:** Production-proven, extremely fast, minimal artifacts, designed for exactly this use case
- **Cons:** Not open source (can't modify internals), NVIDIA GPU lock-in, won't work on A100/H100 (no NVENC)
- **Links:**
  - Blog: https://developer.nvidia.com/blog/improve-human-connection-in-video-conferences-with-nvidia-maxine-eye-contact/
  - NIM docs: https://docs.nvidia.com/nim/maxine/eye-contact/latest/overview.html
  - SDK samples: https://github.com/NVIDIA-Maxine/AR-SDK-Samples
  - NIM clients: https://github.com/NVIDIA-Maxine/nim-clients

#### 2. GazeGaussian (ICCV 2025 Highlight) -- Best Open-Source Quality
- **What:** Two-stream 3DGS model -- separate face/eye Gaussians, novel Eye Rotation field for rigid eyeball rotation
- **Speed:** 74 FPS on GPU (vs GazeNeRF 46 FPS, STED 18 FPS)
- **Availability:** Code + pretrained checkpoint on GitHub/HuggingFace
- **Stars:** 25
- **Dependencies:** Python 3.8, CUDA 11.6, PyTorch 1.12, PyTorch3D, Kaolin, diff-gaussian-rasterization
- **Open source?** Yes, full code + weights
- **Approach:** Per-subject trained 3DGS model with face deformation field + eye rotation field. Optimized on ETH-XGaze. Rasterizes to feature map, then neural renderer for final output.
- **Pros:** State-of-the-art quality, 74 FPS, 3DGS-based (same tech stack as GaussianTalker!)
- **Cons:** Requires per-subject training on ETH-XGaze data, not a simple "plug and play" post-processor. The 3DGS model here is trained specifically for gaze, not a generic post-processing filter.
- **Links:**
  - Paper: https://arxiv.org/abs/2411.12981
  - GitHub: https://github.com/ucwxb/GazeGaussian
  - Project page: https://ucwxb.github.io/GazeGaussian/

#### 3. STED-gaze (NeurIPS 2020, NVIDIA Research)
- **What:** Self-Transforming Encoder-Decoder for gaze + head redirection
- **Speed:** ~18 FPS (from GazeGaussian comparison table)
- **Availability:** Code + pretrained models on GitHub (Google Drive)
- **Stars:** 97
- **Dependencies:** PyTorch 1.7
- **Open source?** Yes, full code + weights
- **Approach:** Learns disentangled representation of gaze vs head pose. Encoder maps face to latent space, decoder redirects gaze. Works on 128x128 full-face crops.
- **Pros:** Well-established, pretrained models available, NVIDIA-backed research
- **Cons:** Only 18 FPS (below 25 fps target), older architecture, 128x128 resolution
- **Links:**
  - Paper: https://ait.ethz.ch/sted-gaze
  - GitHub: https://github.com/zhengyuf/STED-gaze

### Tier 2: Viable Alternatives

#### 4. RTGaze (Nov 2025) -- Fast but No Code Yet
- **What:** Real-time 3D-aware gaze redirection via feedforward network with triplane representation
- **Speed:** ~16 FPS (61ms/image on RTX 3090) -- paper claims "real-time" but this is borderline
- **Availability:** Paper only (arxiv 2511.11289). NO CODE OR WEIGHTS released yet.
- **Approach:** Distills from pre-trained 3D portrait generator. Cross-attention gaze injection. Triplane decoder + neural rendering.
- **Pros:** 800x faster than previous 3D-aware methods, single-image input
- **Cons:** No code available, 16 FPS is below our 25 fps target
- **Links:**
  - Paper: https://arxiv.org/abs/2511.11289

#### 5. Warping-based Gaze Correction (chihfanhsu)
- **What:** CNN-based pixel warping in eye regions
- **Speed:** Designed for real-time video calls (likely 30+ fps but no benchmarks published)
- **Stars:** 338 (most popular repo in this space)
- **Dependencies:** TensorFlow 1.8, Python 3.5, dlib, OpenCV (VERY outdated)
- **Open source?** Yes, but no pretrained weights published
- **Approach:** Detects face landmarks, crops eye regions, CNN predicts warp field, warps pixels to correct gaze
- **Pros:** Simple concept, lightweight, high star count
- **Cons:** Very old codebase (TF 1.8, Python 3.5), no pretrained weights, trained on Asian faces only (37 volunteers)
- **Links:**
  - GitHub: https://github.com/chihfanhsu/gaze_correction
  - Paper: https://people.cs.nycu.edu.tw/~yushuen/data/LookAtMe19.pdf

#### 6. GazeFlow (Normalizing Flows)
- **What:** Gaze redirection using normalizing flows
- **Speed:** Unknown (no benchmarks published)
- **Stars:** 47
- **Dependencies:** TensorFlow 2.3+
- **Open source?** Yes, with ETH-XGaze pretrained model (Google Drive)
- **Approach:** Encode eye patch conditioned on head pose + target gaze, decode redirected eye, Poisson-blend back into face
- **Pros:** Pretrained model available, processes individual eye images, Poisson blending for seamless output
- **Cons:** No speed benchmarks, TensorFlow-based, may be slow
- **Links:**
  - GitHub: https://github.com/CVI-SZU/GazeFlow

#### 7. GazeNeRF (CVPR 2023)
- **What:** 3D-aware gaze redirection with Neural Radiance Fields
- **Speed:** 46 FPS (from GazeGaussian comparison)
- **Stars:** ~50+
- **Dependencies:** PyTorch 1.12, CUDA 11.3
- **Open source?** Yes, with ETH-XGaze trained models
- **Approach:** Two-stream NeRF: separate face and eyeball volumes. Predicts volumetric features independently.
- **Pros:** Good quality, code available, 46 FPS is viable
- **Cons:** NeRF-based (slower than 3DGS), requires per-subject data
- **Links:**
  - GitHub: https://github.com/AlessandroRuzzi/GazeNeRF

### Tier 3: Research / Less Practical

#### 8. Roll Your Eyes (ACM MM 2025)
- **What:** Explicit 3D eyeball rotation in Gaussian Head Avatars
- **Speed:** Not benchmarked yet
- **Availability:** Paper only, no code released
- **Approach:** Integrates 3D eyeball structure into 3DGS avatar. Face stream + eyeball stream. Very relevant to GaussianTalker integration.
- **Links:** https://arxiv.org/abs/2508.06136

#### 9. interpGaze (ACM MM 2020)
- **What:** Controllable continuous gaze redirection
- **Stars:** ~30
- **Availability:** Code + dataset + pretrained model
- **Links:** https://github.com/IIGROUP/interpGaze

---

## Commercial / Closed-Source Solutions

### Apple FaceTime "Eye Contact" (iOS 14+)
- Uses ARKit depth map + face tracking to adjust eye position
- Only works within FaceTime on Apple devices
- Not available as API or for third-party use
- Occasional artifacts especially with glasses

### NVIDIA Broadcast App
- Consumer app with Eye Contact effect built on Maxine SDK
- Free download for RTX GPU owners
- Only works as a virtual camera for video calls
- Cannot process individual frames/images directly

### VEED.io / Descript / BigVU
- Cloud-based video editing tools with AI eye contact correction
- Not real-time, not self-hosted, API access unclear
- Pay-per-use pricing

---

## How the Best Solutions Work (Architecture Summary)

The dominant approach across all solutions:

1. **Detect face** -> extract eye region (crop eye patch, typically 256x64 or 128x128)
2. **Encode** eye patch -> extract gaze angle + appearance embeddings
3. **Decode/redirect** -> generate new eye patch with gaze directed at camera
4. **Blend** corrected eye patch back into original frame (inverse transform or Poisson blending)

This is fundamentally a **local inpainting** task on the eye region only. The rest of the face is untouched.

---

## Recommendation: What to Try First

### Option A: NVIDIA Maxine Eye Contact (Fastest Path to Results)
**If you have an RTX GPU (not A100):**
1. Pull the NIM Docker container
2. Feed rendered GaussianTalker frames through gRPC API
3. Get corrected frames back in <5ms each

This is the fastest, most production-ready solution. The SDK is free, self-hostable, and designed for exactly this use case. The 5ms latency means it can easily run as a post-processing step in the rendering pipeline without impacting the 25fps target.

**Limitation:** Won't work on A100 (no NVENC). For the RunPod A100 setup, you'd need a different approach or a different GPU type.

### Option B: GazeFlow + Custom Integration (Simplest Open-Source)
**If you want open-source and quick experimentation:**
1. Clone GazeFlow repo
2. Load ETH-XGaze pretrained model
3. For each rendered frame: detect face landmarks -> crop eye patches -> encode/redirect -> Poisson blend
4. Benchmark speed

GazeFlow is the simplest open-source option with pretrained weights that works on individual eye images. The Poisson blending approach should preserve facial quality well.

### Option C: GazeGaussian (Best Quality, Most Effort)
**If quality is paramount and you have time:**
1. This uses the same 3DGS tech stack as GaussianTalker
2. Could potentially be integrated directly into the GaussianTalker rendering pipeline (modify the eye Gaussians before rasterization)
3. 74 FPS is well above the 25fps target
4. But requires significant integration work

### Option D: Direct 3DGS Eyeball Manipulation (Long-term Best)
**The "Roll Your Eyes" paper points to the ideal solution:**
- Modify GaussianTalker itself to have explicit eyeball Gaussians
- Control gaze direction by rotating eyeball Gaussians before rasterization
- Zero post-processing overhead -- gaze is controlled at render time
- This is the architecturally cleanest solution but requires the most R&D

---

## Quick Decision Matrix

| Solution | Speed | Open Source | Pretrained | Ease of Integration | Quality |
|----------|-------|-------------|------------|---------------------|---------|
| NVIDIA Maxine | <5ms | No (free SDK) | Yes | Easy (API) | Excellent |
| GazeGaussian | 74 fps | Yes | Yes | Hard (3DGS) | Excellent |
| GazeNeRF | 46 fps | Yes | Yes | Medium | Good |
| STED-gaze | 18 fps | Yes | Yes | Medium | Good |
| GazeFlow | Unknown | Yes | Yes | Easy | Decent |
| RTGaze | 16 fps | No code yet | N/A | N/A | Good |
| Warping CNN | ~30 fps? | Yes (no weights) | No | Hard (old code) | Decent |
