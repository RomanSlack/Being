# The Hybrid Pipeline — Building Our Own Method

**Date:** 2026-02-25

## The Problem

We tried 5 methods over 4 days. None do everything:

| Method | Face Quality | Lip Sync | Full Frame | Real-Time | Upper Face |
|--------|-------------|----------|-----------|-----------|------------|
| GaussianTalker R1 | Great | Amazing | Yes | NO (2.4fps) | Frozen |
| GaussianTalker R2 | Worse | Untested | Yes | NO (2.4fps) | Maybe |
| FlashAvatar | Good | N/A | NO (floating head) | Yes (300fps) | Good |
| SplattingAvatar | Bad (PSNR 20) | Bad | Partial | Yes | Good |
| GAGAvatar | Crystal clear | Unnatural | Partial | Yes (78fps) | Good |

No off-the-shelf method gives us: clear face + natural lips + full expressions + real-time. So we build it ourselves by combining the best pieces.

## The Solution: ARTalk + GAGAvatar + MuseTalk

Three specialized models, each best-in-class at their job:

```
Audio (.wav)
  │
  ▼
ARTalk ──► FLAME motion (100 exp + 3 head + 3 jaw) at 25fps
  │
  ▼
GAGAvatar ──► 512x512 face render (crystal clear, full expressions, blinks, head motion)
  │            78fps, one-shot from single photo, no training
  │
  ▼
MuseTalk 1.5 ──► Inpaint lower face with realistic mouth/teeth from audio
                  30fps, zero-shot, masks lower face only, preserves upper face
  │
  ▼
Final Output ──► Photorealistic talking head with natural lip sync
                  ~30fps (MuseTalk is bottleneck), ~10-12GB VRAM total
```

### Why This Works

- **GAGAvatar** handles what it's good at: face identity, skin texture, hair, expressions, blinks, head motion
- **MuseTalk** handles what GAGAvatar can't: realistic mouth interior, teeth, tongue, lip shapes from audio
- **ARTalk** handles audio → expression mapping (already proven working)
- Upper face (eyebrows, blinks) comes from GAGAvatar via ARTalk FLAME params — MuseTalk only touches the lower face
- Each piece is independently upgradeable

### Why MuseTalk?

Researched 13 lip-sync methods (see `2026-02-25-lip-sync-post-processing-research.md`). Most are either:
- Full face generators (wrong tool — would fight GAGAvatar)
- Diffusion-based (too slow for real-time)
- Low resolution (Wav2Lip at 96x96 = blurry)

MuseTalk 1.5 (Tencent) is purpose-built for this:
- Masks lower face, inpaints in a **single forward pass** (not diffusion — no iterative denoising)
- 30fps+ on V100, zero-shot, Whisper-based audio encoding
- v1.5 adds perceptual + GAN + sync loss for better quality
- Active development, large community

### Concerns

1. **Blend boundary** — Where MuseTalk's mouth meets GAGAvatar's cheeks. Need Gaussian-blurred alpha mask or Poisson blending.
2. **Resolution mismatch** — MuseTalk operates at 256x256 face crop, GAGAvatar at 512x512. Need to crop → inpaint → upscale → blend.
3. **Temporal consistency** — Two independent models may produce flickering at the boundary. May need EMA smoothing.
4. **Latency** — MuseTalk needs ~1 audio chunk processed by Whisper before rendering. Adds ~100ms latency to the pipeline.

## What We Keep

All prior work feeds into this:
- **metrical-tracker data** (838 FLAME frames) — could train a custom mouth model later
- **GAGAvatar setup** — already working on pod with our face tracked at 78fps
- **ARTalk** — already producing FLAME motion from audio
- **All the scripts** in `scripts/splatting_avatar/` — reusable pipeline tooling

## Next Steps

1. **Set up MuseTalk 1.5 on pod** — clone repo, download models
2. **Test on existing GAGAvatar renders** — feed `gaga_roman_clean.mp4` + audio through MuseTalk
3. **Evaluate mouth quality** — is it natural enough? Teeth visible? Lip shapes accurate?
4. **Build the combined pipeline** — single script: audio → ARTalk → GAGAvatar → MuseTalk → output
5. **Benchmark** — measure actual end-to-end FPS on RTX 4090

## The Bigger Picture

This is novel. Nobody has published this exact combination:
- One-shot Gaussian avatar (GAGAvatar) for identity + expressions
- Audio-driven lip inpainting (MuseTalk) for natural mouth
- FLAME-based audio motion (ARTalk) for expression control

Each piece is SOTA independently. The integration is the contribution. If it works well, this could be a paper or at minimum a very compelling open-source project.

## Alternative: Skip MuseTalk, Fix GaussianTalker

If MuseTalk quality isn't good enough, the fallback is:
- Go back to GaussianTalker with the FULL training config (not the small one)
- Use 30s video + real AU45 data (already preprocessed)
- Train for the full 10K fine iterations (~9h) instead of rushing with 3K
- Then solve the speed problem (torch.compile, TensorRT, FP16)

GaussianTalker Round 1 proved the lip sync quality is there — we just undertrained Round 2 and never properly evaluated it.
