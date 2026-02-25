# Methods Comparison — Talking Head Avatar (Feb 22–25)

## The Quest
Record short video → get photorealistic avatar driven by any audio in real-time. Tried 5 different approaches over 4 days.

---

## Method 1: GaussianTalker Round 1 (Feb 22) ★★★★
**The OG. Still the best lip sync we've seen.**

- **Input**: 7-minute video, A100 pod, BFM 2009 face model
- **Training**: ~24 hours (coarse 5min + fine ~3h), 12,600 frames
- **Result**: Full frame (head + shoulders + neck + hair + background), 130fps render
- **Lip sync**: Incredible. Best of any method. Mouth looked totally natural.
- **Problem**: Upper face frozen — no eyebrows, no blinks, no eye movement. BFM tracker only captured mouth/jaw. AU45 (blink) data was likely dummy values.
- **Audio pipeline**: End-to-end built-in (audio → wav2vec features → Gaussian deformation)
- **Real-time**: Yes, 130fps
- **Assets**: `assets/gaussiantalker_round2/checkpoints/` (confusingly named, this is round 1)

## Method 2: FlashAvatar (Feb 23–25) ★★★
**Fast to train, but fundamentally limited.**

- **Input**: 30s video, 838 frames, metrical-tracker FLAME data
- **Training**: ~5 min for 150K iterations
- **Result**: 300+ FPS render, FLAME mesh drives Gaussian splats
- **Problem**: Only renders the face (FLAME mesh = face + scalp). No neck, no shoulders, no hair, no background. Floating head on black.
- **Other issues**: Camera jitter in some renders
- **Audio pipeline**: None. Just a renderer — needs separate audio-to-FLAME driver
- **Real-time**: Yes, 300fps (overkill)
- **Verdict**: Research building block, not a usable avatar. Proved metrical-tracker works though.
- **Assets**: `assets/flashavatar_test_*_fresh.avi`

## Method 3: ARTalk + SplattingAvatar (Feb 25) ★★
**The decoupled approach. Pipeline works, quality doesn't.**

- **Input**: 30s video, 838 frames, metrical-tracker FLAME data → IMavatar format
- **Training**: 30K iterations, ~34 min on RTX 4090
- **Result**: Full frame with Gaussian splats driven by FLAME mesh
- **Lip sync**: Mouth moves but blurry and "monster-ish". Face is jumbled.
- **Problem**: SplattingAvatar's quality hit a ceiling at PSNR ~20. Face crop alignment issues between metrical-tracker and training images. Even after fixing (v4), still blurry. Fundamental quality limitation of the method with our data.
- **Head motion**: Fixed by applying ARTalk head rotation to FLAME global_orient (tiny ±3 deg)
- **Audio pipeline**: Decoupled — ARTalk (audio → 106-dim FLAME) + SplattingAvatar (FLAME → render)
- **Real-time**: SplattingAvatar is fast but we didn't benchmark. ARTalk is fast.
- **Verdict**: Pipeline architecture is sound (decoupled = swappable parts) but renderer quality too low.
- **Assets**: `assets/artalk_phase1/splatting_driven/artalk_driven_v4_headmove.mp4`

## Method 4: ARTalk + GAGAvatar with metrical-tracker data (Feb 25) ★
**Wrong tracker format = garbage output.**

- Same GAGAvatar renderer but fed metrical-tracker FLAME params
- Camera conventions completely wrong (Tz=1.69 vs expected ~8.9, different R convention)
- Result: Blurry blob, distorted face
- **Lesson**: Can't just plug one tracker's output into another method's renderer. Format looks similar (both FLAME) but camera conventions differ.
- **Assets**: `assets/artalk_phase1/splatting_driven/gaga_roman.mp4` (the bad one)

## Method 5: ARTalk + GAGAvatar with proper tracking (Feb 25) ★★★★
**Cleanest render quality. But mouth is unnatural.**

- **Input**: Single photo (!) — one-shot method, no training needed
- **Tracking**: GAGAvatar's own MICA-based pipeline (EMICA + VGGHead + optimization)
- **Result**: Crystal clear face, hair, mustache. Recognizable. 512x512.
- **Lip sync**: Mouth moves but looks unnatural/fake. GAGAvatar hallucinate mouth interior from a single photo — no real teeth/tongue data.
- **Other issues**: Lower resolution (512x512), less shoulder/body visible, slight uncanny valley on mouth
- **Audio pipeline**: Decoupled — ARTalk (audio → FLAME) + GAGAvatar (FLAME → render)
- **Real-time**: Yes, **78 FPS** on RTX 4090 (benchmarked)
- **Verdict**: Best overall render quality, but mouth naturalness is a dealbreaker for talking head use case.
- **Assets**: `assets/artalk_phase1/splatting_driven/gaga_roman_clean.mp4`

---

## Scorecard

| Method | Lip Sync | Face Quality | Full Frame | Real-time | Training Time | Setup |
|--------|----------|-------------|-----------|-----------|---------------|-------|
| GaussianTalker R1 | ★★★★★ | ★★★★ | ★★★★★ | ✅ 130fps | ~24h | 7min video |
| FlashAvatar | N/A | ★★★★ | ★ (head only) | ✅ 300fps | 5min | 30s video |
| SplattingAvatar | ★★ | ★★ | ★★★ | ✅ | 34min | 30s video |
| GAGAvatar (bad) | ★ | ★ | ★★★ | ✅ 78fps | 0 | 1 photo |
| GAGAvatar (proper) | ★★★ | ★★★★★ | ★★★ | ✅ 78fps | 0 | 1 photo |

## Key Takeaways

1. **GaussianTalker Round 1 is still the best overall result** — incredible lip sync, full frame, the only issue was frozen upper face. That's a fixable problem (better tracking data, more expression variety, real AU45 blinks).

2. **GAGAvatar is the cleanest static face quality** — but one-shot methods can't match per-subject training for mouth naturalness. Great for quick demos, not for production talking heads.

3. **The decoupled approach (ARTalk + any renderer) is architecturally correct** — swap audio model, swap renderer. But the renderer quality matters more than the architecture.

4. **Tracker format matters enormously** — metrical-tracker, MICA, BFM all output "FLAME params" but camera conventions and scales differ. Can't mix and match without careful conversion.

5. **Training data volume helps** — 7min video + 24h training (GaussianTalker) >> 30s video + 34min training (SplattingAvatar). More data = better mouth detail.

## What's Next

The most promising path: **GaussianTalker Round 2** with:
- 30s video (enough data based on all experiments)
- Better face tracking — full-face AU extraction (eyebrows, blinks, eyes)
- Real AU45 blink data (not dummy zeros)
- Same architecture that gave us amazing lip sync in Round 1
- If upper face unfreezes, we have the complete solution

The plan is at `notes/2026-02-25-gaussiantalker-round2.md`.

**Confidence level: High.** We've proven every piece works independently. GaussianTalker already gave us the best result — we just need to fix one specific issue (frozen upper face). That's a tractable problem with a clear cause (insufficient expression data in tracking).
