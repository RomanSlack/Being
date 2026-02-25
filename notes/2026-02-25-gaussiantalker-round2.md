# GaussianTalker Round 2 — Full-Face Expressions (30s Video)

**Date:** 2026-02-25
**Pod:** A100-SXM4-80GB RunPod

## Goal
Re-train GaussianTalker with the 30s recording (838 frames @ 25fps) to fix the frozen upper face from Round 1 (which only used 10s). Key success metric: eyebrows and blinks visible in rendered output.

## What Changed from Round 1
| | Round 1 | Round 2 |
|---|---|---|
| Video | 10s, limited expression | 30s, more expression variety |
| Frames | ~250 | 839 |
| AU45 (blinks) | Dummy/zero values | EAR-based extraction (339 blink frames) |
| Fine phase | batch_size=32, 10K iter (~9h!) | batch_size=8, 3K iter (~30min) |
| Config | `64_dim_1_transformer.py` | `64_dim_1_transformer_small.py` |

## Preprocessing Pipeline
All 9 steps completed successfully on 1920x1080 frames:

1. **Audio extraction** → aud.wav, aud_train.wav, aud_novel.wav
2. **Frame extraction** → 839 frames at 25fps in ori_imgs/
3. **Face parsing (BiSeNet)** → 839 parsing PNGs (~6 min)
4. **Background extraction** → bc.jpg (~7 min)
5. **Face landmarks** → 839 .lms files (~3 min)
6. **Torso + GT images** → 839 each (~4 min)
7. **BFM face tracking** → track_params.pt (349MB, ~60 min at full HD)
8. **Transforms** → transforms_train.json, transforms_val.json
9. **Audio features** → aud.npy (855, 16, 44) via custom wav2vec script

### Custom Scripts Created
- **`extract_wav2vec.py`**: Standalone wav2vec feature extraction (pyaudio not needed)
  - Uses `cpierse/wav2vec2-large-xlsr-53-esperanto`, sliding window, unfold(16,2)
  - Output: (855, 16, 44) matching expected format
- **`extract_au45.py`**: AU45 blink extraction via Eye Aspect Ratio (EAR)
  - Computes EAR from 68-landmark .lms files
  - Auto-calibrates open/closed thresholds (85th/2nd percentiles)
  - Results: 339 blink frames, AU45 range 0-5, mean 1.79, zero fraction 15%

### Issues Resolved
- **Face tracking spawned duplicates**: SSH background commands silently forked 4 processes → killed extras
- **pyaudio not installed**: Blocked `data_utils/wav2vec.py` → wrote standalone extractor
- **`nerf/asr.py` missing**: process.py wav2vec mode broken → separate extraction
- **Rasterizer mismatch**: Standard diff_gaussian_rasterization returns (color, radii), code expects (color, radii, depth) → installed custom `submodules/custom-bg-depth-diff-gaussian-rasterization/`
- **Fine phase too slow**: Default config runs 10K fine iterations at batch_size=32 = ~9 hours → reduced to 3K iterations at batch_size=8 = ~30 min

## Training Results

### Config: `64_dim_1_transformer_small.py`
- **Coarse phase**: 7999 iterations, batch_size=1, ~12 min, ~13 it/s
- **Fine phase**: 3000 iterations, batch_size=8, ~30 min, ~1.8 it/s
- **Total training time**: ~42 minutes

### Final Metrics (Fine Phase)
| Iteration | Loss | PSNR | Points |
|-----------|------|------|--------|
| 1 | 0.060 | 30.3 | 40,996 |
| 500 | 0.041 | 32.0 | 40,996 |
| 1000 | 0.035 | 33.4 | 40,996 |
| 1500 | 0.033 | 34.8 | 40,996 |
| 2000 | 0.031 | 34.9 | 40,996 |
| 2500 | 0.031 | 35.0 | 40,996 |
| 3000 | 0.031 | 35.2 | 40,996 |

### Checkpoints Saved
- `output/roman_v2_fast/point_cloud/coarse_iteration_[500-7500]` (every 500)
- `output/roman_v2_fast/point_cloud/iteration_[1-3000]` (fine, every 500)

## Rendered Test Videos
All in `assets/gaussiantalker_round2/`:

| File | Description |
|------|-------------|
| `test_coarse_only_renders.mov` | Coarse phase only (no audio attention) |
| `test_fine1000_renders.mov` | Fine phase @ 1000 iterations |
| `test_fine3000_renders.mov` | **Final model** (3000 fine iterations) |
| `test_fine3000_gt.mov` | Ground truth for comparison |
| `test_fine3000_eye_attention.mov` | Eye attention heatmap |
| `test_fine3000_audio_attention.mov` | Audio attention heatmap |

## Key Observations
- **PSNR improved**: 30.3 → 35.2 over fine phase (significant improvement)
- **Loss decreased**: 0.06 → 0.031 (converged well)
- **Training converged by ~2000 iterations**: Marginal gains from 2000→3000
- **Rendering speed during training**: ~2.3 FPS (shared GPU with training)
- **Point count stable**: ~41K points after coarse, no growth in fine phase (as expected)

## Comparison with Previous Experiments

| Metric | GaussianTalker R1 | FlashAvatar 150K | GaussianTalker R2 |
|--------|-------------------|------------------|-------------------|
| Video length | 10s | 30s | 30s |
| Training time | ~3h | ~5min | ~42min |
| Render FPS | 130 | 300+ | TBD (dedicated) |
| Frame coverage | Full (head+shoulders) | Head only (FLAME) | Full (head+shoulders) |
| Upper face | Frozen | Active | TBD (check video) |
| Audio pipeline | Built-in | None | Built-in |
| PSNR (final) | ~32 | ~30 | 35.2 |

## Files on Pod
- Data: `/workspace/Being/extern/GaussianTalker/data/roman_v2/`
- Model (fast): `/workspace/Being/extern/GaussianTalker/output/roman_v2_fast/`
- Model (original): `/workspace/Being/extern/GaussianTalker/output/roman_v2/`
- Training log: `/workspace/train_round2_fast.log`
- Custom config: `/workspace/Being/extern/GaussianTalker/arguments/64_dim_1_transformer_small.py`

## Next Steps
1. Review rendered videos — check if upper face (blinks, eyebrows) are visible
2. If upper face still frozen: investigate BFM tracking vs audio attention
3. Try longer fine phase (5K-10K iterations at batch_size=8) if more training helps
4. Measure dedicated render FPS (without training competing for GPU)
5. Test with custom audio input for audio-driven generation
