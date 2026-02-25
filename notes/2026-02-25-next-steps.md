# Next Steps — GaussianTalker Round 2

## What We Learned (Feb 22-25)

### GaussianTalker (Feb 22)
- Full head + shoulders + neck rendering — the whole frame, not just a face mesh
- 130fps, real-time capable
- Even 10 seconds of data gave decent results
- Quality was genuinely impressive — Tavus charges for 2 minutes of video and we beat it with 10 seconds
- **Only problem:** BFM tracking only captured mouth/jaw, so upper face was static (no eyebrows, no blinks, no eye expressions)
- Training was slow because we gave it 7 minutes of video — massive overkill

### FlashAvatar (Feb 23-24)
- FLAME-based, 300+ FPS, 5 min training
- **Day 1 failure:** Bypassed metrical-tracker with flame-head-tracker. Custom format conversion introduced unfixable camera bugs. 4 failed training runs.
- **Day 2 success:** Fresh 30s recording → metrical-tracker (native pipeline) → clean training. Face looks great.
- **Fundamental limitation:** Only renders what the FLAME mesh covers — face and scalp. No neck, no shoulders, no hair below ears. A floating head isn't usable for a real avatar.
- **Camera jitter:** metrical-tracker estimates per-frame camera pose, doesn't know camera was static. Minor jitter in output.

### Key Takeaways
1. **Short video is enough.** 30 seconds at 1080p/30fps gives ~750 frames. More than enough for any method.
2. **Don't bypass the intended pipeline.** Custom format conversions between trackers are a trap.
3. **Full-frame rendering matters.** For a usable avatar (video calls, content), you need head + shoulders + neck, not just a face.
4. **Tracking quality determines output quality.** The renderer can only be as good as the tracking it's given.

## Plan: GaussianTalker Round 2

### Why go back to GaussianTalker
- Renders full frame (head, shoulders, neck, hair, background) — not limited to a face mesh
- No camera jitter (handles camera differently than metrical-tracker)
- 130fps is plenty for real-time
- The only problem (partial face tracking) is fixable

### What to fix: Full-face BFM tracking
The first run only tracked mouth/jaw. BFM can track the full face — we need to configure it to capture:
- **Eyebrows** (raise, furrow, asymmetric)
- **Eyes** (blinks, gaze direction, squint)
- **Mouth** (already worked)
- **Jaw** (already worked)
- **Cheeks** (smile lines, puff)

Options:
1. **Fix BFM tracker config** — GaussianTalker uses its own BFM-based tracker. Check if it has options for full-face landmark fitting (not just mouth region). This is the simplest path.
2. **Use FLAME data we already have** — We now have 838 frames of high-quality FLAME tracking from metrical-tracker. Could map FLAME expression params → BFM expression params, or modify GaussianTalker to accept FLAME params directly.
3. **Replace BFM with FLAME in GaussianTalker** — More invasive but gives us 120 expression dims. The deformation network would need to accept FLAME params instead of BFM.

### Training plan
- **Video:** Use the same 30s recording (already on pod at `/workspace/metrical-tracker/input/roman/video.mp4`)
- **Tracking:** Fix the tracker to capture full face, run on 30s video
- **Training:** Single stage, should be ~30-60 min on A100 with 750 frames (vs hours with 7 min video)
- **Target:** Full head+shoulders render with eyebrows, blinks, and mouth all working

### What we keep from FlashAvatar work
- The 30s recording and recording script (`notes/recording-script-30s.md`)
- metrical-tracker output (838 .frame files) — ground truth FLAME expressions paired with audio
- This paired data is training data for a future **audio → expression model** (wav2vec → FLAME/BFM params)
- The FlashAvatar model itself as a comparison baseline

## Future: Audio-Driven Pipeline
Once we have a good GaussianTalker model with full expressions:
1. **Audio → Expression model:** Train a small transformer that maps wav2vec features → face expression params. Training data = the metrical-tracker output (audio + FLAME expressions from the 30s video).
2. **Real-time pipeline:** Audio stream → wav2vec → expression model → GaussianTalker render → output frame
3. **Integrate into Being:** Wire into `being/inference/engine.py` and `being/api/server.py`
