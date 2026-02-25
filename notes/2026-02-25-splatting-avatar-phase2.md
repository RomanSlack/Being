# SplattingAvatar Phase 2 — FLAME-Driven Gaussian Avatar

## Date: 2026-02-25

## Summary
Built the full decoupled pipeline: ARTalk (audio→FLAME) + SplattingAvatar (FLAME→Gaussian rendering). Got end-to-end audio-driven rendering working. Currently retraining with properly aligned face crops for better quality.

---

## ~10:00 — Phase 1: ARTalk Validation (complete)
- ARTalk motion export works: 475 frames, 106 dims (100 exp + 6 pose), 25fps
- 84/100 expression dims active (vs ~10 with GaussianTalker's BFM)
- FLAME wireframe video looks great — clear lip sync, head movement, blinks
- Output: `assets/artalk_phase1/artalk_wireframe_discord.mp4`

## ~11:00 — Phase 2: SplattingAvatar Setup & Training
- Cloned SplattingAvatar on RTX 4090 pod
- Installed deps: PyTorch3D, diff-gaussian-rasterization, simple-knn, simple_phongsurf
- Patched chumpy for Python 3.11 + numpy 2.x compatibility
- Converted metrical-tracker .frame files → IMavatar flame_params.json format

### Three training runs (~11:00–14:00):
1. **v1**: Wrong images (full video resized, not face-cropped) + R not transposed → upside-down blurry mess
2. **v2**: Fixed R.T + face-cropped images, but 4x FLAME scale mismatch → right-side-up but still swirly
3. **v3**: Fixed 4x scale (camera t *= 4) → recognizable face, correct orientation, some blur (PSNR 20.0)

## ~14:00 — Phase 3: ARTalk → SplattingAvatar Driving (pipeline works!)
- Wrote `drive_with_artalk.py` — maps ARTalk 106-dim output → FLAME params → SplattingAvatar render
- ARTalk expression[:50] → FLAME expression, jaw_rot → FLAME jaw pose, global_orient zeroed (trained with camera rotation)
- Rendered 475 frames with audio sync → `assets/artalk_phase1/splatting_driven/artalk_driven.mp4`
- **Result**: Pipeline works end-to-end! Mouth moves with audio, blinks visible. But face is blurry/monster-ish due to v3 training quality.

## ~14:30 — Phase 4: Fixing the Crop Alignment (in progress)
- **Root cause found**: Our face crop used face_alignment with a simple square bbox, but metrical-tracker uses `crop_image_bbox` + `squarefiy` (asymmetric crop → pad to square → resize). The padding shifts the principal point.
- Cloned metrical-tracker source, found exact crop logic in `image.py`
- Wrote `crop_frames_exact.py` reproducing tracker's exact pipeline
- Crop result: 1124x1078 (not square!), squarefiy pads 23px top/bottom
- Re-cropped all 838 frames with correct alignment
- **v4 training started** — 30K iterations, ~30 min on RTX 4090

## Key Bugs Found & Fixed
1. **world_mat needs R.T**: SplattingAvatar's `getWorld2View2()` transposes R internally, so world_mat must store R_opencv.T (not R_opencv)
2. **IMavatar FLAME has factor=4**: All geometry scaled 4x. Camera translation must be multiplied by 4 to compensate
3. **Face crop squarefiy**: Tracker crops non-square bbox then pads to square. Our original script made a square bbox directly, shifting the principal point ~28px
4. **chumpy incompatible with Python 3.11**: `inspect.getargspec` → `getfullargspec`, numpy int/float removed
5. **igl API change**: `igl.read_obj` → `igl.readOBJ` in newer versions

## Files on Pod (RTX 4090: root@103.196.86.168 -p 52372)
- `/workspace/SplattingAvatar/` — code + submodules
- `/workspace/SplattingAvatar/data/roman/` — images + flame_params.json
- `/workspace/SplattingAvatar/data/roman/output-splatting/` — v3 trained model
- `/workspace/SplattingAvatar/data/roman/output-splatting-v4/` — v4 retrain (in progress)
- `/workspace/SplattingAvatar/tracking_data/` — 838 .frame files from metrical-tracker
- `/workspace/ARTalk/outputs/test_audio_natural_0.pt` — ARTalk motion output

## Files Local (permanent)
- `scripts/splatting_avatar/drive_with_artalk.py` — ARTalk → SplattingAvatar driver
- `scripts/splatting_avatar/convert_tracker_to_imavatar.py` — .frame → flame_params.json
- `scripts/splatting_avatar/crop_frames_exact.py` — metrical-tracker's exact face crop
- `scripts/splatting_avatar/verify_alignment.py` — FLAME mesh projection checker
- `assets/artalk_phase1/` — wireframe video, splatting eval renders, driven video
- `assets/artalk_phase1/splatting_driven/artalk_driven.mp4` — first audio-driven render (v3, blurry)

## ~15:00 — v4 Training Complete + Head Motion Fix
- v4 PSNR 19.93 (similar to v3's 20.0 numerically)
- Still blurry/monster-ish — fundamental quality ceiling for SplattingAvatar with our data
- Fixed head motion: ARTalk head_rot (±3 deg) applied via FLAME global_orient instead of zeroed
- Result: head moves now but quality still not good enough
- **Diagnosis**: Expression ranges match (ARTalk mean_abs=0.478 vs training=0.564), problem is base model quality not expression mismatch

## ~15:30 — Pivoted to GAGAvatar
- GAGAvatar (NeurIPS 2024): one-shot FLAME-driven Gaussian renderer, no training needed
- Downloaded GAGAvatar.pt (713MB) + tracked.pt (159MB) into ARTalk's assets
- Built `diff_gaussian_rasterization_32d` (custom 32-dim feature rasterizer from xg-chu)
- **Demo avatar test**: Crystal clear quality at 512x512. Night and day vs SplattingAvatar.

## ~15:45 — Custom Avatar Attempt #1 (metrical-tracker data → GAGAvatar)
- Wrote `create_custom_avatar.py` to convert metrical-tracker .frame data to GAGAvatar format
- Result: Blurry blob — camera conventions completely wrong (Tz=1.69 vs expected ~8.9)
- **Root cause**: GAGAvatar expects its own MICA-based tracking, not metrical-tracker output

## ~16:00 — Custom Avatar with Proper GAGAvatar Tracking ✓
- Cloned GAGAvatar's tracking submodule at `/workspace/GAGAvatar_repo/core/libs/GAGAvatar_track/`
- Downloaded tracking resources (1.06GB): EMICA encoder, VGGHead detector, FLAME model, StyleMatte
- Installed deps: onnx2torch, lmdb (face_alignment, pytorch3d already present)
- Ran `track_image.py` on our face photo → proper tracked entry with correct camera conventions
- Added to ARTalk's tracked.pt, rendered 475 frames with audio
- Removed GAGAvatar watermark (patched `models.py` line 95)
- **Result**: Clean face render, recognizable, 78 FPS on RTX 4090
- **But**: Mouth looks unnatural (one-shot hallucination), lower res (512x512), less shoulders visible

## Verdict
SplattingAvatar quality too low (PSNR ~20, blurry). GAGAvatar face quality excellent but mouth unnatural (one-shot limitation). Neither beats GaussianTalker Round 1's lip sync quality — that's still the best result we've gotten across all methods. See `notes/2026-02-25-methods-comparison.md` for full comparison.

## Next
- **GaussianTalker Round 2**: Fix the frozen upper face issue with better tracking data + real AU45 blinks
- Plan: `notes/2026-02-25-gaussiantalker-round2.md`
