# Mouth Mask Bug — 2026-02-24

## Issue
In `/workspace/generate_masks.py`, the mouth parsing mask uses BiSeNet label **10** (nose) instead of **11** (mouth interior).

```python
# BUG (current):
mouth_mask = (parsing == 10).astype(np.uint8) * 255  # label 10 = NOSE

# FIX:
mouth_mask = (parsing == 11).astype(np.uint8) * 255  # label 11 = mouth interior
```

## BiSeNet label reference
```
0: background    6: eye_glasses  12: upper_lip
1: skin          7: left_ear     13: lower_lip
2: left_brow     8: right_ear    14: neck
3: right_brow    9: earring      15: necklace
4: left_eye     10: NOSE         16: cloth
5: right_eye    11: MOUTH        17: hair
```

## Impact
- All 10,081 `_mouth.png` masks show nose region instead of mouth
- FlashAvatar uses mouth mask for extra supervision on mouth region
- Won't crash training but will give worse mouth rendering quality

## Fix plan
- Fix the label in generate_masks.py (10 → 11)
- Only need to regenerate `_mouth.png` files (alpha + imgs + neckhead are fine)
- BiSeNet inference is fast, regeneration should take ~20 min
