# Recording Script — 30 Second Training Video

## Why 30 seconds?
- 30s at 25fps = 750 frames
- metrical-tracker at ~1.4 min/frame = ~17.5 hours (overnight)
- FlashAvatar test_num = 100 → 650 training frames (paper caps at 2500, 650 is plenty for a clean first result)
- Our 7-min video was massive overkill and took days to track

## Before Recording

- **Resolution:** 1080p (tracker crops to 512x512)
- **FPS:** 30fps (tracker extracts at 25fps)
- **Format:** MP4, H.264
- **Lighting:** Front-lit, even, no harsh shadows — avoid overhead-only or side lighting
- **Framing:** Head + shoulders, space above head for hair, centered
- **Background:** Simple, non-distracting (plain wall ideal)
- Follow [bracketed directions] for movements
- Keep shoulders still, move from the neck
- No hands in frame

---

## 0:00–0:05 — Neutral + Calibration

> [Look straight at camera, neutral face, hold 2 seconds]

> [Slow turn LEFT ~25 degrees, hold 1 second]
> [Slow turn RIGHT ~25 degrees, hold 1 second]
> [Look slightly UP, hold 1 second]
> [Look slightly DOWN, hold 1 second]
> [Back to center]

---

## 0:05–0:10 — Head Calibration Continued

> [Tilt head LEFT, hold 1 second]
> [Tilt head RIGHT, hold 1 second]
> [Slow nod up and down, twice]
> [Back to center]

---

## 0:10–0:20 — Talking + Expressions

> [Natural head movement — shift gently as you would in a real conversation]

The thing about building technology is that you never quite know where it's going to take you.

> [Surprised expression — raise eyebrows]

Sometimes things go exactly as planned. And sometimes they absolutely do not.

> [Smile, showing teeth]

But honestly, those are the stories you end up telling people about.

> [Serious, direct at camera]

The important thing is that you keep going.

---

## 0:20–0:25 — Mouth + Teeth Visibility

> [Speak clearly with slightly exaggerated mouth openings]

Open vowels: Ah. Oh. Ooh. Eee. Ay.

> [Hold each vowel ~1 second, mouth wide open]

> [Smile wide, teeth visible, small laugh]

Ha ha ha.

---

## 0:25–0:30 — Closing Neutral + Movement

> [Slow turn LEFT, then RIGHT while speaking]

She sells seashells by the seashore. Unique New York.

> [Back to center, neutral face, hold 2 seconds]

> [Warm smile, hold 2 seconds]

> [End recording]

---

## Coverage Checklist

| Requirement | Sections |
|---|---|
| Head rotation (yaw) | 0:00–0:10, 0:25–0:30 |
| Head tilt (roll) | 0:05–0:10 |
| Head pitch (up/down) | 0:00–0:05 |
| Expression variety | 0:10–0:20 (neutral, surprised, happy, serious) |
| Teeth / mouth interior | 0:20–0:25 (open vowels, laugh) |
| Speech during movement | 0:25–0:30 (pangrams while turning) |
| Calibration poses | 0:00–0:10 (opening) |
| Natural conversation | 0:10–0:20 (talking with expressions) |

## After Recording — Upload & Track

```bash
# Upload to pod
scp -i ~/.ssh/runpod_key -P 19870 video.mp4 root@154.54.102.23:/workspace/metrical-tracker/input/roman/

# On pod: start tracking (overnight)
cd /workspace/metrical-tracker
python tracker.py --cfg configs/actors/roman.yml
```

Output lands at `/workspace/metrical-tracker/output/roman/checkpoint/*.frame` — FlashAvatar's native format, zero conversion needed.
