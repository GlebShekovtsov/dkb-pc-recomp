# Frame generation (experimental)

Off by default. Code: `runtime/vk_framegen.c`, pacing arithmetic in `runtime/fg_pace.h`, shaders in `runtime/shaders/fg_*.comp`.

## What it does

2x only. For each pair of game frames the present thread makes one extra frame:

1. The cropped window texture is copied into an RGBA8 history image and reduced to quarter-size luma.
2. Block matching of that luma against the previous frame's gives one motion vector per 16x16 pixels (within +-32 pixels), followed by a 3x3 median.
3. The made frame (t = 0.5) blends the previous frame moved forward and the new one moved back along the vectors.

The made frame is shown first, and the game's own frame half an interval later. The interval is an exponential average of the gaps between game frames. Both frames go through the normal present chain (colour, FSR, SMAA, HDR, overlay).

## Turning it on

Set `FRAME_GEN=2x` in `dkb.ini`, or `TOTK_FRAME_GEN=2x` in the environment. It can be changed while running (turning it off frees the images).

## Requirements

Without these, no made frames are produced (one log line says why) and presentation is unchanged:

- Bananza only.
- Present mode `mailbox` or `immediate` (`TOTK_VK_PRESENT`). With `fifo` the two presents per game frame would block the present thread.
- Window texture format 8-bit UNORM (RGBA8 or BGRA8) with blit-source support.
- Not with native HDR output (the picture holds PQ code values, which cannot be blended).

If set-up fails (missing push descriptors, format support, pipeline or image creation, fence timeout, submit failure), the pass turns itself off for the session.

## When a made frame is skipped

- No history: the first frame, after a hitch, a crop change, or after a skip for the reasons above.
- Hitch: more than 100 ms since the previous game frame (loading, pause). The history starts again.
- Crop change: the window texture crop size changed; images are rebuilt and the history starts again.
- Over refresh: 2 / interval exceeds the monitor refresh by more than 5% (60 Hz is assumed if the refresh is unknown).

A skipped frame is shown as the game's frame alone.

## Log line

Every 5 seconds:

`fg: 5.0 s: real R fps, shown S fps, made N; skipped: no history A, hitch B, over refresh C, crop D; real cut short E; late >2 ms F`

- real: game frames per second.
- shown: real plus made frames per second.
- made: made frames in the period.
- skipped: made frames not produced, by reason (see above).
- real cut short: the game's frame was flipped before its half-interval wait ended because a newer frame was already waiting.
- late >2 ms: the game's frame flipped more than 2 ms after its due time.

## Known limits

- The overlay is drawn last on every shown frame and is never interpolated. The game's own UI is part of the game picture the made frame is built from.
- Input latency is unchanged: the made frame is not newer than the real one.
- Extra GPU cost: a blit, luma, block matching, median and synthesis pass per game frame.
- Experimental: artifacts around fast or occluded motion are expected.

## Attribution

The shaders and the approach come from Hookshot (<https://github.com/NotRexed/Hookshot>, GPL-3.0). See `THIRD_PARTY_NOTICES.md`.
