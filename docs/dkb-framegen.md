# Frame generation (experimental)

Off by default. Code: `runtime/vk_framegen.c`, pacing arithmetic in `runtime/fg_pace.h`, shaders in `runtime/shaders/fg_*.comp`; AMD FSR 3.1's frame interpolation as the other way to make the frames at 2x: `runtime/vk_fsr3fi.c` (see "AMD FSR 3.1 frame interpolation" below).

## What it does

Auto, or a fixed 2x to 8x. For each pair of game frames the present thread makes the frames between them:

1. The cropped window texture is copied into an RGBA8 history image and reduced to quarter-size luma.
2. Block matching of that luma against the previous frame's gives one motion vector per 16x16 pixels (within +-32 pixels), followed by a 3x3 median.
3. A made frame at time t between the two blends the previous frame moved forward and the new one moved back along the vectors.

The interval is an exponential average of the gaps between game frames. Every shown frame goes through the normal present chain (colour, FSR, SMAA, HDR, overlay).

### 2x

One made frame (t = 0.5) for each pair of game frames, shown first, and the game's own frame half an interval later. The present thread shows both itself (`vk_fg_show`), as it always did. When the display shows fewer than 2 frames per game frame the made frame is skipped (`over refresh` in the log).

### A fixed multiplier (`3x` to `8x`)

Each game frame gets as many frames as the display can show in the game's frame time, the multiplier at most (`fg_pace_fixed_frames`): 60 fps on a 165 Hz display gives 2 at any multiplier, 30 fps gives up to 5. The made frames are at t = k / n and are shown first, the game's own frame last, each interval / n after the one before. When the display shows only one frame per game frame, none is made (`over refresh` in the log). A log line says when fewer frames are made than asked. These and Auto are shown by a display thread (below).

### Auto

Each refresh of the display between two game frames gets a frame made at its place between them (`fg_pace_auto_slots`); a refresh within a quarter period of a game frame shows that frame itself. The frames are shown one game frame time later, each at its refresh's time, so Auto adds one game frame of delay. A game frame may get up to the refresh over 30 frames (2 to 4). When the game runs faster than the display, a game frame that no refresh falls on is not shown. An unknown refresh counts as 60 Hz.

### Two threads, as in Hookshot (Auto and 3x to 8x)

The design is Hookshot's (`gpu/vk/vk_framegen.cpp`: `framegen_record`, `display_main`). The present thread (`vk_backend.c` `present_main`) calls `vk_fg_record` in place of its blit: it makes the game frame's set (the made frames and a copy of the game's own, since the history image is written again two frames on) into one of three sets, submits it and queues it, and is free for the next game frame at once. It waits only when all three sets are still being shown, which holds the game back as Hookshot's sets do.

The display thread (`display_main`) takes the sets in order and presents each frame through the whole present chain on an even line (`fg_pace_show_due`): the game's frame time divided by the set's frames apart, never more than one game frame behind what the game handed over, and from now again when it fell behind; Auto's frames go out at their own refreshes' times. A made frame is left out only when the next set is already waiting. A frame the display thread does not get (frame generation turned off, a skip) is presented by the present thread once every set before it was shown.

Not ported: Hookshot's "present ahead", which submits a set's copies into the window together and keeps only the presents on the line, so a copy does not wait behind the game's next frame on the GPU. Here each frame's copy is submitted at its time, behind whatever the game queued before it. It needs `vk_present.c` to hold several window images with their presents pending, where it holds one now.

### Measured

RX 7900 XTX (Mesa 26.2.7), 165 Hz display, the first cave of Ingot Isle at about 46 game frames a second, the game's own
upscaler, SDR, mailbox, 2026-11-16 (`bananza` b99dd34 with Auto and the multipliers); each run 90 s, the last 5 s lines:

| Setting | Game fps | Made a second | Frames shown per 5 s | Shown apart |
| --- | --- | --- | --- | --- |
| off | 46.4 | | | |
| 2x | 45.8 | 47 | 472 to 475 (94.5 to 95.0 a second) | 0 to 1 late over 2 ms |
| Auto | 46.5 | 142 | 819 on time, 5 made ones left out | 777 under 7 ms, 38 at 7 to 9 ms; 0 late |
| 4x (3x made: 165 Hz over 44 fps fits 3) | 47.2 | 97 | 732 on time | 679 under 7 ms, 53 at 7 to 9 ms; 0 late |

Hookshot's own log on the same machine at Auto: 806 to 827 frames per 5 s, 714 to 756 of them under 7 ms. The `fg: frame interpolation:` line counts frames presented by the display thread; when they reached the screen was not measured.

## Turning it on

Set `frame_gen` in `dkb.ini` (or `TOTK_FRAME_GEN` in the environment) to `auto` or `2x` to `8x`. It can be changed while running (turning it off frees the images; another multiplier rebuilds them and starts the history again).

## AMD FSR 3.1 frame interpolation (`frame_gen_by = fsr3`, experimental)

`frame_gen_by` (`TOTK_FRAME_GEN_BY`, live) chooses what makes the frame at 2x: `builtin` (the default) is the interpolation above, which sees the picture only; `fsr3` is AMD's FSR 3.1 frame interpolation, which also uses the scene's depth and motion and an optical flow over the two pictures. The passes are AMD's own (FidelityFX SDK v1.1.4, commit `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`: ten frame interpolation passes, seven optical flow passes) compiled from `runtime/third_party/fidelityfx` by `tools/build_shaders.py` (`runtime/fsr3_fi_*_spv.h`, `runtime/fsr3_of_*_spv.h`) and recorded by `runtime/vk_fsr3fi.c` as compute dispatches; no AMD library is loaded.

The depth and motion are the temporal upscaler's: `vk_dlss.c` `vk_temporal_inputs` makes them for DLSS, FSR 3.1 and FSR 4, and `vk_temporal_done` hands them to `vk_fsr3fi_note`. So AMD's frame is made only when:

- `frame_gen = 2x` (FSR 3.1's interpolation makes the midpoint only; Auto and 3x to 8x always use the built-in one);
- a temporal upscaler drew this frame's 3D (`upscaler = dlss`, `fsr3` or `fsr4`, with the 3D below the output size, so that a scene was noted at the input size), and the projection is a perspective one;
- the GPU has push descriptors, storage images without a format and the subgroup arithmetic and quad operations the passes need (the log says what is missing);
- the history is there: the first frame after a gap, a resize or a camera cut is made by the built-in interpolation, as is any frame the passes refuse (a picture whose shape is not the 3D's).

Otherwise the built-in frame is made, with no other change to frame generation. The game's HUD is part of the picture the passes read, so it is interpolated with the scene, as it is by the built-in interpolation; AMD's SDK can take the picture without the HUD as well, which this runtime does not make yet.

Log lines: `fsr3fi: AMD FSR 3.1 frame interpolation can run on this GPU` and `fsr3fi: 17 pass pipelines ready` once the setting is chosen on a capable device, `fsr3fi: making the 2x frames at WxH from RWxRH depth and motion` at the first frame it makes, and `by AMD FSR 3.1 N` in the 2x `fg:` line (below), which counts the made frames AMD's passes made. `fg: frame_gen_by = fsr3 needs the depth and motion of a temporal upscaler ...` says once that a frame had none.

## Requirements

Without these, no made frames are produced (one log line says why) and presentation is unchanged:

- Bananza only.
- Present mode `mailbox` or `immediate` (`TOTK_VK_PRESENT`). With `fifo` the several presents per game frame would block the present thread.
- Window texture format 8-bit UNORM (RGBA8 or BGRA8) with blit-source support.
- Not with native HDR output (the picture holds PQ code values, which cannot be blended).

If set-up fails (missing push descriptors, format support, pipeline or image creation, fence timeout, submit failure), the pass turns itself off for the session.

## When a made frame is skipped

- No history: the first frame, after a hitch, a crop change, or after a skip for the reasons above.
- Hitch: more than 100 ms since the previous game frame (loading, pause). The history starts again.
- Crop change: the window texture crop size changed; images are rebuilt and the history starts again.
- Over refresh: at 2x, 2 / interval exceeds the monitor refresh by more than 5% (60 Hz is assumed if the refresh is unknown); at a fixed multiplier, the display shows one frame per game frame.
- Left out (Auto and 3x to 8x): the next set was already waiting when a made frame's turn came (the display thread is a set behind).

A skipped frame is shown as the game's frame alone.

## Log lines

At 2x, every 5 seconds:

`fg: 5.0 s: real R fps, shown S fps, made N (by AMD FSR 3.1 M); skipped: no history A, hitch B, over refresh C, crop D; real cut short E; late >2 ms F`

- real: game frames per second.
- shown: real plus made frames per second.
- made: made frames in the period; by AMD FSR 3.1: those AMD's passes made (`frame_gen_by = fsr3`), the rest the built-in interpolation.
- skipped: made frames not produced, by reason (see above).
- real cut short: the game's frame was flipped before its half-interval wait ended because a newer frame was already waiting.
- late >2 ms: the game's frame flipped more than 2 ms after its due time.

With Auto and 3x to 8x, from the present thread:

`fg: 5.0 s: real R fps, made N (M a second); game frames without made ones: no history A, hitch B, over refresh C, crop D`

and from the display thread, worded as Hookshot's line so the two logs compare directly:

`fg: frame interpolation: N frames shown on time, L late (over 2 ms), S made ones left out (the game's next frame waiting); shown apart: <7 ms a, 7-9 b, 9-12 c, 12-17 d, 17-25 e, over 25 f`

- on time and late: presented within 2 ms of the frame's due time, or later.
- left out: made frames not shown because the next set was already waiting.
- shown apart: the time between two presents, counted by range.

The set-up line says the mode: `fg: frame generation Auto on (experimental): a frame for each of the display's refreshes (165 Hz; up to 4 a game frame), made at WxH, history RGBA8, 3 sets`, or `fg: frame generation 4x on ...`. A fixed multiplier the display cannot fill logs `fg: 4x asked, 3x made: the display's 165 Hz over the game's 45 fps (the game keeps its rate)`.

## Known limits

- The overlay is drawn last on every shown frame and is never interpolated. The game's own UI is part of the game picture the made frame is built from.
- Input latency: at 2x it is unchanged (the made frame is not newer than the real one); a fixed multiplier above 2x shows the game's frame up to (n - 1) / n of a frame time later; Auto one frame time later.
- Extra GPU cost: a blit, luma, block matching and median pass per game frame, and a full-size synthesis pass per made frame.
- Experimental: artifacts around fast or occluded motion are expected.

## Attribution

The shaders and the approach come from Hookshot (<https://github.com/NotRexed/Hookshot>, GPL-3.0); Auto, the multipliers above 2x, the sets and the display thread from <https://git.notrexed.dev/Hookshot/Hookshot>. AMD FSR 3.1 frame interpolation: the passes are AMD's FidelityFX SDK (MIT), the dispatches follow Hookshot's `vk_fsr3fi.cpp` (<https://git.notrexed.dev/Hookshot/Hookshot>). See `THIRD_PARTY_NOTICES.md`.
