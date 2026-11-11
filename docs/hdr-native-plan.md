# Native HDR10: assessment and plan

Status: route A (the game's own HDR mode) and the highlight rewrite (`spirv_hdr_shoulder.c`) are in the runtime, measured in play and checked by eye on one HDR TV (2026-11-10). Started 2026-11-09 against `bananza` at `9ea44ad`, game 3.2.0 (`dkb458752`).

"Native HDR" here means the picture's highlights come from the game's own high-range scene, not from stretching the
finished 8-bit picture. Today's `hdr = 1` (`runtime/vk_hdr.c`, `runtime/shaders/hdr_encode.frag`) is the second kind:
an inverse tonemap of the SDR window texture.

## Finding: the game has its own HDR output path

It was assumed that the build has no HDR output path. A check of the 3.2.0 modules says otherwise. The host told the
game that HDR is off, and the game then never used that path.

Verified by scanning the symbol names in the decompressed `main` NSO:

- `main` imports `nn::vi::QueryDisplayHdrStatus(bool *, HdrOutputFormat *, u16 *, const Display *)`,
  `nn::vi::QueryDisplayHdrPaperWhite(u16 *, const Display *)` and four layer metadata setters:
  `SetLayerNextHdrContentGamutMetadata`, `SetLayerNextHdrToSdrToneMappingMetadata`,
  `SetLayerNextHdrStandardContentLuminanceRangeMetadata`, `SetLayerNextHdrExtendedContentLuminanceRangeMetadata`.
- `main` names the NVN procs `nvnWindowBuilderSetColorSpace` and `nvnWindow[Builder]{Set,Get}Hdr{Content,Display}Metadata`.
- `main` contains the strings `alHdrCompose` and `alColorSpaceConvert`, which look like names of the engine's own
  HDR composition and colour conversion passes.
- `main` imports none of the HDR status event calls of the SDK (`CreateHdrStatusEvent` and the like), so it must poll
  `QueryDisplayHdrStatus`.

What the host does with them today:

- `runtime/ov_gfx.c` `ov_hdr_status` writes `enabled = 0` and leaves the format and the third output untouched.
  `ov_hdr_paper_white` answers 203 nits. The four metadata setters go to `ov_vi_accept`, which ignores them.
- `runtime/nvn_api_nvn2.c`: `nvnWindowBuilderSetColorSpace` only logs its argument, `GetColorSpace` answers 0, and the
  four HDR metadata setters are empty.

Not verified, and the first things to find out:

- Whether the game calls these at run time, when, and with which arguments. Importing is not calling.
- The values of `HdrOutputFormat`, `HdrContentGamut`, `HdrToneMapping`, `NVNcolorSpace`, the meaning of the `u16 *`
  output, and the layout of the NVN HDR metadata structs.
- Which window texture format and encoding the game picks in HDR mode (10-bit PQ in Rec. 2020 is the likely one; the
  SDK also names a linear encoding, `SetLayerNativeWindowHdrLinearColorEncoding`).
- Whether the HDR shader variants lower and translate. They have never run here.

Known about the frame so far: the scene colour is float, the grading LUT is 8x8x8 R11G11B10F, the game's compute
upscaler writes an R11G11B10F image at output size (2880x1620 seen), and the window texture is 8-bit sRGB in SDR mode.

## Step 1 result: Eglantine has no HDR path

`refs/eglantine` implements none of the `nn::vi` HDR commands (no entries in
`src/core/hle/service/vi/application_display_service.cpp` or `system_display_service.cpp`), has no HDR enums, and
its Vulkan presenter (`src/video_core/renderer_vulkan/vk_swapchain.cpp` `ChooseSwapSurfaceFormat`) picks
B8G8R8A8_UNORM with SRGB_NONLINEAR only. In Eglantine the game therefore always runs its SDR path, as it does here.
Everything below comes from the game's own code.

## Step 2 result: how the game drives HDR (3.2.0 `main`)

Addresses are offsets in the 3.2.0 `main` module. "Display object" is the game's object that owns the `nn::vi`
display (pointer at `+0x178`) and layer (`+0x180`); the offsets below are its fields.

Per-frame update of the display object (function at `0x203728`):

- When `+0x1ce` (HDR allowed) is set, it calls `QueryDisplayHdrStatus(bool *enabled = +0x1cd, HdrOutputFormat *fmt =
  stack local, u16 *peak = +0x1d0, display)`. `fmt` is never read afterwards. When `+0x1ce` is clear, `+0x1cd` is
  cleared and nothing is queried.
- `+0x1d6 = min(+0x1d0, +0x1d2)`: the display's peak in nits, capped by a per-mode game maximum.
- `QueryDisplayHdrPaperWhite(u16 *)` -> `+0x1d4`; `+0x1d8 = +0x1d4 * float +0x1dc` (a per-mode paper-white scale).
- `+0x1e2` is a pending request (1 or 2). It moves to `+0x1e1` (1 for request 1, 0 for request 2), which arms the
  window rebuild below.

Window rebuild (function around `0x199cf0`, runs when `+0x1e1` is set):

- When `+0x1ce` and `+0x1cd` are both set, the window texture format (`[+0x150] + 0x68`) becomes `NVN_FORMAT_RGB10A2`
  (`0x3d`) when `+0x1cf` is clear, `NVN_FORMAT_RGBA16F` (`0x29`) when it is set. Otherwise an HDR format goes back
  to `NVN_FORMAT_RGBA8` (`0x25`). Either change sets the rebuild flag `[+0x150] + 0xc8`.
- Then `SetLayerNextHdrStandardContentLuminanceRangeMetadata(layer, 0, +0x1d8)` (paper white) and
  `SetLayerNextHdrExtendedContentLuminanceRangeMetadata(layer, 0, +0x1d6)` (peak).

Metadata (function at `0x203674`): writes peak (`+0x1d6`) and paper white (`+0x1d8` as float) into a renderer
parameter block, then `SetLayerNextHdrContentGamutMetadata(layer, 1)` and `SetLayerNextHdrToSdrToneMappingMetadata
(layer, 0)`.

Game side (function ending at `0x30fa38`): copies `+0x1cd` into bit 11 of a render flag word, so the renderer knows
HDR is on, and peak and paper white into its own state. When the byte `+0xa5` is set, `+0x1ce` (HDR allowed) follows
bit 12 of that word. `+0x1d2`, `+0x1dc` and, when `+0xa6` is set, `+0x1cf` (bit 2) come from a table of 0x64-byte
entries indexed by a mode byte (`+0x1c5`).

What this means for the host:

- The call answers that matter are `enabled`, the peak (`u16`, nits) and the paper white. The format output is
  unused.
- In HDR mode the window texture becomes RGB10A2 (most likely PQ code values) or RGBA16F (most likely linear
  scRGB-style values). The present must then treat it as HDR, not as 8-bit sRGB.
- Unknown yet: what sets bit 12 (HDR allowed) and the request at `+0x1e2`. If they stay clear, answering
  "enabled" alone changes nothing. Step 4 finds out.

## Step 3 and 4 results: go for route A

Trace (`DKB_HDR_TRACE=1`, title screen, 3.2.0): the game polls `QueryDisplayHdrStatus` and `QueryDisplayHdrPaperWhite`
several times a frame (callers `main+0x2037b8`, `main+0x2037d8`) and sends `SetLayerNextHdrContentGamutMetadata(layer,
1)` and `SetLayerNextHdrToSdrToneMappingMetadata(layer, 0)` with them. So HDR is allowed (`+0x1ce` is set) without any
further host answer. It calls no NVN colour-space or HDR metadata proc.

Experiment (`DKB_HDR_NATIVE=1`: `enabled = 1`, peak = `hdr_peak_nits`, paper white = `hdr_game_nits`):

- The window texture changes from RGBA8 sRGB to A2B10G10R10 (`present: window texture VkFormat 64`), at the window's
  size (3840x2160 on the test machine).
- The game starts calling `SetLayerNextHdrStandardContentLuminanceRangeMetadata(layer, 0, 203)` and
  `...ExtendedContentLuminanceRangeMetadata(layer, 0, 1000)`, the paper white and peak it was given.
- The frame has the same 60 passes as in SDR; only the last one, which reads the game's R11G11B10F scene image,
  writes the HDR10 texture. No shader failures, no validation errors: step 5 has nothing to fix so far.
- Decoded as Rec. 2020 PQ, the title screen matches the SDR frame below paper white (median 17 vs 16 nits, p90 143 vs
  143) with correct colours. The title screen and DK Artist mode have almost nothing above paper white (peaks 315 and
  217 nits); gameplay frames with sun and specular highlights are still to be checked.

First cut of step 7 (`gfx_present.c` `present_image`): when the swapchain is HDR10 and the window texture is
A2B10G10R10, the picture is blitted to the swapchain as it is, without the inverse tonemap, colour adjustment, FSR or
SMAA. The game is told HDR is on only while the swapchain is HDR10 (`gfx_window_hdr10`), and it polls the answer
every frame: with `hdr = 0`, without an HDR10 surface format or after a fallback to SDR (Windows HDR switched off, a
swapchain format without `BLIT_DST`) it goes back to its SDR picture instead of sending PQ to an SDR swapchain.
`hdr_native` needs `hdr = 1` and game 3.2.0; on 1.0.0 it stays off, because 1.0.0's HDR mode has never been run.

Fixed on the way: `HDR_TRACE` had no row in `cfg.c`'s settings table, so a run without the variable set stopped at
start (`cfg: TOTK_HDR_TRACE has no row in the settings table`), and later none in its section table.

## Gameplay result: the game's HDR mode keeps its SDR tonemap

Frame dumps from play (mining tunnel, 3.2.0, `DKB_HDR_NATIVE=1`, peak 1000, paper white 203):

- The lit scene is real HDR. The float targets of passes 37 to 65 (R11G11B10F) reach 30 to 60, with 0.1% of
  pixels above 2.
- Pass 82, the composite (`fs f57691dc0f0288e1`: scene, bloom chain, the 8x8x8 grading LUT, a 4x4 RGBA8 texture),
  tonemaps with `1 - exp(-x)`, blended per channel and on the largest channel, clamped to 0..1 and then graded by the
  LUT. Its output tops out at about 1.4. The shader and its inputs are the same in SDR and HDR mode.
- The final pass (`fs c18eb0f3d4fe9b51`) multiplies by a 3x3 matrix from its uniform buffer (rows 17 to 19, which
  hold Rec. 709 to 2020 and the paper white), divides by 10000 and PQ-encodes. It does not clamp.
- Result: the HDR10 picture equals the SDR grade below paper white and peaks at about 1.35x paper white (273 nits
  here). It is correct and has 10-bit precision, but it has almost none of the scene's range, whatever peak the game
  is told.

So route A is done as far as it goes: it shows the console's HDR. More range needs route B at one place, the
composite shader: keep its LUT grade for the SDR part and extend the curve above 1 towards `peak / paper white`
(e.g. scale the graded colour by the ratio of an extended curve to `1 - exp(-x)` on luminance). The final pass needs
no change. `DKB_SPV_OVERRIDE` (`vk_pipeline.c` `spv_override`) can carry a prototype before a proper rewrite in the
runtime.

## Route B: why the game's curve stops at 1.36, and the prototype

The composite's uniform block (`DKB_DUMP_UBO=1`) holds, in HDR mode, `c[4] = {4.92611, 0.65, 0.75, 0.5}`: the
ceiling M is exactly peak / paper white (1000 / 203), the knee k 0.65, the shoulder power p 0.75 and a chroma
exponent 0.5; `c[5].x` is 203. The SDR variant (another shader) has `c[4].x = 2.21675` (450 / 203).

After the LUT the shader computes the luminance `Ys = 1 - exp(-L)` and, above k, the shoulder
`Y = k + (1 - k) r t / (1 + (r - 1)(1 - t)^p)` with `t = (Ys - k) / (1 - k)` and `r = (M - k) / (1 - k)`, which runs
from k at the knee to M at `Ys = 1`. But L comes from a per-channel logistic stage, `(1 / (1 + exp((c1.w - x) c1.z)) -
e0) n`, which never exceeds `(1 - e0) n = 2.54` with the dumped `c[1].z = 1`, `c[1].w = 0.5`. So Ys never exceeds
`1 - exp(-2.54) = 0.921`, and the shoulder never exceeds 1.357, whatever M is. That is the 273-nit ceiling measured.

Prototype (`DKB_SPV_OVERRIDE` of the final module `5ea7515d96755e19`, files under `build/hdr-proto/`, which stay
there because they are derived from the game's shader): the same shoulder, run from k to that real maximum `top =
1 - exp(-(1 - e0) n)` instead of to 1.0, and only the output luminance replaced; the chroma blend keeps the game's own
luminance. Below the knee nothing changes. Predicted against the game: scene 1 -> 128 nits in both, 3 -> 340 (game 239),
6 -> 811 (274), 10 and up -> about 1000 (276).

Measured in play (mining tunnel, `DKB_HDR_NATIVE=1` with the override set, its log line present): maximum 975 nits (273 without
it); median, p90 and p99.9 unchanged (3, 15 and 202 nits). Only 0.04% of that frame is above paper white, the
highlights at the tunnel's end, so brighter scenes still need a look.

## Route B in the runtime

The edited shader cannot go into the repository (it is decompiled from the game), so `spirv_hdr_shoulder.c` makes the
same change to the game's own SPIR-V when the module is created (`vk_pipeline.c` `hdr_shoulder`, before the NaN guard),
whenever `hdr_native = 1`; the diagnostic `DKB_HDR_SHOULDER=0` keeps the game's curve (a user setting for that, `hdr_range = console`, was dropped: nobody would choose the 1.36x ceiling).

- It finds the composite by structure: three colour output components stored as `FMin(FMax(0, FMul(x, Load Y)),
  U[m][4].x)` for one Function variable Y and one uniform block U, and `Load Ys >= U[m][4].y`. It computes `top` from
  `c[1].z` and `c[1].w`, so it depends on no variable ids.
- Offline (a harness over the 213 distinct fragment modules in the dumps): it matches the gameplay composite
  (`f57691dc0f0288e1`, origin `fa8531841679c4de`) and the title screen's variant without the LUT (`7c69d0df5393f00c`,
  origin `cc63f1aa1b6efbeb`), nothing else; the output passes `spirv-val`, and decompiled it is the prototype's code.
- In play (the same mining tunnel, the rewrite against the game's curve in two runs): 975 against 273 nits at the
  maximum, median, p90 and p99.9 the same (3, 15, 202). No validation errors, no shader failures.
- On the title screen the two are identical: the game's exposure there (`c[0].x = 0.5`) keeps the scene below the knee.

It refuses a module in which the knee comparison's Ys could differ from the one it loads again (a store to Ys between
the two) or Y could change among the three FMul, so another translator or game version is left as it is rather than
rewritten wrongly. `tests/hdr_shoulder_spirv_test.py` checks the match, the refusals and that the output passes
`spirv-val` and the NaN guard.

## Measured with the trace (3.2.0, `hdr_peak_nits = 4000`)

- The game takes the lower of the display's peak and its own limit for the mode (`+0x1d6 = min(+0x1d0, +0x1d2)`):
  it sends `SetLayerNextHdrExtendedContentLuminanceRangeMetadata(layer, 0, 1400)` and the composite's
  `c[4].x = 6.897` (1400 / 203). A higher `hdr_peak_nits` changes nothing in the picture.
- The display was told `hdr_peak_nits` (4000) as MaxCLL and mastering peak, for a picture that stops at 1400, so a
  display that tone-maps by the metadata compressed highlights it could have shown. `vk_hdr.c` `vk_hdr_metadata` now
  sends the game's own peak with native HDR (`ov_gfx.c` `hdr_game_content_peak`).
- One gameplay frame (F9 dump, decoded as Rec. 2020 PQ from the 8-bit dump of the window texture): maximum 1299 nits,
  median 35, p90 150, p99.9 842; 6% of the frame above paper white, 0.14% above 800; no pixel pinned at the peak;
  frame-average light level 89 nits; next to nothing more than 15% outside Rec. 709. A third and a fourth composite variant
  matched the rewrite in play: `4951bdb9c94d40a2` (origin `147ab1f597568b2d`) and `5264abb433fb715c` (origin `2c058fbf1c3c1bdc`). `tools/hdr_frame_stats.py` computes these numbers from a dump.
- Raising the game's own limit is not attempted: it is the grade the game ships, few displays show more than
  1400 nits, and a display above it tone-maps nothing away.

Still open: brighter scenes (outdoors, the sun) and a person's judgement on an HDR display; 1.0.0, whose composite
should match by the same structure but has not been run (`hdr_native` is off there); the HUD, which the game draws into the scene image after the
composite at its own level (about paper white), so it is not affected.

## Two routes

Route A, switch on the game's own HDR mode. Answer `QueryDisplayHdrStatus` with "on", honour the window colour space
and present the game's HDR window texture on the HDR10 swapchain. The game then does its own HDR grading and composes
its HUD at the paper white it asks for. This is what the console shows, and it is the route to try first.

Route B, take the scene before the SDR tonemap and tonemap it ourselves. Find the pass that maps the float scene to
the 8-bit window texture, replace or bypass it by shader hash (as `vk_exec_draw.c` `dlss_upscale` and
`mvk_upscale_blit` do for the upscaler), and feed the float image to an HDR tonemapper of ours. The HUD is drawn after
that pass into the same 8-bit target, so it would have to be kept apart and composed at its own brightness. This is
more work and more fragile (shader hashes change with the translator and between game versions). Use it only when
route A is ruled out.

## Assessment

Route A, if the game's path behaves: small. The HDR10 swapchain, the metadata and the settings exist already; what is
missing is the answer to the game, the window state and one present variant. Expect 300 to 600 lines and three to
five working sessions, most of the time spent on finding out what the game sends and on checking the picture.

Risks, in order of likelihood:

1. The HDR shader variants hit translator or lowering bugs (`runtime/nvn2_lower.cpp`, the translator). Each one is a
   separate bug hunt of unknown size.
2. The enum values and struct layouts must be derived from the SDK code, because no public header exists. A wrong
   guess gives a plausible but wrong picture.
3. The game may also change other behaviour in HDR mode (a menu entry, a calibration screen, more metadata calls).
4. Nobody on the project has an HDR display under test. Every step needs a check that works without one, and a
   final check by a person with one.

Route B: medium to large, two to three times route A, with a maintenance cost per game version.

## Steps

Every step keeps `hdr = 0` and today's `hdr = 1` unchanged. The new mode is its own setting, `hdr_native = 1` with
`hdr = 1` (read at start), off by default until it has been verified on a display.

| # | Step | Done when |
|---|---|---|
| 1 | Read how `refs/eglantine` answers the same `nn::vi` and NVN calls (`tools/fetch_refs.sh` recreates `refs/`). AGENTS.md makes it the default reference. Note where it only stubs. | A short note in this file: per call, what Eglantine does. |
| 2 | Derive the enums and struct layouts from the `sdk` module: follow `nn::vi::QueryDisplayHdrStatus`, the four metadata setters, `nn::gs::sdk::SetNativeWindowColorSpace` and the `nvnWindow*Hdr*Metadata` procs in Ghidra or the `rea` tools. | A table of values and offsets here, each with the function it was read from. |
| 3 | Log every HDR call with its arguments and return address, behind `TOTK_TRACE` or an env variable: the `ov_gfx.c` overrides and the `nvn_api_nvn2.c` window procs. Run to gameplay with HDR still off. | The log shows which calls happen, when and how often. |
| 4 | Experiment: behind an env variable, answer `QueryDisplayHdrStatus` with "on" and the format from step 2. Run with `TOTK_DUMP=key` and `DKB_DUMP_SPV=1`. Read the dump: the window texture's format, the new passes, the arguments of the metadata calls. No presentation work yet; a wrong-looking picture is expected. | Go or no-go for route A, with the dump's pass list and the formats recorded here. On no-go, skip to step 10. |
| 5 | Fix what step 4 broke: HDR shader variants that fail to lower or translate, formats `vk_formats.c` lacks. One fix per bug, each with the translator's own checks (`tools/xlate_smoke.py`, `--probe-shader`). | The game runs in HDR mode without new log errors and the dumped passes look sane. |
| 6 | Window state: keep the colour space and both metadata structs on the NVN window (`nvn_api_nvn2.c`, `nvn_obj.h`) and make the Get procs return them. Carry the colour space with the presented texture to `gfx_present.c`. | `--probe-nvn-tex` and the selftest pass; a new probe case sets and reads the values back. |
| 7 | Present variant in `vk_hdr.c` and `gfx_present.c`: when the window texture is already HDR10, it must not go through the 8-bit `vk_hdr_mid` image. Decide where FSR, SMAA and `vk_color.c` grading run (they need a display-referred, roughly perceptual signal; PQ is close, linear is not) and write the shader that takes the result to the swapchain's PQ. The in-game menu keeps `hdr_ui.frag`. | With `hdr_native = 1` the swapchain shows the game's HDR frame; with FSR and SMAA on and off. |
| 8 | Settings: map `hdr_game_nits` to the answer of `QueryDisplayHdrPaperWhite`, pass the display's peak to the game where step 2 found a place for it, send the game's content metadata to `vkSetHdrMetadataEXT`. Add the setting to `cfg.c`, the menu (`runtime/ui/menu.cpp`) and `totk.ini.example`. | Changing the nits in the menu changes the picture; the metadata in the log matches the game's. |
| 9 | Checks without an HDR display: a script that reads a dumped PQ frame and reports the nits histogram, the peak and the share of pixels above paper white, for the same scene in SDR, inverse-tonemap and native mode. Then the fallbacks: no HDR10 surface format, Windows HDR switched off while running, `--shot` and the probes (`boot.c`, `vk_exec_probe.c` read 8-bit pixels), movies (`ov_movie.c`), 1.0.0 as well as 3.2.0, Linux. | The script's numbers show highlights above paper white in native mode only; every fallback ends in SDR with one log line. |
| 10 | Route B, only after a no-go in step 4: find the tonemap pass with `TOTK_DUMP`, match it by SPIR-V hash, redirect its input to a tonemapper of ours and compose the HUD apart. Write its own plan first. | A separate plan in this file. |
| 11 | Docs: `docs/architecture.md` (the HDR entry), `totk.ini.example`, `docs/dkb-testing.md` and the devlog. Ask a tester with an HDR display for a photo and the `hdr:` log lines. | The docs describe both modes and which one is native. |

Order and gates: 1, 2 and 3 are independent and can run in parallel. Step 4 is the decision point; do not start 6 to
8 before it says go. Step 5 has no bound, so report after each bug instead of batching.

Rules that apply to every step (AGENTS.md): generated C is never edited; dumps, SPIR-V files and anything else taken
from the game stay under `build/`; enum values and layouts derived from the SDK are written as our own description,
without copied code; run `python tools/check_content.py --staged` before proposing a commit.
