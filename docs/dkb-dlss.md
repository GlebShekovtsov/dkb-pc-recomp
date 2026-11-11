# DLSS (experimental)

`upscaler = dlss` in `totk.ini` (or `DKB_UPSCALER=dlss`) replaces the game's own upscaler, the compute dispatch that
scales the rendered frame to the output resolution, with NVIDIA DLSS Super Resolution (`runtime/vk_dlss.c`). FSR still
scales the output to the window when the window is larger. The setting is read at start.

Needs:

- An NVIDIA RTX GPU on Windows, with a driver recent enough for DLSS. The driver installs the NGX core
  (`_nvngx.dll`); the runtime finds it through the driver's registry keys.
- `nvngx_dlss.dll`, the DLSS model, next to `dkb.exe`. DKB Recomp does not ship it. Take it from a game or from
  NVIDIA's DLSS SDK release (310.5 or later for preset M, which the runtime asks for).

What it does: the game has no temporal anti-aliasing of its own (no velocity target, no history), so the runtime
makes the inputs DLSS needs.

- Jitter. The draws into the scene depth are moved by a sub-pixel Halton(2,3) offset through their viewport, and
  DLSS gets the same offset. Draws into other targets (shadow maps, probes, the half-size passes, the HUD) are not
  jittered. The jitter stops when DLSS has not run for a few frames.
- Depth. The scene depth is noted at the 3D draws: the passes whose draws read the camera block (vertex constant bank
  c3, 0x980 bytes) and whose depth target is as large as the upscaler's input. The game's depth is not reversed
  (near 1, far 5000). This block and its layout come from a frame dump of 1.0.0. On 3.2.0 the block read in play has the same layout
  (an orthonormal view, the view-projection as their product, a projection of near 1 and far 5000), the upscaler's
  shaders have the same hashes, and a 180 s run had depth and motion vectors on 10799 of 10800 DLSS frames.
  The camera is read from the copy of the uniform blocks taken at the submit call (`runtime/vk_usnap.c`); a frame
  whose block was not copied runs spatially.
- Motion vectors. The camera block holds the view-projection matrix. A small compute pass
  (`runtime/shaders/dlss_mv.comp`) reprojects every depth pixel from this frame's view-projection to the last frame's.
  These are camera-only vectors: what moves on its own (Donkey Kong, enemies, debris, water, particles) gets the motion
  of the surface behind it. DLSS rejects much of the wrong history there, but animated characters can still leave a
  faint trail or smear, most of all when the camera follows a fast character. The pass assumes the scene viewport
  covers the whole depth image; a scene drawn into part of it (a split view, a letterbox) would get wrong vectors.
- History. DLSS keeps its history from frame to frame. It starts again on the first frame, on a change of size (each
  change of the dynamic resolution recreates the DLSS feature, which can hitch), on a new scene depth image and on a
  camera cut (the camera moved more than 20 units or turned more than about 25 degrees in one frame).

A frame without the scene noted at the input size (a cutscene drawn another way, the first frame at a new size) runs
as before this change: zero motion vectors, a zero depth image, a new history. The quality mode follows the ratio of
output to input (DLAA at 1:1).

DLSS only runs where the game runs its own upscaler. When the game draws the scene at the output size (in 150 s
replays on 1.0.0 with a 1920x1080 output it switched to that partway through the play, at a different point in each
run), there is no upscaler dispatch to replace and the game's picture is shown as it is.

DLSS writes into the image the game's upscaler writes, so its output size is the game's own 3D resolution, not the
window's or `output_res`. On a 2560x1440 virtual display (`output_res`) the game makes that image 1706x960, two
thirds of it on each side (issue #152, update 3.2.0, `dlss: Quality 1138x640 -> 1706x960`); at the default 2880x1620
that is 1920x1080; at 3840x2160 the upscaler goes 2560x1440 -> 3840x2160 and DLSS ran `Quality 2560x1440 -> 3840x2160` on 1.0.0. The frame is stretched from there to the window as with the game's upscaler, so DLSS looks soft
when the window is much larger than that image. `render_scale` is a TotK setting and does nothing here (its menu row is hidden on Bananza); the HUD's "3D scene" line and `dkb.log` (`upscaler: WxH -> WxH`) report the game's own upscaler input and output sizes instead. The game scales its 3D resolution dynamically and runs the upscaler only while that is below the output size; once it reaches the full size (easy at 1080p on a fast GPU) the line reads "no upscaler" and DLSS has nothing to upscale.
`dkb.log` gets a `dlss:` summary at the first frame and every 1800 frames: how many frames had depth and motion
vectors, how many ran spatially, and how many started a new history.

When anything is missing or fails (no NVIDIA GPU, no NGX core, no `nvngx_dlss.dll`, NGX initialisation or feature
creation fails, an input larger than the output), `dkb.log` gets one line starting with `dlss:` that says why, and
the game's own upscaler runs for the rest of the run.
