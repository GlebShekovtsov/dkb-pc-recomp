## Draft: next release (since 0.4.1)

Draft, not released. The 0.4.1 notes follow below, unchanged.

- `console = 0` also closes a Windows Terminal window: the game frees its console instead of hiding the window (#238, #364).
- #372 Leave the translator's emulated compare-exchange loop after 1024 attempts (@roverdrubber)
- dotnet no longer adds a dead folder to the user's PATH on each install (c222918).
- The draws of vertex stage 45e1a6f48cb7872e are skipped on NVIDIA, for the device loss at the Landfill Layer's first checkpoint on 3.2.0 (#304).
- #393 Copy the indirect reset's disjoint records in address order (@wrothgaril)
- #400 Log where a large pool's first GPU use spends its time, and resume write-watch queries where a full answer stopped (@wrothgaril)
- #409 Save TOTK_SHOTS and --shot pictures of an RGB10A2 (HDR) presented image (@wrothgaril)
- #378 Lazy write-back of shader storage writes on Linux, off by default (@roverdrubber)
- #405 Thaw a pool's whole frozen regions without VirtualQuery (@wrothgaril)
- #406 Split the replay of a slow frame into its parts in the perf trace (@wrothgaril)
- #407 Create the DLSS feature and the motion pass's pipeline ahead of the first upscale (@wrothgaril)
- #408 Skip mem_touch's VirtualQuery for lazy-heap chunks known to be committed (@wrothgaril)
- HD rumble is off by default (`rumble_hd = auto` turns it on): three third-party Switch Pro pads broke with it (#363, #383).
- HD rumble sends neutral values while the motors are silent, so a frequency change at zero amplitude no longer floods the pad (#363).
- A refused `output_res` now says which sizes work (#403).
- On a first run with an empty pipeline cache, a draw that has to wait for its pipeline (a cutscene or menu quad) builds it itself instead of queueing behind dozens of pipelines of skipped draws: on the AMD path the longest wait fell from 2.4 s to 0.23 s (#309). A ~10 s stall on the very first heavy frame remains.
- The radial blur a smash starts ends above 60 fps with `realtime = 1` on 1.0.0 too, as #159 did for 3.2.0 (#389, #390).
- The installer keeps the DLC of a container whose base package has title id 0, warns for any add-on it cannot place, and the game logs a missing add-on (#388, fix by @Ki1z).
- #357 Slice passes whose draws sample an image through a depth twin (@GlebShekovtsov)
- #360 Scripted game runs with a verdict: tools/scenario.py, save_warp.py and perf_check.py (@thatbananaman08)
- #373 Skip the futex wake of a Linux condition variable signal when no thread waits (@roverdrubber)
- #374 Give the heap's freed pages back under Linux memory pressure (@roverdrubber)
- #376 Share one sampler per create info and retire the least used near the driver's limit (@roverdrubber)
- #379 Copy the full-size 3D through native anti-aliasing's upscale step (@roverdrubber)
- #385 Stamp the GPU writes of a command buffer in batches at its fence (@wrothgaril)
- #386 Write the GPU writes of a held part back to guest memory once the part is submitted (@wrothgaril)
- #392 Let a recording thread note its GPU writes in a list of its own (@wrothgaril)
- #399 Keep the window scaling filter row open when the window is the picture size (@wrothgaril)
- #401 Answer the pointer guards' readability checks without VirtualQuery (@wrothgaril)
- HD rumble falls back to the two-motor path for the rest of the session when a pad disconnects after HD packets (#363, untested on that pad).
- #354 Read indirect-reset pointer lists from guest memory instead of through a GPU read-back (TOTK_RESET_CPU_LIST) (@thatbananaman08, landed with a host-pool check)
- #361 Scan a pool's pages for its mirror sync under the pool's own lock, and take mirror_lock only to queue the copies (@thatbananaman08)
- #365 Skip the submit scans of copies that hold nothing for them (@thatbananaman08)
- #366 Log the scheduler state when no frame has been presented for a while (TOTK_STALL_DUMP, default 5 s) (@thatbananaman08)
- #367 Opt-in orders for consecutive dispatches: the game's barriers alone, or compute barriers (TOTK_CS_BARRIER) (@GlebShekovtsov)
- #368 Apply the follow-ups from the review of #338: Anti-Lag 2 across a device restart, Auto's flight thumbnails, and the frame_gen_by text (@roverdrubber)

---

DKB Recomp 0.4.1: a static recompilation of *Donkey Kong Bananza* (Switch 2) to a native Windows exe with a Vulkan renderer. It is still experimental (0.x). This is a smaller release: what landed in the two days after 0.4.0, most of it from contributors.

**The packages hold no game data, and no release ever will.** You build the game on your own PC from your own dump of **1.0.0 or the 3.2.0 update**, given as a `.nup` or `.nx` file or as ExeFS and RomFS folders. Other updates are not supported.

Not run in play by us: the Lanczos window filter at 1440p, AMD FSR 3.1 frame interpolation, AMD Anti-Lag 2 on AMD hardware, FSR 4 on AMD hardware and HD rumble on real controllers. The CRT and window filters and the frame interpolation come from community pull requests and have had little testing in play.

## What is new since 0.4.0

- **Crashes.** A null edge target in the job graph builder is guarded on 1.0.0 and 3.2.0 (#317). A call that crashed after the Forest Layer snow tunnel gets a guard that skips it when its slot is null (#136, still open until the reporter retests). A GPU that finishes nothing for 30 seconds is reported as a device loss instead of a silent freeze (#329). A failed swapchain rebuild destroys the retired swapchain (#332). The frame generation display thread no longer takes its start lock under the present lock (#343).
- **Picture and upscalers.** `upscaler = fsr3` is AMD FSR 3.1 temporal upscaling and `upscaler = fsr4` is AMD FSR 4 through your own AMD DLLs (#270, #333); both are off by default and fsr4 has not been run on AMD hardware by us. DLSS, FSR 3.1 and FSR 4 also run as native anti-aliasing at `scene_res = full` on 3.2.0 (#345). Only draws that test or write depth are jittered (#321, #325), and the upscaler dispatch is found by its shape when its hash is not a listed one (#323). The upscaler row stays open on Bananza, and DLSS says in the log why it does not run (#152, #306). Window scaling filter: `window_filter = fsr1`, `linear`, `catmull_rom` or `lanczos`, with Sharpness at any window size (#340); `fsr1` stays the default, and Lanczos at 1440p has not been run by us. CRT filters: `crt_filter = off` (default), `lottes_grille`, `lottes_tv`, `hyllian` or `zfast`, with `crt_lines` (#342). ASTC textures stored as BC3 are copied instead of skipped, which is meant to bring back the Ingot Isle textures (#310, awaiting the reporter). Text above 1080p is sharper behind `DKB_FONT_SCALE=1` (#214, off by default). Dug terrain builds on RADV (#322).
- **Frame generation.** `frame_gen` has an Auto mode and 3x to 8x besides 2x, with the pacing they need (#334); it is off by default. At 2x, `frame_gen_by = fsr3` makes the frame with AMD FSR 3.1 frame interpolation instead of the built-in one (#346, off by default, not run by us), and it gets the frames without their HUD (#348). Frames are also made from the game's own HDR picture (#337). The Display tab has a Frame generation row and a Made by row (#349), and a set that waited is counted in the log (#352).
- **Performance.** The uniform snapshot takes shortcuts (`TOTK_USNAP_FAST`) and deferred guest reads are written back without waiting, on by default, about 1.5 ms a frame in Canyon Layer (#312). `TOTK_MIRROR_OWN` leaves the VRAM mirror's own GPU writes out of the sync (#295), `TOTK_CALL_REF` records some CallCommands as one call (#307), and a submission's GPU writes are copied back in one pass (#328). `TOTK_DRAW_MERGE` draws runs of c4-only draws as instanced draws and is off by default (#300). The end of a sliced pass is held back without the recorder thread too (#292), and sliced passes are recorded by default on AMD's proprietary driver on the RX 9070 and 9070 XT, the chip measured (#298; RDNA 1 and 2 are left out, #84, #155). An upscaler that is no longer chosen releases its images after 120 frames (#351). On Linux the shader disk caches get 10 GiB of room unless you set a size (#330), and on Intel under Windows the pipeline cache data is fetched only while no pipeline is being built (#331).
- **Sound.** The audio renderer is paced evenly instead of in bursts after each device period, which is meant to remove the crackling on a 1024-frame device (#133, #272, #347); #272 awaits the reporter.
- **Languages and text.** Polish (`pl`, `EUpl`, `pl-PL`) is a language choice (#335).
- **Input.** `pad:touchpad` can be bound, so the DualSense or DualShock 4 touchpad click can open the map (#291). HD rumble goes to a Switch Pro Controller or Joy-Con as amplitude and frequency (#285, not run on real controllers by us). The Controls tab builds against SDL older than 3.4.18 (#324).
- **Installer and Linux.** The setup's host check builds against the release sysroot on Linux, so the AppImage no longer fails at step 1 (#293), and its last check accepts a CPU that has no AVX2 when you build for x86-64-v2 (#318). `vulkan_driver` runs a Mesa build of your own on Linux without a wrapper script (#339). The `--warm-shaders` and `--prepare-shaders` library builds run at below-normal priority on one core fewer than the machine has, from 4 cores up (#3).
- **Diagnostics.** The performance overlay has a Breakdown mode that stacks each frame's parts in the graph (#341) and a Counter mode, a large FPS number with a small label (#350); both are values of `perf_overlay`. AMD Anti-Lag 2 is `antilag = 1`, off by default, on an AMD GPU whose driver has VK_AMD_anti_lag (#338); not run on AMD hardware by us. `TOTK_MEM_REPORT=N` logs where the process's commit goes, and `tools/lowend_bench.py` runs a low-end benchmark. The perf trace counts the waits on the generation-table lock.
- **Long sessions.** The VRAM mirror sync asks Windows for written pages before it takes the mirror lock, so the replay and its slices wait less on it (#317, partial: the query itself gets slower as the process ages).

## Pull requests in this release

Thank you to everyone who sent code. Each one below is new since 0.4.0:

- #292 Hold back the end of a sliced pass without the recorder thread too (@GlebShekovtsov)
- #294 Read page stamps under gen_lock in TOTK_MEMORY=verify (@Zorkats)
- #295 Leave the VRAM mirror's own GPU writes out of mirror_sync (TOTK_MIRROR_OWN) (@Zorkats)
- #296 Commit the probe's heap before the freeze checks (exe-probe-gfx-abi with the lazy heap) (@thatbananaman08)
- #298 Record sliced passes by default on AMD's proprietary driver too (@GlebShekovtsov)
- #300 Draw runs of c4-only draws as instanced draws (TOTK_DRAW_MERGE, off by default) (@Zorkats)
- #307 Record CallCommands of a recording without deferred records as one call instead of a copy (TOTK_CALL_REF) (@thatbananaman08)
- #312 Take shortcuts in the uniform snapshot and write deferred reads back without waiting, about 1.5 ms a frame in Canyon Layer (@Zorkats)
- #321 Jitter only the draws that test or write depth, so the temporal upscalers are not handed a picture shifted twice (@roverdrubber)
- #322 Build the geometry stage of dug terrain on RADV, whose input locations past 15 the location check refused (@roverdrubber)
- #323 Take the game's upscaler dispatch for DLSS, FSR 3.1 and FSR 4 by its shape when its hash is not one of the listed ones (@roverdrubber, landed by hand)
- #324 Build the Controls tab against SDL older than 3.4.18, whose headers have no SDL_GAMEPAD_TYPE_STEAM (@roverdrubber)
- #325 Jitter the scene draws by their depth test alone, as the review of #321 noted, and say which draws that leaves out (@roverdrubber)
- #326 Say in geometry_validate that taking the vertex stage's location limit on RADV is deliberate, as the review of #322 asked (@roverdrubber)
- #328 Copy a submission's GPU writes back to guest memory in one pass (@roverdrubber, landed by hand)
- #329 Report a GPU that finishes nothing for 30 seconds as a device loss (@roverdrubber)
- #330 Give Mesa's and NVIDIA's shader disk caches 10 GiB of room on Linux unless the player set a size (@roverdrubber, landed by hand)
- #331 Fetch the pipeline cache's data only while no pipeline is being built on Intel under Windows (@roverdrubber, landed by hand)
- #332 Destroy the retired swapchain when creating its replacement fails (@roverdrubber, landed by hand)
- #333 Give FSR 4 an input pass, FSR 3.1's output pass, a Linux library and depth flags (@roverdrubber)
- #334 Add frame generation's Auto mode and the 3x to 8x multipliers, with the pacing they need (@roverdrubber, landed by hand)
- #337 Make frames from the game's own HDR picture too (@roverdrubber)
- #338 Add AMD Anti-Lag 2 as an option (antilag = 1) (@roverdrubber)
- #339 Add an optional vulkan_driver setting so a Mesa build of the player's own can run the game on Linux without a wrapper script (@roverdrubber)
- #340 Give the window a choice of scaling filter, with Sharpness at any window size and the game's own HDR picture for the filters chosen (@roverdrubber)
- #341 Add a Breakdown mode to the performance overlay (@roverdrubber)
- #342 Add Hookshot's CRT filters to the present (crt_filter) (@roverdrubber)
- #343 Keep the frame generation display thread from taking the start lock under the present lock (@roverdrubber)
- #345 Run DLSS, FSR 3.1 and FSR 4 as native anti-aliasing at scene_res = full (@roverdrubber)
- #346 Add AMD FSR 3.1 frame interpolation as a way to make frame generation's 2x frames (frame_gen_by = fsr3) (@roverdrubber)
- #347 Pace the audio renderer evenly instead of in bursts after each device period (@Gabinou22s)
- #356 Refill a behind audio queue at once, drop the dead device pacing path (@Gabinou22s)
- #348 Give AMD FSR 3.1 frame interpolation the frames without their HUD, so the HUD is not moved with the scene (@roverdrubber)
- #349 Add Frame generation and its Made by row to the Display tab (@roverdrubber)
- #350 Add a Counter value to the performance overlay: a large FPS number with a small label, after Hookshot's (@roverdrubber)
- #351 Let an upscaler no longer chosen release its images, feature or context after 120 frames (@roverdrubber)
- #353 Slice indirect draws and cut the slice threads' lock waits, Canyon Layer replay 11.8 to 10.0 ms a frame (TOTK_PAR_INDIRECT) (@Zorkats)
- #358 Say in inline_recording's comment that TOTK_CALL_REF records a call instead of a copy (@thatbananaman08)
- #352 Count frame generation's set waits, name Auto's flight thumbnails, and correct its pacing comment and docs (@roverdrubber)

## Issues closed or answered by this release

Thank you to everyone who reported a problem. Open ones wait for the reporter to retest.

- #133 Linux audio static with a 1024-frame period (@xenovorx), closed; #272 the same on Windows (@q1209786034-netizen), open, retest asked (#347)
- #152 DLSS not working properly (@Gavury), closed, and #306 DLSS/DLAA is not working (@badman101011), open: the upscaler row stays open on Bananza and DLSS says in the log why it does not run
- #84 (@philneitz) and #155 (@bubblesnout), closed: AMD RDNA 1 and 2 are left out of the default replay threads
- #291 Touchpad button in the controls (@Dimiduh), closed
- #270 FSR 4 / FSR 3.1 temporal upscaler (@MatyMeatBoy), open: fsr3 and fsr4 are on `bananza`
- #285 HD rumble request (@vink2212), open: Switch Pro Controller and Joy-Con only, untested
- #293 AppImage error on step 1 (@Nikoroq), open
- #310 Ingot Isle cave-in missing textures (@shiwntr), open
- #317 FPS drops after a while (@li-beihai), open: one null edge in the job graph builder is guarded
- #318 No AVX2 (@Enzo1383), open
- #335 Polish language (@nario999), open
- #136 Forest Layer snow tunnel crash (@luis-7654), open
- #214 Blurry text at 1440p (@Felitendo), open: `DKB_FONT_SCALE=1`

## Thanks

Pull requests since 0.4.0: roverdrubber, Zorkats, GlebShekovtsov, thatbananaman08, Gabinou22s. Thank you.

## Download

| Package | For | Status |
|---|---|---|
| `DKB-Setup-<version>-windows.zip` | Windows 10/11 x64 | Supported. Carries its own tools and the installer window: unpack, run `DKB-Setup.exe`, pick your dump. |
| `DKB-Setup-<version>-linux-x86_64.tar.gz` | Linux x86-64, kernel 6.7 or later | Preview. Source package with `setup.sh`: you install the system packages it lists (`./setup.sh --check`), then it runs the command-line installer. |
| `DKB-Setup-<version>-macos-arm64.zip` | macOS 14.4 or later, Apple silicon | Preview. Same idea with `setup.command`. Play is not verified and the destructible terrain is missing (Metal has no geometry stage). |

## Requirements (Windows)

- NVIDIA GPU with a recent driver (tested on an RTX 5060 Ti and an RTX 3070 Ti Laptop). AMD and Intel are experimental: on AMD's Windows driver the device loss most players hit is fixed (recorder thread off), and RADV on Linux can still hang (#99, #122, #198, #204). Intel Arc: #3.
- 16 GB of RAM plus a large page file (the process commits about 9 GB), 32 GB is comfortable.
- A controller is recommended.

## Known issues

- GPU hangs under RADV on Linux (#99, #122, #198, #204), and a driver fault on Intel Arc (#3).
- Crashes reported without a reproduction here: #126 (Lagoon to Hilltop tunnel, retest asked), #136 (Forest snow tunnel landing: needs a guard on a call, after this release), #148 (Linux, after the first boss: the next log will name the function).
- Black shards where salt or acid meets slime (#139, #266).
- Text is soft at 1440p and above: the game's fonts are drawn at a fixed size (#214; `DKB_FONT_SCALE=1` helps small text).
- A device loss on an RTX 5070 in the Landfill Layer that we cannot reproduce (#304), and frame rate falling after a long session (#317).
- Heavy scenes can still fall under 60 fps, and the first visit to an area can stutter.
- Checkerboard textures (#5), #112 and unrecorded NVN 2 calls (#6).

## Reporting a bug

Open an issue with the log (`dkb.log`), the build line at its top (`dkb 0.4.1+<commit>`), your GPU and driver, and what you did before it failed. **Never share generated files, keys or screenshots that show game content.**

Roadmap: #10.
