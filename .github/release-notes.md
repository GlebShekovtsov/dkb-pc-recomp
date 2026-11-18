## Draft: next release (since 0.4.2)

- #419 Keep the Bananza installer tests independent of the host's free space and symlink rights (@wrothgaril)

DKB Recomp 0.4.2: a static recompilation of *Donkey Kong Bananza* (Switch 2) to a native Windows exe with a Vulkan renderer. It is still experimental (0.x). A small release: fixes and community pull requests from the day after 0.4.1, with most of the work on frame drops in heavy scenes, on first runs and on PCs with fewer CPU threads.

**The packages hold no game data, and no release ever will.** You build the game on your own PC from your own dump of **1.0.0 or the 3.2.0 update**, given as a `.nup` or `.nx` file or as ExeFS and RomFS folders. Other updates are not supported.

Not run in play by us: Switch 2 controllers over Bluetooth (`switch2 = 1`), HD rumble on the pads that reported problems, and the lookahead on AMD hardware.

## What is new since 0.4.1

- **Heavy scenes.** The layer -200 scene, the one most players reported, holds 60 fps on a Ryzen 7 5700X and RTX 5060 Ti with warm caches (first minute about 57 fps while it loads, second minute 59.9, 1% low 51; 0.4.0 gave 54 and 27). Most of it comes from the pointer guards no longer asking Windows about memory (#401) and the other replay and memory changes listed below.
- **Fewer CPU threads.** PCs with 4 to 7 logical processors (many laptops, 6-core CPUs without SMT) record big render passes on 3 threads by default instead of one: on layer -200, frames over 50 ms in the first minute fell from about 160 to 16 on 6 logical processors and from 203 to 47 on 4 (docs/dkb-lowend.md).
- **First runs.** With an empty pipeline cache, a draw that waits for its pipeline builds it itself instead of queueing behind dozens of skipped draws (longest wait 2.4 s to 0.23 s), and before it waits it queues the pipelines of the frame's next draws, so the first heavy frame stalls about 3 s instead of about 10 s (#309; `TOTK_LOOKAHEAD=0` turns it off).
- **Long sessions.** Frame drops that come after an hour or more of play (#317, #391, #380, #397) are much smaller with #401.
- **Crashes and device loss.** The Landfill Layer device loss on NVIDIA 50-series (#304) is avoided by skipping one vertex stage on NVIDIA.
- **Picture.** The radial blur a smash starts ends above 60 fps with `realtime = 1` on 1.0.0 too, as it already did on 3.2.0 (#389, #390).
- **Input.** HD rumble is off by default (`rumble_hd = auto` turns it on) because three third-party Switch Pro pads broke with it; it sends neutral values while the motors are silent and falls back to two motors when a pad drops (#363, #383). Switch 2 controllers can connect over Bluetooth LE with `switch2 = 1` (#413, off by default).
- **Installer and setup.** DLC from a container whose base package has title id 0 installs, and an add-on that cannot be placed is reported (#388). `console = 0` also closes a Windows Terminal window (#238, #364). A refused `output_res` says which sizes work (#403). The installer's dotnet no longer adds a dead folder to your PATH on every install.
- **Testing.** `TOTK_VRAM_LIMIT=<MiB>` caps the video memory budget the runtime reads, to see how a smaller card behaves.

## Pull requests in this release

Thank you to everyone who sent code. Each one below is new since 0.4.1:

- #354 Read indirect-reset pointer lists from guest memory instead of through a GPU read-back (TOTK_RESET_CPU_LIST) (@thatbananaman08, landed with a host-pool check)
- #357 Slice passes whose draws sample an image through a depth twin (@GlebShekovtsov)
- #360 Scripted game runs with a verdict: tools/scenario.py, save_warp.py and perf_check.py (@thatbananaman08)
- #361 Scan a pool's pages for its mirror sync under the pool's own lock, and take mirror_lock only to queue the copies (@thatbananaman08)
- #365 Skip the submit scans of copies that hold nothing for them (@thatbananaman08)
- #366 Log the scheduler state when no frame has been presented for a while (TOTK_STALL_DUMP, default 5 s) (@thatbananaman08)
- #367 Opt-in orders for consecutive dispatches: the game's barriers alone, or compute barriers (TOTK_CS_BARRIER) (@GlebShekovtsov)
- #368 Apply the follow-ups from the review of #338: Anti-Lag 2 across a device restart, Auto's flight thumbnails, and the frame_gen_by text (@roverdrubber)
- #372 Leave the translator's emulated compare-exchange loop after 1024 attempts (@roverdrubber)
- #373 Skip the futex wake of a Linux condition variable signal when no thread waits (@roverdrubber)
- #374 Give the heap's freed pages back under Linux memory pressure (@roverdrubber)
- #376 Share one sampler per create info and retire the least used near the driver's limit (@roverdrubber)
- #377 Keep an opaque ASTC texture as BC1 instead of BC3 with `astc_bc1 = 1` (off by default) (@roverdrubber)
- #378 Lazy write-back of shader storage writes on Linux, off by default (@roverdrubber)
- #379 Copy the full-size 3D through native anti-aliasing's upscale step (@roverdrubber)
- #385 Stamp the GPU writes of a command buffer in batches at its fence (@wrothgaril)
- #386 Write the GPU writes of a held part back to guest memory once the part is submitted (@wrothgaril)
- #392 Let a recording thread note its GPU writes in a list of its own (@wrothgaril)
- #393 Copy the indirect reset's disjoint records in address order (@wrothgaril)
- #399 Keep the window scaling filter row open when the window is the picture size (@wrothgaril)
- #400 Log where a large pool's first GPU use spends its time, and resume write-watch queries where a full answer stopped (@wrothgaril)
- #401 Answer the pointer guards' readability checks without VirtualQuery (@wrothgaril)
- #405 Thaw a pool's whole frozen regions without VirtualQuery (@wrothgaril)
- #406 Split the replay of a slow frame into its parts in the perf trace (@wrothgaril)
- #407 Create the DLSS feature and the motion pass's pipeline ahead of the first upscale (@wrothgaril)
- #408 Skip mem_touch's VirtualQuery for lazy-heap chunks known to be committed (@wrothgaril)
- #409 Save TOTK_SHOTS and --shot pictures of an RGB10A2 (HDR) presented image (@wrothgaril)
- #410 TOTK_MIRROR_STAGED (off by default): feed pools' VRAM mirror through staging (@wrothgaril)
- #413 Switch 2 controllers over Bluetooth LE (`switch2 = 1`, off by default), ported from mkw-recomp (@zigzagulouszagger)
- #414 Start a new guest thread on its ideal core alone, as Horizon does (@wrothgaril)

## Issues fixed or answered by this release

#238 and #364 (console window, @mashakos, @JiangYunWeiShu), #304 (Landfill Layer device loss, @Dimiduh), #309 (late models and textures on a first run, @Th0rfinnKarlsefni), #317 (@li-beihai), #391 (@1665879844), #380 (@supermahdi-enjoyer) and #397 (@crni20zd) (frame drops over time and in heavy scenes), #363 and #383 (HD rumble, @mrflanderson, @zero-zhouxiang), #388 (DLC with a zero title id, found and fixed by @Ki1z), #389 and #390 (smash blur above 60 fps, @elitalos, @Chadman246), #403 (custom resolution, @NicNamSam). Some of these stay open until the reporter retests.

## Thanks

Pull requests since 0.4.1: wrothgaril, roverdrubber, thatbananaman08, GlebShekovtsov, zigzagulouszagger. Issues, logs, saves and recordings: everyone named above. Thank you.

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
- Black shards where salt or acid meets slime (#139, #266, #387).
- Text is soft at 1440p and above: the game's fonts are drawn at a fixed size (#214; `DKB_FONT_SCALE=1` helps small text).
- Another device loss on an RTX 5080 at B1000 (#336); retests of the Landfill Layer skip (#304) and of long sessions (#317) are welcome.
- On CPUs with 4 to 6 threads heavy scenes stay around 40 fps, bound by the replay thread. A first run with an empty pipeline cache still stalls about 3 s on its first heavy frame, and area loads can stutter about 0.1 to 0.4 s.
- Checkerboard textures (#5), #112 and unrecorded NVN 2 calls (#6).

## Reporting a bug

Open an issue with the log (`dkb.log`), the build line at its top (`dkb 0.4.1+<commit>`), your GPU and driver, and what you did before it failed. **Never share generated files, keys or screenshots that show game content.**

Roadmap: #10.
