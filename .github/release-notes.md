DKB Recomp 0.2.0: a static recompilation of *Donkey Kong Bananza* (Switch 2) to a native Windows exe with a Vulkan renderer. It is still experimental (0.x): it plays the tutorial, its cutscene and the first stage on recent NVIDIA cards, and it can crash.

**The packages hold no game data, and no release ever will.** You build the game on your own PC from your own dump of **1.0.0 or the 3.2.0 update**, given as a `.nup` file or as ExeFS and RomFS folders. Other updates are not supported.

## What is new since 0.1.0

- **Update 3.2.0 (title version 458752).** The installer takes a `.nup` file, picks the package and builds the game from it. A 3.2.0 build installs, boots and held 60 fps for 150 s at the start on an RTX 5060 Ti; it has not been played further. Its patch table is ported from the 1.0.0 one (`tools/port_patches.py`), so the output resolution, ultrawide and real-time frame settings are meant to work there too; only a short run has been checked. When a game version has no real-time frame-rate support, the log now says so plainly (#137).
- **Add-on content (DLC) reader.** Checked on synthetic packs and on a real one; not tried in the game yet.
- **AMD.** The map's long cyan wedges are gone on an RX 9070 XT (reported by the contributor who fixed it, not tested elsewhere). A NaN guard on fragment outputs stops a NaN texel from being spread by the bloom into glowing bands (`DKB_NAN_GUARD`, on by default).
- **Sound.** The left channel was silent at the start; fixed.
- **Installer.** It warns when the Windows commit limit is below 24 GiB, and keeps the newer driver when the loader lists the same GPU twice.
- **Performance and shaders.** Fewer texture re-uploads, more translated shader forms (1D image atomics, packed halves), and the NVN 2 mode calls the game uses are handled instead of ignored.
- **Display and sound.** `window_mode` (windowed, borderless, fullscreen; F11), an optional SMAA pass after the upscale (`smaa` off, low, medium, high; off by default), `audio_channels` (auto, stereo, 5.1), a clean exit with no hang on close, and the guide button opens the menu.
- **Crash guard (1.0.0 only).** The layer-change pool loop is guarded against stale entries (`DKB_FIX_POOL_GUARD`, #126/#135). It is a workaround; the cause of the stale entries is not known.
- **Linux (preview, untested guesses).** The shader translator is made executable if an archive lost the bit, and the sound asks for a 240-frame period by default. Under RADV the feedback loops default to the pass-restart path.

## Download

| Package | For | Status |
|---|---|---|
| `DKB-Setup-<version>-windows.zip` | Windows 10/11 x64 | Supported. Carries its own tools and the installer window: unpack, run `DKB-Setup.exe`, pick your dump. |
| `DKB-Setup-<version>-linux-x86_64.tar.gz` | Linux x86-64, kernel 6.7 or later | Preview. Source package with `setup.sh`: you install the system packages it lists (`./setup.sh --check`), then it runs the command-line installer. |
| `DKB-Setup-<version>-macos-arm64.zip` | macOS 14.4 or later, Apple silicon | Preview. Same idea with `setup.command`. Play is not verified and the destructible terrain is missing (Metal has no geometry stage). |

## Requirements (Windows)

- NVIDIA GPU with a recent driver (tested on an RTX 5060 Ti and an RTX 3070 Ti Laptop). AMD and Intel are experimental: device loss has been reported (#3, #84, #99, #124).
- 16 GB of RAM plus a large page file (the process commits about 14 GB), 32 GB is comfortable.
- A controller is recommended.

## Known issues

- Device lost on several AMD cards without a fault address (#3, #84, #99, #124), and a GPU hang on an RX 6600 under RADV (#122).
- A crash on Linux when walking into the tunnel from the Lagoon Layer to the Hilltop Layer (#126).
- A frame limit below 60 on 1.0.0 only holds with "Game speed follows the clock" on; without it the game keeps its own timing (#118).
- The first visit to an area can stutter. Open-area performance is 31 to 35 fps on an RTX 3070 Ti Laptop; a heavy scene on an RTX 5060 Ti measured 51.8 fps with a 1% low of 20.6 fps.
- The cutscene at the end of the tutorial lost the device on NVIDIA before a workaround (#75); it is on by default and the cause is not known.
- The NVIDIA workaround for #9 skips light quads, which hides some particles (#62).
- Checkerboard textures (#5), one-sided sound in the first 20 seconds (#63), #76, #83, #112 and unrecorded NVN 2 calls (#6).

## Reporting a bug

Open an issue with the log (`dkb.log`), the build line at its top (`dkb 0.2.0+<commit>`), your GPU and driver, and what you did before it failed. **Never share generated files, keys or screenshots that show game content.**

Roadmap: #10. Testers: #3 (other GPUs) and #7 (play further).
