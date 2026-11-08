First release of DKB Recomp: a static recompilation of *Donkey Kong Bananza* (Switch 2) to a native Windows exe with a Vulkan renderer. It is experimental (0.x): it plays the tutorial, its cutscene and the first stage on recent NVIDIA cards, and it is not finished.

**The packages hold no game data, and no release ever will.** You build the game on your own PC from your own dump of **1.0.0 or the 3.2.0 update**, given as a `.nup` file or as ExeFS and RomFS folders. Other updates and the DLC are not supported.

## Download

| Package | For | Status |
|---|---|---|
| `DKB-Setup-<version>-windows.zip` | Windows 10/11 x64 | Supported. Carries its own tools and the installer window: unpack, run `DKB-Setup.exe`, pick your dump. |
| `DKB-Setup-<version>-linux-x86_64.tar.gz` | Linux x86-64, kernel 6.7 or later | Preview. Source package with `setup.sh`: you install the system packages it lists (`./setup.sh --check`), then it runs the command-line installer. |
| `DKB-Setup-<version>-macos-arm64.zip` | macOS 14.4 or later, Apple silicon | Preview. Same idea with `setup.command`. Play is not verified and the destructible terrain is missing (Metal has no geometry stage). |

## Requirements (Windows)

- NVIDIA GPU with a recent driver (tested on an RTX 5060 Ti and an RTX 3070 Ti Laptop). AMD and Intel are experimental: device loss has been reported (#84, #99).
- 16 GB of RAM plus a large page file (the process commits about 14 GB), 32 GB is comfortable.
- A controller is recommended.

## Known issues

- The first visit to an area can stutter. The game imports memory on first use now, which removed most of the long stalls, but shader and pipeline work on a new area can still take a second or more.
- The cutscene at the end of the tutorial lost the device on NVIDIA before a workaround (#75). The workaround is on by default and passed on an RTX 5060 Ti; the cause is not known and other cards are untested.
- The NVIDIA workaround for #9 skips light quads, which hides some particles (#62).
- Checkerboard textures (#5), one-sided sound in the first 20 seconds (#63), a darker opening wall, #76, #83 and unrecorded NVN 2 calls (#6).
- Open-area performance is 31 to 35 fps on an RTX 3070 Ti Laptop (replay cost).

## Reporting a bug

Open an issue with the log (`dkb.log`), the build line at its top (`dkb 0.1.0+<commit>`), your GPU and driver, and what you did before it failed. **Never share generated files, keys or screenshots that show game content.**

Roadmap: #10. What this release needed: #85. Testers: #3 (other GPUs) and #7 (play further).
