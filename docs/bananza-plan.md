# DKB Recomp plan

Goal: *Donkey Kong Bananza* (Switch 2, base 1.0.0) playable on Windows as a static recompilation, built on
[TotK Recomp](https://github.com/valentderah/totk-pc-recomp). No game files in the repository: each user dumps their
own cartridge and the build runs on their machine.

Known input: the base ExeFS has `main` (md5 f68752f5…, the binary GRAnimated/DKBananza's BinSync uses) and `sdk`, no
`subsdk0`. Update 3.2.0 (title version 458752) is a second target with its own ExeFS hashes and patch table
(`dkb458752`). The RomFS is about 19 GB.

Each phase ends in a milestone someone else can check. The order follows the critical path.

## F0 — CPU: lifter and build (done)

- The TotK lifter runs on Bananza: 133,806 functions in `main`, 46,255 in `sdk`.
- ARMv8.3 instructions the Switch 2 code uses (commit 3941578): `retab`/`retaa`/`braa`/`blraa` as branches without
  authentication, `ldapr*` as `ldar*`, PAC hints as NOP. Each new class has a differential test against Unicorn.
- **Milestone reached:** all of `main` and `sdk` lift with no unsupported instruction, compile and link.

## F1 — OS: Horizon on Switch 2 (done)

- NSO loader for Switch 2 modules (RELR, MOD0), TLS, memory layout, `main.npdm`.
- TotK Recomp's host (threads, IPC, file system and RomFS, time, account, applet) with the Switch 2 `GetInfo` ids,
  the 9 GiB memory pool, `IAudioDevice` 19-21, `IAudioRenderer` 12-14 and the Switch 2 `oe`/`vi`/`account` calls.
- **Milestone reached:** the game leaves `nnMain`, mounts the RomFS and reaches its first NVN call.

## F2 — NVN 2 with a null renderer (done)

- NVN 2 bootstrap and proc table read from the game's own `sdk` (the driver is linked into the game, as on Switch 1).
  529 of the 700 entry points share NVN 1's names; the 171 new ones are listed in
  [bananza-nvn-new-procs.txt](bananza-nvn-new-procs.txt).
- With `TOTK_GPU=null` the main loop runs at 20 fps, audio plays and controller input arrives.
- **Milestone reached:** the game loop runs without a picture.

## F3 — Shaders: Ampere to SPIR-V (done for the first scene)

- Eglantine's Ampere-to-Maxwell lowering (`runtime/third_party/eglantine/`, GPL) runs at `nvnProgramSetShaders`. Its
  output goes to TotK Recomp's Maxwell translator (Ryujinx's, `tools/xlate/`) and comes out as SPIR-V.
- Fixes on top of Eglantine are marked "DKB Recomp" in its files. Translator fixes are patches in
  `tools/xlate/patches/`. Notes: [bananza-shaders.md](bananza-shaders.md).
- **Milestone reached:** every shader of the first 3D scene lowers and translates (4,820 of 4,820 stages).

## F4 — Vulkan back end (done for the first scene)

- NVN 2 compute constant bank c0 packed per program from the control section counts. Shared memory size read from
  the control. Bank 17 holds the Switch 2 blit parameters.
- Every single-sample colour texture can be used as a storage image, because NVN 2 does not ask for it.
- **Milestone reached:** the title screen and the tutorial cave render (2,600 frames, 0 validation errors).

## F5 — Playable (in progress)

- [x] Controller input in the menu and in gameplay (walking, punching). Keyboard and mouse come from TotK Recomp's
      input layer ([controls.md](controls.md)).
- [x] Audio: the game's audio renderer (REV15) mixes on the host and plays through SDL3. In a 90 s tutorial run,
      `TOTK_TRACE=audio` measured an RMS of 0.02 to 0.04 and peaks up to 0.41 once gameplay starts, with no underrun
      or overrun.
- [x] Tutorial guide movies: `nn::codec` VP9 decoding on the host with libvpx (`runtime/ov_codec.c`).
- [ ] GPU fault (an invalid instruction pointer; the freeze was the same fault, 86fb7cc) right after the first guide
      movie frame. It is the first draw of one instanced light-quad vertex shader, and its fragment stage (one of the
      123 that read a storage buffer at c0+0x530) does not matter. Its fragment code, its data and rasterisation are
      ruled out; leaving out only the draw command passes. Leaving out that shader moves the fault to another vertex
      shader of the same light-quad family. Seen on NVIDIA 617.14 (a crash inside the driver's shader compiler),
      617.42 and 621.01. Mitigated (84a8f7d, `TOTK_FIX_VS_FAULT`): the quads with ten instance streams are skipped on
      NVIDIA. The later fault (about 2 runs in 4, also with synchronous pipeline builds) follows one more quad of the
      same shape, NVN 2 vertex stage cae0ef8e9fe9147f, which is skipped too (6da41ba): 0 faults in 6 runs of 600 s
      since. The cause is still open: its vertex input and SPIR-V match quads that never fault. [#9]
- [x] Partial stencil clears (write mask other than 0xff): a full-screen triangle replaces only the masked bits (#36,
      by monsterlag).
- [x] `nvnCommandBufferPush{Graphics,Compute}Constants`: recorded (33628c7). Graphics root table 0 is bank 24, read by
      the full-screen blits, which now get the pushed `{1/w, 1/h, w, h}`. Compute root tables sit in the packed c0
      after its 16-byte head, each padded to 16 bytes, with the word counts from the shader control (#20, by
      monsterlag): the upscaler's two image rectangles are root table 0, pushed by the game. [#2]
- [x] Colour grading: the 3D LUT is filled (8x8x8 R11G11B10F, checked in the mining tunnel); the empty LUT was an old build. [#4]
- [ ] Some textures look blocky (checkerboard rocks) [#5]: the deswizzle now uses the Switch 2 GOB layout, which fixes the gold rock in the tutorial cave; the other rock surfaces still need a check.
- [ ] Getting past the tutorial cave. [#7] A hand-played run reached the Ingot Isle Cave-In section; respawning
      there stalled the GPU.
- [ ] Destructible terrain renders black. [#45] Its indirect records are now reset on the host in replay order
      (#48, `TOTK_FIX_INDIRECT_RESET`), which should also end the garbage arguments behind the respawn stall in
      #10 (not retested yet). The index counts come from the geometry stage, still to be lowered and run. Ordinary
      geometry shaders now translate and cache under an explicit topology contract (#49, #50), and replay behind
      `TOTK_EXPERIMENTAL_GEOMETRY=1` (#52). The translator also accepts a portable passthrough subset (#54). The
      terrain's geometry stage is passthrough, and the replay still rejects it until #55 lands.
- [x] Saves: the game writes `progress.sav`, `caption.sav` and `config.sav` as it plays, and a later run on the same
      profile offers Continue and resumes at the last checkpoint (the mining tunnel of the tutorial).
- [x] Performance in the tutorial: the game picks its own present interval (1 to 3 vsyncs) from how long its frames
  take. A profile of the render thread found the #9 skip's stage-name lookup taking 72 % of it (fixed in d5fd595), and
  the host's hold of each present until its display time made the job workers' mid-frame GPU reads wait for vsync
  (off for Bananza since 69d51d0). A 600 s run of the scripted test now runs at 59.4 to 60.0 fps in every minute after
  the first. Other areas and other GPUs are not measured yet.
- [ ] Testing on other GPUs. [#3]
- [x] Installer on TotK Recomp's model (`docs/installer.md`, #19): from the user's base 1.0.0 ExeFS and RomFS folders
      it packs the RomFS, builds the shader translator, lifts with the patch table and builds `dkb100`, run from this
      checkout with the documented toolchain. A release package is not made yet. [#8]

Issues: [#2] [#3] [#4] [#5] [#7] [#8] [#9]; the whole list is the
[roadmap](https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/10). The progress so far, by date:
[devlog.md](devlog.md).

[#2]: https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/2
[#3]: https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/3
[#4]: https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/4
[#5]: https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/5
[#7]: https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/7
[#8]: https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/8
[#9]: https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/9
- **Milestone:** play from the start to the end of the first level without a crash.

## Later

- NVN 2 calls the host accepts and does not record yet (`runtime/nvn_api_nvn2.c`, `NVN2_NOT_RECORDED`): mesh shaders,
  variable-rate shading, DLSS, ray tracing, device command buffers. Each one is logged on its first call. None of
  them has mattered so far.
- Play-testing of update 3.2.0 (`dkb458752`): the installer supports it and it boots, but nobody has played it as
  far as the base has been played. Add-on content (`runtime/aoc_romfs.c`) is likewise untested in game.
- Linux, which TotK Recomp's toolchain supports.
- DLSS and temporal upscalers (checked 2026-11-09). The game does not use DLSS: with `nvnDlssNetworkInitialize`
  answering success it still called no `nvnDlss*` proc in a 330 s run on 1.0.0, and a scan of the guest code finds
  no reference to the DLSS slots of the NVN proc table in 1.0.0 or in 3.2.0 (the loader alone names them). Nor do the
  frame dumps show a velocity target or a history texture, so the game has no temporal AA to borrow from.
  `upscaler = dlss` (docs/dkb-dlss.md) therefore makes its own inputs, as TotK Recomp's `vk_dlaa.c` does: a sub-pixel
  jitter on the scene's viewports, the scene depth, and camera-only motion vectors from depth and the camera block's
  view-projection (vertex bank c3, 0x980 bytes, found with `DKB_DUMP_UBO=1`). Still missing: per-object motion
  (animated characters can smear), DLSS where the game renders at the output size and skips its upscaler, and FSR 3.
  Frame generation has the same input needs. The drivers' own frame generation (NVIDIA Smooth Motion, AMD AFMF) needs
  nothing from this project.

## Alongside

- Symbols in [GRAnimated/DKBananza](https://github.com/GRAnimated/DKBananza) (nerve names, BinSync), which help debug
  every phase.
- Following Eden, oboromi, NeXo2 and any other NVN 2 work, and crediting what is used.

## Rules

- Nothing from the game in git: no ExeFS, RomFS, generated C, extracted shaders or screenshots with game assets.
- Licences: see [AGENTS.md](../AGENTS.md), "Reusing code".
- Small pull requests, one topic each, with honest numbers.
