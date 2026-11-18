# Architecture

How the Switch build of *Tears of the Kingdom* becomes a native Windows x86-64 program. The upstream projects it builds on are listed under Credits in the [README](../README.md).

## Starting points

Two MIT projects cover different halves of the problem:

| | LAS Recomp | MacNativeGames |
|---|---|---|
| Game | *Link's Awakening* | *Tears of the Kingdom* 1.0.0–1.4.3 |
| CPU | ARM64 lifted to C, runs on x86-64 | Original ARM64 code runs directly on Apple silicon |
| OS boundary | IPC services and SVCs | ~163 `nn::*` SDK functions replaced by symbol, 38 SVCs |
| Graphics | Own NVN layer on Vulkan, own Maxwell translator | NVN front end + Metal 4 back end, Ryujinx translator |
| Useful to us | Lifter, NEON-to-C lowering, Windows memory layer, tests | Everything TotK-specific: host behaviour, NVN front end, version tables, patches, shader list |

MacNativeGames cannot run on x86-64 as is: its CPU path executes ARM64 code natively. LAS Recomp's host only covers what *Link's Awakening* calls. The port takes the CPU half from LAS Recomp and the game half from MacNativeGames, and writes the parts neither has: a lifter that scales to TotK, and a Vulkan back end for the NVN front end.

## Pipeline

### 1. Input

- Base 1.0.0 (`0100F2C0115B6000`) and one cumulative update (`0100F2C0115B6800`), NSP or XCI, plus the keys of the user's console.
- ExeFS and RomFS extraction: MacNativeGames' Python extractor (Python 3.14 for `compression.zstd`, `.venv-py314`).
- Update merging: hactool at `1d64a83` with MacNativeGames' BKTR patches, built by `tools/build_hactool.py`.
- Verification: MacNativeGames' `expected-v*.json` hashes of ExeFS, RomFS and fonts.
- Primary version: 1.4.3, the last update; performance work targets it. It replaced all three modules of the earlier updates (new SDK 20.5.22, RELR relocations, audio renderer REV15) and builds with `--version v143`, `recomp/patches-v143.txt`. 1.2.1 is a second target on the same build path; 1.0.0, 1.1.2 and 1.2.1 share `sdk`, `subsdk0` and `rtld` byte for byte, so only `main` differs among them. Saves load on either.
- Windows driver: `tools/extract.py` runs the vendored extractors (`tools/vendor/macnativegames/`) and identifies the update by the SHA-256 of its `main`. hactool has no Darwin `funopen` here, so the base Program NCA is copied out of the NSP before the BKTR merge (about 17 GB of temporary space).

### 2. CPU: lift ARM64 to C

- Modules: `main`, `sdk`, `subsdk0`. `rtld` is dropped and linking is done by the host, as MacNativeGames does.
- Base: LAS Recomp's hand-written decoder (`lifter.py`) and NEON/FP lowering (`simd_c.py`, `simd.h`), MIT. Output: one C function per guest function, direct calls for `BL`, an address-keyed table for indirect branches.
- Rewritten for TotK scale:
  - generator runs in parallel processes, with bounded memory per worker;
  - jump tables resolved by our own symbolic pass instead of Unicorn, which keeps GPL code out of the shipped generator;
  - `LDXP`/`STXP` (128-bit exclusive pairs) supported;
  - no 64 MB limit on `main`; `subsdk*` loaded;
  - NZCV and the six most used general registers of each function kept in C locals instead of the `Cpu`
    (`recomp/flags.py`, `recomp/regs.py`): registers are stored back where they changed, before calls, SVCs and
    exits, and reloaded after them. A call reloads only X0-X18 and X30, as AAPCS64 has the callee preserve X19-X29 and
    SP; `gen.py --no-callee-saved-trust` reloads those too.
- Coverage oracle: MacNativeGames' NSO rewriter lists every site where game code touches the OS (SVCs, TLS reads, counter reads, `x18`, exclusive windows, jump tables): 68,968 sites in 1.4.3 `main`. The lifter must handle every one.
- Memory: guest address equals host address, all below 2^39 (IPC encodes 39-bit addresses). The heap, stacks and the alias region are reserved with `MEM_WRITE_WATCH`, so the graphics back end can ask which pages the CPU wrote since the last upload (`runtime/memory.c`).
- Indirect branches go through `rt_dispatch` (`runtime/dispatch.c`), which looks the target up in the address table and tail-calls it.
- Testing: LAS Recomp's differential harness, every lifted instruction class checked against Unicorn (test-only, never linked into the game).

### 3. Host

- Portable C from MacNativeGames: threads and synchronisation, IPC, memory services, fs/RomFS, time, account, applet, audio renderer and effects. The Apple pieces (mach VM, CommonCrypto, AudioToolbox, GameController) are replaced with Win32 and SDL3.
- SDK boundary at the `nn::*` function level, matching MacNativeGames. Only `audren:u` and `sm:` are real IPC services.
- Window, keyboard and mouse: Win32 (`gfx_window.c`, `input_win.c`). Gamepad and audio output: SDL3. Voice: libopus (BSD). Movies: the SDK's `movie::Decoder` is replaced by a host VP9 decoder over libvpx (BSD, static; `runtime/ov_movie.c`), because FFmpeg from MSYS2 is a GPL build.
- zstd: most RomFS files of 1.4.3 are zstd frames. The patch set `zstd` sends the game's two decoder entry points to
  libzstd on the host (`ov_zstd.c`, `TOTK_ZSTD_NATIVE`); the lifted decoder took almost three times as long.
- Settings and menu: every key with its default is in one table (`cfg.c`), read from `totk.ini` and `TOTK_*`
  variables and written back when the in-game menu changes it. The menu and the performance overlay are Dear ImGui,
  drawn into the swapchain image (`runtime/ui/`); their text is in `ui_text.c` (English, Russian).
- OS calls the host makes go through `runtime/platform/` (`platform.h`, `win32.c`), so a Linux or macOS host adds one file instead of touching the rest. The split is not finished: `tools/check_platform.py` lists the Win32 uses left in the other files.

#### Threads and cores

Every guest thread is a host thread. TotK assumes Horizon's scheduler: three application cores, a thread runs only on a core in its affinity mask, and a higher-priority thread takes a core from a lower-priority one. The game keeps per-core data without locks and relies on that, so the host models it (`runtime/kernel.c`):

- Three virtual cores. A guest thread runs guest code only while it holds a core; the others wait in a priority-ordered list.
- Core masks as Horizon sets them: a new thread may run on its ideal core alone (core -2 is the process's, main.npdm's
  core 0) until SetThreadCoreMask widens it. Bananza leaves some threads on that default, which then run only when the
  main thread waits; with cores 0-2 instead, one of them changed a job graph while the main thread built it (#317).
- Preemption. When a waiting thread has a better priority than a holder on a core it may use, exactly one holder is flagged. The holder checks the flag at its next SVC and at every indirect call (`rt_dispatch`: two loads when nobody waits), checks again under the kernel lock that a better thread still waits, and only then hands its core over. The indirect-call check comes from SavageWindRecomp's BotW port; checking only at SVCs left a better thread waiting up to 26 ms.
- Asynchronous preemption (from MacNativeGames 8bf0893, on unless `TOTK_ASYNC_PREEMPT=0`). Game code can run for
  milliseconds with neither (`TOTK_CORE_DIAG=1` measures it): a waiter that has spun 250 us suspends the holder and,
  if its host rip is in lifted code at host depth 0 (not inside a host call or SVC), moves it to `kern_ap_stub`, which
  saves the volatile registers and gives the core up there. A holder of a guest mutex counts at the priority of its
  best waiter, and one inside an SDK spin lock (TLR+0x100 `disable_count`) is not preempted: its interrupt flag is set
  and the switch happens at SVC 0x36. `TOTK_ASYNC_PREEMPT=2` keeps those two without interrupting. The interruption is
  Win32 (SuspendThread, SetThreadContext); another host needs a signal handler there, as MacNativeGames has.
- Host work gives the core back. Queue submits, presents, GPU waits, shader translation, `nn::fs` calls and NVN memory pool setup run without a core (`NVN_BLOCKING`, `kern_core_away`), as the Switch blocks those threads in IPC, so a thread blocked in the driver or on a file does not hold up guest threads.
- Spin waits. Loops that only another thread can end (857 in 1.4.3, 864 in 1.2.1, found by `tools/spin_waits.py`, the
  nxnative search) send their back-edges through `rt_spin_gate`, which pauses the core once per pass and hands it to a
  waiter after 50 us of polling (`TOTK_SPIN`, `TOTK_SPIN_HANDOFF_US`).
- `TOTK_FREE_CORES=1` turns the gate off, for experiments.

### 4. Graphics

- NVN front end from MacNativeGames (`nvn_api_*.c`, `nvn_obj.h`, texture layout and deswizzle). The game records command buffers through it; nothing reaches the GPU until the game submits.
- Vulkan 1.3 back end in place of MacNativeGames' Metal one. At `QueueSubmitCommands` the recorded NVN commands are replayed into Vulkan command buffers (`vk_exec*.c`).
- Memory pools. Apple's unified memory let MacNativeGames use the game's pools zero-copy. The back end imports the pools as host memory (`VK_EXT_external_memory_host`). On a discrete GPU, pools bound as buffers are mirrored in device-local memory and the pages the CPU wrote (write watch, above) are uploaded before a replay; idle pool regions are made read-only so the write-watch queries skip them (`vk_memory.c`). GPU-rendered textures are written back when the CPU reads them: on a pool invalidate, on a CPU copy, and for small targets once the CPU waits for a fence (`vk_texture.c`).
- Replay thread. `TOTK_REPLAY_ASYNC=1` moves the replay off the submitting thread onto a queue at most one frame behind the game; reads of GPU results wait for the newest writer instead of draining the queue (`vk_backend.c`). It is off by default because no timing has shown a gain yet.
- Recorder thread. The Vulkan calls the replay makes into its command buffer are made by a thread of their own (`vk_rec.c`, after Savage Wind Recomp and LAS Recomp): each `vkCmd*` copies its arguments into a packet of a ring and the recorder makes the calls in order; calls into other command buffers are made at once; the recorder also ends the command buffer, and the submit waits until it has. With `TOTK_REC_SUBMIT=1` (off by default) a part that is not the submission's last is submitted by the recorder too, while the replay goes on with the next part. In the town on an RX 9070 the replay takes 17% less time on the submitting thread. `TOTK_VK_RECORDER=0` makes every call on the submitting thread.
- Helper translation threads (`TOTK_REPLAY_PAR=1`, off by default). Most of a replay goes into long runs of draws that only bind, look up and record. With the helpers on, when the replay enters such a run it hands the draws from a cut to an end to up to five helper threads, one consecutive part each (`vk_spec.c`, the `par_*` functions of `vk_exec.c`). Every helper translates its part with the helpers' copy of the draw translator (`vk_exec_draw_helper.c`: `vk_exec_draw.c` compiled once more, so that the replay thread's copy addresses its own state directly) and a draw context of its own, a copy of the state a draw changes, into a command buffer of its own pool; at the cut the replay submits its part, queues the helpers' parts behind it in order and goes on after the last. A helper works on the replay state the ops before its part give, and never writes memory the guest reads, uploads a texture, makes an image or waits: where a draw needs that (a GPU write into guest memory, a texture upload, a new image, a pipeline still building) it gives its part up and the replay translates those draws itself, and a part is also dropped when the replay changed something the helper read meanwhile. What a helper does make is what the replay would have made at the same draw and stays valid when the part is dropped: image views in the caches both threads use, a missing pipeline entered into the cache with its build queued, and the freshness marks of textures it found unchanged (`vk_spec.h` lists them). So a part is what the replay would have recorded or it is thrown away, and a dropped part does not cost the parts behind it. A scene does the same things at the same draws frame after frame: each run remembers which draws a helper cannot hold and where its reads change, plans the next frame's parts from that, and moves the cut and the parts' lengths until all threads finish together. A further part saves less than the one before and costs a join and a submission of its own, so the number of parts grows with the square root of the run's length: two from 640 draws, three from 1,440, four from 2,560. `TOTK_REPLAY_HELPERS` sets the number of helper threads (1 to 5); without it there are five from 24 logical processors, four from 16, three from 8, two from 4 and one below (`vk_spec.c` `vk_spec_helpers_for`). `TOTK_REPLAY_PAR=check` has the helpers and the replay translate every part and compares, draw by draw, what they bound and recorded (the replay's is the one submitted); `TOTK_REPLAY_PAR=0`, the default, has the replay translate every draw alone. On a Ryzen 7 5700X3D with one helper, in a scene bound by the replay, the frame goes from 12.9 to 11.3 ms (77 to 89 fps), and by the same 1.7 ms with the process held to four cores. In a scene of 27,000 draws a frame, whose four longest runs have 2,700 to 8,192 draws, four helpers take the frame from 43 to 33 ms (23 to 30 fps); three give 36 ms and five 33. Held to its first 8 logical processors the same scene takes 71 ms without helpers, 52 with one, 44 with two and 42 with three; held to 4, 74 without, 58 with one and 52 with two.
- Uniform snapshot. The game rewrites per-object uniform blocks on other threads soon after it submits the frame
  that reads them, while the replay still records it. At the start of a submission's replay every uniform range it
  binds from CPU-written pools is copied, and the draws bind the copy (`vk_usnap.c`, after MacNativeGames'
  `TOTK_UBO_SNAPSHOT`); without it, late draws read another object's matrix.
- Driver data. The NVN driver keeps a per-stage constant bank (c0) that shaders read for things like point size, dispatch size and base instance. The back end fills it the way the driver does.
- Shaders: the Ryujinx translator 1.3.2 with MacNativeGames' patches and ours (`tools/xlate/patches/`), built with .NET 10 NativeAOT as `totk_xlate.exe` and run as a separate process. Its SPIR-V goes to Vulkan directly. `tools/prepare_shaders.py` and `totk.exe --prepare-shaders` translate and compile the variants of MacNativeGames' shared list ahead of play (29,587 for 1.4.3); the rest are translated on first use and cached on disk.
- Pipelines: `VK_EXT_graphics_pipeline_library` where the driver has it. Shader stages are compiled ahead into libraries; a draw that meets a new state links them at once (fast link), and an optimised pipeline replaces the linked one when a background thread finishes it. Pipeline keys seen in play are saved and built at boot. A draw whose pipeline is not ready is skipped when one of its pass's targets was rendered in the previous frame, and waits otherwise, so one-shot passes are never lost; a full-screen draw is skipped as well rather than waiting (`TOTK_PIPE_SKIP`); the stage libraries a waiting draw needs are built on idle workers in parallel (`TOTK_PIPE_WAIT`). Without the extension, a missing pipeline is built on a worker and the draw waits or is skipped by the same rule. Before a one-shot draw waits, the pipelines of the submission's next 256 draws are keyed from their ops on a scratch copy of the replay state and queued, so a burst of new one-shot passes builds on the workers in parallel instead of one pipeline at a time (`TOTK_LOOKAHEAD`).
- Resolution. Patch sets make the game read its output, swapchain and scene sizes from the patch page
  (MacNativeGames' sites); `resolution.c` sets them from `output_res` and `render_scale` and grows the texture heap
  and the application pool to match. On the present, `vk_fsr.c` scales the picture to the window by `window_filter`
  (`window_filter.h`: AMD FSR 1 (EASU, RCAS) where the window is larger, the default; or linear, Catmull-Rom,
  Lanczos 2). `shaders/win_scale.comp` filters over the picture's pixels one window pixel covers, so a smaller window
  is filtered too, then sharpens by `TOTK_SHARPNESS` for the filters the player chose; with FSR 1 (the default) and the
  window the picture's size nothing runs and the picture is blitted. Then `vk_smaa.c` runs SMAA 1x on the finished
  picture when `TOTK_SMAA` asks for it, then `vk_crt.c` a CRT filter when `TOTK_CRT_FILTER` asks for one (Hookshot's:
  the picture reduced by area to a virtual raster of `TOTK_CRT_LINES` lines, then drawn back at full size as
  scanlines and a mask by Lottes', Hyllian's or zfast's shader), before the overlay is drawn. SMAA and the CRT filter
  are SDR passes: the game's own HDR picture skips them, and with the lifted HDR output they run on the 8-bit
  picture before the HDR pass. An `output_res` that is not 16:9 (more than 1% off, `res_is_16x9`) also switches on the
  `aspect` patch set, which hands the output's ratio to the camera, and `romfs.c` `overlay_init` loads the
  `romfs-patches-<w>x<h>` overlay for it when it exists (`tools/make_romfs_patches.py --aspect`, laid out by
  `tools/ultrawide_layout.py`; the installer builds ten sizes: 21:9 2560x1080, 3440x1440, 3840x1600, 5120x2160, 32:9
  3840x1080, 5120x1440 and 16:10 1280x800, 1680x1050, 1920x1200, 2560x1600). A wider layout moves the HUD to the
  screen's corners, a narrower one to its top and bottom edges. Other sizes get the camera of their shape and the 16:9
  HUD stretched over it. The `aspect` setting (`fit`, the default; `stretch`) decides what the present does
  when the window's shape differs from the output's: `fit` letterboxes, where a 16:9 game in a wider window used to be
  stretched.
- HDR (`hdr = 1`, experimental, `vk_hdr.c`). The game's window texture is 8-bit sRGB, so HDR is an inverse tonemap of
  it, not the scene's real range. On an HDR10 swapchain (A2B10G10R10, ST 2084; `VK_EXT_swapchain_colorspace`,
  `VK_EXT_hdr_metadata`) the present chain is game, FSR, SMAA, the CRT filter, the HDR pass, the menu. FSR, SMAA and the CRT
  filter write an 8-bit
  image of the swapchain's size; `shaders/hdr_encode.frag` then linearises it, stretches the largest channel above a
  knee towards `hdr_peak_nits` (all channels scaled together, so hue is kept), puts white at `hdr_game_nits`, converts
  Rec. 709 to Rec. 2020 and PQ-encodes into the swapchain image. The in-game menu is drawn into an RGBA8 image and
  blended on at `hdr_ui_nits`. Without an HDR10 surface format (Windows with "Use HDR" off, a compositor without
  colour management) the swapchain stays SDR and none of this runs.
- Native HDR, Bananza 3.2.0 (`hdr_native = 1` with `hdr = 1`). `ov_gfx.c` answers the game's
  `nn::vi::QueryDisplayHdrStatus` with "on" and `hdr_peak_nits`, and `QueryDisplayHdrPaperWhite` with
  `hdr_game_nits`. The game then switches to its own HDR mode: its window texture becomes A2B10G10R10 holding
  Rec. 2020 PQ, which `vk_present.c` `present_image` puts on the HDR10 swapchain as the codes it is: blitted, or,
  when `window_filter` and Sharpness ask for it (a filter other than FSR 1, and the window differs from the picture or
  Sharpness is above 0), scaled and sharpened by `vk_fsr.c` with `win_scale_pq` (`win_scale.comp` with `WIN_SCALE_PQ`:
  the filters and the sharpen work on the PQ code values and write an A2B10G10R10 image, so the 10 bits stay;
  `window_filter.h` `wf_plan`). No colour adjustment, SMAA or CRT filter, which expect an SDR picture. The game's
  tonemap stops its highlights at about 1.36 times paper white,
  whatever peak it is told; `spirv_hdr_shoulder.c` (`DKB_HDR_SHOULDER=0` turns it off) rewrites its composite stage
  so they reach the peak, leaving everything below the shoulder's knee as the game grades it. Findings and
  measurements: [hdr-native-plan.md](hdr-native-plan.md).
- DLSS (`upscaler = dlss`, experimental, NVIDIA only, `vk_dlss.c`): the game's upscaler dispatch is replaced by NVIDIA
  NGX DLSS on the same images, spatial only for now. Setup and limits: [dkb-dlss.md](dkb-dlss.md).
- Occlusion queries never block the CPU; a result that is not ready counts as visible.
- ASTC textures are uploaded as they are where the device reports `textureCompressionASTC_LDR`, and decoded by a compute shader everywhere else (`vk_astc.c`, `TOTK_ASTC`). The CPU decoder (`tex_astc*.c`) is the fallback and the test oracle.
- Draw state. A draw whose program, bound state and textures repeat reuses the previous draw's shaders and skips rebuilding its descriptor sets; descriptor sets are written through update templates (`vk_exec_draw.c`, `vk_pipeline.c`).
- Allocation: Vulkan Memory Allocator. Debugging: Vulkan validation layers, per-pass frame dumps (`TOTK_DUMP`, `vk_dump.c`) and a flight recorder (`TOTK_FLIGHT`). RenderDoc cannot capture, because it hides `VK_EXT_external_memory_host` from the application.

Maxwell behaviour the back end has to match, and the TotK bugs other ports fixed: [maxwell.md](maxwell.md).

Why Vulkan rather than D3D12 or wgpu:

- The translator emits SPIR-V, which Vulkan consumes as is. D3D12 would add a SPIR-V to DXIL step; wgpu would add SPIR-V to WGSL through Naga and drop features the game uses (bindless textures, some atomics).
- LAS Recomp's Vulkan back end is a working reference on Windows.
- The same back end runs on Linux.

### 5. Toolchain

- Generated C: MSYS2 UCRT64 Clang + LLD. The generated code uses GNU extensions (`__int128`, `musttail`, vector types) that `cl.exe` cannot compile.
- Host code: the same Clang, C11.
- Builds: `tools/build_game.py` writes a Ninja file and runs MSYS2 Ninja. Every object is hashed before linking, because a faulty RAM module once flipped bits in stored objects.
- Analysis: Ghidra with the Switch loader, Capstone, `llvm-objdump`.
- Players: the installer ([installer.md](installer.md)) builds the game code on the player's machine with a bundled llvm-mingw, so no recompiled code is ever distributed.

