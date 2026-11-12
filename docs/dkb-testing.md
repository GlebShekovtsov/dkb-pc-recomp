# Testing DKB Recomp

This guide is for running DKB Recomp on your own PC and reporting what happens. You build it from source with your own
copy of the game. Nothing from the game is in this repository, and nothing you build from it may be shared: not the
executable, not the generated code, not the shader cache.

## What you need

- Windows 10 or 11 on an x86-64 CPU with AVX2.
- 16 GB of RAM plus a large page file (32 GB is comfortable). The game commits about 14 GB (13.5 GiB at the start, 13.75 GiB after 40 minutes in the tutorial and first mine,
  rising about 6 MB per minute, peak 15.3 GiB; sessions over an hour are untested), so close other large programs first.
- A GPU with Vulkan 1.3 and a current driver.
- About 15 GB free for the build (the packed RomFS is about 9 GB), besides the space your dump takes.
- *Donkey Kong Bananza* base version 1.0.0 or update 3.2.0 (title version 458752), dumped from your own Switch 2: the
  ExeFS (`main`, `sdk`, `main.npdm`) and the RomFS as a folder, or a `.nup` file. The installer picks the target by the
  `main` hash and, for 3.2.0 only, keeps the generated C and objects in `temp/gen` and `temp/build` so a second install
  does not relift or recompile; the base install removes them. This project does not explain how to dump the game.
  A `.nup` file (the container Eglantine and Eden load) goes straight to the installer (`--nup`, [installer.md](installer.md)).
  For a build by hand, `tools/extract_nup.py` writes the folders: `python tools\extract_nup.py <game>.nup --list` shows
  the packages, and `python tools\extract_nup.py <game>.nup --select latest --out build\extracted\dkb-dump` writes
  `exefs` and `romfs` there as Eglantine runs them (the newest update's ExeFS, the base RomFS with the update's laid
  over it; `--select base` for 1.0.0), checking every file's SHA-256. Pass those two folders to the commands in the
  README, with `dkb458752` and `recomp\patches-dkb458752.txt` in place of `dkb100` for 3.2.0.

On Apple Silicon macOS the same steps work ([building.md](building.md), "macOS"): the commands use
`.venv/bin/python`, the executable is `build/dkb100/dkb`, and the renderer is MoltenVK. The game boots to the mode
select, but the destructible terrain is not drawn yet, because Metal has no geometry stage.

## Build

Install the toolchain from [building.md](building.md) (MSYS2 UCRT64 Clang, Ninja, Python 3.12 in `.venv`, .NET SDK),
then run the commands in the README, "Building from source". The executable ends up in `build\dkb100\dkb.exe`.

The generation command must include `--patches recomp\patches-dkb100.txt`. This enables the controller setup patch
that bypasses the game's HID server proxy, which the host does not implement. If you generated the code without
that argument, rerun the README's generation command with it and then rebuild with `tools\build_game.py`.
Rebuilding alone leaves the unpatched generated code in place and the game stops during controller setup.

## Run

```
build\dkb100\dkb.exe
```

Run it from inside the repository: it finds the shader translator (`build\xlate\totk_xlate.exe`) by looking up from
its own folder. The first run translates every shader it meets and is slow; later runs use the cache.

Prepare the shaders once (the Bananza installer does this as its `dkb_shaders` step): the game's own shader files hold
about 18,800 shader stages, and translating one the first time a draw needs it takes about 35 ms, with the draw waiting.
On a first visit to an area that adds up to seconds. These two steps translate them all up front, from your own game
files:

```
.venv\Scripts\python tools\dkb_shader_census.py --version dkb100 --out build\dkb100
build\dkb100\dkb.exe --prepare-shaders
```

The first reads the shader folders of the extracted RomFS (4 seconds) and writes `seed-nvn2.bin` (about 187 MiB) next to
`dkb.exe`; the second lowers and translates every stage into the shader cache. Each stage is translated with the default
answers (every texture a 2D texture, float vertex inputs and colour outputs), and each vertex stage again for the integer
vertex layouts met so far (Locations 6, 6 and 7, 3 and 4) plus any layout your own `shader-answers.bin` recorded. That
takes about 150 seconds on 20 threads and about 2.9 GB of cache. The pack is deleted afterwards by the installer.
`--prepare-shaders` also builds the driver's pipeline libraries in the same process. The installer instead runs
`--translate-shaders` (translation only) and then `--warm-shaders` (libraries only), so a driver crash in the second
cannot fail the first.

In a cold run of the first area this cut the shaders translated during play from 714 to 62, the worst translation stall
from 6.6 s to 0.34 s, and the 1% low of the last window from 24 to 39 fps. What is left is about 1.4% of the stages the
game loads: 38 embedded in `main` and about 50 whose code is in no file the census reads. They translate on first use, and
play records them. `DKB_ANSWER_STATS=1` logs each translation made in play with what made it necessary.

Useful settings, as environment variables (`set NAME=value` in the same Command Prompt):

| Setting | Effect |
| --- | --- |
| `TOTK_PROFILE_DIR=<folder>` | Use this folder for the save and settings. A new, empty folder gives a fresh start, which is what reports should use. |
| `TOTK_GPU_NAME=<text>` | Use the first GPU whose name contains this text, on machines with more than one. |
| `TOTK_GPU=null` | Run without a picture (no Vulkan). Useful to tell CPU and OS bugs from graphics bugs. |
| `TOTK_XLATE_PROCS=<n>` | Shader translator processes at run time (default 2). Lower it if RAM runs out. |
| `TOTK_TRACE=perf` | Log the frame time, what holds frames back and the GPU time every 60 frames. For stalls and low frame rates. |
| `DKB_REPLAY_THREADS=N` | Experimental, off by default (`docs/parallel-replay.md`). A render pass whose targets took at least 256 draws the last time is recorded in slices of 1024 draws, each into a secondary command buffer that the pass executes in order (`vk_exec.c` `par_start`, `par_flush`). `N` threads record the slices: the submitting thread and `N-1` workers (`1`: the submitting thread alone, to compare). Try `4`. The log says `vk: passes of 256 or more draws are recorded in slices ...` at the first sliced pass, `vk: N replay slice workers started`, and `a sliced pass needed a pass or barrier change inside a slice` when a pass cannot be sliced (it is then left alone). With `N` above 1 the command buffer parts default to at least 300 us (`TOTK_SPLIT_US`, else 1000), so the GPU starts sooner on the sliced passes. |
| `TOTK_MEM_REPORT=N` | Logs `mem: commit frame F: guest heap+stack, alias, modules, kernel, other private, mapped, image; process commit (limit)` every N presents (0, the default, off), and once at exit with `exit`. One walk of the address space (`memory.c` `mem_commit_report`), with the GPU side (device-local, mirrors, staging pool, usnap arenas) appended. For where the process's commit goes. Windows only. |
| `TOTK_HEAP_LAZY=1` | Experimental, off by default (0), Windows only. For 16 GB machines with a small page file, where the commit limit is what ends the game: the guest heap (the game asks for about 9 GiB with SetHeapSize, about 4.2 GiB of it ever resident) stays reserved and each 64 KiB chunk is committed when first touched (`win32.c` `fault_veh`), or by `mem_touch` before a file read or a Vulkan host-pointer import. See the effect with `TOTK_MEM_REPORT=N`: `process commit` and `guest heap+stack` drop, and a `lazy heap` line counts the commit faults. |
| `DKB_DRAW_PROFILE=1` | Logs where a draw's replay time goes every 60000 draws, in nanoseconds per draw: `vk_exec: draw profile ...` with the sections pass, shaders, pipeline, state, bind-sig, ubo, ssbo, textures, sets, feedback, vertex-buffers, tail and other-ops (`vk_exec_draw.c` `prepare_draw`). For busy areas where `TOTK_TRACE=perf` reports a large `replay avg`. One clock read per section mark, so leave it off otherwise. |
| `TOTK_DYN_UBO=1` | Makes the uniform buffers of the allocated fragment set dynamic (`vk_pipeline.c` `dyn_ubo_convert`), so consecutive draws that differ only in a buffer's offset reuse one written set (`vk_exec_draw.c` `bind_resources`). Off by default. Its author measured +16% fps in dense areas on an RTX 3050 laptop; untested on AMD and Intel, and not yet run under the validation layer. Any value other than `0` (including `off`) turns it on; `ab` alternates it with the A/B blocks. |
| `TOTK_TEX_REPEAT=0` | Looks up every graphics draw's texture descriptors again. By default a draw with the same pipeline as its thread's last descriptor build, the same sampled texture handles, the same descriptor context and generations and the same feedback-layout attachments reuses the descriptors that build wrote (`vk_exec_draw.c` `tex_repeat`); draws that bind storage images always look up. `ab` alternates it with the A/B blocks. For checking a texture glitch against the reuse. |
| `TOTK_WB_ASYNC=1` | Writes a deferred guest read's render target back without making the replay wait: the copy is submitted and lands in the storage at a later poll (`vk_texture.c` `vk_texture_wb_poll`), unless the CPU rewrote the storage meanwhile. Off by default: no fps gain beyond run-to-run noise was measured, and 30-45% of the late copies found the storage rewritten and were dropped. |
| `TOTK_PERF_SLOW_GPU_MS=<ms>` | With `TOTK_TRACE=perf`, a replay part that runs this long on the GPU (default 100) gets a line with its draw and dispatch counts and the NVN 2 names of its stages, a second line with the arguments of its indirect draws as read back afterwards, and a third saying whether a shader in that part wrote the largest one's arguments, and which. |
| `TOTK_FIX_SCREEN_UV=0` (or `DKB_FIX_SCREEN_UV=0`) | Leaves the map's mask shader as the game wrote it. By default its screen position is computed from `gl_FragCoord` instead of read from varyings the vertex stage divided by w: the camera sits inside the box that shader draws, and on AMD (Eglantine as well) the faces behind the camera turned the varyings into long cyan wedges across the map. The shader is recognised by its NVN 2 name or the hash of its translated SPIR-V, and the log names it when it is rewritten (`vk: fs 544677ae4ec509be ... rebuilt from gl_FragCoord`). |
| `TOTK_DUMP=key` (or `DKB_DUMP=key`) | Frame dump on a key: press F9 in the game window and the frame after next is dumped to `dump\` (`passes.txt`, one PNG per pass target; `TOTK_DUMP=<frame>` dumps a fixed frame instead). With `DKB_DUMP_UBO=1`, `passes.txt` also lists each pipeline's uniform blocks; it always lists the vertex inputs. The PNGs show the game: never attach them publicly. One dump per run. |
| `TOTK_LIST_RESTART=0` (or `DKB_LIST_RESTART=0`) | Draws triangle lists without primitive restart, as before. By default a list draw restarts at its restart index, as on the Switch, when the GPU has `primitiveTopologyListRestart` (the log says which at start-up); without it the restart indices are fetched as vertices and give long stretched triangles, as on the map. For comparisons only. |
| `TOTK_FIX_INDIRECT_RESET=0` | Draws the terrain's indirect-record reset program as a shader instead of the host doing its stores (#48). The shader's stores are lost, so the records keep zero or stale contents; for comparisons only. |
| `TOTK_EXPERIMENTAL_GEOMETRY=1` | Replays draws with an ordinary geometry stage through it (#52). Off by default; other geometry draws are rejected once in the log. |
| `TOTK_NATIVE_GS=0` (or `DKB_NATIVE_GS=0`) | Turns off the replay of the game's native passthrough geometry stages on the portable path. That replay draws the destructible terrain (#45) and is on by default; with it off the terrain is black. The varyings are inferred, not read from the hardware skip mask. |
| `TOTK_REC_BEGIN=1` | The Vulkan recorder thread also begins each part: `vkBeginCommandBuffer`, the submit breadcrumb and the start timestamp go through the ring instead of being made by the submitting thread. 0 (default) keeps the old order. For AMD proprietary-driver device loss on Windows that goes away with `TOTK_VK_RECORDER=0`; the same idea as `TOTK_REC_END` (#5). |
| `TOTK_REC_BATCH=N` (or `DKB_REC_BATCH=N`) | Publishes the Vulkan recorder ring's head every N packets (1..64) instead of after each one (#93), so the recorder thread's spin reads a cache line the producer rewrites less often. 0 (default) publishes after every packet. Waits and handoffs publish the rest. With large N the last packets are recorded inside the final drain while the submitting thread waits, so watch `drain_ms` in `TOTK_TRACE=perf`; start with 8. |
| `TOTK_TEX_HASH=0` (or `DKB_TEX_HASH=0`) | Uploads a texture again on every check its storage pages cannot answer, as before (#116). Without a write watch (macOS), the default hashes the storage bytes before each upload and skips the next upload while the hash is unchanged. With `TOTK_TRACE=perf` the `texture re-uploads` line counts the uploads kept. The hash runs on any platform for a check the write watch cannot answer (on Windows when `GetWriteWatch` fails for a range, on Linux without the write-tracking descriptor); where the watch answers, nothing changes. |
| `TOTK_POOL_FINI_WAIT=idle\|fence` (or `DKB_POOL_FINI_WAIT`) | What releasing a used memory pool waits for before the game gets the storage back. `idle` (default) idles the whole graphics queue; `fence` waits only for the newest finished submission on the timeline and does nothing when it is already done; it covers the work the game has synchronized on, not the parts already submitted for the submission still being recorded. A level change frees many pools one after another, and a `TOTK_PROFILE_SLOW` dump of a 1.6 s frame showed five guest threads in turn in the idle wait. With either value the log gets a `used pool ... released` line for each release that waited over 10 ms (the first 64). |
| `TOTK_HOST_PROPS_CACHE=0` (or `DKB_HOST_PROPS_CACHE=0`) | Asks `vkGetMemoryHostPointerPropertiesEXT` for every imported memory pool instead of keeping the first answer for the others (the default). The first answer is assumed to hold for every guest pool (all are plain committed read/write memory); if `vkAllocateMemory` fails with the kept answer, the query is made for that pool and the import is tried once more. On NVIDIA the query took long for the big pools of a level change, with several guest threads in it in turn (1.8 s in a `TOTK_PROFILE_SLOW` dump). A query that takes over 10 ms is logged with the pool size (the first 16); a `vkAllocateMemory` import that takes over 10 ms is logged too (the first 32). |
| `TOTK_POOL_LAZY=0` (or `DKB_POOL_LAZY=0`) | Imports every pool at once, as before. By default `nvnMemoryPoolInitialize` only records an imported (CPU-visible) memory pool, and the import of its guest pages into Vulkan waits for the first time the GPU resolves an address in it. On NVIDIA each import cost about 20 ms of driver time whatever the pool size, one at a time, and the guest threads waited in them (hundreds per level change); the `mem: of the N imported pools` census line shows how many pools the GPU ever used (26 of 408 at frame 600 in one run). The price moves to the replay thread: the thread that first resolves a pool for the GPU (replay, texture creation or a snapshot) imports it. It is the default now (it cut a scripted 72 s route from 3219 imports and 27 s in them to 901 and 1.2 s, and four stalls of 360 to 950 ms to one of 993 ms). Shadow storage always imports at once. The census line also counts the pools imported on first use. |
| `python tests/cfg_settings_test.py [--exe build/dkb100-la/dkb.exe]` | Settings table check (`runtime/cfg.c`): every row is in `known[]` and `sections[]`, defaults satisfy their own values, menu keys have rows; docs naming unknown `TOTK_`/`DKB_` keys are printed as warnings. With `--exe` it copies the exe and DLLs to a scratch folder and, through `DKB_PROBE_CFG_INI=<file> dkb.exe --probe-cfg` (prints `key = value` per row), writes every accepted value of every row (list entries, range min/middle/max) into a `dkb.ini` and checks it reads back; a range value that does not parse must give the default. List values are not validated by `cfg.c`, so a bad list entry is not tested. |
| `DKB_GEOMETRY_CONTEXT=1` | Logs the first 12 draws with a geometry stage and their buffers (#58). `DKB_GEOMETRY_CONTEXT_ORIGIN=<hex>` keeps only draws with that stage origin. |
| `DKB_GPU_CHECKPOINTS=1` | On NVIDIA (VK_NV_device_diagnostic_checkpoints), marks every pipeline bind, compute dispatch, copy, blit and colour clear so that a lost device logs the last checkpoints the GPU reached, with the NVN 2 names of the shaders of each pipeline (#75). With VK_EXT_device_address_binding_report it also keeps the last 4096 GPU address bind/unbind events and, for each address in the device fault report, logs the bindings that contain it or the nearest one ending within 1 MiB below it. Also saves the driver's fault dump to `logs\dkb-fault-<time>.bin`; it can contain shader code, never attach it. With VK_KHR_pipeline_executable_properties it also logs, the first time a pipeline is bound, one line per driver executable (stage, NVN 2 stage names, every statistic such as binary size) and saves its internal representation (ISA) to `logs\dkb-isa-<pipeline handle>-<stage>.txt` (or `.bin`; at most 4096 pipelines and 512 MiB; derived from the game, never attach). The capture runs on the recorder thread at each new pipeline's first bind, so it can slow recording, and toggling the variable costs one cold pipeline-cache run. Off by default. AMD always has its own breadcrumbs. |
| `TOTK_CRUMB_BIND=1` | Debug only, for GPUs with VK_AMD_buffer_marker (AMD) and no NVIDIA checkpoints: marks every pipeline bind with a breadcrumb in its own marker words, and a lost device logs the last bind the GPU reached and finished plus the last 8 binds with their pipeline and NVN 2 stage names. Off by default; costs two buffer-marker writes per bind. |
| `TOTK_FIX_VS_FAULT=all` | Debug only, for A/B tests of the AMD device loss: runs the light-quad skip of #9, which is NVIDIA-only by default (`1`), on every GPU. `0` turns it off. |
| `DKB_SKIP_VS=<origin>[,<origin>...]`, `DKB_SKIP_FS=<origin>[,<origin>...]` | Debug only, for diagnosing #75: skips every draw whose NVN 2 vertex (`DKB_SKIP_VS`) or fragment (`DKB_SKIP_FS`) stage origin (the 16-digit hex name logged by `DKB_GPU_CHECKPOINTS=1`) is listed, up to 16. Off by default. |
| `DKB_DUMP_SPV=1` | Debug only, for diagnosing #75: writes the exact SPIR-V passed to `vkCreateShaderModule` for each graphics and compute shader (after every rewrite) to `spv/<vs\|fs\|gs\|cs>-<NVN 2 stage origin>-<FNV-1a 64>.spv` next to `dkb.log`, so a SPIR-V hash from an Nsight Aftermath dump maps to its NVN 2 stage; up to 1 GiB, one `vk: spv dump` log line per file. Off by default. The files can hold code derived from the game: never attach them. |
| `DKB_SPV_OVERRIDE=<hash>:<path.spv>[,<hash>:<path.spv>...]` | Debug only, for #75: a shader module whose final SPIR-V has this FNV-1a 64 hash (16 hex digits, the last part of a `DKB_DUMP_SPV` file name) is created from the given file instead (multiple of 4 bytes, up to 16 MiB; max 8 entries). One `vk: SPIR-V override` log line per replaced module. Off by default. |
| `DKB_NAN_GUARD=zero` (default), `mark` or `0` | Fragment stages store 0 in place of NaN in their float colour outputs (`spirv_nan_guard.c`). A NaN texel in the HDR scene colour is spread by the bloom into a glowing band crossed by black lines; two frame dumps from a challenge room on an RX 9070 XT had NaN right after the lit opaque pass, which the console does not show. `mark` writes a colour named by the stage instead: r, g and b are each 0, 1, 4 or 16 from bits of the stage's SPIR-V hash (the `fs` hash in a dump's `passes.txt`), and the log lists `vk: fs <hash>: NaN marked as <r> <g> <b>` for each stage, so a dump's `_tm` image shows which stage writes the NaN. `0` stores what the shader computed. The guard tests the float's bits (`(bits & 0x7fffffff) > 0x7f800000`), not `OpIsNan`, which a driver may fold to false because the translator never enables `SignedZeroInfNanPreserve`. It changes the final SPIR-V hash of every fragment stage, so `DKB_SPV_OVERRIDE` hashes and `DKB_DUMP_SPV` file names taken under another `DKB_NAN_GUARD` value do not match. Tested by `tests/nan_guard_spirv_test.py --cc clang --spirv-val spirv-val --out <dir>`. |
| `DKB_HDR_NATIVE=1` (`hdr_native = 1`, the menu's "Game's own HDR" under HDR), with `hdr = 1` | Game 3.2.0 only, untested on 1.0.0, where it stays off; checked by eye on one HDR TV so far, so reports from other displays are welcome. Tells the game the display is HDR while the swapchain is HDR10 (`ov_gfx.c` `hdr_answer`), so it renders its own Rec. 2020 PQ picture into an A2B10G10R10 window texture that the present copies onto the HDR10 swapchain without FSR, SMAA or the colour adjustment. On a fallback to SDR the game is told so and goes back to its SDR picture. `spirv_hdr_shoulder.c` takes its composite's highlights up to the peak (`hdr_peak_nits`, at most the game's own 1400 nits), which the display is also told; the game's own curve stops at about 1.36 times `hdr_game_nits`. Existing `DKB_SPV_OVERRIDE` files keyed by hash stop matching the rewritten composite. The rewrite is tested by `tests/hdr_shoulder_spirv_test.py --cc clang --spirv-val spirv-val --out <dir>`. To check a picture without an HDR display, dump a bright frame (`DKB_DUMP=key`, F9) and run `tools/hdr_frame_stats.py <dump folder>`: peak, paper white share, clipping and gamut in nits. See `docs/hdr-native-plan.md`. |
| `DKB_HDR_TRACE=1` (`hdr_trace = 1`) | Logs the game's HDR and colour-space calls (`hdr-trace:` lines: the first 20 of each, then every 600th, with the caller) and the window texture's format when it changes. |
| `DKB_HDR_SHOULDER=0` | With `DKB_HDR_NATIVE=1`: leaves the game's HDR composite as the game made it, so its highlights stop at about 1.36 times `hdr_game_nits` (`spirv_hdr_shoulder.c` off). For a comparison, or to rule out the rewrite when a scene looks wrong; default 1. |
| `DKB_FMZ=0`, `vertex` (default) or `all` | Workaround for #75, on for vertex stages by default (`0` turns it off): rewrites 32-bit float `OpFMul` and GLSL.std.450 `Fma` in the final SPIR-V so `0 * x = 0` even for Inf and NaN (NVIDIA's fused-multiply-zero rule). `vertex` touches vertex-stage modules only, `all` every stage. One `vk: FMZ rewrite <hash> mul=<n> fma=<m>` log line per rewritten module; `DKB_DUMP_SPV` shows the rewritten bytes. `DKB_FMZ_SEL=<mask>:<value>` (hex) limits it to the modules whose FNV-1a 64 hash before the rewrite satisfies `(hash & mask) == value`, so `1:0` and `1:1` split the set in halves for a bisection. |
| `DKB_FIX_POOL_GUARD=0` | Turns off the workaround for #126 (on by default, 1.0.0 only): five guards in the layer-change loop `main+0x410680` keep list entries and cells that hold reused memory out of its work (`recomp/patches-dkb100.txt`, set `poolguard`). Failed checks are counted and logged as `patches: guard ... failed` lines (the first 32, then at each power of two). |
| `TOTK_TRACE=codec` | Log each call of the video decoder that moves a buffer or a picture. For guide movies that stop or show a flat colour. |
| `TOTK_FRAME_GEN=2x` (`frame_gen = 2x`, default `off`) | Experimental, Bananza only (TotK keeps its back-edge pacing and gets no made frames). The present thread (`vk_backend.c` `present_main`) calls `vk_framegen.c`: each game frame is copied from the window texture's crop, reduced to luma, matched against the frame before it (block matching, 3x3 median) and a frame at t = 0.5 is made; it is presented first and the game's frame half the game's frame interval later, both through the normal present chain (FSR, SMAA, HDR, overlay). No made frame after a gap of over 100 ms, a change of the crop size with the game's own HDR picture, with a window texture that is not 8-bit UNORM, when the display's refresh is below the doubled rate (`over refresh` in the log), or unless `vk_present` is `mailbox` or `immediate` (FIFO would block the present thread). The log has `fg:` lines, one every 5 s. Shaders and algorithm from Hookshot (see `THIRD_PARTY_NOTICES.md`). |

The settings keep TotK Recomp's `TOTK_` names, so its fixes apply without a rename. Each one can also be written with
`DKB_` in its place (`DKB_GPU=null`); when both are set, the `DKB_` one wins.

Command-line options: `--seconds S` stops the run after S seconds of game time, and `--shot frame.png` with it saves
that frame. A shot is for your own debugging; do not post it if it shows game content.

## Add-on content (untested in game)

Nothing in the game has been run with add-on content (DLC) yet: the reader (`runtime/aoc_romfs.c`) is checked on a
synthetic pack (`tests/aoc_romfs_test.py`) and by `--probe-aoc` on a real one, and that is all. Update 3.2.0 is the
same: the installer builds it and it boots, and nobody has played it. If you try either, say so in the report.

The reader looks for `aoc-<index>/` folders next to `dkb.exe` (`<index>` is 0 to 31), each holding `romfs.bin` and
`aoc-index.bin`. The installer does not write them. By hand, from the add-on content's extracted RomFS folder, with
`<dlc>` its root and `<n>` its index:

```
.venv\Scripts\python tools\dkb_pack_romfs.py --dir <dlc> --out build/aoc-pack
.venv\Scripts\python tools\romfs_index.py --romfs build/aoc-pack --out build/aoc-pack/index
mkdir "<exe folder>\aoc-<n>"
copy build\aoc-pack\romfs.bin "<exe folder>\aoc-<n>\romfs.bin"
copy build\aoc-pack\index\romfs-index.bin "<exe folder>\aoc-<n>\aoc-index.bin"
```

`dkb_pack_romfs.py` writes `romfs.bin` and `index.json`; `romfs_index.py` turns those into the TOTKRFS1 file
`romfs-index.bin`, which the reader wants under the name `aoc-index.bin`. Then `dkb.exe --probe-aoc` mounts the first
pack and reads up to 64 files back whole; with `DKB_AOC_SOURCE=<dlc>` it also compares them with the folder.

## Experimental ultrawide mode

Bananza 1.0.0 supports an optional wider horizontal camera view with proportional UI.
The title/main-menu background fills the wider viewport: its wall and atmosphere are
enlarged uniformly and cropped vertically, while the logo and text keep their proportions.
Corner HUD elements, including saving/loading indicators, can move to the screen edges
or remain in a centered 16:9 area. Menu controls and world markers remain centered.

Open F1 → Display → Resolution to set **Render resolution**. The **Ultrawide** section on the Display tab contains the wide view toggle and **HUD position**.
The **Graphics** tab contains **Render Options**, **Color**, and **HDR**.
Ultrawide is off by default; HUD placement defaults to **Screen edges** when enabled.
All three settings currently require a restart. Available resolution presets are:

| Ultrawide | Aspect-ratio group | Presets |
| --- | --- | --- |
| Off | 16:9 | 1280x720, 1600x900, 1920x1080, 2560x1440, 2880x1620, 3840x2160 |
| On | 21:9 class | 2560x1080, 3440x1440, 3840x1600, 5120x2160 |
| On | 32:9 | 3840x1080, 5120x1440 |

The original virtual-display functions return 2880x1620 (16:9); this remains the
default when no resolution is selected. These dimensions do not establish the native
3D rendering resolution or the Switch 2 screen's resolution. Window size controls the
host window separately. Use **Black bars** (`DKB_ASPECT=fit`) to preserve proportions
when the window and game have different aspect ratios.

Build with `recomp/patches-dkb100.txt` and generate matching overlays from your own packed
RomFS (requires Python `zstandard`). For a 3440x1440 build, generate both HUD variants:

```powershell
python tools/dkb_wide_layout.py --romfs build/extracted/dkb100-packed/romfs --out build/ultrawide/romfs-patches-3440x1440 --aspect 3440x1440
python tools/dkb_wide_layout.py --romfs build/extracted/dkb100-packed/romfs --out build/ultrawide/romfs-patches-3440x1440-hud-edges --aspect 3440x1440 --hud edges
build/ultrawide/dkb.exe
```

Repeat for each installed ultrawide resolution, placing the folders beside `romfs-index.bin`.
The overlays contain game data and must stay private under `build/`. Regenerate older
overlays to include the title-background fix; updating the executable alone is insufficient.
Missing edge-HUD overlays fall back to the centered variant with a log message.

Environment equivalents are `DKB_ULTRAWIDE=0|1`, `DKB_OUTPUT_RES=<width>x<height>`,
and `DKB_HUD_POSITION=edges|centered`. Overrides lock the corresponding F1 controls;
otherwise choices are saved in `dkb.ini` (an existing `totk.ini` is copied to it on the first run; `TOTK_*` environment variables still work, and `DKB_*` ones win over them). Turning Ultrawide off allows the 16:9 presets
and skips the wide overlay. Invalid sizes fall back to the original 2880x1620 dimensions.

5120x1440 (32:9) and 3840x2160 used to fail during render-resource creation because the game's
render-target pool is a fixed 200 MiB; a patch grows it with the output size. Both now run on 1.0.0
on an RTX 5060 Ti (short runs); 3840x2160 also runs on update 3.2.0 (pools of 356 MiB, 90 s at 60 fps).
5120x1440 on 3.2.0 and other GPUs are untested.

Run the layout, projection/register-preservation, and configuration checks with:

```powershell
python tests/dkb_wide_layout_test.py
python tests/dkb_wide_projection_test.py
python tests/dkb_wide_projection_test.py --table recomp/patches-dkb458752.txt
clang -O2 -march=x86-64-v3 -I runtime tests/dkb_wide_test.c -o build/dkb-wide-test.exe
build/dkb-wide-test.exe
clang -O2 -I runtime tests/res_upscaler_test.c -o build/res-upscaler-test.exe
build/res-upscaler-test.exe
build/ultrawide/dkb.exe --probe-cfg
```

The 3.2.0 patch table is generated from the 1.0.0 one. After touching `recomp/patches-dkb100.txt` or the tool, run
`python tools/port_patches.py --from dkb100 --to dkb458752 --sibling 5ad9f0=5ada00` again and diff the result against
the committed `recomp/patches-dkb458752.txt` (it prints every loose, sibling and re-targeted site to stderr for
review), and run `python tools/port_patches.py --self-check` (the 1.0.0 table mapped onto itself must have no
deltas).

## What to report

Open an issue with the issue form, or comment on [#3](https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/3)
(GPU testing) or [#7](https://github.com/Lucas-blip-png/dkb-pc-recomp/issues/7) (playing further):

- the commit you built (`git rev-parse --short HEAD`);
- GPU, driver version and Windows version;
- how far the game got and what you did just before it stopped;
- the log, `logs\dkb.log` next to `dkb.exe` (`logs\totk.log` in builds before the rename), as an attachment. After a crash, copy it before you start the game
  again: the next run renames it to `dkb.1.log`.

Never attach game files, keys, builds or the shader cache, and never paste key values into a log or an issue.
