# Running on a weak PC

Measurements from `tools/lowend_bench.py`, which runs `dkb.exe` on the first N logical processors with the perf trace and
the draw profile on, and prints one table. This page is the baseline the low-end work is compared against.

For systems with 4 GB of VRAM and gaming laptops, see [4GB VRAM & Laptop Optimization Guide](4gb-vram-optimization.md).

## Baseline

Bananza 1.0.0, first level, the scripted input of about 2.4 minutes, 70 s per run, RTX 5060 Ti, 16 GB of RAM, before the
descriptor memos of pull request 181. This route is light: it does not show the cost of an open area (a 3070 Ti Laptop
reports 31 to 35 fps there with about 8,300 draws a frame), so treat it as a floor for the CPU limit, not a ceiling.

| Logical processors | Mean fps | Steady fps | Worst window (ms) | Replay avg (ms) | GPU avg (ms) | Commit max (MB) | Working set max (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 57.0 | 60.0 | 941 | 3.89 | 1.80 | 14,306 | 5,837 |
| 8 | 55.9 | 60.0 | 938 | 6.36 | 3.02 | 14,133 | 5,704 |
| 4 | 52.7 | 58.9 | 997 | 6.61 | 3.11 | 14,087 | 5,564 |
| 2 | 42.9 | 55.6 | 1,249 | 13.12 | 1.87 | 14,075 | 5,636 |

- Down to 4 logical processors the steady rate holds near 60 fps. At 2 it falls to about 56 fps and the replay thread
  takes 13 ms a frame.
- The worst window is about one second at every count. It falls in the first seconds, where shaders are translated.
- Process commit is 14 GB at every count, so memory, not the CPU, is what a 16 GB machine runs out of first.

## Where the commit goes

`TOTK_MEM_REPORT=N` logs the breakdown every N frames and at exit. Around frame 2400 of the first level, of about 14.4 GiB:

- guest heap and stack: 9.2 GiB, of which only about 4.2 GiB is resident in the working set;
- GPU device-local memory and mirrors: 2.6 to 5.1 GiB, growing during the run;
- the exe image: about 0.5 GiB.

`TOTK_APP_POOL` below 8 GiB makes the game crash at start (access violation reading 0xc8), so it is not a way to save memory.

## Thread knobs on 2 and 4 logical processors

Same route, 60 s runs, one knob changed from the defaults at a time. Repeated runs of the same setting differ by about
1 fps in the mean and 0.5 ms in the replay average.

| Logical processors | Setting | Mean fps | Steady fps | Replay avg (ms) |
| ---: | --- | ---: | ---: | ---: |
| 2 | defaults | 46.5 to 42.9 | 55.6 to 55.7 | 11.8 to 13.1 |
| 2 | `VK_RECORDER=0` | 48.2 to 48.4 | 58.9 to 60.0 | 5.9 to 6.6 |
| 2 | `REPLAY_ASYNC=1` | 45.6 | 56.1 | 12.4 |
| 2 | `SPLIT_DRAWS=0` | 46.4 | 57.4 | 9.2 |
| 2 | `SPLIT_DRAWS=256` | 46.2 | 57.5 | 12.4 |
| 2 | `FREE_CORES=1` | the run stopped at frame 1353 | | |
| 4 | defaults | 52.7 to 53.1 | 58.9 to 59.0 | 6.3 to 6.6 |
| 4 | `VK_RECORDER=0` | 50.3 | 59.0 | 6.3 |

- With 2 logical processors the recorder thread competes with the replay for a core. Turning it off halves the replay
  time and brings the steady rate to 60 fps. With 4 it changes nothing.
- `FREE_CORES=1` with 2 logical processors stopped producing frames after about 25 s. It is off by default and stays off.
- The worst window stays near 1 to 1.7 s whatever the setting: it is the shader translation at the start, not a thread
  setting.

This route is light, so these are not yet reasons to change a default. A low-end profile would turn the recorder off at
2 logical processors once a heavy scene confirms it.

## Heavy scene: layer -101

A player's recording in layer -101 on an RTX 5060 Ti at 1920x1080 with a 60 fps limit, measured on 13 November over
the first minute with a warm shader cache. Each setting ran once or twice; two values are the two runs.

0.3.0 build:

| Setting | fps | 1% low | Frames over 25 ms |
| --- | --- | --- | --- |
| defaults | 50.5 and 49.8 | 21.4 and 23.1 | 346 and 370 |
| `TOTK_REPLAY_PAR=1` | 51.3 | 23.4 | 314 |
| `TOTK_REPLAY_THREADS=4` | 54.3 | 25.6 | 146 |
| `TOTK_REPLAY_THREADS=8` | 53.1 | 23.1 | 198 |

Build 89645fc (warm cache):

| Setting | fps | Frames over 25 ms |
| --- | --- | --- |
| defaults | 50.3 and 51.6 | not recorded |
| `TOTK_REPLAY_THREADS=4` | 54.2 and 53.6 | 188 and 181 |

- A cold shader cache cost about 5 fps in the first run.
- A stall of about 0.9 s around frame 1775, almost all of it replay, shows in every run.
- Cause of that stall: the first frame of the new area imports guest memory pools on their first GPU use
  (`TOTK_POOL_LAZY`, `vk_memory.c` `lazy_import`). A sampling profile of the frame (`TOTK_PROFILE_SLOW`) put about 70%
  of the render thread in `import_pool` (`vkAllocateMemory` of host memory); 673 pools were imported between frames 600
  and 1800, about 960 ms in all, 1.4 ms each, most of them in that frame. The GPU uses only two pool kinds: of 3325
  imported pools at frame 3600, 430 of 1651 `CPU_UNCACHED|GPU_CACHED` and 638 of 1076 `CPU_CACHED|GPU_CACHED` were
  resolved, and almost none of the other kinds.
- Tried and dropped: a background thread that imported every lazy pool older than 2 s. The worst frame fell from
  about 750 ms to 350 ms, but the import time rose from 1.5 s to 5.7 s, held the pool lock the guest's threads wait on,
  and the run got worse overall: 47.1 and 49.1 fps against 52.7 and 53.6, a 1% low of 9.4 and 10.6 against 25.6 and
  27.8, and 29 to 32 frames over 100 ms against 4 to 5.
- Merging neighbouring pools into one import was ruled out by measurement (the census line "imports took", new
  defaults, 13 November). At frame 1800, 717 lazy imports took 431 ms in all, about 0.6 ms each: 301 ms in
  `vkAllocateMemory` and 129 ms committing and touching the pages. The cost follows the size, not the number of calls:
  a single 450 MiB pool took 114 ms of `vkAllocateMemory` in the 485 ms worst frame (1753). 541 of the 711 used pools
  start within 1 MiB of the previous one, but joining small pools would not shrink the large ones. What is left to try
  is importing only the part of a large pool the GPU uses.

Defaults chosen from these runs (13 November, build of bananza with the pre-import dropped, warm cache, two runs each):

| Setting | fps | 1% low | Frames over 25 ms | Worst frame |
| --- | --- | --- | --- | --- |
| old defaults | 52.7 and 53.6 | 25.6 and 27.0 | 177 and 226 | about 760 ms |
| `TOTK_REPLAY_THREADS=4` | 57.0 and 57.2 | 28.8 and 29.0 | 63 and 77 | about 775 ms |
| `TOTK_HEAP_LAZY=1` | 54.3 and 54.3 | 28.2 and 28.4 | 168 and 193 | about 473 ms |
| new defaults (both) | 57.9 and 57.6 | 29.6 and 29.6 | 62 and 65 | about 456 ms |
| 8 logical processors, old defaults | 49.8 and 50.5 | 23.7 and 26.3 | 371 and 389 | |
| 8 logical processors, `TOTK_REPLAY_THREADS=4` | 54.1 and 54.3 | 26.1 and 23.5 | 165 and 185 | |

So `TOTK_REPLAY_THREADS` defaults to 4 from 8 logical processors (6 from 12, 3 from 4 since 18 November; on AMD only the RX 9070 and 9070 XT on the proprietary driver, the setup measured) and `TOTK_HEAP_LAZY`
to 1 on Windows.
- Run-to-run noise is about 2 fps, and 3 in the 1% low.

## Linux: the write-back of GPU results (14 November)

Measured on an RX 7900 XTX with RADV (Mesa 26.2.7) and a 16-thread CPU, standing still in the cave after the tutorial
on 1.0.0 at 1920x1080: 5,230 draws a frame, the GPU busy about 5 ms a frame, and 46 to 47 fps. No thread was busy for a
whole frame (the busiest, the one that replays, 51 % of a core in a 20 s `perf` sample), so the frame was waiting, not
working: the game's threads waited 7.8 ms a frame on fences.

Those fences are held up by the shadow storage's write-back (`vk_memory.c`, "shadow storage"): the frame reports about
4,400 GPU writes into CPU-visible memory, a thousand 15,000-byte storage buffers among them, 15 MiB in all, and the
completion thread copied each back with its own search of the pending list, its own locks and a write-protection
fault per page, 6.7 ms of CPU a frame. `vk_gpu_written_many` does a submission's ranges in one pass.

| Build | fps | Frame | Fence wait of the game's threads |
| --- | --- | --- | --- |
| one call per range | 46.8 and 47.1 | 21.3 ms | 7.8 ms |
| one pass per submission | 56.9 and 57.1 | 17.6 ms | 4.5 ms |
| the 15,000-byte write-backs skipped (a test, not a fix: the game reads some results back) | 59.8 and 59.8 | 16.7 ms | 2.2 to 3.8 ms |

Ruled out in the same scene: `DKB_REPLAY_THREADS` 2, 4 and 8 (46 to 48 fps), the generated code at `-O2` instead of
`-O1` (46.7 and 47.4 against 46.8), `DKB_REPLAY_ASYNC=1` (32 fps: the game falls to two vsyncs a frame) and
`DKB_SHADOW=0` (2 fps on Linux, as the shadow storage's comment says). What is left on the replaying thread is the
write-watch query before each submit, about 2.5 ms a frame over 530 MiB. Copying a GPU write back only when the CPU
touches its pages would remove the rest of the write-back; the last table row is its upper bound here.

Again on 3.2.0 (16 November, `bananza` dba3a15 without and with this change, the same card and driver): standing still
in the first cave of Ingot Isle at 1920x1080, the game's own upscaler, frame generation off: 49.2 and 49.3 fps before,
59.6 and 59.1 after; the slowest frame 71.5 and 64.5 ms before, 25.7 and 26.3 ms after; the replaying thread busy
20.3 ms a frame before, 14.2 and 16.1 ms after.

`DKB_WB_LAZY=1` (`wb_lazy = 1`, Linux, off by default) does that (`vk_memory.c`, "lazy write-back"): a storage
buffer a shader wrote is not copied back at its fence; its pages are closed, and the first CPU access to one copies
what it is owed and opens it. At the start of Ingot Isle on 3.2.0 (17 November, `bananza` 9ed1784 with it, RX 7900 XTX,
Mesa 26.2.7, the default settings: the game's own upscaler from 1280x720 to 1920x1080, frame generation off, SDR),
four runs each way in turn: render busy 18.42, 17.79, 18.44 and 17.49 ms off, 14.52, 13.40, 12.23 and 13.60 ms on,
lower in every pair; 52.7, 55.1, 51.9 and 56.8 fps off, 55.7, 57.6, 59.1 and 56.7 on, higher in three pairs of four.
About 12,000 pages stay closed and 14 MiB a frame is left uncopied, with 4 to 8 pages touched every 60 frames. In the
crater scene the game already holds 60 fps, and it made no difference there. It stays off by default until more of
the game has been played with it.

## Layer -200 on 6 and 4 logical processors (18 November)

A Ryzen 7 5700X held to its first 6 or 4 logical processors (the process starts with that affinity, so the runtime
counts only those), RTX 5060 Ti, the layer -200 recording on a player's save (3.2.0), 1920x1080, 60 fps limit, warm
caches. The second minute and the end move with where the recording ends, so the first minute compares best.

| Logical processors, `TOTK_REPLAY_THREADS` | first minute fps | frames over 50 ms | second minute fps, 1% low |
| --- | --- | --- | --- |
| 16, default (6) | 57.4 | 9 | 60.0, 53.3 |
| 6, 0 (the old default) | 40.1 and 40.3 | 169 and 155 | 42.8 and 48.2, 19.4 and 23.0 |
| 6, 2 | 40.6 | 138 | 42.5, 20.0 |
| 6, 3 (the new default) | 43.7 and 43.2 | 15 and 17 | 49.4 and 50.0, 27.4 and 27.4 |
| 6, 4 | 43.7 and 43.6 | 20 and 25 | 53.8 and 48.3, 28.8 and 28.0 |
| 4, 0 (the old default) | 37.6 | 203 | 36.9, 18.3 |
| 4, 2 | 38.8 | 148 | 36.8, 18.1 |
| 4, 3 (the new default) | 41.0 | 47 | 41.6, 21.2 |

So from 4 logical processors the default is 3 threads. Below 60 fps the scene stays bound by the replay thread.
