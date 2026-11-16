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

So `TOTK_REPLAY_THREADS` defaults to 4 from 8 logical processors (6 from 12; not on RADV or AMD's RDNA 1/2 driver, not measured there) and `TOTK_HEAP_LAZY`
to 1 on Windows.
- Run-to-run noise is about 2 fps, and 3 in the 1% low.
