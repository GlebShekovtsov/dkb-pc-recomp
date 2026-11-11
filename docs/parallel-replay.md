# Recording the replay on several threads

Status: plan, stage 1 done (measurements and descriptor pools). Branch `claude/parallel-replay`.

## Why

In busy areas the frame rate is set by the replay, not by the GPU. A `TOTK_TRACE=perf` run on an RX 9070 XT
(3.2.0, the hills region, many objects in view) logged frames of 33-39 ms (25-30 fps) with `replay avg` 25-31 ms
and `gpu avg` 11-13 ms; turning the camera to an empty part of the same area gave 60 fps with a replay of 5-12 ms.
`TOTK_REPLAY_ASYNC=1` was slower (the replay costs the same, and the game then waits for it on other threads).

A `TOTK_PROFILE` of the render thread over that run (`tools/prof_report.py`):

| Function (inclusive) | Share of samples |
|---|---|
| `nvnQueueSubmitCommands` > `vk_exec_record` (the replay) | 60% |
| `vk_exec_op_draw` > `prepare_draw` | 47% |
| `bind_resources` (10% in itself, then the texture lock, `vk_usnap_find`, `alloc_set`, the driver) | 27.5% |
| `bind_vertex_buffers` (8.4% in itself) | 10.6% |
| `block` (guest waits) and `nvnWindowAcquireTexture` (pacing), over the whole run with menus and loading | 38% |

The driver (`amdvlk64.dll`) takes under 4%. No single function dominates: each draw costs a fraction of a
microsecond, and there are many draws. Shaving the per-draw path can win a few percent; reaching 60 fps there needs
the replay to take about half the time, which means recording on more than one thread.

### Stage 1 measurements

The same spot with the stage 1 build (0901269, `TOTK_TRACE=perf`, 60-frame windows over about four minutes):

- 29,350-29,400 draws and dispatches per frame, steady; 1.00-1.10 us of replay per draw; replay 29-32 ms, 24-27 fps.
  In the empty view before it: 60-90 draws per frame.
- 17 descriptor pools taken per frame, none created: about 17,000 allocated descriptor sets per frame, written with
  update templates. With the old 256-set pools that is about 68 pools a frame, more than the 64 spares, so pools were
  created inside the frame before; the frame rate did not change measurably.
- 2.3 command buffer parts per submission (about 3 submissions per frame): parts are only cut between render passes,
  so a frame of 29,000 draws has around 7 parts, and a single pass can hold thousands of draws.

The last point decides the design: parallel recording by parts alone cannot spread a large pass. Its draws have to be
recorded into secondary command buffers (`VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT` with dynamic rendering)
by several workers, and executed in order by the primary (`vkCmdExecuteCommands`). A frame dump of the spot
(`passes.txt`, 103 render passes and 27 dispatches, 28,428 draws) confirms it:

| Pass | Targets | Draws | Share |
|---|---|---|---|
| 54 | r11g11b10f, rgba8, r16f, d32f 1397x786 (lit opaque) | 11,984 | 42% |
| 36 | r11g11b10f, rgba8, d32f 1397x786 (opaque before lighting) | 11,780 | 41% |
| 27, 26 | 1496x1496 (shadow map) | 3,932 | 14% |
| the other 99 passes | | 732 | 3% |

Two passes hold 84% of the draws. In pass 36 one pipeline draws 5,973 times (a mesh of about a hundred vertices, its
own transform in uniform block c4 each time, no instancing) and another 3,236 times; pass 54 repeats the pattern
(4,119 + 1,854 and 3,236). Splitting between passes cannot help here; slices of these passes recorded in parallel into
secondary command buffers can.

The repeated pipelines also suggest a cheaper per-draw path for runs of draws that change only uniform block
addresses and vertex buffers: dynamic uniform buffer offsets instead of a new push of the whole set.

### Per-draw breakdown

`DKB_DRAW_PROFILE=1` in the same spot (727d63e, 1642 windows of 60000 draws, median ns per draw; the profile itself
adds about 4 ms of replay a frame, and `TOTK_TRACE=perf` was on, which adds `note_work` to other-ops):

| Section | ns | Section | ns |
|---|---|---|---|
| other ops (about 3.2 bind ops and the draw command per draw) | 234 | ubo | 81 |
| bind-sig (signature and the binding loop itself) | 144 | textures | 78 |
| sets (push or allocate) | 103 | pipeline | 48 |
| vertex buffers | 99 | ssbo | 42 |
| tail | 28 | shaders | 40 |
| state | 23 | pass, feedback | 17, 14 |
| **total** | **964** | | |

No section dominates. Shortening the larger ones (other ops, the signature and loop, sets, vertex buffers) can take a
share of the per-draw time, not halve it: the frame stays above 16.7 ms on one thread, so recording in parallel is still
the way to 60 fps in this spot.

## What already exists

- The replay splits a submission into parts between passes (`vk_exec.c` `maybe_split`, `TOTK_SPLIT_DRAWS`,
  `TOTK_SPLIT_US`): each part is its own primary command buffer, submitted in order (`vk_sync.c` `split_impl`). Only
  command-buffer state is lost at a split (`vk_exec_draw_new_cb`, `ex->dirty`).
- The Vulkan calls themselves are made by a recorder thread (`vk_rec.c`); the replay thread does the NVN-to-Vulkan
  translation (`prepare_draw`), which is what the profile shows.

So the natural unit of parallel work is a part: its command buffer is independent, and the queue order of parts is
already handled.

## What stands in the way

The replay keeps its state in file-level variables, written for one thread:

- `vk_exec.c`: the replay state `X` (`VkExec`: bound NVN state, render targets, dirty bits, the current command
  buffer), `split_draws`, `part_start`, `part_t0`, the views made during the submission.
- `vk_exec_draw.c`: what the Vulkan command buffer holds (`B`), the descriptor signatures (`DS`), the bind-skip
  record (`RS`), the bind statistics (`BS`), the shader and pipeline memos (`SM`, `PM`, `stagem`, `slotm`,
  `shader_tab`, `AC`), the per-command-buffer resources (`cur`: descriptor pools, scratch chunks, written ranges),
  the support buffer, the twin buffers, `UPD`, `FM`.
- Shared with other threads behind locks: memory resolution and mirrors (`vk_memory.c`), the texture registry
  (`vk_texture_lock`), the pipeline cache and its background builds, the uniform snapshot (`vk_usnap.c`, filled at
  the start of a submission and read-only during the walk).
- Order-dependent side effects: texture uploads queued by `tex_info` (`vk_upload_flush` must reach the queue before
  the part that samples them), written ranges and feedback tracking (`note_written`, `feedback_*`), GPU-written
  notes, the dump. A part recorded early must not publish these before the parts ahead of it.

## Stage A: slices on one thread (DKB_REPLAY_THREADS=1)

A pass whose targets drew at least 256 draws the last time (`par_hist`, keyed by the `OP_SET_RT` payload) opens with
`VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT`. From its first draw the walk collects ops (`par_take`): state
and bind ops are applied to the replay state as usual, draws are only kept, and every 1024 draws a copy of the replay
state starts a new slice. The first op of another kind (other targets, a clear, a copy, a barrier, an indirect draw, an
uniform update, the end of the submission) flushes (`par_flush`): each slice is replayed from its copy into its own
secondary command buffer, inheriting the pass's formats, and the primary executes them in order and ends the pass.

Kept out: passes without attachments, passes with feedback-layout targets, passes while occlusion queries count, and
frames being dumped. Inside a slice, anything that would end, restart or begin a pass, record a full barrier or an
inline texture command is dropped and marks a hazard; the pass's targets are then not sliced again.

This stage records the slices on the same thread, so it only checks that slicing keeps the frame unchanged (and what it
costs). The next stage moves `prepare_draw`'s state to the recording thread and hands slices to workers.

## Stage B: workers (DKB_REPLAY_THREADS=N)

Stage A has not run in game yet: the test runs so far had `TOTK_DUMP=key` set, which keeps every pass unsliced (the
dump follows draws one by one), so their unchanged frames and timings say nothing about slicing. The log now says so,
and `TOTK_TRACE=perf` reports the passes, slices and draws sliced.

With `N` > 1, `par_flush` hands the slices to `N-1` workers and the submitting thread; each takes the next slice from a
counter tagged with the flush's generation (a late worker finds another generation and takes nothing), records it into a
secondary command buffer of its own pool, and the submitting thread executes them in order once all are recorded.

- The draw path's state (`vk_exec_draw.c`: `B`, `DS`, `RS`, `BS`, `SM`, `PM`, `DP`, the shader, stage, slot,
  descriptor and sub-range caches, the per-submission resources `cur`, the descriptor-build scratch) is reached through
  thread-local pointers to the file's objects; `vk_exec_draw_ctx_new` gives a worker its own copies.
- Shared state a binding may change is locked: the VRAM mirror's creation, first sync and written ranges
  (`vk_memory.c` `use_lock`), the uniform snapshot drops (`vk_usnap.c` `drop_lock`), the storage-image dummies.
  Textures, pipelines, shaders, descriptor pools and the completion queue were already locked.
- The slow-part notes of `TOTK_TRACE=perf` are skipped in slices; frame counters may lose counts.
- `vk_exec_recording_fence` answers in slices too, so storage images bound there get the submission's write fence.

### Occlusion queries

The first run that really sliced (3bd1079, `DKB_REPLAY_THREADS=4`) started its workers and sliced passes in the menus,
but in the busy area almost none: the game counts samples there, and passes were left alone while occlusion queries
counted (60 to 70 passes per 240 submissions against 6 to 8 sliced). A pass of secondary command buffers takes no query
command, so a sliced pass's query now begins before `vkCmdBeginRendering` and ends after `vkCmdEndRendering`
(`vk_exec_xfer.c` `vk_exec_oq_begin_outside`, `vk_exec_oq_pass_ended`), and the slices inherit it
(`occlusionQueryEnable`, device feature `inheritedQueries`; without it the pass is not sliced).

### Passes a clear opens

With queries handled (31b640e), the busy area still sliced one pass of about 350 draws per four frames. The perf line
showed why: counter ops ended ten collections per 240 submissions, and the two big passes never reached `par_start`,
which only looked at draws outside a pass. Those passes begin with a clear, and the clear opens the pass. `par_start`
now also takes the first draw of an open pass: the pass is ended and opened again with secondary contents (one more
`vkCmdBeginRendering` that loads the attachments). The draw history of a pass's targets also falls by an eighth instead
of following a smaller pass at once, so the piece after a counter op is sliced too.

### Feedback targets

The next run (261dff0) reopened no pass: three passes a frame, the big ones, were refused before that because a target
of theirs is attached in the feedback layout (a draw of the pass samples an attachment). Such passes are sliced now. A
slice does not know what earlier slices wrote, so it starts with every attachment marked written, and its first draw
that samples one records the by-region barrier in the secondary command buffer. A target learned inside a slice is a
hazard (its pass restart is dropped); a target set is left alone after eight hazards instead of one.

### The key of the draw history

With all of that, the big passes were still unsliced, and the log line for them (6838ba9) gave the reason: "targets not
seen before", with another signature every frame for what is the same pass. The history was keyed by a hash of the
`OP_SET_RT` payload, whose bytes are not the same from frame to frame for the same targets, so a big pass never found
the count its targets kept. The key is now the resolved images with their level and layers.

### First run with the big passes sliced: slower, on the texture lock

With the history keyed by images (164615a) about 15,000 draws a frame went through the slices on four threads, without
hazards, and the replay took 35 ms against 28 (20 fps against 27). A `TOTK_PROFILE_CHAINS` profile of 15 s there:

- the three workers waited for work 61% of the time (the walk collects a pass on one thread before any slice starts);
- of the rest, 56% was `plat_rmutex_lock` under `bind_resources`: the texture lock, taken for every texture binding of
  every draw. The submitting thread spent 47% of its slice time there too. Four threads handing one lock to each other
  cost more than one thread taking it alone.

A slice now takes a texture descriptor from its thread's memo without the lock when the memo hits and the slot was
already bound in this submission (`tex_info_hit`), and takes the lock only for the rest. The census counters, the
snapshot hit counter and the frame's draw counter are not written from slices, so their cache lines stay with one thread.

### First gain, and slices handed out while the pass is collected

With the texture memo (8d39619) the same spot ran at 30 fps where it had dropped to 25-27: replay 19-20 ms for 28,000
draws (0.70 us a draw against 1.00 on one thread), 15.5 ms for 25,000, no hazard, same picture. The frame is then paced
to 30 fps, since it still misses 16.7 ms.

The workers still waited while the walk collected a whole pass. A slice is now published as soon as the next one starts
(`par_publish`), so the workers record the first slices while the walk collects the later ones; the lists of ops and
copies have a fixed place for that (`PAR_MAX_OPS`, `PAR_MAX_SLICES`). At the end the submitting thread records what is
left and waits for the rest.

### Sixteen threads, and half the draws still unsliced

With slices handed out during collection (689c575) and 16 threads: replay 17.6 ms for 28,600 draws, 34-35 fps, no
hazard, same picture. Only 14,400 of the 28,600 draws a frame went through slices. The unsliced big passes logged
"targets not seen before" for two signatures that differ in their high bits only and fell on one slot of the
direct-mapped history, evicting each other every frame. The history has 256 entries now, a mixed index and four
neighbours per signature.

The render thread is busy 27-28 ms a frame there, 10 ms of it outside the replay (the game building its commands), so
the replay alone cannot bring the frame under 16.7 ms on the submitting thread; `TOTK_REPLAY_ASYNC=1` with the sliced
replay is the next thing to measure.

### Where it stands (278aab1, 16 threads)

Every big pass is sliced after its first frame (about 17,500 of 29,000 draws a frame go through slices, the rest are
small passes), with no hazard and the same picture. The replay stays at 17.8 ms (0.61 us a draw), 31-35 fps against
25-27 on one thread, and 16 threads do little better than 4: what is left is the serial part (collecting the ops and
applying binds on the submitting thread, the small passes, the copies of the replay state) and the wait for the slices.

`TOTK_REPLAY_ASYNC=1` with slices was slower again (29 fps): the render thread is busy 28 ms a frame even without the
replay on it, and the game's other threads wait 18 ms a frame on fences. The frame there is about the replay (17 ms)
followed by the GPU's 11-14 ms, which begins only when the big passes are submitted, so 60 fps in this spot needs both
shorter and overlapped, not the replay alone.

### Slices sized by the pass

With the timing line (81c8015, 6 threads, per frame): about 22,000 of 29,000 draws in slices, 13 sliced passes of 1,600
draws on average in 33 slices. In those passes the submitting thread spent 0.9 ms collecting, 4.8 ms recording its own
slices and 4.9 ms waiting for the workers: with 1024 draws a slice, a pass gave each thread little or nothing to take.
Unsliced passes kept only a few hundred draws a frame. The slice size is now a third of the pass's draw count per thread,
between 128 and 1024 draws.

### Workers asleep, and pools for every thread

The smaller slices (ea688fe) did not change the replay (18.5-19 ms): the submitting thread recorded 7.8 ms of slices a
frame itself. A `TOTK_PROFILE_CHAINS` profile on six threads showed why: the five workers were asleep on their condition
variable 83% of the time. A pass is collected in about 0.1 ms, less than it takes the OS to wake a thread, so the
submitting thread took most slices before the workers ran. Of the workers' busy time, 17% went to kernel calls under
`vkAllocateDescriptorSets`: each recording thread takes descriptor pools of its own per submission, and with 16 spares
they were created in the frame.

The workers are now woken when a collection starts and spin while it lasts (`pw_active`), and the spare pools grow by 8
per recording thread, up to 64. On the submitting thread, the uniform snapshot and the mirror bookkeeping of each
submission (`vk_usnap_begin`, `copy_ranges`, `plat_mem_written`) take about 3.5 ms a frame, which slicing does not
touch.

### Spinning workers: 33-38 fps

With the workers spinning (7273a3f, six threads) the hills spot's replay fell to 16-17 ms for 28,000-29,000 draws, the
frame rate to 33-38 fps (from 30-31), and the area's opening view to 46-47 fps with an 11.5-12 ms replay. The
submitting thread still records 4.7 ms of slices a frame and waits about 0.5 ms. Under Bananza's original game timing a
frame rate between 30 and 60 slows the game down (it steps by the present interval it chose), which the tester noticed;
`DKB_REALTIME=1` makes game time follow the clock.

### Per-draw cost by thread, and a pass end that did not wait: reverted

The draw profile per thread (6 threads, `DKB_DRAW_PROFILE=1`, the hills) put a draw of a slice at 1.5-1.9 us against
1.3 us for the submitting thread's draws outside slices; every section was slower, the descriptor sets (280 against 130
ns), the textures (210 against 85) and the bind signature (185 against 100) most. The submitting thread recorded 19% of
the slices itself (5.3 ms a frame) and waited 2.1 ms a frame for the workers at the 13 collection ends of a frame.

358db4d let the walk go on after a sliced pass and had the recorder thread wait for the slices before its
`vkCmdExecuteCommands`. It saved about 0.7 ms a frame (replay 16.2-16.7 ms), because most collections end at an op
that had to wait for the slices anyway, and two runs in the hills hung after a few minutes: the submitting thread may
wait for the recorder (a drain, a full ring, a split) while it holds a lock a worker needs, and the recorder waits for
that worker. It was reverted. A recorder that waits for other threads needs the submitting thread to hold no lock they
take whenever it waits for the recorder.

### Slices that shrink towards the end of a pass: 35-36 fps (fd9782e)

Six threads, the hills, about 19,000 frames without a hang: 35.8-36.3 fps (27.6-27.9 ms a frame) where 830e509 gave
32-34, replay 15.1-15.3 ms for 26,600 draws, waiting for the workers 1.1 ms a frame (1.4 before). The submitting
thread still records about 3,100 sliced draws a frame (20 short slices), because by the end of a pass the workers are
busy too: 1.3 million sliced draws a minute at 1.55-1.65 us a draw make about 34 ms of recording a frame for six
threads. The cost per draw in a slice is now the limit, against about 1 us on the single-thread path, where the
recorder thread makes the driver calls; a slice makes its own.

### Where eight threads wait (profile at fd9782e)

`DKB_REPLAY_THREADS=8`, `TOTK_PROFILE_CHAINS=1`, the hills. Of the samples under `par_record` (all threads), 14% wait
for the texture lock (half in `vk_texture_slot_token` from `feedback_scan` and `shader_for`, half in the texture loop of
`bind_resources`), 6% for `use_lock` in `vk_mem_gpu_wrote`, 3% for the global lock of `nvn2_origin` (the draw asks for
every stage's origin), 3% for `pool_lock` in `resolve_locked` (mostly from `bind_vertex_buffers`), 2% for `drop_lock`
in `vk_usnap_drop`; 5.5% is the driver's push descriptors. The next commit keeps `nvn2_origin` answers per thread, the
slot tokens of a slice per thread while the slot and memory generations stay, and skips a write-back range a thread
already added in the submission.

### Asynchronous replay, and the submitting thread's own costs (5eae626)

Eight threads with the replay on its own thread (`DKB_REPLAY_ASYNC=1`): 31.5 fps, four threads: 30 fps, against 36-37
without it. The replay itself took 12.5 ms, but the game's other threads waited 21 ms a frame on fences (10.5 ms
without): the game waits for the GPU's results within the frame, so the replay stays on its critical path either way.

The profile of the same build (eight threads, synchronous) splits the submitting thread's replay: about 6 ms a frame
go to per-submission bookkeeping (`vk_usnap_begin` 2.7 ms, `vk_mem_mirror_end` 2 ms, `vk_exec_reset_indirect` 1 ms),
and most of the first two is the CRT's `qsort` over the write lists (`wr_prune`, `copy_ranges`). A second game thread
that submits a little work spends about 4 ms a frame waiting for `sub_lock` while the main submission replays.

### An inlined sort: replay 13 ms, frame unchanged (af96356)

Eight threads, the hills, standing still: replay 12.8-13.2 ms for 28,400 draws (0.46 us a draw against 0.57 before),
but the frame stayed at 28.0-28.5 ms (35-35.6 fps). The game's other threads waited 12.4 ms a frame on fences (10.5
before), the GPU took 13.4 ms, and a submission went to the queue in 1.62 parts: the game waits for the GPU's results
within the frame, and the GPU started on a submission only once most of its replay was done, so the frame was about the
replay plus the GPU work. Draws collected for slices did not count towards a command buffer split; from the next commit
they do, so a split follows a big sliced pass and the GPU starts on it while the rest is replayed.

### Splits after sliced passes: 43-47 fps (0b239bc)

Eight threads, the hills, standing still: 42.3-46.7 fps (21.4-23.7 ms a frame) against 35-35.6 at af96356, with the
same 28,400 draws a frame. A submission now goes to the queue in 2.2 parts (1.62 before), the game's other threads
wait 5.6 ms a frame on fences (12.4 before), the GPU takes 11.1 ms, and the replay 13.5-14.4 ms (the splits cost a
little). The frame is still bound by the CPU: the render thread is busy 21.4-23 ms.

### Shorter parts: 47-49 fps (0b239bc with `DKB_SPLIT_US`)

The same spot on eight threads: parts of at least 300 us gave 46.8-48.8 fps (20.5-21.4 ms a frame, 3.04 parts a
submission, the game's threads waiting 4.9 ms a frame on fences, GPU 10.5 ms); no minimum gave 43.7-46.9 fps with
3.09 parts. With more than one replay thread, `TOTK_SPLIT_US` now defaults to 300.

### The descriptor memos of #181 merged (Zorkats:descriptor-memos 26ad9b8)

Zorkats's branch keeps the texture descriptor memos across submissions (`TOTK_DESC_KEEP`), reuses written descriptor
sets within a submission (`DSC`), copies the compiler-constant banks once per submission (`C1`) and keeps each stage's
NVN 2 origin with the stage memo. Merged here with this branch's per-thread state: `DSC`, `C1`, `ssbo_serial` and the
stage memo are per recording thread (`DRAW_CTX_OBJS`), a slice still reads a texture memo hit without the texture
lock, and `tex_info_hit` uses the memo's new index.

### Profile at 1ae8384: the submitting thread's remaining costs

Eight threads, the hills. Of the submitting thread's replay: collecting a sliced pass (`par_take`) is about 15%, half of
it the copy of `VkExec` (25 KiB) at every slice start; `token_of` waited for the texture lock in 4% (the per-thread
token memo covered slices only), `vk_exec_reset_indirect` takes 5% with a read back of the pointer list
(`vk_exec_read_buffer` waits for the GPU on the prefix) and a `qsort`, and the command buffer splits 2.5%. A second
game thread waits for `sub_lock` about as long as the main submission replays. The next commit keeps the slot tokens
per thread on every thread and sorts the reset targets with the inlined sort.

## Plan

1. **Measure and remove the cheap waste** (done here).
   - `TOTK_TRACE=perf` now logs draws and dispatches per frame and the replay time per draw, and the descriptor pools
     taken and created per frame.
   - Descriptor pools hold 1024 sets instead of 256: in the profile `alloc_set` spent half of its 5% in kernel calls
     under `vkAllocateDescriptorSets`.
2. **Separate state from emission.** Split `vk_exec_op_draw` into the part that updates the NVN state in `VkExec`
   (binds, viewports, programs) and the part that records Vulkan commands. A pre-pass over a submission's ops then
   runs only the first, cheaply, and notes at each split point a copy of `VkExec`. Split points come from draw counts
   alone (the time criterion of `TOTK_SPLIT_US` cannot be known ahead).
3. **Per-recorder state.** Move `B`, `DS`, `RS`, `BS`, the memos and `cur` into a context struct that each recorder
   owns, so two parts can be recorded at once. Single-threaded behaviour stays the same; this stage is testable on
   its own (same images, same frame time).
4. **Ordered side effects.** Each part collects its uploads, written ranges and feedback notes in a list; the
   submitting thread applies them in part order when it submits the part.
5. **Workers.** N workers record chunks of draws from their `VkExec` copies: whole parts into their own primary
   command buffers, and slices of a large pass into secondary command buffers that the primary executes in order
   (without the recorder thread for those); the submitting thread submits the parts in order. Off by default (`TOTK_REPLAY_THREADS`), compared
   against the single thread with the dump (`TOTK_DUMP=key`) and `TOTK_TRACE=perf` in the hills.

Each stage is a commit that keeps the game working; stage 3 onward is checked in game on the RX 9070 XT before the
next starts.
