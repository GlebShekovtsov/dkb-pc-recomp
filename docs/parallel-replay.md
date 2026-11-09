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
5. **Workers.** N workers record parts from their `VkExec` copies into their own command buffers (without the
   recorder thread); the submitting thread submits them in order. Off by default (`TOTK_REPLAY_THREADS`), compared
   against the single thread with the dump (`TOTK_DUMP=key`) and `TOTK_TRACE=perf` in the hills.

Each stage is a commit that keeps the game working; stage 3 onward is checked in game on the RX 9070 XT before the
next starts.
