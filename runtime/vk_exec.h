// The op-stream replay of the Vulkan back end. vk_exec.c walks the command
// handles of a submission, keeps render targets, passes, clears and fixed-function state; draws, binds and dispatches
// (vk_exec_draw.c) and copies, counters, events and conditional rendering (vk_exec_xfer.c) plug in through the two
// hooks at the bottom.
// Runs on the submitting thread under vk_sync_submit's lock: no locking inside.
// The op loop follows MacNativeGames nvn_mtl.m (MIT, Copyright (c) 2026 jhaizhou) and the pass handling LAS Recomp
// (MIT, Copyright (c) 2026 Peiikai); vk_exec.c names the sources.
#pragma once
#include <stdint.h>

#include "gfx.h"
#include "gfx_c0.h"
#include "nvn_api.h"
#include "vk.h"

// Dynamic state the next draw must record (VkExec.dirty). vk_exec_record sets them all at the start of each command
// buffer; the state ops set theirs; vk_exec_draw.c clears what it recorded.
enum {
    VK_EXEC_DIRTY_VIEWPORT = 1 << 0,     // viewports or depth range
    VK_EXEC_DIRTY_SCISSOR = 1 << 1,
    VK_EXEC_DIRTY_DEPTH_BIAS = 1 << 2,   // polygon offset
    VK_EXEC_DIRTY_BLEND_CONST = 1 << 3,
    VK_EXEC_DIRTY_STENCIL = 1 << 4,      // compare and write masks, references
    VK_EXEC_DIRTY_ALL = (1 << 5) - 1,
};

// One render target of OP_SET_RT, resolved to its vk_texture.c image for the current command buffer.
typedef struct {
    const tex_t *tex;                    // payload copy (lives for the run); NULL: no target in this slot
    VkImage image;                       // VK_NULL_HANDLE: no target or not renderable (logged once)
    VkFormat format;                     // view format: the view's when it reinterprets a same-size format
    VkImageViewType view_type;           // 2D, or 2D_ARRAY for more than one layer
    VkImageAspectFlags aspect;
    uint32_t level, base_layer, layers;  // subresource of the view; of a 3D image: depth planes
    uint8_t is3d;                        // a 3D image (colour only), viewed through 2D views of its slices
    uint8_t blocks;                      // a BC image viewed as one texel per block (VkTexImage.block_views)
    uint32_t width, height, samples;     // extent of that level; with blocks, in blocks
    int32_t nvn_format;                  // NVNformat of the view (VkPipeState.color_fmt / depth_fmt)
    const VkTexImage *img;               // the image (fb_usage, fb_learned)
} VkExecTarget;

// The replay state. One instance lives for the run: NVN state is queue state and carries over from one submission to
// the next, as in MacNativeGames' and LAS Recomp's replays. The "per command buffer" fields are reset by vk_exec_record.
typedef struct VkExec {
    // ---- per command buffer
    VkCommandBuffer cb;                  // being recorded
    uint64_t fence;                      // timeline value the submission signals (vk_completion_add)
    uint64_t submitted_ticks;            // immutable front-end entry tick; valid even when 0, not GPU time
    int recorded;                        // Vulkan commands recorded so far; every handler adds the ones it records
    int in_pass;                         // inside vkCmdBeginRendering
    uint32_t dirty;                      // VK_EXEC_DIRTY_*

    // ---- render targets (OP_SET_RT) and the current pass
    p_set_rt rt;                         // last OP_SET_RT payload (a change ends the open pass); rt.n 0, no depth: none
    int rt_resolved;                   // color[] and depth hold this command buffer's images for rt
    VkExecTarget color[8], depth;        // resolved targets
    uint32_t pass_w, pass_h, pass_layers;            // render area and layer count of the open (or next) pass
    int pass_noatt;                                   // the pass has no attachment: its size follows the draws (vk_exec_noatt_dims)
    VkFormat pass_color[8], pass_depth, pass_stencil; // attachment formats of the pass (VkPipelineRenderingCreateInfo);
                                                      // VK_FORMAT_UNDEFINED: none
    uint32_t pass_ncolor, pass_samples;               // colorAttachmentCount and rasterization samples of the pass

    // ---- fixed-function state (vk_exec.c keeps it; vk_exec_draw.c records it before draws)
    float vp[16][6];                     // OP_VIEWPORTS: x, y, w, h, near, far
    int has_vp;                          // a viewport was set; else the render area
    p_scissor scissor;                   // OP_SCISSOR (scissor 0)
    int has_scissor;                     // a scissor was set; else unbounded
    float depth_range[2];                // OP_DEPTH_RANGE: overrides the viewports' near and far
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];   // OP_STENCIL per face (0 front, 1 back): compare mask, write mask, ref
    float point_size;                    // OP_BIND_STATE kind 6: NVN point size for the support buffer; 0 = never set = 1
    p_poly_offset poly_offset;           // OP_POLY_OFFSET: depth bias factor, units, clamp
    p_prim_restart prim_restart;         // OP_PRIM_RESTART: applied to indexed strip and fan draws (op_draw_idx)
    float blend_color[4];                // OP_BLEND_COLOR
    int64_t copy_stride;                 // OP_COPY_STRIDE: row stride of buffer <-> texture copies (vk_exec_xfer.c)
    dpool_ref tex_pool, samp_pool;       // OP_SET_TEX_POOL, OP_SET_SAMP_POOL, captured by value
    // Pipeline inputs. vk_exec.c writes depth_clamp (OP_DEPTH_CLAMP), raster_discard (OP_RASTER_DISCARD) and, at pass
    // begin, color_fmt, depth_fmt and samples; vk_exec_draw.c owns every other field (shaders, prim, y_flip, state
    // objects, vertex input).
    VkPipeState pipe;

    // ---- vk_exec_draw.c slots (draws, binds, dispatch): left empty by vk_exec.c
    const prog_t *prog;                  // OP_BIND_PROGRAM
    int32_t prog_stages;
    NVNbufferRange vbuf[16];             // OP_BIND_VBUF
    p_bind_buf ubo[6][18], ssbo[6][16];  // OP_BIND_UBO / OP_BIND_SSBO per stage and index
    uint64_t tex[6][128], samp[6][128], img[6][32];   // OP_BIND_TEX / OP_BIND_SAMP / OP_BIND_IMG handles per stage
    uint32_t push[2][NVN2_ROOT_TABLES][NVN2_ROOT_WORDS];   // graphics root 0 (bank 24), compute roots in packed c0
    uint64_t push_set[2][NVN2_ROOT_TABLES];                // words written; used by the graphics blit fallback

    // ---- vk_exec_xfer.c slots (render enable, conditional rendering): left empty by vk_exec.c
    int render_disabled;                 // OP_RENDER_ENABLE false or a failed OP_RENDER_COND: clears and draws skip
    p_render_cond render_cond;           // last OP_RENDER_COND
    int has_render_cond;

    // ---- internal to vk_exec.c
    void *views;                         // image views created for this command buffer (freed at its fence)
    int nviews, cap_views;
    int barrier_at;                      // recorded right after the last full barrier (-1: none yet)

    // Feedback loops (vk.h VkCtx.feedback, vk_exec_draw.c feedback_prepare). Learned targets are attached in
    // ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT; they stay in it across consecutive passes and go back to GENERAL at the next
    // full barrier that is not a pass begin.
    VkImage fbl_img[9];                  // images in the feedback layout now
    VkImageAspectFlags fbl_aspect[9];
    int nfbl;
    uint32_t fbl_mask;                   // attachments of the open pass in that layout: bit i colour i, bit 8 depth
    uint32_t fb_dirty;                   // attachments of the open pass written by a draw or clear since the pass began
                                         // or the last in-pass barrier
    const void *fb_fs;                   // feedback_scan's memo: fragment shader, texture bind serial, pass serial
    uint32_t fb_serial, fb_pass, fb_sampled;
    uint32_t tex_serial, pass_serial;    // bumped by every texture or image bind and pass begin
} VkExec;

typedef void (*VkExecWorkFn)(VkExec *ex, void *context);
// Synchronous CPU recording from the submit callback above: no nested submission/replay and no other queue owner.
int vk_exec_record_work(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks,
                        VkExecWorkFn work, void *context);
// Reset a validated selected pointer list in replay order. Returns 0 without record writes on invalid input.
int vk_exec_reset_indirect(VkExec *ex, uint64_t list_addr, uint32_t descriptor_bytes, int64_t first, int64_t count);

// Begins dynamic rendering with the current targets unless a pass is open (a full memory barrier precedes it). Call
// before a draw or an attachment clear.
void vk_exec_pass_begin(VkExec *ex);
// Ends the open pass (no-op without one) and marks its targets GPU-written. Call before copies and dispatches.
void vk_exec_pass_end(VkExec *ex);
// Full memory barrier (all commands, all memory); ends the open pass first. Copies and dispatches order themselves with
// it. Skipped when nothing was recorded since the last one.
void vk_exec_barrier(VkExec *ex);
// vk_texture.c (TOTK_TEX_ALIAS): commands that must run at the replay's current position. vk_exec_inline_begin returns
// the command buffer being recorded after a full barrier (an open pass is ended; *reopen says so), or VK_NULL_HANDLE
// when this thread is not recording a replay. vk_exec_inline_end begins the pass again when it was open.
// vk_exec_inline_buffer: a device buffer for transfers, destroyed when the submission completes.
VkCommandBuffer vk_exec_inline_begin(int *reopen);
// Ordered GPU-visible read during active replay. Requires one contiguous mapping, four-byte alignment and size
// from 4 bytes through 8 MiB. Submits/waits the prefix and continues recording; never dereferences a guest address.
int vk_exec_read_buffer(VkExec *ex, uint64_t addr, uint64_t size, void *out);
void vk_exec_inline_end(int reopen);
// vk_exec_draw.c: forget what the command buffer being recorded has bound (pipeline, sets, dynamic state, vertex
// buffers): the replay continues in a new one (vk_sync_split).
void vk_exec_draw_new_cb(void);
// vk_exec_draw.c: the draw path keeps its bindings, memos and caches per recording thread. vk_exec_draw_ctx_new makes a
// copy of them for another thread (made on the replaying thread while it records nothing); vk_exec_draw_ctx_use, on that
// thread, makes its draws use the copy. Threads that never call it share the file's own objects, as before.
void *vk_exec_draw_ctx_new(void);
void vk_exec_draw_ctx_use(void *ctx);
// vk_exec_draw.c: forget the bindings, memos and descriptor-build records of the thread's last command buffer before it
// records a slice of a sliced pass from a copy of the replay state, and after the primary executed the slices.
void vk_exec_draw_slice_begin(void);
// Sliced passes (vk_exec.c par_start): 1 while this thread records a slice; vk_exec_par_hazard marks that the slice
// needed something only the primary may do (the call is then dropped and the pass's targets are not sliced again).
int vk_exec_par_slice(void);
void vk_exec_par_hazard(void);
// vk_exec_xfer.c: 1 while occlusion queries count draws; a sliced pass's query begins before the pass and ends after
// it, and its slices inherit it.
int vk_exec_oq_counting(void);
int vk_exec_oq_begin_outside(VkExec *ex, VkQueryControlFlags *flags, int *open);
void vk_exec_oq_pass_ended(VkExec *ex);
VkBuffer vk_exec_inline_buffer(VkDeviceSize size);
// vk_exec.c: the stand-in a 3D BC render target is drawn through goes with its image (vk_exec_image_gone).
void vk_exec_bc_gone(VkImage image);
// Scissor 0 clipped to the render area of the open pass; extent 0: nothing to draw.
VkRect2D vk_exec_scissor(const VkExec *ex);
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h);   // pass size when no attachment is set
// 1: every target of the current pass (any 1: at least one) was rendered in the previous presented frame (per-frame
// targets); 0 otherwise, or when the pass has none.
int vk_exec_targets_recent(VkExec *ex, int any);
// The current pass's targets with their formats, extents and the frames they were last rendered in (diagnostics).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap);

// upscaler = dlss (vk_dlss.c), from vk_exec_draw.c. vk_dlss_scene_draw: a draw about to be recorded with a depth
// target; it notes the scene's depth and camera for the frame's evaluation. vk_dlss_jitter: sets j (pixels) to the
// viewport offset of draws into depth when it is the scene depth DLSS last ran on; j stays as it is otherwise.
// vk_dlss_image_gone: image is being destroyed (vk_exec_image_gone).
void vk_dlss_scene_draw(const VkExec *ex);
void vk_dlss_jitter(VkImage depth, float j[2]);
void vk_dlss_image_gone(VkImage image);

// TOTK_TRACE=gfx: replay counters of the current frame, logged and reset at each present (vk_exec_frame_log). The
// struct is the back end's (gfx.h GfxExecFrame), since the flight recorder reads it through GfxBackend.exec_frame.
typedef GfxExecFrame VkExecFrame;
extern VkExecFrame vk_exec_frame;
void vk_exec_frame_written(const tex_t *t);                    // vk_texture_gpu_written: t rendered or copied into
uint64_t vk_exec_recording_fence(void);                        // fence of the submission this thread records, else 0
void vk_exec_frame_log(const tex_t *presented, uint64_t frame);
// Running totals since the start: draws and dispatches the replay walked, and descriptor pools taken by command buffers
// (spare or new) and created. Replay thread (TOTK_TRACE=perf, vk_sync.c).
uint64_t vk_exec_draws_total(void);
void vk_exec_pool_counts(uint64_t *created, uint64_t *taken);

// TOTK_DUMP=<frame> (vk_dump.c): between presents frame-1 and frame, every pass's targets and every dispatch's storage
// images are copied out and written as PNGs to TOTK_DUMP_DIR (default "dump"), with passes.txt (targets, pipelines,
// textures, samplers) and images.txt (per image value statistics). The hooks run only when vk_dump_armed is set.
extern int vk_dump_armed;
void vk_dump_init(void);                 // reads the environment; vk_exec_record's first call
void vk_dump_pass_end(VkExec *ex);       // after vkCmdEndRendering
void vk_dump_draw(VkExec *ex, const VkPipeState *s, VkPipeline p, const VkPipeInfo *info);   // after a draw's binds
void vk_dump_dispatch(VkExec *ex, const GfxShader *cs, VkPipeline p, const VkPipeInfo *info, const p_dispatch *d);
void vk_dump_copy(VkExec *ex, uint32_t op, const void *payload);   // before a copy or texture clear is replayed
void vk_dump_present(uint64_t frame);    // vk_be_present: waits for the dump frame's copies and closes the logs

// c0, the NVN driver constant bank of one stage (vk_exec_draw.c; TOTK_FIX_C0=0 binds zeros instead): bank is
// VK_C0_BYTES, stage an NVN stage (0 vertex, 1 fragment, 5 compute), ssbo and ubo the 16 bound slots of that stage.
#define VK_C0_BYTES GFX_C0_BYTES
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance);

// Handler hooks. vk_exec.c defines both WEAK: they stop with rt_fatal naming the op.
// vk_exec_draw.c: OP_DRAW, OP_DRAW_IDX, OP_DISPATCH, OP_BIND_* (program, state, vattr, vstream, vbuf, ubo, ssbo, tex,
// samp, img), OP_UPDATE_UB, OP_DRAW_IDX_IND, OP_DISPATCH_IND. payload: nvn_op_payload of the record.
void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload);
// The aspect a sampled view of img reads (vk_exec_draw.c get_view): stencil for a view in STENCIL depth-stencil mode.
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv);
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks);   // kept for the image's life
// vk_exec_xfer.c: OP_COPY_B2B, OP_COPY_B2T, OP_COPY_T2B, OP_COPY_T2T, OP_CLEAR_TEX, OP_SIGNAL_EVENT, OP_REPORT_COUNTER,
// OP_RESET_COUNTER, OP_RENDER_ENABLE, OP_RENDER_COND, OP_CLEAR_BUF.
void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload);
// vk_exec_xfer.c, samples-passed counting: a draw about to be recorded in the open pass (begins an occlusion query
// when counting and none is active), and the open pass about to end (ends the active query).
void vk_exec_oq_draw(VkExec *ex);
void vk_exec_oq_pass_end(VkExec *ex);
