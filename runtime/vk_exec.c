// The op-stream replay core of the Vulkan back end: walks the command handles of
// one submission into its command buffer, keeps render targets and fixed-function state, opens and closes dynamic
// rendering passes and records clears and barriers. Draws, binds and dispatches go to vk_exec_op_draw
// (vk_exec_draw.c); copies, counters, events and conditional rendering to vk_exec_op_xfer (vk_exec_xfer.c). Images
// stay in VK_IMAGE_LAYOUT_GENERAL (vk_texture.c); full memory barriers order passes, clears and whatever those two
// record between them.
//
// The op loop (END / BEGIN / JUMP / CALL, which ops are pure state, the state defaults, render targets kept until they
// change, clears skipped while rendering is disabled) follows MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:10588-10700, 7409-7460, 5449-5457
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou). Pass begin and end (GENERAL-layout attachments, LOAD/STORE, render area from the smallest attachment, a
// full barrier before each pass), the scissor clip and attachment clears follow LAS Recomp
// runtime/nvn/vulkan/exec.c:855-1149, 3553-3778 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "platform/platform.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_spec.h"

#define WEAK __attribute__((weak))
#define CALL_DEPTH 8

typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    VkImageAspectFlags aspect;
    uint32_t level, base, layers;
    VkImageView view;
} ViewEnt;

static VkExec X;
static int x_ready;
static _Thread_local int x_here;   // this thread is inside vk_exec_record (vk_exec_inline_begin)

// Parallel passes: the state shared with the pass handlers (the rest is at the op loop).
#define PAR_MIN_DRAWS 256
#define PAR_SLICE_DRAWS 1024
#define PAR_HIST 256
enum { PAR_WHY_UNSEEN, PAR_WHY_NONE, PAR_WHY_NEW, PAR_WHY_SMALL, PAR_WHY_BANNED, PAR_WHY_NOATT, PAR_WHY_QUERY };
#define PAR_BAN_AFTER 8   // hazards after which a target set is not sliced again

typedef struct { uint64_t sig; uint32_t draws; uint8_t banned; } ParHist;
static ParHist par_hist[PAR_HIST];
static int par_mode = -1;                       // DKB_REPLAY_THREADS: 0 off, else the threads (slices recorded on the replaying thread)
static uint64_t par_t_collect, par_t_own, par_t_exec;   // ns on the replaying thread (TOTK_TRACE=perf)
static uint64_t par_t0; 
                        // the collection's start
static uint64_t par_unsliced[4];                // draws of unsliced passes: under 64, under 256, under 1024, more
static uint64_t par_restarts;                   // passes a clear opened that were opened again for slices
static uint64_t par_passes, par_slices, par_sliced_draws, par_skipped_oq;   // since the last perf line
static uint64_t par_end_op[OP_COUNT_ + 1];     // collections ended by each op (OP_COUNT_: the submission's end)
static uint64_t par_no[5];                      // big-pass draws left unsliced: 0 small or new targets, 1 banned,
                                                // 2 no attachment, 3 hazards in slices, 4 pass did not open
static _Thread_local int par_slice, par_hazard; // recording a slice; a slice dropped a pass or barrier call
static int par_restart;                         // par_start ends a pass a clear opened, to open it again for slices
static int pass_secondary;                      // the next vk_exec_pass_begin opens a pass of secondary command buffers
static int pass_tried, pass_sliced;             // the open pass: par_start looked at it; it has secondary contents
static int pass_why;                            // why par_start left the open pass alone (PAR_WHY_*), for the log
static uint32_t pass_draws;                     // draws of the open pass, for par_hist
#define PAR_MAX_SLICES 256       // a pass with more draws than these hold keeps adding to its last slice
#define PAR_MAX_OPS (1 << 20)    // collected ops of one pass; a full list ends the collection
static struct {
    int on;
    uint64_t sig;
    const op_hdr **ops;                          // the collected ops in order (PAR_MAX_OPS)
    int nops;
    VkExec *snap;                                // X before each slice's first op (PAR_MAX_SLICES)
    int start[PAR_MAX_SLICES], end[PAR_MAX_SLICES];   // each slice's ops; end is set when the next one starts
    int nslices;
    int draws, slice_draws;
    int oq;                                      // an occlusion query spans the pass: the slices inherit it
    VkQueryControlFlags oq_flags;
} PC;

// A recording thread's secondary command buffers: its own pool (pools are used by one thread at a time) and the ones whose
// submission completed (the completion thread adds them under lock).
typedef struct {
    VkCommandPool pool;
    PlatMutex lock;
    VkCommandBuffer *free;
    int nfree, cfree;
} ParPool;
typedef struct { ParPool *pp; VkCommandBuffer cb; } ParCb;
typedef struct { int n; ParCb cb[]; } ParDone;

static void *par_grow(void *p, int *cap, size_t size) {
    int c = *cap ? *cap * 2 : 64;
    void *q = realloc(p, (size_t)c * size);
    if (!q) rt_fatal("vk: out of memory");
    *cap = c;
    return q;
}

// The pass's targets as one number: the resolved images with their level and layers. Not the OP_SET_RT payload, whose
// bytes differ from frame to frame for the same targets (every pass of a run logged "targets not seen before" with it).
// The targets must be resolved (resolve_rt).
static uint64_t par_sig(const VkExec *ex) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        uint64_t v[2] = {0, 0};
        if ((i == 8 || i < ex->pass_ncolor) && t->image)
            v[0] = (uint64_t)(uintptr_t)t->image, v[1] = (uint64_t)t->level << 40 | (uint64_t)t->base_layer << 20 | t->layers;
        h = (h ^ v[0]) * 0x100000001b3ull;
        h = (h ^ v[1]) * 0x100000001b3ull;
    }
    return h | 1;
}

// The entry of target set sig, among 4 neighbours of its slot. add: a missing one takes a free neighbour, else the one
// with the fewest draws. Two of the busy area's big target sets fell on one slot of the first, direct-mapped table (their
// signatures differed in the high bits only) and evicted each other every frame, so neither was ever sliced.
static uint32_t par_hist_slot(uint64_t sig) { return (uint32_t)((sig ^ sig >> 29) * 0x9E3779B97F4A7C15ull >> 56) % PAR_HIST; }

static ParHist *par_find(uint64_t sig, int add) {
    uint32_t i0 = par_hist_slot(sig);
    ParHist *low = NULL;
    for (uint32_t k = 0; k < 4; k++) {
        ParHist *e = &par_hist[(i0 + k) % PAR_HIST];
        if (e->sig == sig) return e;
        if (!low || (!e->sig && low->sig) || (e->sig && low->sig && e->draws < low->draws)) low = e;
    }
    if (!add) return NULL;
    *low = (ParHist){sig, 0, 0};
    return low;
}


// ---- TOTK_TRACE=gfx frame counters

VkExecFrame vk_exec_frame;
static struct { const uint8_t *mem; uint32_t w, h, fmt, n; } fw[64];   // textures written this frame
static int nfw;

// The helper threads' draw translators (vk_spec.h), by helper number.
static const VkExecHelper *const helper_tr[] = {&vk_exec_draw2_helper, &vk_exec_draw3_helper, &vk_exec_draw4_helper,
                                                &vk_exec_draw5_helper, &vk_exec_draw6_helper};
_Static_assert(sizeof helper_tr / sizeof helper_tr[0] == VK_SPEC_HELPERS, "a draw translator for each helper thread");

// The counters of the calling thread: a helper thread keeps its own for the part it records, and they are added to
// vk_exec_frame when the part is taken (par_join).
#define FRAME (vk_spec_here ? helper_tr[vk_spec_here - 1]->frame : &vk_exec_frame)

uint64_t vk_exec_recording_fence(void) { return x_here || par_slice ? X.fence : vk_spec_helper() ? vk_spec_fence() : 0; }

void vk_exec_frame_written(const tex_t *t) {
    if (!(rt_trace & RT_TRACE_GFX) || !t) return;
    for (int i = 0; i < nfw; i++)
        if (fw[i].mem == t->mem) { fw[i].n++; return; }
    if (nfw == 64) return;
    fw[nfw].mem = t->mem, fw[nfw].w = t->d.width, fw[nfw].h = t->d.height, fw[nfw].fmt = (uint32_t)t->d.format;
    fw[nfw++].n = 1;
}

void vk_exec_frame_log(const tex_t *p, uint64_t frame) {
    if (!(rt_trace & RT_TRACE_GFX)) return;
    VkExecFrame *f = &vk_exec_frame;
    int hit = 0;
    char list[512] = "";
    size_t o = 0;
    for (int i = 0; i < nfw; i++) {
        if (p && fw[i].mem == p->mem) hit = 1;
        if (o < sizeof list - 48)
            o += (size_t)snprintf(list + o, sizeof list - o, " %p:%ux%u/%#x*%u", (const void *)fw[i].mem, fw[i].w,
                                  fw[i].h, fw[i].fmt, fw[i].n);
    }
    rt_log("frame %llu: submits %llu passes %llu draws %llu/%llu clears %llu copies %llu dispatches %llu; present %p "
           "%ux%u/%#x written %d; feedback draws %llu barriers %llu restarts %llu layouts %llu; targets%s",
           (unsigned long long)frame, (unsigned long long)f->submits,
           (unsigned long long)f->passes, (unsigned long long)f->draws, (unsigned long long)f->draws_seen,
           (unsigned long long)f->clears, (unsigned long long)f->copies, (unsigned long long)f->dispatches,
           p ? (const void *)p->mem : NULL, p ? p->d.width : 0, p ? p->d.height : 0, p ? (uint32_t)p->d.format : 0,
           hit, (unsigned long long)f->fb_draws, (unsigned long long)f->fb_barriers, (unsigned long long)f->fb_restarts,
           (unsigned long long)f->fb_layouts, list);
    memset(f, 0, sizeof *f);
    nfw = 0;
}

WEAK void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

WEAK void vk_exec_draw_slice_begin(void) {}
WEAK void vk_exec_draw_spare_threads(int threads) { (void)threads; }
WEAK int vk_exec_oq_counting(void) { return 0; }
WEAK int vk_exec_oq_begin_outside(VkExec *ex, VkQueryControlFlags *flags, int *open) { return (void)ex, (void)flags, *open = 0, 0; }
WEAK void vk_exec_oq_pass_ended(VkExec *ex) { (void)ex; }

WEAK void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

enum { W_UNRENDERABLE, W_LEVEL, W_LAYER, W_VIEW_FORMAT, W_BLOCK_VIEW, W_BLOCK_LAYERS, W_CLEAR_TARGET, W_CLEAR_MASK, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

// NVN defaults of the replay state (nvn_mtl.m).
static void state_reset(VkExec *ex) {
    memset(ex, 0, sizeof *ex);
    ex->st_vmask[0] = ex->st_vmask[1] = ex->st_wmask[0] = ex->st_wmask[1] = 0xff;
    ex->depth_range[1] = 1.0f;
    vk_pipe_state_defaults(&ex->pipe);
    ex->barrier_at = -1;
}

// ---- render targets

static uint32_t level_extent(uint32_t v, uint32_t level) {
    v >>= level;
    return v ? v : 1;
}

// A view format other than the texture's is used when both map directly and have the same texel size (the image is
// created MUTABLE_FORMAT for colour formats). blocks: t is a BC texture with block_views, and the view an uncompressed
// format whose texel is one of its blocks.
static int view_format(const tex_t *t, const view_t *v, int blocks, VkFormat *out) {
    TexVkFormat a, b;
    int bw, bh, vbw, vbh;
    if (!tex_vk_format(t->d.format, &a) || a.vk != a.ideal || !tex_vk_format(v->format, &b) || b.vk != b.ideal ||
        b.depth || b.stencil)
        return 0;
    int bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), vbpb = nvn_fmt_bpb(v->format, &vbw, &vbh);
    if (!bpb || bpb != vbpb || (!blocks && (bw != 1 || bh != 1)) || vbw != 1 || vbh != 1) return 0;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, b.vk, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return 0;
    *out = b.vk;
    return 1;
}

// ---- 3D BC render targets
// A 3D BC texture has no block views (vk_texture.c create_tex, TOTK_BLOCK_VIEWS_3D): on RDNA2 cards such an image
// made BLOCK_TEXEL_VIEW_COMPATIBLE and 2D_ARRAY_COMPATIBLE took GPU writes past the end of its memory (issue #5). Draws
// into it through an uncompressed view of its blocks (agl's BC4 encoder fills 32x32x32 and 64x64x64 volumes and their
// levels at the logo) go to a stand-in: a 2D array of the blocks of one level in the view's format, one layer per
// depth plane. It starts with the level's content, so a pass that writes part of a plane keeps the rest, and each pass
// end copies the planes it drew into the level (a copy between a compressed and a size-compatible uncompressed format).
typedef struct { VkImage bc, img; VmaAllocation alloc; VkFormat fmt; uint32_t level, w, h, d; } BcStandIn;
#define BC_STANDINS 64   // one per level drawn: 6 volumes of up to 7 levels at the logo
static BcStandIn bcs[BC_STANDINS];
static int nbcs;
static PlatMutex bcs_lock = PLAT_MUTEX_INIT;   // the replay adds and reads; destroy_tex (any thread) removes

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, uint32_t layers) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
}

static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                           VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// The stand-in of level level of a 3D BC image for draws through view format fmt, made on first use; 0 when none can
// be made (logged). Replay thread.
static int bc_standin(const VkTexImage *img, uint32_t level, VkFormat fmt, BcStandIn *out) {
    static int warned;
    plat_mutex_lock(&bcs_lock);
    int full = nbcs == BC_STANDINS;
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == img->image && bcs[i].level == level && bcs[i].fmt == fmt) {
            *out = bcs[i];
            plat_mutex_unlock(&bcs_lock);
            return 1;
        }
    plat_mutex_unlock(&bcs_lock);
    if (vk_spec_helper()) return vk_spec_fail(VK_SPEC_TARGET), 0;   // made in the upload batch: the replay thread's
    if (full) {
        if (!warned++) rt_log("vk: more than %d 3D BC render targets: the others are left out", BC_STANDINS);
        return 0;
    }
    vk_spec_event();
    uint32_t lw = level_extent(img->width, level), lh = level_extent(img->height, level);
    BcStandIn s = {img->image, VK_NULL_HANDLE, NULL, fmt, level, (lw + 3) / 4, (lh + 3) / 4,
                   level_extent(img->depth, level)};
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {s.w, s.h, 1}, .mipLevels = 1, .arrayLayers = s.d, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ci, &ai, &s.img, &s.alloc, NULL) != VK_SUCCESS) {
        if (!warned++) rt_log("vk: stand-in for a 3D BC render target could not be created: left out");
        return 0;
    }
    // In the upload batch, which goes to the queue before this part: GENERAL, then the texture's blocks.
    VkCommandBuffer cb = vk_upload_begin();
    image_barrier(cb, s.img, VK_IMAGE_LAYOUT_UNDEFINED, s.d);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, s.d},
                      .extent = {lw, lh, s.d}};   // texels of the compressed source
    vkCmdCopyImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vk_upload_end();
    plat_mutex_lock(&bcs_lock);
    bcs[nbcs++] = s;
    plat_mutex_unlock(&bcs_lock);
    rt_log("vk: 3D BC render target %ux%ux%u level %u drawn through a %ux%u x %u stand-in (format %d)", img->width,
           img->height, img->depth, level, s.w, s.h, s.d, (int)fmt);
    *out = s;
    return 1;
}

// vk_exec_draw.c vk_exec_image_gone: the stand-in of a BC image that goes, freed once the GPU is done with it.
void vk_exec_bc_gone(VkImage image) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == image) {
            vk_defer_image(bcs[i].img, bcs[i].alloc);
            bcs[i--] = bcs[--nbcs];
        }
    plat_mutex_unlock(&bcs_lock);
}

// At a pass end: the planes the pass drew into a stand-in go to its texture.
static void bc_copy_back(VkExec *ex, const VkExecTarget *t) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++) {
        if (bcs[i].img != t->image) continue;
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t->base_layer, t->layers},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, bcs[i].level, 0, 1},
                          .dstOffset = {0, 0, (int32_t)t->base_layer},
                          .extent = {bcs[i].w, bcs[i].h, t->layers}};   // texels of the uncompressed source
        vkCmdCopyImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, bcs[i].bc, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        ex->recorded++;
        break;
    }
    plat_mutex_unlock(&bcs_lock);
}

// A draw into a 3D BC image without block views through an uncompressed view of its blocks: the target is the
// image's stand-in (bc_standin). 1 when o is filled (target or left out), 0 for any other target.
static int bc_target(const tex_t *t, const view_t *v, int has_v, const VkTexImage *img, VkExecTarget *o) {
    int bw, bh;
    if (img->type != VK_IMAGE_TYPE_3D || img->block_views || !nvn_fmt_bpb(t->d.format, &bw, &bh) || bw != 4 || bh != 4)
        return 0;
    VkFormat fmt;
    BcStandIn s;
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return 1;
    }
    if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt) || !bc_standin(img, level, fmt, &s)) {
        warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                  t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
        return 1;
    }
    uint32_t base = 0, layers = s.d;
    if (v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < s.d ? s.d - base : 0);
    }
    if (base >= s.d || !layers) return 1;
    if (base + layers > s.d) layers = s.d - base;
    o->image = s.img;   // not img->image: view_of makes this command buffer's view of the stand-in
    o->img = img;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = 0, o->base_layer = base, o->layers = layers;
    o->width = s.w, o->height = s.h, o->samples = 1;
    o->nvn_format = v->format;
    return 1;
}

static void resolve_target(const tex_t *t, const view_t *v, int has_v, int depth_slot, VkExecTarget *o) {
    memset(o, 0, sizeof *o);
    o->tex = t;
    if (!t) return;
    const VkTexImage *img = vk_texture_get(t);
    // The helper thread makes no texture entry (vk_texture.c entry_of): its part is lost, and the target is not one
    // to report as unrenderable.
    if (!img && vk_spec_helper()) { vk_spec_fail(VK_SPEC_TARGET); return; }
    if (img && !depth_slot && bc_target(t, v, has_v, img, o)) return;   // a 3D BC image: its stand-in
    VkImageUsageFlags need = depth_slot ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    int is3d = img && img->type == VK_IMAGE_TYPE_3D && !depth_slot;   // slices drawn through 2D views (2D_ARRAY_COMPATIBLE)
    if (!img || !(img->usage & need) || (img->type != VK_IMAGE_TYPE_2D && !is3d)) {
        warn_once(W_UNRENDERABLE, "render target format %#x target %d (%ux%u) is not renderable here: left out",
                  t->d.format, t->d.target, t->d.width, t->d.height);
        return;
    }
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return;
    }
    // A 3D level's depth planes take the place of layers (Vulkan views them as the array layers of a 2D array view).
    uint32_t nlay = is3d ? level_extent(img->depth, level) : img->layers;
    uint32_t base = 0, layers = nlay;
    if (has_v && v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < nlay ? nlay - base : 0);
    }
    if (base >= nlay || !layers) {
        warn_once(W_LAYER, "render target layer %u of a %u-layer texture: left out", base, nlay);
        return;
    }
    if (base + layers > nlay) layers = nlay - base;
    VkFormat fmt = img->format;
    int32_t nvn_fmt = t->d.format;
    int blocks = !depth_slot && img->block_views;
    if (blocks) {   // a BC texture takes draws only through an uncompressed view of its blocks
        if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt)) {
            warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                      t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
            return;
        }
        nvn_fmt = v->format;
        if (layers > 1) {   // an uncompressed view of a BC image has one layer (no maintenance6 here)
            warn_once(W_BLOCK_LAYERS, "render target format %#x: block view of %u layers, first layer only", t->d.format,
                      layers);
            layers = 1;
        }
    } else if (!depth_slot && has_v && v->set_format && v->format != t->d.format) {
        if (view_format(t, v, 0, &fmt)) nvn_fmt = v->format;
        else warn_once(W_VIEW_FORMAT, "render target view format %#x of a format %#x texture: texture format used",
                       v->format, t->d.format);
    }
    o->image = img->image;
    o->img = img;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = depth_slot ? img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                           : VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = level, o->base_layer = base, o->layers = layers;
    o->is3d = (uint8_t)is3d;
    o->blocks = (uint8_t)blocks;
    o->width = level_extent(img->width, level), o->height = level_extent(img->height, level);
    if (blocks) o->width = (o->width + 3) / 4, o->height = (o->height + 3) / 4;   // the view's extent: 4x4 blocks
    o->samples = img->samples ? img->samples : 1;
    o->nvn_format = nvn_fmt;
    if (!o->aspect) o->image = VK_NULL_HANDLE;
}

/* The size of a pass without attachments: the far corner of viewport 0, else of the scissor, 1..16384, as
   MacNativeGames and nxnative size it. TotK draws a fragment shader that only writes a storage buffer into a 1x1
   viewport; a pass sized by the viewport's width alone puts an offset viewport outside the scissor, and the cave rock
   loses its ambient light. */
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h) {
    double x1 = 0, y1 = 0;
    if (ex->has_vp) {
        x1 = ceil(fmax(ex->vp[0][0], ex->vp[0][0] + ex->vp[0][2]));
        y1 = ceil(fmax(ex->vp[0][1], ex->vp[0][1] + ex->vp[0][3]));
    }
    if ((x1 < 1 || y1 < 1) && ex->has_scissor) x1 = (double)ex->scissor.x + ex->scissor.w, y1 = (double)ex->scissor.y + ex->scissor.h;
    *w = x1 < 1 ? 1 : x1 > 16384 ? 16384 : (uint32_t)x1;
    *h = y1 < 1 ? 1 : y1 > 16384 ? 16384 : (uint32_t)y1;
}

// Resolves ex->rt to this command buffer's images and derives the pass parameters and the pipeline's target formats.
static void resolve_rt(VkExec *ex) {
    if (ex->rt_resolved) return;
    ex->rt_resolved = 1;
    const p_set_rt *r = &ex->rt;
    int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n;
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 0;
    for (int i = 0; i < 8; i++) {
        VkExecTarget *t = &ex->color[i];
        resolve_target(i < n ? r->color[i] : NULL, &r->cview[i], i < n && r->has_cview[i], 0, t);
        ex->pass_color[i] = t->image ? t->format : VK_FORMAT_UNDEFINED;
        ex->pipe.color_fmt[i] = t->image ? t->nvn_format : 0;
        if (!t->image) continue;
        if (t->width < w) w = t->width;
        if (t->height < h) h = t->height;
        if (t->layers < layers) layers = t->layers;
        if (!samples) samples = t->samples;
    }
    VkExecTarget *d = &ex->depth;
    resolve_target(r->depth, &r->dview, r->has_dview, 1, d);
    ex->pass_depth = d->image && (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pass_stencil = d->image && (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pipe.depth_fmt = d->image ? d->nvn_format : 0;
    if (d->image) {
        if (d->width < w) w = d->width;
        if (d->height < h) h = d->height;
        if (d->layers < layers) layers = d->layers;
        if (!samples) samples = d->samples;
    }
    ex->pass_noatt = w == UINT32_MAX;
    if (ex->pass_noatt) {
        vk_exec_noatt_dims(ex, &w, &h);
        layers = 1;
        static int logged;
        if (logged < 8 && ex->has_vp && (ex->vp[0][0] != 0 || ex->vp[0][1] != 0 || ex->vp[0][2] != w || ex->vp[0][3] != h))
            logged++, rt_log("vk_exec: pass without attachments %ux%u for viewport %g,%g %gx%g", w, h, ex->vp[0][0],
                             ex->vp[0][1], ex->vp[0][2], ex->vp[0][3]);
    }
    ex->pass_w = w, ex->pass_h = h, ex->pass_layers = layers;
    ex->pass_ncolor = (uint32_t)n;
    ex->pass_samples = samples ? samples : 1;
    ex->pipe.samples = (uint8_t)ex->pass_samples;
}

// The attachment view of a target. A registered texture's view lives as long as its image (vk_exec_target_view); any
// other image's view is made for this command buffer and freed at its fence.
static VkImageView view_of(VkExec *ex, const VkExecTarget *t) {
    if (t->img && t->img->image == t->image)
        return vk_exec_target_view(t->img, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, t->blocks);
    ViewEnt *v = ex->views;
    for (int i = 0; i < ex->nviews; i++)
        if (v[i].image == t->image && v[i].format == t->format && v[i].type == t->view_type && v[i].aspect == t->aspect &&
            v[i].level == t->level && v[i].base == t->base_layer && v[i].layers == t->layers)
            return v[i].view;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = t->view_type, .format = t->format,
                                .subresourceRange = {t->aspect, t->level, 1, t->base_layer, t->layers}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    if (t->blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
        rt_fatal("vk: render target view (format %d, %ux%u) could not be created", (int)t->format, t->width, t->height);
    if (ex->nviews == ex->cap_views) {
        ex->cap_views = ex->cap_views ? ex->cap_views * 2 : 16;
        ex->views = realloc(ex->views, sizeof(ViewEnt) * (size_t)ex->cap_views);
        if (!ex->views) rt_fatal("vk: out of memory");
        v = ex->views;
    }
    v[ex->nviews++] = (ViewEnt){t->image, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, view};
    return view;
}

typedef struct { int n; VkImageView v[]; } ViewList;

static void free_views(void *arg) {   // completion thread, once the submission that used them is done
    ViewList *l = arg;
    VkDevice dev = vk_ctx()->device;
    for (int i = 0; i < l->n; i++) vkDestroyImageView(dev, l->v[i], NULL);
    free(l);
}

// ---- target history
// The presented frame (gfx_counters.frames + 1; 0: never) each target subresource was last rendered into, and the one
// before. prepare_draw lets a draw into a target the previous frame did not render (a one-shot draw: a volume or LUT
// generated once) wait for its shaders and pipeline instead of skipping it. Per subresource, not per image: a volume
// may be generated a few slices per frame. Direct-mapped: a slot another subresource took reads as not rendered, so a
// collision makes a draw wait rather than skip.
typedef struct { VkImage image; uint32_t level, layer; uint64_t last, prev; } TargetFrames;
static TargetFrames tfr[1024];

static TargetFrames *tfr_slot(const VkExecTarget *t) {
    uint64_t h = ((uint64_t)(uintptr_t)t->image ^ (uint64_t)t->level << 56 ^ (uint64_t)t->base_layer << 40) *
                 0x9E3779B97F4A7C15ull;
    return &tfr[h >> 54];
}

static int tfr_match(const TargetFrames *e, const VkExecTarget *t) {
    return e->image == t->image && e->level == t->level && e->layer == t->base_layer;
}

static void tfr_note(const VkExecTarget *t, uint64_t stamp) {
    TargetFrames *e = tfr_slot(t);
    if (!tfr_match(e, t)) *e = (TargetFrames){t->image, t->level, t->base_layer, 0, 0};
    if (e->last != stamp) e->prev = e->last, e->last = stamp;
}

static int tfr_recent(const VkExecTarget *t, uint64_t stamp) {
    const TargetFrames *e = tfr_slot(t);
    return tfr_match(e, t) && (e->last == stamp ? e->prev : e->last) == stamp - 1;
}

// A target the GPU writes: a pass that ends (stamp: the frame, for the target history) or a whole-target clear (0).
// A helper thread's writes wait in a list of its own until the replay thread takes its part (par_join), which is their
// place in submission order; a target whose marking cannot wait (the texture would change hands,
// vk_texture_gpu_written_ready) costs the helper its part.
typedef struct { VkExecTarget t; uint64_t stamp; } ParWrite;
static struct { ParWrite *w; int n, cap; } PW[VK_SPEC_HELPERS];

static void target_written(const VkExecTarget *t, uint64_t stamp) {
    if (!vk_spec_helper()) {
        vk_texture_gpu_written(t->tex);
        if (stamp) tfr_note(t, stamp);
        return;
    }
    if (!vk_texture_gpu_written_ready(t->tex)) {
        vk_spec_fail(VK_SPEC_TARGET);
        return;
    }
    typeof(PW[0]) *pw = &PW[vk_spec_here - 1];
    if (pw->n == pw->cap) {
        pw->cap = pw->cap ? pw->cap * 2 : 64;
        pw->w = realloc(pw->w, sizeof *pw->w * (size_t)pw->cap);
        if (!pw->w) rt_fatal("vk: out of memory");
    }
    pw->w[pw->n++] = (ParWrite){*t, stamp};
}

int vk_exec_targets_recent(VkExec *ex, int any) {
    resolve_rt(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int n = 0, recent = 0;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        n++;
        if (tfr_recent(t, stamp)) recent++;
        else if (!any) return 0;
    }
    return any ? recent > 0 : n > 0;
}

// One line naming each target of the current pass: format, extent, layer and the two frames it was last rendered in
// (prepare_draw's pipeline wait log).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap) {
    resolve_rt(ex);
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < 9 && n < cap; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        const TargetFrames *e = tfr_slot(t);
        int m = tfr_match(e, t);
        int w = snprintf(buf + n, cap - n, "%s%c%d fmt %d %ux%u L%u seen %llu/%llu", n ? "; " : "", i < 8 ? 'c' : 'd',
                         i < 8 ? i : 0, (int)t->format, t->width, t->height, t->base_layer,
                         (unsigned long long)(m ? e->last : 0), (unsigned long long)(m ? e->prev : 0));
        if (w < 0) break;
        n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "no targets");
}

// ---- passes and barriers

static int has_image(const VkImage *list, int n, VkImage img) {
    for (int i = 0; i < n; i++)
        if (list[i] == img) return 1;
    return 0;
}

static VkImageMemoryBarrier2 layout_barrier(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    return (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                                   .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
}

// Full barrier. The images in the feedback layout become want[] (pass begin) or none (every other barrier): the others
// go back to GENERAL and the new ones leave it, in the same command.
static void barrier_to(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant) {
    vk_exec_pass_end(ex);
    VkImageMemoryBarrier2 ib[18];
    uint32_t n = 0;
    for (int i = 0; i < ex->nfbl; i++)
        if (!has_image(want, nwant, ex->fbl_img[i]))
            ib[n++] = layout_barrier(ex->fbl_img[i], ex->fbl_aspect[i], VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
                                     VK_IMAGE_LAYOUT_GENERAL);
    for (int i = 0; i < nwant; i++)
        if (!has_image(ex->fbl_img, ex->nfbl, want[i]))
            ib[n++] = layout_barrier(want[i], wasp[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT);
    for (int i = 0; i < nwant; i++) ex->fbl_img[i] = want[i], ex->fbl_aspect[i] = wasp[i];
    ex->nfbl = nwant;
    if (ex->barrier_at == ex->recorded && !n) return;
    FRAME->fb_layouts += n;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = n, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->barrier_at = ++ex->recorded;
}

void vk_exec_barrier(VkExec *ex) { barrier_to(ex, NULL, NULL, 0); }

void vk_exec_pass_begin(VkExec *ex) {
    if (ex->in_pass) return;
    if (par_slice) { par_hazard = 1; return; }   // a slice cannot open a pass (parallel passes)
    resolve_rt(ex);
    // Learned targets (VkTexImage.fb_learned) are attached in the feedback layout
    VkImage want[9];
    VkImageAspectFlags wasp[9];
    int nwant = 0;
    uint32_t fbl = 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || !t->image || !t->img || !vk_tex_fb_learned(t->img)) continue;
        fbl |= 1u << i;
        if (has_image(want, nwant, t->image)) continue;
        want[nwant] = t->image, wasp[nwant++] = i < 8 ? VK_IMAGE_ASPECT_COLOR_BIT : t->img->aspect;
    }
    barrier_to(ex, want, wasp, nwant);   // passes, clears, copies and dispatches see each other's results
    ex->fbl_mask = fbl;
    ex->fb_dirty = 0;
    ex->pass_serial++;
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    for (uint32_t i = 0; i < ex->pass_ncolor; i++) {
        const VkExecTarget *t = &ex->color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (!t->image) continue;
        ca[i].imageView = view_of(ex, t);
        ca[i].imageLayout = fbl >> i & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;    }
    const VkExecTarget *d = &ex->depth;
    if (d->image) {
        VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view_of(ex, d),
                                       .imageLayout = fbl >> 8 & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                   : VK_IMAGE_LAYOUT_GENERAL,
                                       .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        if (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) da = a;
        if (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) sa = a;
    }
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {ex->pass_w, ex->pass_h}},
                          .layerCount = ex->pass_layers, .colorAttachmentCount = ex->pass_ncolor,
                          .pColorAttachments = ca,
                          .pDepthAttachment = ex->pass_depth ? &da : NULL,
                          .pStencilAttachment = ex->pass_stencil ? &sa : NULL,
                          .flags = pass_secondary ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT : 0};
    vkCmdBeginRendering(ex->cb, &ri);
    pass_draws = 0, pass_tried = 0, pass_sliced = pass_secondary, pass_why = PAR_WHY_UNSEEN;
    FRAME->passes++;
    ex->recorded++;
    ex->in_pass = 1;
}

void vk_exec_pass_end(VkExec *ex) {
    if (!ex->in_pass) return;
    if (par_slice) { par_hazard = 1; return; }   // a slice cannot end the pass (parallel passes)
    // The draws this pass's targets take, for the next time (par_start). A pass with fewer draws lowers the count by an
    // eighth only: a big pass comes in several pieces when a counter op ends a collection, and its targets may also
    // take a small pass in the same frame.
    if (ex == &X && !ex->pass_noatt && !par_restart) {
        uint64_t sig = par_sig(ex);
        static int told;
        if (par_mode > 0 && !pass_sliced && pass_draws >= PAR_SLICE_DRAWS && (told < 24 || told % 1000 == 0 || !++told)) {
            static const char *const why[] = {"no direct draw reached par_start", "not refused", "targets not seen before",
                                              "targets drew little before", "banned", "no attachment", "queries"};
            ParHist *h = par_find(sig, 0);
            told++;   // the first 24, then one in a thousand
            rt_log("vk: a pass of %u draws was not sliced: %s (targets %016llx, slot %u, kept count %u, slot holds %016llx)",
                   pass_draws, why[pass_why], (unsigned long long)sig, (unsigned)par_hist_slot(sig), h ? h->draws : 0,
                   (unsigned long long)par_hist[par_hist_slot(sig)].sig);
        }
        if (!pass_sliced && par_mode > 0)
            par_unsliced[pass_draws < 64 ? 0 : pass_draws < 256 ? 1 : pass_draws < 1024 ? 2 : 3] += pass_draws;
        ParHist *e = par_find(sig, pass_draws >= PAR_MIN_DRAWS);
        if (e) e->draws = pass_draws >= e->draws ? pass_draws : e->draws - e->draws / 8;
    }
    vk_exec_oq_pass_end(ex);
    vkCmdEndRendering(ex->cb);
    ex->recorded++;
    ex->in_pass = 0;
    vk_exec_oq_pass_ended(ex);   // a sliced pass's query, begun before the pass
    for (int i = 0; i < ex->pass_ncolor && i < 8; i++)   // stand-ins of 3D BC targets go to their textures
        if (ex->color[i].image && ex->color[i].img && ex->color[i].img->image != ex->color[i].image)
            bc_copy_back(ex, &ex->color[i]);
    if (vk_dump_armed && !vk_spec_helper()) vk_dump_pass_end(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    for (int i = 0; i < 8; i++)
        if (ex->color[i].image) target_written(&ex->color[i], stamp);
    if (ex->depth.image) target_written(&ex->depth, stamp);
}

VkCommandBuffer vk_exec_inline_begin(int *reopen) {
    VkExec *ex = &X;
    *reopen = 0;
    if (par_slice) { par_hazard = 1; return VK_NULL_HANDLE; }   // not inside a slice (parallel passes)
    if (!x_here || !ex->cb) return VK_NULL_HANDLE;
    *reopen = ex->in_pass;
    vk_exec_barrier(ex);   // ends the pass; feedback-layout targets go back to GENERAL
    return ex->cb;
}

void vk_exec_inline_end(int reopen) {
    X.recorded++;
    if (reopen) vk_exec_pass_begin(&X);
}

typedef struct { VkBuffer b; VmaAllocation a; } InlineBuf;
static void free_inline_buf(void *arg) {   // completion thread
    InlineBuf *t = arg;
    vmaDestroyBuffer(vk_ctx()->vma, t->b, t->a);
    free(t);
}

VkBuffer vk_exec_inline_buffer(VkDeviceSize size) {
    InlineBuf *t = calloc(1, sizeof *t);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (!t || vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &t->b, &t->a, NULL) != VK_SUCCESS) { free(t); return VK_NULL_HANDLE; }
    vk_completion_add(X.fence, free_inline_buf, t);
    return t->b;
}

VkRect2D vk_exec_scissor(const VkExec *ex) {
    int64_t w = ex->pass_w, h = ex->pass_h;
    if (!ex->has_scissor) return (VkRect2D){{0, 0}, {(uint32_t)w, (uint32_t)h}};
    int64_t x0 = ex->scissor.x, y0 = ex->scissor.y;
    int64_t x1 = x0 + ex->scissor.w, y1 = y0 + ex->scissor.h;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;
    if (x1 <= x0 || y1 <= y0) return (VkRect2D){{0, 0}, {0, 0}};
    return (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
}

// ---- clears

// Outside a pass, a clear of a whole target (no narrower scissor, the target as large as the render area) clears the
// image directly instead of opening a pass for it.
static int whole_target(const VkExec *ex, const VkExecTarget *t) {
    // vkCmdClear*Image addresses a 3D level whole, not one depth plane: slices of a 3D target clear in a pass; it
    // cannot clear a compressed image: views of BC blocks clear in a pass too
    if (ex->in_pass || t->is3d || t->blocks || t->width != ex->pass_w || t->height != ex->pass_h) return 0;
    if (!ex->has_scissor) return 1;
    const p_scissor *s = &ex->scissor;
    return s->x <= 0 && s->y <= 0 && (int64_t)s->x + s->w >= t->width && (int64_t)s->y + s->h >= t->height;
}

static void clear_attachment(VkExec *ex, const VkClearAttachment *a) {
    vk_exec_pass_begin(ex);
    VkClearRect cr = {vk_exec_scissor(ex), 0, ex->pass_layers};
    if (!cr.rect.extent.width || !cr.rect.extent.height) return;
    vkCmdClearAttachments(ex->cb, 1, a, 1, &cr);
    ex->recorded++;
    ex->fb_dirty |= a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? 1u << a->colorAttachment : 0x100u;
}

static void clear_color(VkExec *ex, const p_clear_color *c) {
    FRAME->clears++;
    if (ex->render_disabled) return;
    resolve_rt(ex);
    int i = c->index;
    if (i < 0 || i >= (int)ex->pass_ncolor || !ex->color[i].image) {
        warn_once(W_CLEAR_TARGET, "colour clear of target %d, which is not bound or not renderable: skipped", i);
        return;
    }
    uint32_t mask = (uint32_t)c->mask & 0xf;
    if (!mask) return;
    if (mask != 0xf) {   // vkCmdClearAttachments ignores write masks; a masked clear needs a draw (vk_exec_draw.c)
        warn_once(W_CLEAR_MASK, "colour clear with channel mask %#x is not replayed: skipped", mask);
        return;
    }
    const VkExecTarget *t = &ex->color[i];
    VkClearColorValue cv;
    memcpy(cv.float32, c->rgba, sizeof cv.float32);   // integer targets: the payload's bits as the front end stored them
    if (whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, 1, t->base_layer, t->layers};
        vkCmdClearColorImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r);
        ex->recorded++;
        target_written(t, 0);
        return;
    }
    VkClearAttachment a = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)i};
    a.clearValue.color = cv;
    clear_attachment(ex, &a);
}

static void clear_ds(VkExec *ex, const p_clear_ds *c) {
    if (ex->render_disabled) return;
    resolve_rt(ex);
    const VkExecTarget *t = &ex->depth;
    if (!t->image) return;
    VkImageAspectFlags asp = 0;
    if (c->depth_mask && (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
    uint32_t sm = (uint32_t)c->stencil_mask & 0xff;
    if (sm && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) {
        if (sm == 0xff) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
        else {
            VkRect2D scissor = vk_exec_scissor(ex);
            if (scissor.extent.width && scissor.extent.height) {
                // Passes of its own and a reset of what is bound: not in a part of the helper thread.
                if (vk_spec_helper()) {
                    vk_spec_fail(VK_SPEC_OP);
                    return;
                }
                vk_spec_event();
                // ClearAttachments ignores the write mask. Draw into each layer with ALWAYS/REPLACE instead,
                // preserving the depth plane and the stencil bits outside sm. A separate pass also excludes the
                // clear's fragments from the game's occlusion queries and avoids touching its colour targets.
                vk_exec_barrier(ex);
                VkExecTarget layer = *t;
                layer.layers = 1;
                layer.view_type = VK_IMAGE_VIEW_TYPE_2D;
                for (uint32_t i = 0; i < t->layers; i++) {
                    layer.base_layer = t->base_layer + i;
                    vk_clear_stencil_masked(ex->cb, t->format, t->samples, t->aspect, view_of(ex, &layer),
                                             ex->pass_w, ex->pass_h, scissor, sm, (uint32_t)c->stencil);
                    ex->recorded += 9;
                    vk_exec_frame.passes++;
                }
                vk_texture_gpu_written(t->tex);
                vk_exec_draw_new_cb();
                ex->dirty = VK_EXEC_DIRTY_ALL;
                vk_exec_barrier(ex);
            }
        }
    }
    if (!asp) return;
    float d = c->depth < 0.0f ? 0.0f : c->depth > 1.0f ? 1.0f : c->depth;
    VkClearDepthStencilValue v = {d, (uint32_t)c->stencil & 0xff};
    if (whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {asp, t->level, 1, t->base_layer, t->layers};
        vkCmdClearDepthStencilImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        ex->recorded++;
        target_written(t, 0);
        return;
    }
    VkClearAttachment a = {.aspectMask = asp};
    a.clearValue.depthStencil = v;
    clear_attachment(ex, &a);
}

// ---- the op loop

static void set_rt(VkExec *ex, const p_set_rt *r) {
    if (!memcmp(&ex->rt, r, sizeof *r)) return;   // the same targets again: the pass goes on
    vk_exec_pass_end(ex);
    ex->rt = *r;
    ex->rt_resolved = 0;
}

// ---- command buffer parts
// NVN starts the GPU as soon as commands are submitted; one Vulkan command buffer per submission starts it only after
// the whole replay. Between passes, once TOTK_SPLIT_DRAWS (64; 0 off) draws and dispatches were recorded since the
// last part, the part goes to the queue (vk_sync_split) and the replay continues in a new command buffer. Only
// command-buffer state is lost there (bindings and dynamic state: vk_exec_draw_new_cb, ex->dirty); barriers, layouts and
// queries are not: a barrier's first scope is everything earlier in submission order, image layouts belong to the image,
// and occlusion queries end with their pass. Fences, completion callbacks and reports stay per submission: only the last
// part signals the value. nxnative commits every 64 draws between passes; MacNativeGames after 30, then every 40.
// A part also spans at least TOTK_SPLIT_US (1000) of replay time: a split costs about 40 us of the render thread
// (the wait for the recorder, a command buffer, the mirror prologue; vkQueueSubmit2, as much again, is the recorder's
// with TOTK_REC_SUBMIT, vk_rec.c rec_submit), and a fight frame of 2,000 draws made 15 parts on the 7900X;
// a millisecond of recording is a fraction of a millisecond of GPU work, so the GPU still starts early.
static int split_n = -1, split_draws, part_start;
static int par_no_split;   // TOTK_REPLAY_PAR=check: the draws being compared stay in one command buffer, as the helper's
static uint64_t split_ns, part_t0;
static uint64_t n_parts, n_subs;

static void maybe_split(VkExec *ex) {
    if (split_n < 0) {
        split_n = cfg_int("TOTK_SPLIT_DRAWS");
        if (split_n < 0) split_n = 0;
        int us = cfg_int("TOTK_SPLIT_US");
        split_ns = us > 0 ? (uint64_t)us * 1000 : 0;
        rt_log("vk: replay command buffer split every %d draws between passes, parts of at least %d us%s", split_n,
               us > 0 ? us : 0, split_n ? "" : " (off)");
    }
    if (!split_n || ex->in_pass || split_draws < split_n || par_no_split) return;
    if (split_ns && kern_now_ns() - part_t0 < split_ns) return;   // split_draws stays: the next draw asks again
    split_draws = 0;
    if (ex->recorded == part_start) return;
    ex->cb = vk_sync_split();
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
}

// An ordered read of the GPU-visible backing at this replay position. In particular, imported guest bytes can be
// older than their shadow or VRAM mirror. Read through the same resolver/pick path as a buffer copy, never the VA.
int vk_exec_read_buffer(VkExec *ex, uint64_t addr, uint64_t size, void *out) {
    if (!x_here || ex != &X || !ex->cb || !out || !addr || !size || size > (8ull << 20) ||
        addr > UINT64_MAX - size || (addr & 3) || (size & 3)) return 0;
    VkMemSpan source;
    if (!vk_mem_resolve(addr, size, &source) || !source.buffer || source.avail < size || (source.offset & 3)) return 0;
    VK_MEM_PICK(source);   // choose an existing mirror; a read does not create a new one
    if (!source.buffer) return 0;
    VkCtx *c = vk_ctx();
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    VmaAllocationInfo mapped = {0};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (vmaCreateBuffer(c->vma, &bi, &ai, &buffer, &allocation, &mapped) != VK_SUCCESS) return 0;
    if (!mapped.pMappedData) { vmaDestroyBuffer(c->vma, buffer, allocation); return 0; }
    vk_exec_barrier(ex);   // end rendering and order all earlier writes before the transfer
    VkBufferCopy copy = {.srcOffset = source.offset, .dstOffset = 0, .size = size};
    vkCmdCopyBuffer(ex->cb, source.buffer, buffer, 1, &copy);
    VkBufferMemoryBarrier2 host = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer, .offset = 0, .size = size};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                           .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &host};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 2;
    vk_mem_count(source.cls, VK_MUSE_COPY_SRC, size);
    VK_MEM_BOUND(source);
    ex->cb = vk_sync_split_wait();   // this prefix, not ex->fence (only the final part signals that value)
    split_draws = 0;
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->barrier_at = -1;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
    VkResult result = vmaInvalidateAllocation(c->vma, allocation, 0, size);
    if (result == VK_SUCCESS) memcpy(out, mapped.pMappedData, (size_t)size);
    // split_wait is fatal on submission/wait failure, so this allocation is never freed while the copy is in flight.
    vmaDestroyBuffer(c->vma, buffer, allocation);
    return result == VK_SUCCESS;
}

static uint64_t draws_total;   // draws and dispatches walked since the start (vk_exec_draws_total)

uint64_t vk_exec_draws_total(void) { return draws_total; }

// The ops that only store fixed-function state for the draws after them. 0: op is not one of them.
static int state_op(VkExec *ex, uint32_t op, const void *a) {
    switch (op) {
    case OP_VIEWPORTS: {
        const p_viewports *v = a;
        for (int i = 0; i < v->n && v->first + i < 16; i++)
            if (v->first + i >= 0) memcpy(ex->vp[v->first + i], v->v[i], sizeof ex->vp[0]);
        ex->has_vp = 1;
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    }
    case OP_SCISSOR:
        ex->scissor = *(const p_scissor *)a;
        ex->has_scissor = 1;
        ex->dirty |= VK_EXEC_DIRTY_SCISSOR;
        break;
    case OP_DEPTH_RANGE: {
        const p_depth_range *r = a;
        ex->depth_range[0] = r->n, ex->depth_range[1] = r->f;
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    }
    case OP_STENCIL: {
        const p_stencil *s = a;
        for (int f = 0; f < 2; f++) {
            if (!(s->face & (1 << f))) continue;
            if (s->which == 0) ex->st_vmask[f] = (uint32_t)s->value_mask;
            else if (s->which == 1) ex->st_wmask[f] = (uint32_t)s->mask;
            else ex->st_ref[f] = (uint32_t)s->ref;
        }
        ex->dirty |= VK_EXEC_DIRTY_STENCIL;
        break;
    }
    case OP_POLY_OFFSET:
        ex->poly_offset = *(const p_poly_offset *)a;
        ex->dirty |= VK_EXEC_DIRTY_DEPTH_BIAS;
        break;
    case OP_BLEND_COLOR:
        memcpy(ex->blend_color, ((const p_blend_color *)a)->rgba, sizeof ex->blend_color);
        ex->dirty |= VK_EXEC_DIRTY_BLEND_CONST;
        break;
    case OP_DEPTH_CLAMP: ex->pipe.depth_clamp = ((const p_flag *)a)->value != 0; break;
    case OP_RASTER_DISCARD: ex->pipe.raster_discard = ((const p_flag *)a)->value != 0; break;
    case OP_PRIM_RESTART: ex->prim_restart = *(const p_prim_restart *)a; break;
    case OP_COPY_STRIDE: ex->copy_stride = ((const p_copy_stride *)a)->stride; break;
    case OP_SET_TEX_POOL: ex->tex_pool = ((const p_set_pool *)a)->pool, ex->tex_serial++; break;
    case OP_SET_SAMP_POOL: ex->samp_pool = ((const p_set_pool *)a)->pool, ex->tex_serial++; break;
    default: return 0;
    }
    return 1;
}

static void views_to_fence(VkExec *ex, uint64_t fence) {   // the views made for a command buffer go when it is done
    if (!ex->nviews) return;
    ViewList *l = malloc(sizeof *l + sizeof(VkImageView) * (size_t)ex->nviews);
    if (!l) rt_fatal("vk: out of memory");
    l->n = ex->nviews;
    for (int i = 0; i < ex->nviews; i++) l->v[i] = ((ViewEnt *)ex->views)[i].view;
    vk_completion_add(fence, free_views, l);
    ex->nviews = 0;
}

// ---- a run shared with the helper threads (vk_spec.h, TOTK_REPLAY_PAR)
// A run is a stretch of the op stream made of state, binds, clears and draws only: nothing in it reads back what the
// GPU wrote, signals the guest or changes the order of GPU writes into guest memory. At the start of a run the replay
// thread looks ahead to its end (par_scan) and picks the draws it leaves to the helpers (par_plan): those from a cut
// to an end, cut into one consecutive part for each helper. It launches the helpers (par_launch) and goes on
// translating up to the cut. Each helper first walks the ops before its part for their state alone, which gives it
// the replay state the replay thread will have where the part begins, then translates its part into a command buffer
// of its own (par_job). At the cut the replay thread ends its pass, waits for the first helper and checks that what
// that helper assumed held (par_join): then its own part goes to the queue, the helper's behind it, and the replay
// goes on after the helper's part with the helper's state. That is where the next helper's part begins, and it is met
// the same way. A part that did not hold is dropped, and the replay thread walks its draws as if no helper had run, up
// to the next part, which can still be taken.
//
// What a helper cannot do it does not do (vk_spec_fail): it never writes memory the guest reads, uploads a texture or
// makes an image, a texture entry or a stand-in. What a dropped part leaves behind is what the replay thread would
// have made at the same draws (vk_spec.h lists it): image views in the caches both threads use, a missing pipeline
// entered into the cache with its build queued, the freshness marks of textures found unchanged, and mirror syncs
// made a little earlier than their first use. The replay thread notes, draw by draw, the two things that decide where the
// parts can lie (par_watch): where it changed what a helper part reads (texture entries, mirrors, uniform snapshots:
// no such change may fall between the launch and the end of the last part) and where it did what a helper gives up
// for (none of that inside a part). A scene does the same things at the same draws frame after frame, so one frame
// plans the next, and a part that fails is walked by the replay thread, which then knows why.
#define PAR_RUN_MAX 8192   // draws of one run; a longer run goes on as the next one
#define PAR_RUN_MIN 160    // draws a run needs before a part is considered
#define PAR_PART_MIN 48    // draws a helper part needs to pay for its launch and join
#define PAR_GUARD 8        // draws left between a part and a draw it cannot hold: draws shift a little between frames
#define PAR_SLOTS 32       // runs that are remembered
#define PAR_SPAN 160       // k helpers take parts where PAR_SPAN * k * k draws lie between the launch and the end of
                           // the parts (par_count): two from 640, three from 1,440, four from 2,560. With a part
                           // for every 160 draws a scene whose runs have 800 to 1,100 draws lost 1.9 ms a frame
#define PAR_PACE 2.0f      // draws the replay thread translates while a helper translates one, to start from: a
                           // helper's draws cost more, it makes its Vulkan calls itself (the replay thread's go
                           // through the recorder, vk_rec.c)

// What is known about one run. gen and ev have a bit per draw: bit i is set when, between its looks before draw
// i - 1 and before draw i (bit n: after the last draw), the replay thread changed what a helper part reads (gen) or did
// what a helper does not do (ev). The bits of the draws a taken part held stay clear: the helper found neither.
typedef struct {
    uint64_t key;                // the render targets the run draws to (par_rt_key)
    uint64_t seen;               // the recording it was last met in
    uint32_t n;                  // its draws
    uint32_t ks;                 // the helpers share was learned with (par_count); 0: not yet
    float share;                 // the replay thread's share of the draws between the launch and the end of the parts
    uint32_t k;                  // the parts w is for
    float w[VK_SPEC_HELPERS];    // each part's share of the helpers' draws
    uint32_t fails, wait;        // runs in a row without a part taken; recordings to sit out before the next try
    uint32_t l, y;               // the last plan: launched after l draws, the parts end at draw y; y 0: no part
    uint32_t c[VK_SPEC_HELPERS]; // the draw each of its k parts begins at
    uint32_t tried, taken;       // parts since the last log, with the sums below
    uint64_t busy_ns, state_ns, early_ns, wait_ns;
    uint8_t gen[PAR_RUN_MAX / 8 + 1], ev[PAR_RUN_MAX / 8 + 1];
} ParSlot;
static ParSlot par_slots[PAR_SLOTS];

// One part of the run being walked: part i is translated by helper i (vk_spec.h) with helper_tr[i].
typedef struct {
    const op_hdr *cut, *hend;           // its first op, and the op after its last (the next part's first)
    uint32_t c, y;                      // its draws: c to y
    int out;                            // launched and not met yet
    int taken;                          // met and taken
    uint64_t busy_ns, state_ns, t_done; // helper: time spent on the part, on the state before it, and when it ended
    int64_t lag_ns;                     // from the replay thread's wait for the part to its end; negative: ended before
    VkExec H, S;                        // the helper's replay state, and what it started its part from
} ParPart;

enum { PS_SCAN, PS_LAUNCH, PS_JOIN, PS_END, PS_CHECK };
static struct {
    const op_hdr *next;                 // the op par_step acts at; NULL: none in this recording
    int state;                          // what it does there
    uint64_t rec;                       // recordings that looked for runs
    ParSlot *slot;                      // the run being walked; NULL: not a long one
    const op_hdr *from;                 // where the helpers are launched
    const op_hdr *stop, *after;         // end of the run, start of the next scan
    uint32_t n, l, k;                   // draws of the run; the plan: launched after l draws, k parts
    uint32_t ji, ci;                    // the part met next (PS_JOIN); the part being compared (PS_CHECK)
    uint32_t ord;                       // draws of the run the replay thread reached or left to the helpers
    uint64_t g;                         // par_watch: the generations and the events at its last look
    uint32_t e;
    int live;                           // parts that are out
    uint32_t ntaken;                    // parts of this run taken so far
    int planned;                        // every part of the plan was launched: the plan learns from this run
    uint64_t world, mirror, usnap;      // the generations at the launch
    VkExecFrame chk_frame;              // TOTK_REPLAY_PAR=check: this thread's counters where the part begins
    uint64_t chk_parts, chk_bad;        // parts compared, and those that differed
    uint64_t chk_word[VK_EXEC_DIG_W];   // by the first word of the digest that differed
    const op_hdr *draws[PAR_RUN_MAX];   // the run's draw ops
    ParPart part[VK_SPEC_HELPERS];
} P;

// 1: an op of a run, 2: a draw. Everything else ends the run.
static int par_run_op(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND: return 2;
    case OP_BEGIN: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET:
    case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART: case OP_COPY_STRIDE:
    case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: case OP_SET_RT: case OP_CLEAR_COLOR: case OP_CLEAR_DS:
    case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
    case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CONST:
    case OP_BARRIER: case OP_WAIT_SYNC: return 1;
    default: return 0;
    }
}

static uint64_t par_gens(void) { return vk_spec_world() + vk_mem_mirror_gen() + vk_usnap_gen(); }

static void par_bit(uint8_t *b, uint32_t i, int on) {
    if (on) b[i >> 3] |= (uint8_t)(1u << (i & 7));
    else b[i >> 3] &= (uint8_t)~(1u << (i & 7));
}

static int par_bit_at(const uint8_t *b, uint32_t i) { return b[i >> 3] >> (i & 7) & 1; }

// Before each draw of a long run the replay thread translates: what happened since the last look.
static void par_watch(void) {
    ParSlot *s = P.slot;
    uint64_t g = par_gens();
    uint32_t e = vk_spec_events();
    par_bit(s->gen, P.ord, g != P.g);
    par_bit(s->ev, P.ord, e != P.e);
    if (g != P.g && P.live)   // the parts that are out are lost: their helpers stop at their next op
        vk_spec_fail(vk_spec_world() != P.world ? VK_SPEC_WORLD : vk_mem_mirror_gen() != P.mirror ? VK_SPEC_MIRROR : VK_SPEC_USNAP);
    P.g = g, P.e = e;
    P.ord++;
}

// Helper thread, with its translator t: the ops from h up to end. state_only: only what they leave in the replay
// state (the ops the replay thread and the helpers before translate meanwhile); otherwise recorded into ex->cb. Stops
// at the first op after the part was given up.
static void par_walk(const VkExecHelper *t, VkExec *ex, const op_hdr *h, const op_hdr *end, int state_only) {
    while (h != end && !vk_spec_failed()) {
        const void *a = nvn_op_payload(h);
        switch (h->op) {
        case OP_BEGIN: case OP_WAIT_SYNC: break;
        case OP_JUMP: h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target; continue;
        case OP_SET_RT:
            if (!state_only) set_rt(ex, a);
            else if (memcmp(&ex->rt, a, sizeof ex->rt)) ex->rt = *(const p_set_rt *)a, ex->rt_resolved = 0;
            break;
        case OP_CLEAR_COLOR:
            if (!state_only) clear_color(ex, a);
            break;
        case OP_CLEAR_DS:
            if (!state_only) clear_ds(ex, a);
            break;
        case OP_BARRIER:
            if (!state_only) vk_exec_barrier(ex);
            break;
        case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
            if (state_only) break;
            t->frame->draws_seen++;
            t->digest_draw(ex);
            // fall through
        case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
        case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CONST:
            t->op(ex, h->op, a);
            break;
        default:
            if (!state_op(ex, h->op, a)) vk_spec_fail(VK_SPEC_OP);   // par_scan lets no other op into a run
        }
        h = nvn_op_next(h);
    }
}

// Helper thread: its part. The command buffer is ended whatever happens: it is begun again for a later part.
static void par_job(void *arg) {
    ParPart *p = arg;
    const VkExecHelper *t = helper_tr[p - P.part];
    uint64_t t0 = kern_now_ns();
    VkExec *ex = &p->H;
    par_walk(t, ex, P.from, p->cut, 1);
    // The command buffer starts as the replay thread's do (record_impl); every image is in GENERAL: the part before
    // ends with a full barrier, the replay thread's (par_join) or a helper's (below).
    ex->recorded = 0, ex->in_pass = 0, ex->dirty = VK_EXEC_DIRTY_ALL, ex->rt_resolved = 0, ex->barrier_at = -1;
    ex->nfbl = 0, ex->fbl_mask = 0, ex->fb_dirty = 0, ex->fb_fs = NULL;
    memcpy(&p->S, ex, sizeof p->S);
    p->state_ns = kern_now_ns() - t0;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(ex->cb, &bi);
    if (vk_spec_checking()) t->digest_begin(ex);
    par_walk(t, ex, p->cut, p->hend, 0);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);   // the part behind begins after a full barrier, as after the replay thread's own parts
    vkEndCommandBuffer(ex->cb);
    p->t_done = kern_now_ns();
    p->busy_ns = p->t_done - t0;
}

// The launch conditions that can change during a run are asked here; record_impl asked the others. Every helper of
// the plan is launched from the same point: what it reads may not change from here to the end of its part.
static void par_launch(VkExec *ex) {
    P.state = PS_END, P.next = P.stop;
    // A helper part records no occlusion query, dump copy or breadcrumb.
    if (!vk_exec_oq_idle() || vk_dump_armed || vk_crumb_bind_on() || vk_checkpoints_on()) return;
    P.world = vk_spec_world(), P.mirror = vk_mem_mirror_gen(), P.usnap = vk_usnap_gen();
    uint32_t k = 0;
    for (; k < P.k; k++) {
        ParPart *p = &P.part[k];
        const VkExecHelper *t = helper_tr[k];
        VkCommandBuffer cb = vk_spec_cb((int)k, ex->fence);
        if (!cb) break;
        if (!t->begin()) {
            vk_spec_cb_drop((int)k);
            break;
        }
        void *views = p->H.views;   // the helper's own list of views, kept from part to part
        int cap = p->H.cap_views;
        memcpy(&p->H, ex, sizeof p->H);
        p->H.cb = cb, p->H.in_pass = 0, p->H.views = views, p->H.nviews = 0, p->H.cap_views = cap;
        PW[k].n = 0;
        memset(t->frame, 0, sizeof *t->frame);
        p->busy_ns = p->state_ns = 0;
        p->out = 1, p->taken = 0;
        vk_dlss_part_begin((int)k);
        vk_spec_launch((int)k, par_job, p, ex->fence);
    }
    if (!k) return;
    // A helper without a command buffer or stand-ins leaves its part and the ones behind it to the replay thread.
    P.planned = k == P.k;
    P.k = k, P.live = (int)k, P.ji = 0, P.ntaken = 0;
    P.slot->tried += k;
    P.state = PS_JOIN, P.next = P.part[0].cut;
}

// The targets by their storage: Bananza initializes its target textures anew every frame, so neither the address of
// the object nor its serial says which target it is.
static uint64_t par_rt_key(uint64_t k, const p_set_rt *r) {
    k = (k ^ (uint64_t)(uint32_t)r->n) * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < r->n && i < 8; i++) k = (k ^ (r->color[i] ? (uintptr_t)r->color[i]->mem : 0)) * 0x9E3779B97F4A7C15ull;
    return (k ^ (r->depth ? (uintptr_t)r->depth->mem : 0)) * 0x9E3779B97F4A7C15ull;
}

// The remembered run this one is: the same targets and nearly the same number of draws, and not met yet in this
// recording. A run met for the first time takes the place of the one that was not met for the longest.
static ParSlot *par_slot(uint64_t key, uint32_t n) {
    ParSlot *s = NULL, *old = &par_slots[0];
    uint32_t best = n / 8 + 1;
    for (int i = 0; i < PAR_SLOTS; i++) {
        ParSlot *t = &par_slots[i];
        if (t->seen < old->seen) old = t;
        if (!t->n || t->key != key || t->seen == P.rec) continue;
        uint32_t d = t->n > n ? t->n - n : n - t->n;
        if (d < best) best = d, s = t;
    }
    if (!s) {
        s = old;
        memset(s, 0, sizeof *s);
        s->key = key;
    }
    s->n = n, s->seen = P.rec;
    return s;
}

// The helpers that the draws between a launch and the end of the parts are worth. A further part saves less than the
// one before it and costs as much: a join, a submission and a pass of its own. So k parts take PAR_SPAN * k * k draws.
static uint32_t par_count(uint32_t span) {
    uint32_t k = 1, max = (uint32_t)vk_spec_count();
    while (k < max && span >= PAR_SPAN * (k + 1) * (k + 1)) k++;
    return k;
}

// The draws of a run of n draws that are left to the helpers, where that saves the most: draws c to y, the helpers
// launched after l draws. Returns the helpers the draws from l to y are worth (par_count), 0: no part. Between the
// launch and the end of the parts nothing a helper reads may change (gen clear from l to y), and the parts hold
// nothing a helper gives up for (ev clear from c to y). The replay thread should reach the cut when the helpers end,
// so it takes its share of the draws between l and y, or more where the parts have to begin later.
static uint32_t par_plan(const ParSlot *s, uint32_t n, uint32_t *pl, uint32_t *pc, uint32_t *py) {
    uint32_t best = 0, bk = 0;
    uint32_t lg = 0;   // the helper cannot be launched before this many draws: a gen bit lies right below
    uint32_t x = 0;    // both bits are clear from here on
    for (uint32_t i = 0; i <= n + 1; i++) {
        int g = i <= n && par_bit_at(s->gen, i), e = i <= n && par_bit_at(s->ev, i);
        if (i <= n && !g && !e) continue;
        // Bits x to i - 1 are clear. A part may end at the end of the run, or a guard below a set bit.
        uint32_t y = i > n ? n : i > PAR_GUARD ? i - 1 - PAR_GUARD : 0;
        uint32_t l = lg ? lg + PAR_GUARD : 0, lo = x ? x + PAR_GUARD : 0;
        if (y > l && y > lo) {
            uint32_t k = par_count(y - l);
            float share = k == s->ks ? s->share : PAR_PACE / ((float)k + PAR_PACE);
            uint32_t c = l + (uint32_t)ceilf((float)(y - l) * share);
            if (c < lo) c = lo;
            if (c <= l) c = l + 1;
            if (c < y && y - c >= PAR_PART_MIN && y - c > best) best = y - c, bk = k, *pl = l, *pc = c, *py = y;
        }
        x = i + 1;
        if (g) lg = i + 1;
    }
    return bk;
}

// At the first op of a run: its extent and draws, and whether and where the helpers take parts of it.
static void par_scan(VkExec *ex, const op_hdr *h) {
    uint32_t n = 0;
    int kind;
    const op_hdr *p = h;
    uint64_t key = par_rt_key(1, &ex->rt);
    for (;;) {
        if (p->size < sizeof *p || (p->size & 7) || p->op >= OP_COUNT_) {   // walk reports it when it gets there
            P.next = NULL;
            return;
        }
        if (p->op == OP_JUMP) {
            p = (const op_hdr *)(uintptr_t)((const p_jump *)nvn_op_payload(p))->target;
            continue;
        }
        if (!(kind = par_run_op(p->op)) || (kind == 2 && n == PAR_RUN_MAX)) break;
        if (kind == 2) P.draws[n++] = p;
        else if (p->op == OP_SET_RT) key = par_rt_key(key, nvn_op_payload(p));
        p = nvn_op_next(p);
    }
    P.stop = p;
    P.after = kind ? p : p->op == OP_END ? NULL : nvn_op_next(p);   // a run cut short by its length goes on at p
    P.slot = NULL;
    P.state = PS_SCAN, P.next = P.after;
    if (n < PAR_RUN_MIN) return;
    ParSlot *s = par_slot(key, n);
    P.slot = s, P.n = n, P.ord = 0;
    P.g = par_gens(), P.e = vk_spec_events();
    P.state = PS_END, P.next = P.stop;
    s->y = 0;
    if (s->wait) {
        s->wait--;
        return;
    }
    uint32_t c, y, ks = par_plan(s, n, &P.l, &c, &y);
    if (!ks) return;
    if (ks != s->ks) s->ks = ks, s->share = PAR_PACE / ((float)ks + PAR_PACE);
    uint32_t k = ks;
    while (k > 1 && y - c < k * PAR_PART_MIN) k--;
    if (k != s->k) {
        s->k = k;
        for (uint32_t i = 0; i < k; i++) s->w[i] = 1.0f / (float)k;
    }
    // The helpers' draws by each part's share, every part long enough to pay for itself.
    float upto = 0;
    uint32_t at = c;
    for (uint32_t i = 0; i < k; i++) {
        ParPart *p = &P.part[i];
        upto += s->w[i];
        uint32_t end = c + (uint32_t)((float)(y - c) * upto + 0.5f), max = y - (k - 1 - i) * PAR_PART_MIN;
        if (i == k - 1 || end > max) end = max;
        if (end < at + PAR_PART_MIN) end = at + PAR_PART_MIN;
        p->c = s->c[i] = at, p->y = end;
        p->cut = nvn_op_next(P.draws[at - 1]);
        p->hend = end == n ? P.stop : nvn_op_next(P.draws[end - 1]);
        at = end;
    }
    P.k = k;
    s->l = P.l, s->y = y;
    if (!P.l) {
        P.from = h;
        par_launch(ex);
    } else {
        P.from = nvn_op_next(P.draws[P.l - 1]);
        P.state = PS_LAUNCH, P.next = P.from;
    }
}

// 1: the op-determined replay state of a and b is the same: the render targets, the fixed-function state, the state
// objects and every binding. What draws and passes derive from it (shaders, formats, resolved targets) is left out.
static int par_same(const VkExec *a, const VkExec *b) {
#define PAR_SAME(from, to) \
    !memcmp((const char *)a + offsetof(VkExec, from), (const char *)b + offsetof(VkExec, from), \
            offsetof(VkExec, to) - offsetof(VkExec, from))
    return !memcmp(&a->rt, &b->rt, sizeof a->rt) && PAR_SAME(vp, pipe) && a->pipe.depth_clamp == b->pipe.depth_clamp &&
           a->pipe.raster_discard == b->pipe.raster_discard && PAR_SAME(pipe.blend, pipe.gs) &&
           !memcmp(&a->pipe.alpha_ref, &b->pipe.alpha_ref, sizeof a->pipe.alpha_ref) && PAR_SAME(prog, views);
#undef PAR_SAME
}

// The last part of a run was met. A run without a part taken counts towards sitting the next recordings out. A run
// with every part taken moves the plan: the replay thread should reach the cut when the helpers end, and the helpers
// should all end together, but the draws of a run do not cost the same.
static void par_learn(void) {
    ParSlot *s = P.slot;
    if (!P.ntaken) {
        s->fails++;
        // What the replay thread notes on the draws it now walks itself moves the next parts (par_watch); a reason it
        // cannot see (a pipeline still building) is waited out.
        s->wait = s->fails < 3 ? 0 : s->fails < 9 ? 1u << (s->fails - 3) : 64;
        return;
    }
    s->fails = 0;
    if (P.ntaken != P.k || !P.planned || P.k != s->k || vk_spec_checking()) return;
    uint32_t k = P.k, draws = P.part[k - 1].y - P.part[0].c;
    double busy = 0, done = 0;
    int64_t off = INT64_MIN;   // how long this thread waited for the helper it waited longest for; negative: the
                               // time every helper stood idle before this thread came for its part
    for (uint32_t i = 0; i < k; i++) {
        const ParPart *p = &P.part[i];
        busy += (double)(p->busy_ns - p->state_ns);
        done += (double)(int64_t)(p->t_done - P.part[0].t_done) / (double)k;
        if (p->lag_ns > off) off = p->lag_ns;
    }
    // The cut moves by the draws that would have made up for the wait or the idle time. A draw of the parts cost a
    // helper b, and this thread would have spent a little less on it; a draw this thread takes over comes off all k
    // parts.
    double b = busy / (double)draws;
    if (b > 0) {
        s->share += 0.4f * (float)((double)off / ((0.8 + 1.0 / (double)k) * b) / (double)(P.part[k - 1].y - P.l));
        // The fewest draws this thread keeps: with several helpers it is the one the others wait for.
        s->share = s->share < 0.05f ? 0.05f : s->share > 0.95f ? 0.95f : s->share;
    }
    // A part whose helper ended after the others gets fewer of the helpers' draws, one that ended before them more.
    float sum = 0;
    for (uint32_t i = 0; i < k; i++) {
        const ParPart *p = &P.part[i];
        double t = (double)(p->busy_ns - p->state_ns), late = (double)(int64_t)(p->t_done - P.part[0].t_done) - done;
        double f = t > 0 ? 1.0 - 0.5 * late / t : 1.0;
        s->w[i] *= (float)(f < 0.5 ? 0.5 : f > 1.5 ? 1.5 : f);
        sum += s->w[i];
    }
    for (uint32_t i = 0; i < k; i++) s->w[i] /= sum;
}

// At a part's first op: the part is taken, and the replay goes on at its end (returned), which is where the next
// part begins; or it is dropped and the replay goes on here.
static const op_hdr *par_join(VkExec *ex, const op_hdr *h) {
    ParSlot *s = P.slot;
    uint32_t i = P.ji;
    ParPart *p = &P.part[i];
    const VkExecHelper *t = helper_tr[i];
    int last = i + 1 == P.k;
    if (last) P.state = PS_END, P.next = P.stop;
    else P.ji = i + 1, P.next = P.part[i + 1].cut;   // PS_JOIN again
    // Straight from the part before, which was taken: this thread's command buffer is new and holds nothing, so
    // there is no pass to end and no part of its own to send first.
    int behind = i && P.part[i - 1].taken && ex->recorded == part_start;
    if (!behind && !vk_spec_part_failed((int)i)) {   // a part that is lost already does not cost the replay thread its open pass
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
    }
    uint64_t t_wait = kern_now_ns();
    int why = vk_spec_join((int)i);
    uint64_t wait_ns = kern_now_ns() - t_wait;
    p->out = 0, P.live--;
    // What the helper read while it translated: unchanged since the launch, the end of this thread's pass included.
    if (!why) {
        why = vk_spec_world() != P.world ? VK_SPEC_WORLD : vk_mem_mirror_gen() != P.mirror ? VK_SPEC_MIRROR :
              vk_usnap_gen() != P.usnap ? VK_SPEC_USNAP : !par_same(ex, &p->S) ? VK_SPEC_STATE : VK_SPEC_NONE;
    }
    views_to_fence(&p->H, ex->fence);
    vk_spec_note(why, p->y - p->c, wait_ns, p->busy_ns);
    s->busy_ns += p->busy_ns, s->state_ns += p->state_ns, s->wait_ns += wait_ns;
    if (p->t_done < t_wait) s->early_ns += t_wait - p->t_done;
    p->lag_ns = (int64_t)(p->t_done - t_wait);
    if (!why) P.ntaken++;
    if (last) par_learn();
    if (why) {
        vk_spec_cb_drop((int)i);
        PW[i].n = 0;
        return h;
    }
    if (vk_spec_checking()) {
        // The helper's part is dropped and this thread translates the same draws from the same start: nothing bound
        // on the command buffer, no memo of the state, no barrier to rely on. par_check compares at the end.
        vk_spec_cb_drop((int)i);
        PW[i].n = 0;
        ex->dirty = VK_EXEC_DIRTY_ALL, ex->rt_resolved = 0, ex->barrier_at = -1;
        ex->nfbl = 0, ex->fbl_mask = 0, ex->fb_dirty = 0, ex->fb_fs = NULL;
        vk_exec_draw_rebase();
        P.chk_frame = vk_exec_frame;
        vk_exec_draw_digest_begin(ex);
        par_no_split = 1;
        P.ci = i;
        P.state = PS_CHECK, P.next = p->hend;
        return h;
    }
    // This thread's part if it recorded one, then the helper's, then a new command buffer for what follows.
    if (!behind) ex->cb = vk_sync_split(), n_parts++;
    vk_sync_split_insert(p->H.cb);
    n_parts++;
    for (int j = 0; j < PW[i].n; j++) {
        vk_texture_gpu_written_late(PW[i].w[j].t.tex);
        if (PW[i].w[j].stamp) tfr_note(&PW[i].w[j].t, PW[i].w[j].stamp);
    }
    PW[i].n = 0;
    vk_dlss_part_taken((int)i);
    for (size_t j = 0; j < sizeof vk_exec_frame / sizeof(uint64_t); j++)
        ((uint64_t *)&vk_exec_frame)[j] += ((const uint64_t *)t->frame)[j];
    draws_total += t->frame->draws_seen;   // walk did not see the part's draws (a part holds no dispatch)
    // The replay state after the part is the helper's: everything from the render targets to the bindings. The command
    // buffer is new, the targets are resolved again, and no memo of this thread's translator knows the state.
    uint32_t tex_serial = (ex->tex_serial > p->H.tex_serial ? ex->tex_serial : p->H.tex_serial) + 1;
    uint32_t pass_serial = (ex->pass_serial > p->H.pass_serial ? ex->pass_serial : p->H.pass_serial) + 1;
    memcpy((char *)ex + offsetof(VkExec, rt), (const char *)&p->H + offsetof(VkExec, rt),
           offsetof(VkExec, views) - offsetof(VkExec, rt));
    ex->in_pass = 0, ex->dirty = VK_EXEC_DIRTY_ALL, ex->rt_resolved = 0;
    ex->nfbl = 0, ex->fbl_mask = 0, ex->fb_dirty = 0, ex->fb_fs = NULL;
    ex->tex_serial = tex_serial, ex->pass_serial = pass_serial;
    vk_exec_draw_rebase();
    split_draws = 0;
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    p->taken = 1;
    s->taken++;
    // The helper's draws are behind this thread: par_watch looks again from here.
    P.ord = p->y == P.n ? P.n + 1 : p->y;
    P.g = par_gens(), P.e = vk_spec_events();
    return p->hend;
}

// TOTK_REPLAY_PAR=check, at the end of a part its helper and this thread translated: the same draws bound and
// recorded the same, and left the same replay state.
static void par_check(VkExec *ex) {
    const ParPart *p = &P.part[P.ci];
    const VkExecHelper *t = helper_tr[P.ci];
    const uint64_t *a, *b;
    vk_exec_pass_end(ex);   // as the helper ends its part (par_job)
    vk_exec_barrier(ex);
    uint32_t na = vk_exec_draw_digest_end(ex, &a), nb = t->digest_end(&p->H, &b), at = 0;
    const VkExecFrame *f = &vk_exec_frame, *f0 = &P.chk_frame, *hf = t->frame;
    static const char *const word[VK_EXEC_DIG_W] = {
        "the image views a draw binds", "the samplers a draw binds", "the image layouts a draw binds",
        "the uniform buffer ranges a draw binds", "the uniform buffers a draw binds",
        "the uniform bytes a translator filled in for a draw", "the storage buffer ranges a draw binds",
        "the storage buffers a draw binds", "the pipeline or the dynamic state of a draw",
        "the vertex or index buffers of a draw", "the commands a draw records"};
    const char *bad = na != nb ? "the number of draws" : NULL;
    for (uint32_t i = 0; !bad && i < na * VK_EXEC_DIG_W; i++)
        if (a[i] != b[i]) bad = word[i % VK_EXEC_DIG_W], at = i / VK_EXEC_DIG_W, P.chk_word[i % VK_EXEC_DIG_W]++;
    if (!bad && !par_same(ex, &p->H)) bad = "the replay state after the part";
    if (!bad && memcmp(&ex->pipe, &p->H.pipe, sizeof ex->pipe)) bad = "the pipeline state after the part";
    if (!bad && (f->draws - f0->draws != hf->draws || f->passes - f0->passes != hf->passes ||
                 f->clears - f0->clears != hf->clears || f->fb_draws - f0->fb_draws != hf->fb_draws ||
                 f->fb_barriers - f0->fb_barriers != hf->fb_barriers || f->fb_restarts - f0->fb_restarts != hf->fb_restarts))
        bad = "the draws, passes, clears or feedback barriers recorded";
    P.chk_parts++;
    if (bad && P.chk_bad++ < 16)
        rt_log("vk_spec: check: helper %u's part of draws %u to %u of a run of %u differs in %s (draw %u of the part; "
               "draws %llu and %llu, passes %llu and %llu)",
               P.ci + 1, p->c, p->y, P.n, bad, at, (unsigned long long)(f->draws - f0->draws),
               (unsigned long long)hf->draws, (unsigned long long)(f->passes - f0->passes),
               (unsigned long long)hf->passes);
    par_no_split = 0;
    if (P.ci + 1 == P.k) P.state = PS_END, P.next = P.stop;
    else P.state = PS_JOIN, P.next = P.part[P.ci + 1].cut;   // par_join set P.ji to that part
}

// At the end of a long run: the ops after the last draw, if this thread walked them.
static void par_end(void) {
    ParSlot *s = P.slot;
    if (s && P.ord == P.n) {
        par_bit(s->gen, P.n, par_gens() != P.g);
        par_bit(s->ev, P.n, vk_spec_events() != P.e);
    }
    P.slot = NULL;
    P.state = PS_SCAN, P.next = P.after;
}

// The replay thread reached P.next. Returns the op to go on at: h, or the end of a part that was taken.
static const op_hdr *par_step(VkExec *ex, const op_hdr *h) {
    switch (P.state) {
    case PS_SCAN: par_scan(ex, h); break;
    case PS_LAUNCH: par_launch(ex); break;
    case PS_JOIN: return par_join(ex, h);
    case PS_CHECK: par_check(ex); break;
    default: par_end(); break;
    }
    return h;
}

// The recording is over: no part is out in a well-formed stream, where the replay thread always reaches the cut.
static void par_close(VkExec *ex) {
    P.next = NULL, P.slot = NULL;
    par_no_split = 0;
    if (!P.live) return;
    vk_spec_fail(VK_SPEC_CANCEL);
    for (int i = 0; i < VK_SPEC_HELPERS; i++) {
        ParPart *p = &P.part[i];
        if (!p->out) continue;
        vk_spec_note(vk_spec_join(i), 0, 0, p->busy_ns);
        p->out = 0;
        views_to_fence(&p->H, ex->fence);
        vk_spec_cb_drop(i);
        PW[i].n = 0;
    }
    P.live = 0;
}

static void par_log(void) {
    static uint64_t since;
    vk_spec_log();
    if (vk_spec_checking())
        rt_log("perf: helper check: %llu parts compared, %llu differ (first in image views %llu, samplers %llu, "
               "image layouts %llu, uniform ranges %llu, uniform buffers %llu, uniform bytes %llu, storage ranges %llu, "
               "storage buffers %llu, pipeline and state %llu, vertex and index buffers %llu, commands %llu)",
               (unsigned long long)P.chk_parts, (unsigned long long)P.chk_bad, (unsigned long long)P.chk_word[0],
               (unsigned long long)P.chk_word[1], (unsigned long long)P.chk_word[2], (unsigned long long)P.chk_word[3],
               (unsigned long long)P.chk_word[4], (unsigned long long)P.chk_word[5], (unsigned long long)P.chk_word[6],
               (unsigned long long)P.chk_word[7], (unsigned long long)P.chk_word[8], (unsigned long long)P.chk_word[9],
               (unsigned long long)P.chk_word[10]);
    for (int i = 0; i < PAR_SLOTS; i++) {
        ParSlot *s = &par_slots[i];
        if (!s->n || s->seen <= since) continue;
        uint32_t ne = 0, ng = 0, e0 = 0, e1 = 0, g0 = 0, g1 = 0;
        for (uint32_t j = 0; j <= s->n; j++) {
            if (par_bit_at(s->ev, j)) e1 = j, e0 = ne++ ? e0 : j;
            if (par_bit_at(s->gen, j)) g1 = j, g0 = ng++ ? g0 : j;
        }
        double t = s->tried ? 1e3 * (double)s->tried : 1;
        char at[64] = "";
        for (uint32_t j = 0, o = 0; s->y && j < s->k && o < sizeof at - 12; j++)
            o += (uint32_t)snprintf(at + o, sizeof at - o, "%s%u", j ? ", " : "", s->c[j]);
        rt_log("perf: helper run of %u draws: parts from draws %s to %u launched after %u, replay share %.2f; %u of %u "
               "taken, helper %.0f us (%.0f on the state before) and %.0f us early, replay waited %.0f us a part; no "
               "part taken in a row %u; helpers cannot hold %u draws (%u to %u), their reads change at %u (%u to %u)",
               s->n, s->y ? at : "none", s->y, s->l, s->share, s->taken, s->tried, (double)s->busy_ns / t,
               (double)s->state_ns / t, (double)s->early_ns / t, (double)s->wait_ns / t, s->fails, ne, e0, e1, ng, g0, g1);
        s->tried = s->taken = 0;
        s->busy_ns = s->state_ns = s->early_ns = s->wait_ns = 0;
    }
    since = P.rec;
}

// ---- sliced passes (DKB_REPLAY_THREADS=N, off by default; docs/parallel-replay.md)
// A render pass whose targets drew at least PAR_MIN_DRAWS draws the last time is recorded in slices. From its first
// direct draw the walk collects the pass's state, bind and draw ops instead of replaying them: it applies the state and
// bind ops to X and starts a slice with a copy of X every PAR_SLICE_DRAWS draws. At the first op of another kind (or
// the end of the submission) each slice is replayed from its copy into a secondary command buffer, and the primary
// executes them in order inside the pass, which began with VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT; the
// pass then ends. This step records the slices on the replaying thread, whatever N is: it checks that slicing leaves
// the frame unchanged, before other threads record them.
// A slice that would end, begin or restart the pass, or record an inline texture command, drops the call and marks a
// hazard: a target set is left unsliced after PAR_BAN_AFTER of those.
int vk_exec_par_slice(void) { return par_slice; }
static int par_collecting(void) { return PC.on; }
void vk_exec_par_hazard(void) { if (!par_hazard) par_hazard = 1; }

static void par_done(void *arg) {   // completion thread: the secondaries of a finished submission can be reused
    ParDone *d = arg;
    for (int i = 0; i < d->n; i++) {
        ParPool *pp = d->cb[i].pp;
        plat_mutex_lock(&pp->lock);
        if (pp->nfree == pp->cfree) pp->free = par_grow(pp->free, &pp->cfree, sizeof *pp->free);
        pp->free[pp->nfree++] = d->cb[i].cb;
        plat_mutex_unlock(&pp->lock);
    }
    free(d);
}

static VkCommandBuffer par_secondary(ParPool *pp) {   // on the thread that owns pp
    VkCtx *c = vk_ctx();
    VkCommandBuffer cb = VK_NULL_HANDLE;
    plat_mutex_lock(&pp->lock);
    if (pp->nfree) cb = pp->free[--pp->nfree];
    plat_mutex_unlock(&pp->lock);
    if (cb) {
        vkResetCommandBuffer(cb, 0);
        return cb;
    }
    if (!pp->pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &pp->pool) != VK_SUCCESS) rt_fatal("vk: secondary command pool failed");
    }
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pp->pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY, .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(c->device, &ai, &cb) != VK_SUCCESS) rt_fatal("vk: secondary command buffer failed");
    return cb;
}

// Inherits the open pass: its attachment formats as vkCmdBeginRendering saw them (a slot without a view: UNDEFINED).
static void par_begin_secondary(const VkExec *ex, VkCommandBuffer cb) {
    VkFormat color[8];
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++) color[i] = ex->color[i].image ? ex->pass_color[i] : VK_FORMAT_UNDEFINED;
    const VkExecTarget *d = &ex->depth;
    VkCommandBufferInheritanceRenderingInfo ir = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
        .colorAttachmentCount = ex->pass_ncolor, .pColorAttachmentFormats = color,
        .depthAttachmentFormat = d->image && (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? ex->pass_depth : VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = d->image && (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? ex->pass_stencil : VK_FORMAT_UNDEFINED,
        .rasterizationSamples = (VkSampleCountFlagBits)(ex->pass_samples ? ex->pass_samples : 1)};
    VkCommandBufferInheritanceInfo ii = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, .pNext = &ir,
                                         .occlusionQueryEnable = PC.oq ? VK_TRUE : VK_FALSE,
                                         .queryFlags = PC.oq ? PC.oq_flags : 0};
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
                                            VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
                                   .pInheritanceInfo = &ii};
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) rt_fatal("vk: secondary command buffer begin failed");
}

static void resolve_rt(VkExec *ex);

// At a draw outside a pass: 1 when the pass it opens is sliced (the pass is then open, with secondary contents).
// TOTK_REPLAY_THREADS (0: off), read once.
static int par_threads(void) {
    if (par_mode < 0) {
        const char *e = cfg_get("TOTK_REPLAY_THREADS");
        par_mode = e && *e ? atoi(e) : 0;
        if (par_mode < 0) par_mode = 0;
        if (par_mode) rt_log("vk: passes of %d or more draws are recorded in slices of up to %d draws into secondary command buffers "
                             "(TOTK_REPLAY_THREADS=%d: recorded by the replaying thread)", PAR_MIN_DRAWS, PAR_SLICE_DRAWS, par_mode);
    }
    return par_mode;
}

static int par_start(VkExec *ex) {
    par_threads();
    // An open pass (a clear opened it, or its first draws were indirect) can still be sliced: it is ended and opened
    // again below. One look per pass. Disabled rendering does not matter: the slices skip their draws as the walk would.
    if (!par_mode || ex != &X || (ex->in_pass && pass_tried)) return 0;
    pass_tried = 1, pass_why = PAR_WHY_NONE;
    if (vk_dlss_wanted()) return 0;   // vk_dlss.c notes the scene pass from its draws, on the replaying thread
    if (vk_dump_armed) {   // the dump follows draws one by one, on one thread
        static int told;
        if (!told++) rt_log("vk: TOTK_DUMP is set: no pass is sliced in this run (TOTK_REPLAY_THREADS has no effect)");
        return 0;
    }
    resolve_rt(ex);
    uint64_t sig = par_sig(ex);
    ParHist *e = par_find(sig, 0);
    if (!e) return pass_why = PAR_WHY_NEW, par_no[0]++, 0;
    if (e->draws < PAR_MIN_DRAWS) return pass_why = PAR_WHY_SMALL, par_no[0]++, 0;
    if (e->banned >= PAR_BAN_AFTER) return pass_why = PAR_WHY_BANNED, par_no[1]++, 0;
    if (ex->pass_noatt) return pass_why = PAR_WHY_NOATT, par_no[2]++, 0;
    if (ex->in_pass) {
        par_restart = 1;
        vk_exec_pass_end(ex);
        par_restart = 0;
        par_restarts++;
    }
    VkQueryControlFlags qf = 0;
    int qopen = 0, q = vk_exec_oq_begin_outside(ex, &qf, &qopen);
    if (q < 0) return pass_why = PAR_WHY_QUERY, pass_tried = 1, par_skipped_oq++, 0;
    pass_secondary = 1;
    vk_exec_pass_begin(ex);
    pass_secondary = 0;
    pass_tried = 1;
    if (!ex->in_pass) return par_no[4]++, vk_exec_oq_pass_ended(ex), 0;
    PC.oq = qopen, PC.oq_flags = qf;
    if (!PC.ops) {
        PC.ops = malloc(sizeof *PC.ops * PAR_MAX_OPS), PC.snap = malloc(sizeof *PC.snap * PAR_MAX_SLICES);
        if (!PC.ops || !PC.snap) rt_fatal("vk: out of memory");
    }
    PC.on = 1, PC.sig = sig, PC.nops = 0, PC.nslices = 0, PC.draws = 0, PC.slice_draws = PAR_SLICE_DRAWS;
    par_t0 = kern_now_ns();
    return 1;
}

static void par_op(VkExec *ex, uint32_t op, const void *a) {   // replay of one collected op
    switch (op) {
    case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET:
    case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART:
    case OP_COPY_STRIDE: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL:
        state_op(ex, op, a);
        break;
    default: vk_exec_op_draw(ex, op, a); break;
    }
}

// The op h while collecting: 1 when it was taken (state and bind ops applied to ex, draws only kept), 0 when it ends
// the collection.
static int par_take(VkExec *ex, const op_hdr *h, const void *a) {
    if (PC.nops == PAR_MAX_OPS) return 0;
    switch (h->op) {
    case OP_SET_RT:
        return !memcmp(&ex->rt, a, sizeof ex->rt);   // the same targets again: the pass goes on
    case OP_WAIT_SYNC:
        return 1;                                     // waited for before the replay (walk)
    case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET:
    case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART:
    case OP_COPY_STRIDE: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL:
        state_op(ex, h->op, a);
        break;
    case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
    case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CONST:
        vk_exec_op_draw(ex, h->op, a);   // binds only change ex
        break;
    case OP_DRAW: case OP_DRAW_IDX:
        if (PC.slice_draws >= PAR_SLICE_DRAWS && PC.nslices < PAR_MAX_SLICES) {
            // A new slice starts at this draw, from ex as it is now; the one before is complete.
            if (PC.nslices) PC.end[PC.nslices - 1] = PC.nops;
            PC.snap[PC.nslices] = *ex;
            PC.start[PC.nslices++] = PC.nops;
            PC.slice_draws = 0;
        }
        PC.slice_draws++, PC.draws++;
        vk_exec_frame.draws_seen++;
        break;
    default:
        return 0;
    }
    PC.ops[PC.nops++] = h;
    return 1;
}

// What recording one slice left: its command buffer, and what the primary's state takes from the slice's copy.
typedef struct {
    ParCb cb;
    uint32_t fb_dirty;
    int recorded, hazard;
    ViewEnt *views;      // views the slice made (freed with the submission)
    int nviews;
} ParOut;

static ParOut par_out[PAR_MAX_SLICES];

// Records slice i into a secondary command buffer of pp, from the slice's copy of the replay state, into w.
static void par_record(int i, ParPool *pp, VkExec *w) {
    VkCommandBuffer cb = par_secondary(pp);
    par_begin_secondary(&PC.snap[i], cb);
    *w = PC.snap[i];
    w->cb = cb, w->recorded = 0, w->dirty = VK_EXEC_DIRTY_ALL;
    // Feedback targets: what earlier slices wrote is not known here, so every attachment counts as written and the
    // slice's first draw that samples one records its by-region barrier (vk_exec_draw.c feedback_barrier).
    w->fb_dirty = 0x1ff;
    w->views = NULL, w->nviews = 0, w->cap_views = 0;   // the slice's own; merged into the primary's after
    vk_exec_draw_slice_begin();
    par_slice = 1, par_hazard = 0;
    for (int k = PC.start[i], end = PC.end[i]; k < end; k++) par_op(w, PC.ops[k]->op, nvn_op_payload(PC.ops[k]));
    par_slice = 0;
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) rt_fatal("vk: secondary command buffer end failed");
    par_out[i] = (ParOut){{pp, cb}, w->fb_dirty, w->recorded, par_hazard, w->views, w->nviews};
}

static ParPool par_main_pool;
static VkExec par_main_w;

static void par_flush(VkExec *ex) {
    if (!ex) {   // TOTK_TRACE=perf (record_impl): what was sliced since the last line
        if (par_mode > 0)
            rt_log("perf: sliced passes %llu with %llu slices and %llu draws; %llu passes not sliced for want of "
                   "inherited queries", (unsigned long long)par_passes, (unsigned long long)par_slices,
                   (unsigned long long)par_sliced_draws, (unsigned long long)par_skipped_oq);
        if (par_mode > 0) {
            char line[400];
            size_t o = 0;
            line[0] = 0;
            for (uint32_t i = 0; i <= OP_COUNT_ && o < sizeof line - 48; i++)
                if (par_end_op[i])
                    o += (size_t)snprintf(line + o, sizeof line - o, " %s %llu", i < OP_COUNT_ ? nvn_op_name(i) : "submission-end",
                                          (unsigned long long)par_end_op[i]);
            rt_log("perf: collections ended by:%s; draws that opened an unsliced pass: small or new targets %llu, banned %llu, "
                   "no attachment %llu, hazards in slices %llu, pass not opened %llu; passes opened again after a clear %llu", line, (unsigned long long)par_no[0],
                   (unsigned long long)par_no[1], (unsigned long long)par_no[2], (unsigned long long)par_no[3],
                   (unsigned long long)par_no[4], (unsigned long long)par_restarts);
        }
        if (par_mode > 0)
            rt_log("perf: sliced passes: collecting %.1f ms, recording slices %.1f ms, executing %.1f ms; draws of unsliced "
                   "passes: under 64 %llu, 64-255 %llu, 256-1023 %llu, 1024 and more %llu", par_t_collect / 1e6,
                   par_t_own / 1e6, par_t_exec / 1e6,
                   (unsigned long long)par_unsliced[0], (unsigned long long)par_unsliced[1],
                   (unsigned long long)par_unsliced[2], (unsigned long long)par_unsliced[3]);
        par_t_collect = par_t_own = par_t_exec = 0;
        memset(par_unsliced, 0, sizeof par_unsliced);
        par_restarts = 0;
        memset(par_end_op, 0, sizeof par_end_op), memset(par_no, 0, sizeof par_no);
        par_passes = par_slices = par_sliced_draws = par_skipped_oq = 0;
        return;
    }
    if (!PC.on) return;
    PC.on = 0;
    par_passes++, par_slices += (uint64_t)PC.nslices, par_sliced_draws += (uint64_t)PC.draws;
    uint64_t t1 = kern_now_ns();
    par_t_collect += t1 - par_t0;
    if (PC.nslices) PC.end[PC.nslices - 1] = PC.nops;
    for (int i = 0; i < PC.nslices; i++) par_record(i, &par_main_pool, &par_main_w);
    uint64_t t3 = kern_now_ns();
    par_t_own += t3 - t1;
    ParDone *done = malloc(sizeof *done + sizeof(ParCb) * (size_t)PC.nslices);
    VkCommandBuffer *cbs = malloc(sizeof *cbs * (size_t)(PC.nslices ? PC.nslices : 1));
    if (!done || !cbs) rt_fatal("vk: out of memory");
    done->n = 0;
    int hazard = 0;
    for (int i = 0; i < PC.nslices; i++) {
        ParOut *o = &par_out[i];
        done->cb[done->n++] = o->cb;
        cbs[i] = o->cb.cb;
        ex->fb_dirty |= o->fb_dirty;
        ex->recorded += o->recorded;
        if (o->hazard > hazard) hazard = o->hazard;
        for (int k = 0; k < o->nviews; k++) {   // the slice's views go with the submission like the primary's
            if (ex->nviews == ex->cap_views) ex->views = par_grow(ex->views, &ex->cap_views, sizeof(ViewEnt));
            ((ViewEnt *)ex->views)[ex->nviews++] = o->views[k];
        }
        free(o->views);
    }
    rec_drain();   // the recorder has made every earlier call into the primary
    if (PC.nslices) vkCmdExecuteCommands(ex->cb, (uint32_t)PC.nslices, cbs);
    free(cbs);
    ex->recorded++;
    vk_completion_add(ex->fence, par_done, done);
    par_t_exec += kern_now_ns() - t3;
    vk_exec_draw_slice_begin();   // the primary's bindings are undefined after vkCmdExecuteCommands
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_pass_end(ex);         // a pass of secondary command buffers takes nothing else
    // The sliced draws count towards the next command buffer split (maybe_split), which the walk asks for after this
    // op: the GPU can start on a big pass while the rest of the submission is replayed. Collected draws skipped the
    // walk's count, so a submission of big sliced passes went to the queue in one or two parts, and the game, which
    // waits for the GPU within the frame, waited for the whole replay and then the whole GPU work.
    split_draws += PC.draws;
    if (hazard) {
        ParHist *e = par_find(PC.sig, 1);
        // A feedback target learned inside a slice is one: its pass restart was dropped, and the next pass attaches
        // the target in the feedback layout, so a few hazards are expected; a set that keeps having them is left alone.
        par_no[3]++;
        if (e->banned < 255 && ++e->banned == PAR_BAN_AFTER)
            rt_log("vk: a sliced pass needed a pass or barrier change inside a slice %d times: those targets are not sliced again",
                   PAR_BAN_AFTER);
    }
}

static void walk(VkExec *ex, const op_hdr *h, int depth) {
    for (;;) {
        while (h == P.next && !depth) h = par_step(ex, h);
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        if (h->op != OP_END && h->op != OP_BEGIN && h->op != OP_JUMP && h->op != OP_CALL) {
            if (PC.on) {
                if (par_take(ex, h, a)) {
                    if (h->op == OP_DRAW || h->op == OP_DRAW_IDX) draws_total++, pass_draws++;
                    h = nvn_op_next(h);
                    continue;
                }
                par_end_op[h->op]++;
                par_flush(ex);
            } else if ((h->op == OP_DRAW || h->op == OP_DRAW_IDX) && par_start(ex) && par_take(ex, h, a)) {
                draws_total++, pass_draws++;
                h = nvn_op_next(h);
                continue;
            }
        }
        switch (h->op) {
        case OP_END: return;
        case OP_BEGIN: break;
        case OP_JUMP: h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target; continue;
        case OP_CALL:
            if (depth >= CALL_DEPTH) rt_fatal("vk: command buffer calls nested deeper than %d", CALL_DEPTH);
            walk(ex, nvn_op_first(((const p_jump *)a)->target), depth + 1);
            break;
        case OP_SET_RT: set_rt(ex, a); break;
        case OP_CLEAR_COLOR: clear_color(ex, a); break;
        case OP_CLEAR_DS: clear_ds(ex, a); break;
        case OP_BARRIER: vk_exec_barrier(ex); break;
        case OP_WAIT_SYNC: break;   // submit preparation waited before any replay/snapshot; source may be gone
        case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: case OP_UPDATE_UB: case OP_DRAW_IDX_IND: case OP_DISPATCH_IND:
            if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND) {
                vk_exec_frame.draws_seen++, split_draws++;
                if (P.slot) par_watch();
                if (par_no_split) vk_exec_draw_digest_draw(ex);
            }
            if (h->op == OP_DISPATCH || h->op == OP_DISPATCH_IND) vk_exec_frame.dispatches++, split_draws++;
            draws_total += h->op != OP_UPDATE_UB;

        case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
        case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CONST:
            vk_exec_op_draw(ex, h->op, a);
            break;
        case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX:
            if (h->op != OP_CLEAR_TEX) vk_exec_frame.copies++;
            if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
            // fall through
        case OP_SIGNAL_EVENT: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND:
        case OP_CLEAR_BUF:
            vk_exec_op_xfer(ex, h->op, a);
            break;
        default:
            if (!state_op(ex, h->op, a)) rt_fatal("vk: op %s has no replay handler", nvn_op_name(h->op));
        }
        if ((h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND) && ex->in_pass) pass_draws++;
        if (split_draws >= split_n) maybe_split(ex);
        h = nvn_op_next(h);
    }
}

static int record_impl(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks,
                       VkExecWorkFn work, void *context) {
    VkExec *ex = &X;
    if (!x_ready) state_reset(ex), vk_dump_init(), x_ready = 1;
    ex->cb = cb;
    ex->fence = fence;
    ex->submitted_ticks = submitted_ticks;
    ex->recorded = 0;
    ex->in_pass = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->rt_resolved = 0;   // images may have changed since the last command buffer
    ex->nviews = 0;
    ex->barrier_at = -1;
    ex->nfbl = 0;   // the previous command buffer ended with a full barrier: every image is in GENERAL
    vk_exec_frame.submits++;
    split_draws = part_start = 0;
    part_t0 = kern_now_ns();
    x_here = 1;
    int memory = n || work;
    if (memory) vk_mem_mirror_begin(), vk_usnap_begin(handles, n, fence);
    P.next = NULL;
    if (work) work(ex, context);
    else {
        // Shadow storage notes every GPU write at record time, in order: no helper there.
        // Sliced passes and helper runs both record a submission's draws on other threads, from other copies of the
        // draw path's state: with TOTK_REPLAY_THREADS set, the helpers of TOTK_REPLAY_PAR take no run.
        int par = vk_spec_on() && !vk_mem_shadow_on() && !par_threads();
        if (vk_spec_on() && par_mode > 0) {
            static int told;
            if (!told++) rt_log("vk: TOTK_REPLAY_THREADS is set: the helper threads of TOTK_REPLAY_PAR take no run");
        }
        P.rec += par;
        for (int i = 0; i < n; i++) {
            if (!handles[i]) rt_fatal("vk: command handle %d of a submission is null", i);
            const op_hdr *first = nvn_op_first(handles[i]);
            if (par) P.state = PS_SCAN, P.next = first;
            walk(ex, first, 0);
        }
        par_close(ex);
    }
    if (par_collecting()) par_end_op[OP_COUNT_]++;
    par_flush(ex);
    vk_exec_pass_end(ex);
    x_here = 0;
    if (ex->recorded) vk_exec_barrier(ex);   // later submissions see this one's writes
    if (memory) ex->recorded += vk_mem_mirror_end(ex->cb, fence), vk_usnap_end(fence);   // VRAM mirror bytes the GPU wrote go back to guest memory
    views_to_fence(ex, fence);
    ex->cb = VK_NULL_HANDLE;
    if (memory) n_subs++;
    if ((rt_trace & RT_TRACE_PERF) && n_subs >= 240) {
        rt_log("perf: replay command buffer parts per submission %.2f", (double)(n_subs + n_parts) / (double)n_subs);
        par_flush(NULL);   // its perf line
        n_subs = n_parts = 0;
        par_log();
    }
    return ex->recorded - part_start;   // what the last part holds
}

int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks) {
    return record_impl(cb, handles, n, fence, submitted_ticks, NULL, NULL);
}

int vk_exec_record_work(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks, VkExecWorkFn work, void *context) {
    if (!work) rt_fatal("vk: work recording without a callback");
    return record_impl(cb, NULL, 0, fence, submitted_ticks, work, context);
}
