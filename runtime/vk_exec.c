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
#define PAR_HIST 64
enum { PAR_WHY_UNSEEN, PAR_WHY_NONE, PAR_WHY_NEW, PAR_WHY_SMALL, PAR_WHY_BANNED, PAR_WHY_NOATT, PAR_WHY_QUERY };
#define PAR_BAN_AFTER 8   // hazards after which a target set is not sliced again

typedef struct { uint64_t sig; uint32_t draws; uint8_t banned; } ParHist;
static ParHist par_hist[PAR_HIST];
static int par_mode = -1;                       // DKB_REPLAY_THREADS: 0 off, else the threads (stage A: 1)
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
static struct {
    int on;
    uint64_t sig;
    const op_hdr **ops;                          // the collected ops in order
    int nops, cops;
    VkExec *snap;                                // X before each slice's first op
    int *start;                                  // each slice's first op
    int nslices, cslices;
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

static ParHist *par_find(uint64_t sig, int add) {
    ParHist *e = &par_hist[sig % PAR_HIST];
    if (e->sig == sig) return e;
    if (!add) return NULL;
    *e = (ParHist){sig, 0, 0};
    return e;
}


// ---- TOTK_TRACE=gfx frame counters

VkExecFrame vk_exec_frame;
static struct { const uint8_t *mem; uint32_t w, h, fmt, n; } fw[64];   // textures written this frame
static int nfw;

uint64_t vk_exec_recording_fence(void) { return x_here || par_slice ? X.fence : 0; }   // slices: workers too

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
    if (full) {
        if (!warned++) rt_log("vk: more than %d 3D BC render targets: the others are left out", BC_STANDINS);
        return 0;
    }
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
    vk_exec_frame.fb_layouts += n;
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
        if ((i < 8 && i >= ex->pass_ncolor) || !t->image || !t->img || !t->img->fb_learned) continue;
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
    vk_exec_frame.passes++;
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
        if (par_mode > 0 && !pass_sliced && pass_draws >= PAR_SLICE_DRAWS && told < 24) {
            static const char *const why[] = {"no direct draw reached par_start", "not refused", "targets not seen before",
                                              "targets drew little before", "banned", "no attachment", "queries"};
            ParHist *h = par_find(sig, 0);
            told++;
            rt_log("vk: a pass of %u draws was not sliced: %s (targets %016llx, slot %u, kept count %u, slot holds %016llx)",
                   pass_draws, why[pass_why], (unsigned long long)sig, (unsigned)(sig % PAR_HIST), h ? h->draws : 0,
                   (unsigned long long)par_hist[sig % PAR_HIST].sig);
        }
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
    if (vk_dump_armed) vk_dump_pass_end(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    for (int i = 0; i < 8; i++)
        if (ex->color[i].image) vk_texture_gpu_written(ex->color[i].tex), tfr_note(&ex->color[i], stamp);
    if (ex->depth.image) vk_texture_gpu_written(ex->depth.tex), tfr_note(&ex->depth, stamp);
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
    vk_exec_frame.clears++;
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
        vk_texture_gpu_written(t->tex);
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
        vk_texture_gpu_written(t->tex);
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
// (vkQueueSubmit2, a command buffer, the mirror prologue), and a fight frame of 2,000 draws made 15 parts on the 7900X;
// a millisecond of recording is a fraction of a millisecond of GPU work, so the GPU still starts early.
static int split_n = -1, split_draws, part_start;
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
    if (!split_n || ex->in_pass || split_draws < split_n) return;
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

// The fixed-function and pool state ops: they only change VkExec (the draws record what they set).
static void state_op(VkExec *ex, uint32_t op, const void *a) {
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
    default: break;
    }
}

// ---- parallel passes (DKB_REPLAY_THREADS, docs/parallel-replay.md)
// A render pass whose targets drew at least PAR_MIN_DRAWS draws the last time is recorded in slices: from its first
// draw the walk collects the pass's state, bind and draw ops instead of replaying them, applying only the state and bind
// ops to X and keeping a copy of X every PAR_SLICE_DRAWS draws. When an op that is none of those comes (or the
// submission ends), each slice is replayed from its copy into a secondary command buffer, and the primary executes them
// in order inside the pass, which began with VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT; the pass then ends.
// Stage A of the plan: the slices are recorded one after another on the replaying thread, to check that slicing keeps
// the frame the same before workers record them at once. A slice that would end or restart the pass, or record a full
// barrier (a feedback target learned, an inline texture command) is a hazard: the call is dropped, and the targets are
// not sliced again.
int vk_exec_par_slice(void) { return par_slice; }
static int par_collecting(void) { return PC.on; }
void vk_exec_par_hazard(void) { par_hazard = 1; }

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
static int par_start(VkExec *ex) {
    if (par_mode < 0) {
        const char *e = cfg_get("TOTK_REPLAY_THREADS");
        par_mode = e && *e ? atoi(e) : 0;
        if (par_mode < 0) par_mode = 0;
        if (par_mode) rt_log("vk: passes of %d or more draws are recorded in slices of %d into secondary command buffers "
                             "(TOTK_REPLAY_THREADS=%d: recorded by that many threads)", PAR_MIN_DRAWS, PAR_SLICE_DRAWS, par_mode);
    }
    // An open pass (a clear opened it, or its first draws were indirect) can still be sliced: it is ended and opened
    // again below. One look per pass. Disabled rendering does not matter: the slices skip their draws as the walk would.
    if (!par_mode || ex != &X || (ex->in_pass && pass_tried)) return 0;
    pass_tried = 1, pass_why = PAR_WHY_NONE;
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
    PC.on = 1, PC.sig = sig, PC.nops = 0, PC.nslices = 0, PC.draws = 0, PC.slice_draws = PAR_SLICE_DRAWS;
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
        if (PC.slice_draws >= PAR_SLICE_DRAWS) {   // a new slice starts at this draw, from ex as it is now
            if (PC.nslices == PC.cslices) {
                int c = PC.cslices;
                PC.snap = par_grow(PC.snap, &c, sizeof *PC.snap);
                PC.start = realloc(PC.start, (size_t)c * sizeof *PC.start);
                if (!PC.start) rt_fatal("vk: out of memory");
                PC.cslices = c;
            }
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
    if (PC.nops == PC.cops) PC.ops = par_grow(PC.ops, &PC.cops, sizeof *PC.ops);
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

static ParOut *par_out;
static int cpar_out;

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
    int end = i + 1 < PC.nslices ? PC.start[i + 1] : PC.nops;
    for (int k = PC.start[i]; k < end; k++) par_op(w, PC.ops[k]->op, nvn_op_payload(PC.ops[k]));
    par_slice = 0;
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) rt_fatal("vk: secondary command buffer end failed");
    par_out[i] = (ParOut){{pp, cb}, w->fb_dirty, w->recorded, par_hazard, w->views, w->nviews};
}

// ---- workers (DKB_REPLAY_THREADS=N: N-1 of them; the submitting thread records slices too)
#define PAR_MAX_WORKERS 15
typedef struct {
    PlatThread *t;
    void *dctx;          // vk_exec_draw_ctx_new
    ParPool pp;
    VkExec w;
    uint32_t seen;
} ParWorker;
static ParWorker *par_workers[PAR_MAX_WORKERS];
static int par_nworkers;
static PlatMutex pw_lock = PLAT_MUTEX_INIT;
static PlatCond pw_go, pw_idle;
static uint32_t pw_gen;                          // bumped per flush (pw_lock)
static volatile uint64_t pw_job;                 // (flush generation << 32) | the next slice to take
static volatile int32_t pw_n, pw_left;           // slices of the flush; slices not recorded yet
static ParPool par_main_pool;
static VkExec par_main_w;

// Takes and records slices of flush gen until none is left. A thread that comes late (its flush is over, or another
// one began) finds another generation in pw_job and takes nothing.
static void par_take_slices(uint32_t gen, ParPool *pp, VkExec *w) {
    for (;;) {
        uint64_t v = __atomic_load_n(&pw_job, __ATOMIC_ACQUIRE);
        if ((uint32_t)(v >> 32) != gen || (int32_t)(uint32_t)v >= __atomic_load_n(&pw_n, __ATOMIC_ACQUIRE)) return;
        if (!__atomic_compare_exchange_n(&pw_job, &v, v + 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
        par_record((int)(uint32_t)v, pp, w);
        if (__atomic_sub_fetch(&pw_left, 1, __ATOMIC_ACQ_REL) == 0) {
            plat_mutex_lock(&pw_lock);
            plat_cond_broadcast(&pw_idle);
            plat_mutex_unlock(&pw_lock);
        }
    }
}

static void par_worker(void *arg) {
    ParWorker *pw = arg;
    vk_exec_draw_ctx_use(pw->dctx);
    for (;;) {
        plat_mutex_lock(&pw_lock);
        while (pw->seen == pw_gen) plat_cond_wait(&pw_go, &pw_lock);
        uint32_t gen = pw->seen = pw_gen;
        plat_mutex_unlock(&pw_lock);
        par_take_slices(gen, &pw->pp, &pw->w);
    }
}

static void par_start_workers(void) {
    int want = par_mode - 1;
    if (want > PAR_MAX_WORKERS) want = PAR_MAX_WORKERS;
    if (par_nworkers >= want) return;
    if (!par_nworkers) plat_cond_init(&pw_go), plat_cond_init(&pw_idle);
    while (par_nworkers < want) {
        ParWorker *pw = calloc(1, sizeof *pw);
        if (!pw) rt_fatal("vk: out of memory");
        pw->dctx = vk_exec_draw_ctx_new();   // here, while this thread records nothing
        pw->seen = pw_gen;
        if (!(pw->t = plat_thread_new_ex(par_worker, pw, "replay slices", 0))) {
            rt_log("vk: a replay slice worker could not start: %d of %d run", par_nworkers, want);
            free(pw);
            par_mode = par_nworkers + 1;
            return;
        }
        par_workers[par_nworkers++] = pw;
    }
    rt_log("vk: %d replay slice workers started", par_nworkers);
}

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
        par_restarts = 0;
        memset(par_end_op, 0, sizeof par_end_op), memset(par_no, 0, sizeof par_no);
        par_passes = par_slices = par_sliced_draws = par_skipped_oq = 0;
        return;
    }
    if (!PC.on) return;
    PC.on = 0;
    par_passes++, par_slices += (uint64_t)PC.nslices, par_sliced_draws += (uint64_t)PC.draws;
    if (PC.nslices > cpar_out) {
        free(par_out);
        cpar_out = PC.nslices * 2;
        if (!(par_out = calloc((size_t)cpar_out, sizeof *par_out))) rt_fatal("vk: out of memory");
    }
    if (par_mode > 1 && PC.nslices > 1) par_start_workers();
    plat_mutex_lock(&pw_lock);
    uint32_t gen = ++pw_gen;
    __atomic_store_n(&pw_n, PC.nslices, __ATOMIC_RELEASE);
    __atomic_store_n(&pw_left, PC.nslices, __ATOMIC_RELEASE);
    __atomic_store_n(&pw_job, (uint64_t)gen << 32, __ATOMIC_RELEASE);
    if (par_nworkers && PC.nslices > 1) plat_cond_broadcast(&pw_go);
    plat_mutex_unlock(&pw_lock);
    par_take_slices(gen, &par_main_pool, &par_main_w);
    plat_mutex_lock(&pw_lock);
    while (__atomic_load_n(&pw_left, __ATOMIC_ACQUIRE) > 0) plat_cond_wait(&pw_idle, &pw_lock);
    plat_mutex_unlock(&pw_lock);
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
        hazard |= o->hazard;
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
    vk_exec_draw_slice_begin();   // the primary's bindings are undefined after vkCmdExecuteCommands
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_pass_end(ex);         // a pass of secondary command buffers takes nothing else
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
        case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET:
        case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART:
        case OP_COPY_STRIDE: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL:
            state_op(ex, h->op, a);
            break;
        case OP_BARRIER: vk_exec_barrier(ex); break;
        case OP_WAIT_SYNC: break;   // submit preparation waited before any replay/snapshot; source may be gone
        case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: case OP_UPDATE_UB: case OP_DRAW_IDX_IND: case OP_DISPATCH_IND:
            if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND) vk_exec_frame.draws_seen++, split_draws++;
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
        default: rt_fatal("vk: op %s has no replay handler", nvn_op_name(h->op));
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
    if (work) work(ex, context);
    else {
        for (int i = 0; i < n; i++) {
            if (!handles[i]) rt_fatal("vk: command handle %d of a submission is null", i);
            walk(ex, nvn_op_first(handles[i]), 0);
        }
    }
    if (par_collecting()) par_end_op[OP_COUNT_]++;
    par_flush(ex);
    vk_exec_pass_end(ex);
    x_here = 0;
    if (ex->recorded) vk_exec_barrier(ex);   // later submissions see this one's writes
    if (memory) ex->recorded += vk_mem_mirror_end(ex->cb, fence), vk_usnap_end(fence);   // VRAM mirror bytes the GPU wrote go back to guest memory
    if (ex->nviews) {
        ViewList *l = malloc(sizeof *l + sizeof(VkImageView) * (size_t)ex->nviews);
        if (!l) rt_fatal("vk: out of memory");
        l->n = ex->nviews;
        for (int i = 0; i < ex->nviews; i++) l->v[i] = ((ViewEnt *)ex->views)[i].view;
        vk_completion_add(fence, free_views, l);
        ex->nviews = 0;
    }
    ex->cb = VK_NULL_HANDLE;
    if (memory) n_subs++;
    if ((rt_trace & RT_TRACE_PERF) && n_subs >= 240) {
        rt_log("perf: replay command buffer parts per submission %.2f", (double)(n_subs + n_parts) / (double)n_subs);
        par_flush(NULL);   // its perf line
        n_subs = n_parts = 0;
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
