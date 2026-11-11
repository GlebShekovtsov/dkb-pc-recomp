// Frame generation (TOTK_FRAME_GEN = 2x; off by default, experimental): a frame made between each two of the game's, from
// their motion, shown first, then the game's own frame half an interval later. A C rewrite of Hookshot's
// gpu/vk/vk_framegen.cpp (https://github.com/NotRexed/Hookshot @ fab8cb37641714bd839ae83c2a7bc582bedc4d70, GPL-3.0) with
// its shaders runtime/shaders/fg_*.comp:
//   1. the crop of the window texture is copied into an RGBA8 history image (the game draws over its texture next frame)
//      and reduced to quarter-size luma;
//   2. block matching of the luma against the frame before: a vector per 16x16 pixels within +-32 pixels, then a 3x3 median;
//   3. the frame at t = 0.5 blends the last frame moved back and the new one forward along the vectors.
// The present thread (vk_backend.c present_main) calls vk_fg_record with the game's texture before its blit, then, when a
// made frame exists, vk_fg_show in place of the blit: the made frame goes through vk_present_image, the history copy of
// the game's frame through vk_present_blit, and its flip waits until half an interval after the made one. Both go
// through the whole present chain (colour, FSR, SMAA, HDR, overlay), so the overlay is never interpolated. The game's
// texture is read only by the command buffer vk_fg_record submits, and the swapchain is touched only through
// vk_present_image and vk_present_flip.
// Any failure turns the pass off for the session with one log line, and the present goes on as without it.
#include "vk.h"

#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "fg_flow_spv.h"
#include "fg_luma_spv.h"
#include "fg_median_spv.h"
#include "fg_pace.h"
#include "fg_synth_spv.h"
#include "flight.h"
#include "host.h"
#include "platform/platform.h"
#include "ui/menu.h"   // gfx_window_present_mode

#define HIST_FMT VK_FORMAT_R8G8B8A8_UNORM

typedef struct {
    VkImage img;
    VmaAllocation alloc;
    VkImageView view;
} Image;

// A compute pipeline over `samplers` combined image samplers (bindings 0..) and one storage image (the next binding),
// pushed with push descriptors, and `push` bytes of push constants.
typedef struct {
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    uint32_t samplers;
} Pipe;

static struct {
    int tried, dead, said;   // said: bits of the reasons already logged (SAY_*)
    PFN_vkCmdPushDescriptorSetKHR push;
    Pipe luma, flow, median, synth;
    VkSampler nearest, linear;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    Image hist[2], luma_img[2], flow_img[2], made;
    uint32_t w, h;        // the crop size the images were made for; 0 = none
    VkFormat fmt_ok;      // the last window texture format found to have BLIT_SRC
    VkFormat fmt_bad;     // the last one found without it
    int cur;              // the history image the next frame is copied into; the other holds the frame before
    int shown;            // the history image holding the newest frame (what vk_fg_show presents as the real one)
    int have_prev, fresh;
    uint64_t last_at;     // when the previous job reached vk_fg_record; 0 = none
    int64_t interval;     // the game's frame interval, fg_pace_interval
    // the 5 s log
    uint64_t t0, real, made_n, no_hist, hitch, over, crop, cut, late;
} S;
// S and the images are touched by vk_fg_record (the present thread) and vk_fg_release (the exit, under vk_present.c's g_pl
// first: that order everywhere), so each holds this lock for its whole run; vk_fg_show copies what it needs and presents
// without it.
static PlatMutex fgl = PLAT_MUTEX_INIT;
enum { SAY_HDR = 1, SAY_MODE = 2, SAY_GAME = 4, SAY_FMT = 8, SAY_BLIT = 16 };

static void say(int bit, const char *msg, int arg) {
    if (S.said & bit) return;
    S.said |= bit;
    rt_log(msg, arg);
}

static VkResult make_pipe(VkCtx *c, Pipe *p, const uint32_t *code, size_t size, uint32_t samplers, uint32_t push) {
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i <= samplers; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    b[samplers].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = samplers + 1, .pBindings = b};
    VkResult r = vkCreateDescriptorSetLayout(c->device, &dci, NULL, &p->dsl);
    if (r != VK_SUCCESS) return r;
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &p->dsl, .pushConstantRangeCount = push ? 1 : 0, .pPushConstantRanges = &pr};
    if ((r = vkCreatePipelineLayout(c->device, &li, NULL, &p->layout)) != VK_SUCCESS) return r;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    if ((r = vkCreateShaderModule(c->device, &mi, NULL, &m)) != VK_SUCCESS) return r;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = p->layout};
    r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    p->samplers = samplers;
    return r;
}

static void free_pipe(VkCtx *c, Pipe *p) {
    if (p->pipe) vkDestroyPipeline(c->device, p->pipe, NULL);
    if (p->layout) vkDestroyPipelineLayout(c->device, p->layout, NULL);
    if (p->dsl) vkDestroyDescriptorSetLayout(c->device, p->dsl, NULL);
    memset(p, 0, sizeof *p);
}

static void free_image(VkCtx *c, Image *i) {
    if (i->view) vkDestroyImageView(c->device, i->view, NULL);
    if (i->img) vmaDestroyImage(c->vma, i->img, i->alloc);
    memset(i, 0, sizeof *i);
}

static int make_image(VkCtx *c, Image *i, VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &ai, &i->img, &i->alloc, NULL) != VK_SUCCESS) {
        i->img = VK_NULL_HANDLE;
        return 0;
    }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = i->img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &i->view) != VK_SUCCESS) {
        i->view = VK_NULL_HANDLE;
        free_image(c, i);
        return 0;
    }
    return 1;
}

static void free_images(VkCtx *c) {
    for (int i = 0; i < 2; i++) free_image(c, &S.hist[i]), free_image(c, &S.luma_img[i]), free_image(c, &S.flow_img[i]);
    free_image(c, &S.made);
    S.w = S.h = 0;
    S.have_prev = 0;
}

// Everything the pass made, for the session's end or a failed set-up. The device is idle or nothing was submitted.
static void free_all(VkCtx *c) {
    free_images(c);
    free_pipe(c, &S.luma), free_pipe(c, &S.flow), free_pipe(c, &S.median), free_pipe(c, &S.synth);
    if (S.nearest) vkDestroySampler(c->device, S.nearest, NULL);
    if (S.linear) vkDestroySampler(c->device, S.linear, NULL);
    if (S.fence) vkDestroyFence(c->device, S.fence, NULL);
    if (S.pool) vkDestroyCommandPool(c->device, S.pool, NULL);   // frees the command buffer
    S.nearest = S.linear = VK_NULL_HANDLE, S.fence = VK_NULL_HANDLE, S.pool = VK_NULL_HANDLE, S.cb = VK_NULL_HANDLE;
}

static int fail(VkCtx *c, const char *what) {
    rt_log("fg: %s: off", what);
    S.dead = 1;
    vk_queue_lock();   // an earlier present may still read the images
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    free_all(c);
    return 0;
}

static int init(VkCtx *c) {
    S.tried = 1;
    S.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    if (!S.push) return fail(c, "no push descriptors");
    // storage and sampled use of what the passes make, with linear filtering where the linear sampler reads
    const struct { VkFormat f; VkFormatFeatureFlags need; } fmts[] = {
        {VK_FORMAT_R16_SFLOAT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT},
        {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT},
        {HIST_FMT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                       VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                       VK_FORMAT_FEATURE_BLIT_DST_BIT},
    };
    for (size_t i = 0; i < sizeof fmts / sizeof *fmts; i++) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(c->phys, fmts[i].f, &fp);
        if ((fp.optimalTilingFeatures & fmts[i].need) != fmts[i].need) {
            char msg[96];
            snprintf(msg, sizeof msg, "format %d lacks storage, sampled or blit support", (int)fmts[i].f);
            return fail(c, msg);
        }
    }
    const struct { Pipe *p; const char *name; const uint32_t *code; size_t size; uint32_t samplers, push; } pipes[] = {
        {&S.luma, "luma", fg_luma_spv, sizeof fg_luma_spv, 1, 0},
        {&S.flow, "flow", fg_flow_spv, sizeof fg_flow_spv, 2, 0},
        {&S.median, "median", fg_median_spv, sizeof fg_median_spv, 1, 0},
        {&S.synth, "synth", fg_synth_spv, sizeof fg_synth_spv, 3, 4},
    };
    for (size_t i = 0; i < sizeof pipes / sizeof *pipes; i++) {
        VkResult r = make_pipe(c, pipes[i].p, pipes[i].code, pipes[i].size, pipes[i].samplers, pipes[i].push);
        if (r != VK_SUCCESS) {
            char msg[64];
            snprintf(msg, sizeof msg, "pipeline %s not made (%d): off for this session", pipes[i].name, (int)r);
            rt_log("fg: %s", msg);
            S.dead = 1;
            free_all(c);
            return 0;
        }
    }
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f};
    VkSamplerCreateInfo li = si;
    li.magFilter = li.minFilter = VK_FILTER_LINEAR;
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    VkResult r = vkCreateSampler(c->device, &si, NULL, &S.nearest);
    if (r == VK_SUCCESS) r = vkCreateSampler(c->device, &li, NULL, &S.linear);
    if (r == VK_SUCCESS) r = vkCreateCommandPool(c->device, &pi, NULL, &S.pool);
    if (r == VK_SUCCESS) {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = S.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        r = vkAllocateCommandBuffers(c->device, &ai, &S.cb);
    }
    if (r == VK_SUCCESS) r = vkCreateFence(c->device, &fi, NULL, &S.fence);
    if (r != VK_SUCCESS) {
        char msg[64];
        snprintf(msg, sizeof msg, "set-up failed (%d)", (int)r);
        return fail(c, msg);
    }
    return 1;
}

// Images for a w x h crop. A new size waits for the device (the earlier jobs' images may still be read) and starts the
// history again.
static int ensure_images(VkCtx *c, uint32_t w, uint32_t h) {
    if (S.w == w && S.h == h) return 1;
    if (S.w) {   // earlier jobs' images may still be read; the first allocation has nothing to wait for
        vk_queue_lock();   // vkDeviceWaitIdle needs every queue left alone
        vkDeviceWaitIdle(c->device);
        vk_queue_unlock();
    }
    free_images(c);
    uint32_t qw = (w + 3) / 4, qh = (h + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    const VkImageUsageFlags work = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    int ok = make_image(c, &S.made, HIST_FMT, w, h, work | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    for (int i = 0; i < 2 && ok; i++)
        ok = make_image(c, &S.hist[i], HIST_FMT, w, h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
             make_image(c, &S.luma_img[i], VK_FORMAT_R16_SFLOAT, qw, qh, work) &&
             make_image(c, &S.flow_img[i], VK_FORMAT_R16G16B16A16_SFLOAT, cw, ch, work);
    if (!ok) {
        char msg[64];
        snprintf(msg, sizeof msg, "images %ux%u not made", w, h);
        return fail(c, msg);
    }
    S.w = w, S.h = h, S.fresh = 1;
    rt_log("fg: frame generation 2x on (experimental): made at %ux%u, history RGBA8", w, h);
    return 1;
}

static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 m = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
                          .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m};
    vkCmdPipelineBarrier2(cb, &d);
}

// The first command buffer after (re)allocation moves every image to GENERAL, where they stay.
static void to_general(VkCommandBuffer cb) {
    const Image *all[] = {&S.hist[0], &S.hist[1], &S.luma_img[0], &S.luma_img[1], &S.flow_img[0], &S.flow_img[1], &S.made};
    VkImageMemoryBarrier2 b[7];
    for (int i = 0; i < 7; i++)
        b[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                       .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = all[i]->img,
                                       .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 7, .pImageMemoryBarriers = b};
    vkCmdPipelineBarrier2(cb, &d);
}

// One dispatch over w x h in 8x8 groups: `in` are the sampled inputs (bindings 0..), out the storage image after them.
static void pass(VkCommandBuffer cb, const Pipe *p, const VkDescriptorImageInfo *in, VkImageView out, uint32_t w,
                 uint32_t h, const void *push, uint32_t push_bytes) {
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet ws[4];
    for (uint32_t i = 0; i <= p->samplers; i++)
        ws[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1,
                                       .descriptorType = i < p->samplers ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                         : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .pImageInfo = i < p->samplers ? &in[i] : &oi};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, p->samplers + 1, ws);
    if (push_bytes) vkCmdPushConstants(cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
}

static void log_stats(uint64_t now) {
    if (!S.t0) S.t0 = now;
    if (now - S.t0 < 5000000000ull) return;
    double s = (double)(now - S.t0) / 1e9;
    rt_log("fg: %.1f s: real %.1f fps, shown %.1f fps, made %u; skipped: no history %u, hitch %u, over refresh %u, crop %u; "
           "real cut short %u; late >2 ms %u", s, (double)S.real / s, (double)(S.real + S.made_n) / s, (unsigned)S.made_n,
           (unsigned)S.no_hist, (unsigned)S.hitch, (unsigned)S.over, (unsigned)S.crop, (unsigned)S.cut, (unsigned)S.late);
    S.t0 = now, S.real = S.made_n = S.no_hist = S.hitch = S.over = S.crop = S.cut = S.late = 0;
}

static int record(VkImage src, VkFormat fmt, int w, int h, const int crop[4]) {
    VkCtx *c = vk_ctx();
    if (S.dead || !c) return 0;
    if (gfx_window_closing()) return 0;   // the exit is about to free everything below
    if (strcmp(cfg_value("TOTK_FRAME_GEN"), "2x")) {   // off, or turned off while running: hand the images back
        if (S.w) {
            vk_queue_lock();
            vkDeviceWaitIdle(c->device);
            vk_queue_unlock();
            free_images(c);
        }
        S.last_at = 0;
        return 0;
    }
    // the crop as vk_present_image reads it
    int cx = 0, cy = 0, cw = w, ch = h;
    if (crop && crop[2] > 0 && crop[3] > 0) cx = crop[0], cy = crop[1], cw = crop[2], ch = crop[3];
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx + cw > w) cw = w - cx;
    if (cy + ch > h) ch = h - cy;
    if (cw <= 0 || ch <= 0) return 0;
    // Where no frames are made (one log line each): the HDR picture (PQ code values: not for a blend); TotK, whose
    // back-edge pacing (vk_backend.c present_hold) the paced show replaces; window textures that are not 8-bit UNORM; and
    // FIFO, where the two acquires and presents of a job would block the present thread and halve the game's rate.
    const char *mode = gfx_window_present_mode();
    int skip = 0;
    if (fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32 && hdr_native_on()) skip = 1, say(SAY_HDR, "fg: native HDR picture: no made frames", 0);
    else if (rt_game_version != 0) skip = 1, say(SAY_GAME, "fg: not Bananza: no made frames", 0);
    else if (fmt != VK_FORMAT_R8G8B8A8_UNORM && fmt != VK_FORMAT_B8G8R8A8_UNORM)
        skip = 1, say(SAY_FMT, "fg: window texture format %d is not 8-bit UNORM: no made frames", (int)fmt);
    else if (!*mode) skip = 1;   // no swapchain yet
    else if (strcmp(mode, "mailbox") && strcmp(mode, "immediate"))
        skip = 1, say(SAY_MODE, "fg: present mode fifo: no made frames (needs mailbox or immediate)", 0);
    if (skip) {
        S.have_prev = 0, S.last_at = 0;
        return 0;
    }
    if (!S.tried && !init(c)) return 0;
    if (fmt != S.fmt_ok) {
        if (fmt != S.fmt_bad) {
            VkFormatProperties fp;
            vkGetPhysicalDeviceFormatProperties(c->phys, fmt, &fp);
            S.fmt_bad = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ? 0 : fmt;
        }
        if (fmt == S.fmt_bad) {   // this texture only: the frame goes out plain
            say(SAY_BLIT, "fg: format %d has no blit source: no made frames", (int)fmt);
            S.have_prev = 0, S.last_at = 0;
            return 0;
        }
        S.fmt_ok = fmt;
    }
    uint64_t now = kern_now_ns();
    int64_t gap = S.last_at ? (int64_t)(now - S.last_at) : 0;
    int hitch = S.last_at && fg_pace_hitch(gap);
    S.last_at = now;
    log_stats(now);
    if (gap && !hitch) S.interval = fg_pace_interval(S.interval, gap);
    if (S.w && ((uint32_t)cw != S.w || (uint32_t)ch != S.h)) S.crop++;
    if (hitch) S.hitch++;
    if (!ensure_images(c, (uint32_t)cw, (uint32_t)ch)) return 0;
    int blend = S.have_prev && !hitch;   // ensure_images cleared have_prev on a new size
    if (blend && fg_pace_over_refresh(S.interval, gfx_window_refresh_hz())) blend = 0, S.over++;   // the display cannot show 2x
    if (!blend && !hitch && !S.fresh && !S.have_prev) S.no_hist++;
    S.real++;

    // the previous job's command buffer is done long ago; the wait is for the fence's sake
    if (vkWaitForFences(c->device, 1, &S.fence, VK_TRUE, 5000000000ull) != VK_SUCCESS) return fail(c, "fence wait timed out");
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkResetCommandBuffer(S.cb, 0);
    vkBeginCommandBuffer(S.cb, &bi);
    VkCommandBuffer cb = S.cb;
    if (S.fresh) to_general(cb);
    S.fresh = 0;
    const int cur = S.cur, prev = cur ^ 1;
    const uint32_t uw = (uint32_t)cw, uh = (uint32_t)ch, qw = (uw + 3) / 4, qh = (uh + 3) / 4, fw = (qw + 3) / 4,
                   fh = (qh + 3) / 4;
    full_barrier(cb);   // the game's writes (earlier submissions) and the earlier presents' reads of our images
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .srcOffsets = {{cx, cy, 0}, {cx + cw, cy + ch, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {cw, ch, 1}}};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, S.hist[cur].img, VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
    full_barrier(cb);
    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo in[3];
    in[0] = (VkDescriptorImageInfo){S.nearest, S.hist[cur].view, G};
    pass(cb, &S.luma, in, S.luma_img[cur].view, qw, qh, NULL, 0);
    if (blend) {
        in[0] = (VkDescriptorImageInfo){S.nearest, S.luma_img[prev].view, G};
        in[1] = (VkDescriptorImageInfo){S.nearest, S.luma_img[cur].view, G};
        pass(cb, &S.flow, in, S.flow_img[0].view, fw, fh, NULL, 0);
        in[0] = (VkDescriptorImageInfo){S.nearest, S.flow_img[0].view, G};
        pass(cb, &S.median, in, S.flow_img[1].view, fw, fh, NULL, 0);
        const float t = 0.5f;
        in[0] = (VkDescriptorImageInfo){S.linear, S.hist[prev].view, G};
        in[1] = (VkDescriptorImageInfo){S.linear, S.hist[cur].view, G};
        in[2] = (VkDescriptorImageInfo){S.nearest, S.flow_img[1].view, G};
        pass(cb, &S.synth, in, S.made.view, uw, uh, &t, sizeof t);
    }
    // the last barrier orders later writes to the game's texture (the next submissions) after our blit's read of it
    vkEndCommandBuffer(cb);
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi};
    vkResetFences(c->device, 1, &S.fence);
    vk_queue_lock();
    VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, S.fence);
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        char msg[64];
        snprintf(msg, sizeof msg, "submit failed (%d)", (int)r);
        return fail(c, msg);
    }
    S.shown = cur;
    S.cur = prev;
    S.have_prev = 1;
    if (blend) S.made_n++;
    return blend;
}

int vk_fg_record(VkImage src, VkFormat fmt, int w, int h, const int crop[4]) {
    plat_mutex_lock(&fgl);
    int r = record(src, fmt, w, h, crop);
    plat_mutex_unlock(&fgl);
    return r;
}

void vk_fg_show(int (*pending)(void), uint64_t frame) {
    plat_mutex_lock(&fgl);
    VkImage made = S.made.img, real = S.hist[S.shown].img;
    int w = (int)S.w, h = (int)S.h;
    int64_t interval = S.interval;
    plat_mutex_unlock(&fgl);
    if (!made || !real) return;   // released meanwhile
    vk_present_image(made, VK_IMAGE_LAYOUT_GENERAL, HIST_FMT, w, h, NULL);
    uint64_t made_at = kern_now_ns();
    if (frame) vk_thumbs_frame(frame);   // the thumbnail and the dump name the game's frame, not the made one
    vk_present_blit(real, VK_IMAGE_LAYOUT_GENERAL, HIST_FMT, w, h, NULL);
    // the game's frame goes out half an interval after the made one, or at once when a newer job waits: the present
    // thread must not hold it back (vk_backend.c present_hold)
    uint64_t due = (uint64_t)fg_pace_due((int64_t)made_at, interval), now;
    int cut = 0;
    while ((now = kern_now_ns()) < due) {
        if (pending()) {
            cut = 1;
            break;
        }
        uint64_t step = due - now;
        plat_wait_ns(step > 1000000 ? 1000000 : step);
    }
    vk_present_flip();
    plat_mutex_lock(&fgl);
    S.cut += cut;
    S.late += now > due + 2000000;
    plat_mutex_unlock(&fgl);
}

void vk_fg_release(void) {
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&fgl);
    if (c && S.tried) {
        free_all(c);
        memset(&S, 0, sizeof S);
    }
    plat_mutex_unlock(&fgl);
}
