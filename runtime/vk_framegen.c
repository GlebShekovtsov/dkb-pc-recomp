// Frame generation (TOTK_FRAME_GEN = auto, 2x to 8x; off by default, experimental): frames made between each two of the
// game's, from their motion. A C rewrite of Hookshot's gpu/vk/vk_framegen.cpp (https://github.com/NotRexed/Hookshot @
// fab8cb37641714bd839ae83c2a7bc582bedc4d70, GPL-3.0; Auto, the multipliers above 2x, the sets and the display thread from
// https://git.notrexed.dev/Hookshot/Hookshot @ 7191250b7b0f8a4becd79ee086229a2620872fc7) with its shaders
// runtime/shaders/fg_*.comp:
//   1. the crop of the window texture is copied into a history image (RGBA8; A2B10G10R10 for the game's own HDR
//      picture: the game draws over its texture next frame) and reduced to quarter-size luma;
//   2. block matching of the luma against the frame before: a vector per 16x16 pixels within +-32 pixels, then a 3x3 median;
//   3. a frame at each time t between the two blends the last frame moved back and the new one forward along the vectors.
//
// 2x: one made frame at t = 0.5, shown first, then the game's own frame half an interval later. The present thread
// (vk_backend.c present_main) calls vk_fg_record with the game's texture before its blit (it returns 1), then vk_fg_show
// in place of the blit: the made frame goes through vk_present_image, the history copy of the game's frame through
// vk_present_blit, and its flip waits until half an interval after the made one. With frame_gen_by = fsr3
// (TOTK_FRAME_GEN_BY) the made frame comes from AMD FSR 3.1's frame interpolation instead of steps 2 and 3
// (vk_fsr3fi.c), over the depth and motion of the frame's temporal upscaler (vk_fg_record's fi); a frame it cannot
// make gets the built-in one. AMD's passes are also given the game's frames without their HUD, so the HUD is not moved
// with the scene: the upscaler's output is copied before the game draws its HUD into it (vk_fsr3fi_note_picture), and
// shaders/fg_hudless.comp makes from that copy the frame as the window texture would hold it without the HUD.
//
// Auto and 3x to 8x (vk_fg_record returns 2), two threads as in Hookshot. A fixed multiplier m gives each game frame as
// many frames as the display can show, m at most (fg_pace_fixed_frames): the made ones at t = k / n, then the game's own.
// Auto gives each of the display's refreshes between two game frames a frame made at its place between them
// (fg_pace_auto_slots), shown one game frame time later at its refresh's time; a refresh next to a game frame shows that
// frame itself. The present thread makes the game frame's set (the made frames and a copy of the game's own, so the history
// can move on) into one of three sets, submits it, and queues it. It waits only when all three sets are still being shown,
// which holds the game back as Hookshot's sets do. The display thread (display_main) takes the sets in order and shows each
// frame through vk_present_image on an even line: the game's frame time over the set's frames apart, never more than one
// game frame behind what the game handed over, a line that fell behind starting again from now (fg_pace_show_due); a made
// frame is left out when the next set is already waiting.
//
// Every frame goes through the whole present chain (colour, FSR, SMAA, HDR, overlay), so the overlay is never
// interpolated. The game's texture is read only by the command buffer vk_fg_record submits, and the swapchain is touched
// only through vk_present_image and vk_present_flip (and, from the display thread, under the present lock).
//
// The game's own HDR picture (hdr_native = 1, ov_gfx.c hdr_native_on) is Rec. 2020 PQ code values in an A2B10G10R10
// window texture. Its frames are made in the same way, in A2B10G10R10 history, made-frame and set images, so the
// picture keeps its 10 bits and vk_present_image shows each frame as the game's own HDR picture. The blend works on the
// code values, as it does on sRGB's: PQ is, like sRGB, a perceptual coding. The comparisons in the luma and
// synthesis passes (block matching's cost, what counts as moved or agreeing) take each value to the SDR picture's
// scale first, the sRGB code of its light over paper white (hdr_game_nits), so their thresholds mean what they do in
// SDR. The lifted SDR picture (hdr = 1 without hdr_native) needs nothing of this: the window texture is the SDR
// picture, and the HDR pass runs on every frame shown.
//
// Any failure turns the pass off for the session with one log line, and the present goes on as without it.
#include "vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "fg_flow_spv.h"
#include "fg_hudless_spv.h"
#include "fg_luma_spv.h"
#include "fg_median_spv.h"
#include "fg_pace.h"
#include "fg_synth_spv.h"
#include "flight.h"
#include "host.h"
#include "platform/platform.h"
#include "ui/menu.h"   // gfx_window_present_mode

#define HIST_FMT VK_FORMAT_R8G8B8A8_UNORM
#define HIST_FMT_PQ VK_FORMAT_A2B10G10R10_UNORM_PACK32   // the game's own HDR picture
#define SETS 3   // one shown, one waiting its place on the line, one being made: the display thread never holds the game back

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

// One game frame's frames, in the order shown (Auto and 3x to 8x): frame[k] is a made frame or a copy of the game's own
// (game[k]).
typedef struct {
    Image frame[FG_MAX_MULT];   // frame[0..S.frames) exist: the most a game frame may be given in this mode
    int n, timed;               // the frames shown; timed (Auto): each at due[k], else on the display thread's line
    int game[FG_MAX_MULT];
    int64_t due[FG_MAX_MULT];
    uint32_t w, h;
    VkFormat fmt;               // the images' format (HIST_FMT or HIST_FMT_PQ), which vk_present_image shows them as
    uint64_t frame_no;          // the game frame for the thumbnails and the dump (vk_thumbs_frame), 0 none
    uint64_t antilag;           // the game frame's AMD Anti-Lag 2 index (vk_antilag.c), 0 none
    int busy;                   // made and not yet all shown (D.mu)
} Set;

static struct {
    int tried, dead, said;   // said: bits of the reasons already logged (SAY_*)
    PFN_vkCmdPushDescriptorSetKHR push;
    Pipe luma, flow, median, synth, hudless;
    VkSampler nearest, linear;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    Image hist[2], luma_img[2], flow_img[2], made;   // made: the 2x frame
    // The frame without the game's HUD, for AMD FSR 3.1's interpolation (shaders/fg_hudless.comp), beside the history
    // image of the same index; hudless_ok: made for the frame that history image holds. Made at their first use.
    Image hudless_img[2];
    int hudless_ok[2];
    Set sets[SETS];
    int frames;           // 0: the 2x path (made); else the images of each set: Auto's cap, or the multiplier
    int next_set;         // the set the next record fills
    int64_t slot_next;    // Auto: the display's next refresh to be given a frame (fg_pace_auto_slots)
    int said_n;           // the frames a game frame got at the last "asked, made" log line; 0 = none yet
    uint64_t said_at;
    uint32_t w, h;        // the crop size the images were made for; 0 = none
    VkFormat img_fmt;     // the history and made-frame images' format (HIST_FMT or HIST_FMT_PQ)
    int pq_ok;            // HIST_FMT_PQ's features: 0 not looked at yet, 1 enough, -1 not
    VkFormat fmt_ok;      // the last window texture format found to have BLIT_SRC
    VkFormat fmt_bad;     // the last one found without it
    int cur;              // the history image the next frame is copied into; the other holds the frame before
    int shown;            // the history image holding the newest frame (what vk_fg_show presents as the real one)
    int have_prev, fresh;
    uint64_t last_at;     // when the previous job reached vk_fg_record; 0 = none
    int64_t interval;     // the game's frame interval, fg_pace_interval
    // the 5 s log
    uint64_t t0, real, made_n, by_fsr, hudless_used, no_hist, hitch, over, crop, cut, late, set_waits;
} S;
// S and the images are touched by vk_fg_record (the present thread) and vk_fg_release (the exit, under vk_present.c's g_pl
// first: that order everywhere), so each holds this lock for its whole run; vk_fg_show copies what it needs and presents
// without it. The display thread reads a set only between its queueing and its busy flag going down, and takes neither
// this lock nor S.
static PlatMutex fgl = PLAT_MUTEX_INIT;
enum { SAY_HDR = 1, SAY_MODE = 2, SAY_GAME = 4, SAY_FMT = 8, SAY_BLIT = 16, SAY_SLOW = 32, SAY_NOFI = 64 };

// The display thread's queue: the sets made and not yet taken, in order (D.mu, signalled on D.cv). dead is set by
// vk_fg_release under the present lock, and the display thread reads it under that lock before each present, so it
// never presents an image the release freed; nothing waits on the display thread once it is set.
static struct {
    PlatMutex mu;
    PlatCond cv;
    int jobs[SETS], at_head, count;
    int64_t job_at[SETS];
    int started, dead;
    uint64_t gen;   // bumped by vk_fg_release (D.mu): a set made before it is not presented after it
} D = {.mu = PLAT_MUTEX_INIT, .cv = PLAT_COND_INIT};

static void say(int bit, const char *msg, int arg) {
    if (S.said & bit) return;
    S.said |= bit;
    rt_log(msg, arg);
}

// Waits until the display thread has shown every set it was given (or the release ended it). Before the sets' images
// are freed or remade, and before the present thread presents a frame of its own again.
static void display_idle(void) {
    plat_mutex_lock(&D.mu);
    for (;;) {
        int busy = D.count;
        for (int i = 0; i < SETS; i++) busy |= S.sets[i].busy;
        if (!busy || D.dead || !D.started) break;
        plat_cond_wait_ms(&D.cv, &D.mu, 100);
    }
    plat_mutex_unlock(&D.mu);
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
    free_image(c, &S.hudless_img[0]), free_image(c, &S.hudless_img[1]);
    S.hudless_ok[0] = S.hudless_ok[1] = 0;
    for (int s = 0; s < SETS; s++)
        for (int i = 0; i < FG_MAX_MULT; i++) free_image(c, &S.sets[s].frame[i]);
    S.w = S.h = 0;
    S.frames = 0;
    S.have_prev = 0;
}

// Everything the pass made, for the session's end or a failed set-up. The device is idle or nothing was submitted.
static void free_all(VkCtx *c) {
    free_images(c);
    free_pipe(c, &S.luma), free_pipe(c, &S.flow), free_pipe(c, &S.median), free_pipe(c, &S.synth),
        free_pipe(c, &S.hudless);
    if (S.nearest) vkDestroySampler(c->device, S.nearest, NULL);
    if (S.linear) vkDestroySampler(c->device, S.linear, NULL);
    if (S.fence) vkDestroyFence(c->device, S.fence, NULL);
    if (S.pool) vkDestroyCommandPool(c->device, S.pool, NULL);   // frees the command buffer
    S.nearest = S.linear = VK_NULL_HANDLE, S.fence = VK_NULL_HANDLE, S.pool = VK_NULL_HANDLE, S.cb = VK_NULL_HANDLE;
}

static int fail(VkCtx *c, const char *what) {
    rt_log("fg: %s: off", what);
    S.dead = 1;
    display_idle();    // the sets already queued go out first
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
        {&S.luma, "luma", fg_luma_spv, sizeof fg_luma_spv, 1, 8},
        {&S.flow, "flow", fg_flow_spv, sizeof fg_flow_spv, 2, 0},
        {&S.median, "median", fg_median_spv, sizeof fg_median_spv, 1, 0},
        {&S.synth, "synth", fg_synth_spv, sizeof fg_synth_spv, 3, 12},
        {&S.hudless, "hudless", fg_hudless_spv, sizeof fg_hudless_spv, 2, 4},
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

// TOTK_FRAME_GEN: 0 off, 1 Auto, 2 to FG_MAX_MULT the fixed multiplier. Another value is off.
static int fg_mode(void) {
    const char *v = cfg_value("TOTK_FRAME_GEN");
    if (!strcmp(v, "auto")) return 1;
    if (v[0] >= '2' && v[0] <= '0' + FG_MAX_MULT && v[1] == 'x' && !v[2]) return v[0] - '0';
    return 0;
}

// Images for a w x h crop in format f (HIST_FMT, or HIST_FMT_PQ for the game's own HDR picture). frames 0 is the 2x
// path (one made image); otherwise `frames` images in each of the SETS sets. A new size, count or format waits for the
// display thread and the device (the earlier jobs' images may still be read, the sets still shown) and starts the
// history again.
static int ensure_images(VkCtx *c, uint32_t w, uint32_t h, int frames, int mode, VkFormat f) {
    if (S.w == w && S.h == h && S.frames == frames && S.img_fmt == f) return 1;
    if (S.w) {   // the first allocation has nothing to wait for
        display_idle();
        vk_queue_lock();   // vkDeviceWaitIdle needs every queue left alone
        vkDeviceWaitIdle(c->device);
        vk_queue_unlock();
    }
    free_images(c);
    uint32_t qw = (w + 3) / 4, qh = (h + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    const VkImageUsageFlags work = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    int ok = 1;
    if (!frames) ok = make_image(c, &S.made, f, w, h, work | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // a set's frames are written by the synthesis pass or copied from the history, and blitted to the window
    for (int s = 0; s < SETS && ok; s++)
        for (int i = 0; i < frames && ok; i++)
            ok = make_image(c, &S.sets[s].frame[i], f, w, h,
                            work | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    for (int i = 0; i < 2 && ok; i++)
        ok = make_image(c, &S.hist[i], f, w, h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
             make_image(c, &S.luma_img[i], VK_FORMAT_R16_SFLOAT, qw, qh, work) &&
             make_image(c, &S.flow_img[i], VK_FORMAT_R16G16B16A16_SFLOAT, cw, ch, work);
    if (!ok) {
        char msg[64];
        snprintf(msg, sizeof msg, "images %ux%u not made", w, h);
        return fail(c, msg);
    }
    S.w = w, S.h = h, S.frames = frames, S.img_fmt = f, S.fresh = 1;
    const char *hist = f == HIST_FMT_PQ ? "RGB10A2 (the game's own HDR picture, PQ)" : "RGBA8";
    if (mode == 1)
        rt_log("fg: frame generation Auto on (experimental): a frame for each of the display's refreshes (%.0f Hz; up to %d "
               "a game frame), made at %ux%u, history %s, %d sets", gfx_window_refresh_hz(), frames, w, h, hist, SETS);
    else if (frames)
        rt_log("fg: frame generation %dx on (experimental): made at %ux%u, history %s, %d sets", mode, w, h, hist,
               SETS);
    else
        rt_log("fg: frame generation 2x on (experimental): made at %ux%u, history %s", w, h, hist);
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
    enum { MOST = 7 + SETS * FG_MAX_MULT };
    const Image *all[MOST] = {&S.hist[0], &S.hist[1], &S.luma_img[0], &S.luma_img[1], &S.flow_img[0], &S.flow_img[1]};
    uint32_t n = 6;
    if (!S.frames) all[n++] = &S.made;
    for (int s = 0; s < SETS; s++)
        for (int i = 0; i < S.frames; i++) all[n++] = &S.sets[s].frame[i];   // only those that exist: a barrier on no image is invalid
    VkImageMemoryBarrier2 b[MOST];
    for (uint32_t i = 0; i < n; i++)
        b[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                       .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = all[i]->img,
                                       .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = n, .pImageMemoryBarriers = b};
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

// HIST_FMT_PQ's uses here, looked at once: storage (the synthesis pass writes it), sampled with linear filtering, and
// blits both ways. Vulkan requires none of them of A2B10G10R10 beyond sampling.
static int pq_format_ok(VkCtx *c) {
    if (!S.pq_ok) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(c->phys, HIST_FMT_PQ, &fp);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                          VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
        S.pq_ok = (fp.optimalTilingFeatures & need) == need ? 1 : -1;
    }
    return S.pq_ok > 0;
}

// The two images of the frame without its HUD, at the crop size, made at their first use; the command buffer being
// recorded moves them to GENERAL.
static int hudless_images(VkCtx *c, VkCommandBuffer cb) {
    if (S.hudless_img[1].img) return 1;
    for (int i = 0; i < 2; i++)
        if (!make_image(c, &S.hudless_img[i], HIST_FMT, S.w, S.h,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
            free_image(c, &S.hudless_img[0]), free_image(c, &S.hudless_img[1]);
            return 0;
        }
    VkImageMemoryBarrier2 b[2];
    for (int i = 0; i < 2; i++)
        b[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                       .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = S.hudless_img[i].img,
                                       .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 2,
                          .pImageMemoryBarriers = b};
    vkCmdPipelineBarrier2(cb, &d);
    S.hudless_ok[0] = S.hudless_ok[1] = 0;
    return 1;
}

static void log_stats(uint64_t now) {
    if (!S.t0) S.t0 = now;
    if (now - S.t0 < 5000000000ull) return;
    double s = (double)(now - S.t0) / 1e9;
    if (S.frames)   // Auto and 3x to 8x: the display thread shows the frames and counts what it shows (display_main)
        rt_log("fg: %.1f s: real %.1f fps, made %u (%.1f a second); game frames without made ones: no history %u, hitch %u, "
               "over refresh %u, crop %u; waited for a set %u", s, (double)S.real / s, (unsigned)S.made_n,
               (double)S.made_n / s, (unsigned)S.no_hist, (unsigned)S.hitch, (unsigned)S.over, (unsigned)S.crop,
               (unsigned)S.set_waits);
    else
        rt_log("fg: %.1f s: real %.1f fps, shown %.1f fps, made %u (by AMD FSR 3.1 %u, %u of them with the frames "
               "without their HUD); skipped: no history %u, hitch %u, over refresh %u, crop %u; real cut short %u; "
               "late >2 ms %u",
               s, (double)S.real / s, (double)(S.real + S.made_n) / s, (unsigned)S.made_n, (unsigned)S.by_fsr,
               (unsigned)S.hudless_used, (unsigned)S.no_hist, (unsigned)S.hitch, (unsigned)S.over, (unsigned)S.crop,
               (unsigned)S.cut, (unsigned)S.late);
    S.t0 = now, S.real = S.made_n = S.by_fsr = S.hudless_used = S.no_hist = S.hitch = S.over = S.crop = S.cut = S.late = S.set_waits = 0;
}

// The display thread (Hookshot vk_framegen.cpp display_main): takes the sets in the order they were made and shows
// their frames on the pacer's line (fg_pace_show_due). Its 5 s line is Hookshot's, so the two logs compare directly.
static void display_main(void *arg) {
    (void)arg;
    int64_t interval = 16666667, last_at = 0, next_show = 0, shown_at = 0, since = 0;
    int on_time = 0, late = 0, skipped = 0, hist[6] = {0};
    plat_mutex_lock(&D.mu);
    for (;;) {
        while (!D.count) plat_cond_wait(&D.cv, &D.mu);
        const int si = D.jobs[D.at_head];
        const int64_t at = D.job_at[D.at_head];
        const uint64_t gen = D.gen;
        D.at_head = (D.at_head + 1) % SETS, D.count--;
        plat_mutex_unlock(&D.mu);
        Set *st = &S.sets[si];
        if (last_at) {
            const int64_t d = at - last_at;
            if (d > 0 && !fg_pace_hitch(d)) interval = fg_pace_interval(interval, d);
        }
        last_at = at;
        if (!since) since = at;
        if (at - since > 5000000000ll) {
            rt_log("fg: frame interpolation: %d frames shown on time, %d late (over 2 ms), %d made ones left out (the game's "
                   "next frame waiting); shown apart: <7 ms %d, 7-9 %d, 9-12 %d, 12-17 %d, 17-25 %d, over 25 %d", on_time,
                   late, skipped, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
            memset(hist, 0, sizeof hist);
            on_time = late = skipped = 0, since = at;
        }
        const int timed = st->timed, shown = st->n;
        // The frame that stands for the game's: its own when the set shows it, else the set's last frame, the one nearest
        // it (Auto shows the game's frame only when a refresh falls on it). It names the flight thumbnail
        // (vk_thumbs_frame) and, like the game's own frame, is never left out, so no set loses its thumbnail.
        int tag = shown - 1;
        for (int k = 0; k < shown; k++)
            if (st->game[k]) tag = k;
        const int64_t step = interval / (shown > 1 ? shown : 1);
        for (int k = 0; k < shown; k++) {
            // A made frame is left out when the next set is already waiting: the display thread behind the game holds
            // its sets, and the game waits for them. The frame that stands for the game's (tag) is always shown.
            if (!st->game[k] && k != tag) {
                plat_mutex_lock(&D.mu);
                const int newer = D.count > 0;
                plat_mutex_unlock(&D.mu);
                if (newer) {
                    skipped++;
                    continue;
                }
            }
            const int64_t now = (int64_t)kern_now_ns();
            const int64_t due = fg_pace_show_due(timed, st->due[k], next_show, now, at, interval, step);
            vk_wait_until((uint64_t)due);
            const int64_t t = (int64_t)kern_now_ns();
            // The start lock (gfx_window_start) before the present lock, never under it: the exit holds the start lock
            // while vk_present_release waits for the present lock (vk_present_image_held).
            gfx_window_start();
            vk_present_lock();   // the release frees the images under this lock and sets D.dead first
            plat_mutex_lock(&D.mu);   // g_pl, then D.mu: the release's order
            const int live = !D.dead && gen == D.gen;
            plat_mutex_unlock(&D.mu);
            if (live) {
                if (k == tag && st->frame_no) vk_thumbs_frame(st->frame_no);
                // AMD Anti-Lag 2: only the game's own frame of the set has a present stage, not the made ones; the set
                // keeps the game frame's index from when it was recorded
                if (st->game[k]) vk_antilag_present(st->antilag);
                vk_present_image_held(st->frame[k].img, VK_IMAGE_LAYOUT_GENERAL, st->fmt, (int)st->w, (int)st->h, NULL);
            }
            vk_present_unlock();
            if (shown_at) {
                const int64_t d = t - shown_at;
                hist[d < 7000000 ? 0 : d < 9000000 ? 1 : d < 12000000 ? 2 : d < 17000000 ? 3 : d < 25000000 ? 4 : 5]++;
            }
            shown_at = t;
            next_show = t + step;
            if (t - due > 2000000) late++;
            else on_time++;
        }
        plat_mutex_lock(&D.mu);
        st->busy = 0;
        plat_cond_broadcast(&D.cv);
    }
}

static int record(VkImage src, VkFormat fmt, int w, int h, const int crop[4], const VkFiInputs *fi, uint64_t frame_no,
                  uint64_t antilag) {
    VkCtx *c = vk_ctx();
    if (S.dead || !c) return 0;
    if (gfx_window_closing()) return 0;   // the exit is about to free everything below
    const int mode = fg_mode();
    if (!mode) {   // off, or turned off while running: hand the images back once the display thread showed its sets
        if (S.w) {
            display_idle();
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
    // Where no frames are made (one log line each): TotK, whose back-edge pacing (vk_backend.c present_hold) the paced
    // show replaces; window textures that are neither 8-bit UNORM nor the game's own HDR picture; that picture on a GPU
    // without HIST_FMT_PQ's features; and FIFO, where the two acquires and presents of a job would block the present
    // thread and halve the game's rate (the display thread's presents would block on the display and hold the sets
    // back).
    const char *pmode = gfx_window_present_mode();
    const int pq = fmt == HIST_FMT_PQ && hdr_native_on();   // the game's own HDR picture: PQ code values, 10 bits
    int skip = 0;
    if (rt_game_version != 0) skip = 1, say(SAY_GAME, "fg: not Bananza: no made frames", 0);
    else if (!pq && fmt != VK_FORMAT_R8G8B8A8_UNORM && fmt != VK_FORMAT_B8G8R8A8_UNORM)
        skip = 1, say(SAY_FMT, "fg: window texture format %d is not 8-bit UNORM: no made frames", (int)fmt);
    else if (pq && !pq_format_ok(c))
        skip = 1, say(SAY_HDR, "fg: A2B10G10R10 lacks storage, linear filtering or blits here: no made frames from the "
                               "game's own HDR picture", 0);
    else if (!*pmode) skip = 1;   // no swapchain yet
    else if (strcmp(pmode, "mailbox") && strcmp(pmode, "immediate"))
        skip = 1, say(SAY_MODE, "fg: present mode fifo: no made frames (needs mailbox or immediate)", 0);
    if (skip) {
        S.have_prev = 0, S.last_at = 0;
        return 0;
    }
    if (!S.tried) {
        if (!init(c)) return 0;
        // a device made again after a release: the display thread presents again. D.mu only: vk_present_release holds
        // the present lock and then takes fgl, so taking the present lock here, under fgl, would invert that order.
        plat_mutex_lock(&D.mu);
        D.dead = 0;
        plat_mutex_unlock(&D.mu);
    }
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
    const uint64_t before = S.last_at;
    int64_t gap = S.last_at ? (int64_t)(now - S.last_at) : 0;
    int hitch = S.last_at && fg_pace_hitch(gap);
    S.last_at = now;
    log_stats(now);
    if (gap && !hitch) S.interval = fg_pace_interval(S.interval, gap);
    if (S.w && ((uint32_t)cw != S.w || (uint32_t)ch != S.h)) S.crop++;
    if (hitch) S.hitch++;
    const double hz = gfx_window_refresh_hz();
    // the images are made once for the most a game frame may be given: Auto's cap, or the multiplier's frames (2x: one)
    const int frames = mode == 2 ? 0 : mode == 1 ? fg_pace_auto_cap(hz) : mode;
    if (!ensure_images(c, (uint32_t)cw, (uint32_t)ch, frames, mode, pq ? HIST_FMT_PQ : HIST_FMT)) return 0;
    int blend = S.have_prev && !hitch;   // ensure_images cleared have_prev on a new size, count or format
    if (!frames && blend && fg_pace_over_refresh(S.interval, hz)) blend = 0, S.over++;   // the display cannot show 2x
    if (!blend && !hitch && !S.fresh && !S.have_prev) S.no_hist++;
    S.real++;

    // Auto and 3x to 8x: the set to fill. It waits while the display thread still shows all three (the game is held back
    // then, as by Hookshot's sets).
    Set *st = NULL;
    float t[FG_MAX_MULT];
    int n = 0, any_made = blend;
    if (frames) {
        plat_mutex_lock(&D.mu);
        const uint64_t wait0 = kern_now_ns();
        S.set_waits += S.sets[S.next_set].busy && !D.dead;   // the game frame waits for the display thread
        while (S.sets[S.next_set].busy && !D.dead) plat_cond_wait_ms(&D.cv, &D.mu, 100);
        const uint64_t waited = kern_now_ns() - wait0;
        const int dead = D.dead;
        st = &S.sets[S.next_set];
        st->busy = !dead;
        S.next_set = (S.next_set + 1) % SETS;
        plat_mutex_unlock(&D.mu);
        if (dead) return 0;
        if (waited > 1000000000ull) say(SAY_SLOW, "fg: a set waited over 1 s for the display thread", 0);

        // the set: its frames' places between the last game frame and this one in t (1: this game frame)
        st->timed = mode == 1;
        if (blend && mode == 1) {   // Auto: a frame for each of the display's refreshes, at its own time
            int64_t at[FG_MAX_MULT];
            n = fg_pace_auto_slots((int64_t)before, (int64_t)now, (int64_t)(1e9 / (hz > 0 ? hz : 60.0)), &S.slot_next, t, at,
                                   S.frames);
            for (int k = 0; k < n; k++) st->game[k] = t[k] > 0.999f, st->due[k] = at[k] + S.interval;
        } else if (blend) {   // a fixed multiplier: the made frames at k / n, then the game's own
            n = fg_pace_fixed_frames(mode, hz, S.interval);
            if (n != S.said_n && (!S.said_n || now - S.said_at > 5000000000ull)) {
                if (n < mode)
                    rt_log("fg: %dx asked, %dx made: the display's %.0f Hz over the game's %.0f fps (the game keeps its rate)",
                           mode, n, hz, S.interval > 0 ? 1e9 / (double)S.interval : 0.0);
                else if (S.said_n) rt_log("fg: %dx made", n);
                S.said_n = n, S.said_at = now;
            }
            if (n < 2) S.over++;   // the display shows the game's frames alone
            for (int k = 0; k < n; k++) st->game[k] = k == n - 1, t[k] = (float)(k + 1) / (float)n;
        } else {   // nothing to blend from: the game's frame, and Auto's timeline starts at it
            n = 1, st->game[0] = 1, t[0] = 1.0f, st->due[0] = (int64_t)now + S.interval;
            S.slot_next = (int64_t)now + (int64_t)(1e9 / (hz > 0 ? hz : 60.0));
        }
        st->n = n, st->w = (uint32_t)cw, st->h = (uint32_t)ch, st->fmt = S.img_fmt, st->frame_no = frame_no, st->antilag = antilag;
        any_made = 0;
        for (int k = 0; k < n; k++) any_made |= !st->game[k];
    }

    // the previous job's command buffer is done long ago; the wait is for the fence's sake
    if (vkWaitForFences(c->device, 1, &S.fence, VK_TRUE, 5000000000ull) != VK_SUCCESS) {
        if (st) {
            plat_mutex_lock(&D.mu);
            st->busy = 0;
            plat_mutex_unlock(&D.mu);
        }
        return fail(c, "fence wait timed out");
    }
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
    // the game's writes (earlier submissions), and the earlier presents' reads of our images, which went to the same
    // queue before this
    full_barrier(cb);
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .srcOffsets = {{cx, cy, 0}, {cx + cw, cy + ch, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {cw, ch, 1}}};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, S.hist[cur].img, VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
    full_barrier(cb);
    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL;
    // the passes' comparisons on the SDR picture's scale (shaders/fg_luma.comp, fg_synth.comp): pq says the frames hold
    // PQ code values, white the paper white their light is taken over
    const int32_t pq_push = pq;
    const float white = pq ? vk_hdr_game_nits() : 0.0f;
    const struct { int32_t pq; float white; } luma_push = {pq_push, white};
    VkDescriptorImageInfo in[3];
    in[0] = (VkDescriptorImageInfo){S.nearest, S.hist[cur].view, G};
    pass(cb, &S.luma, in, S.luma_img[cur].view, qw, qh, &luma_push, sizeof luma_push);
    // 2x with frame_gen_by = fsr3: AMD's interpolation makes the frame where the upscaler gave this frame's depth and
    // motion (vk_fsr3fi.c; a size that is not the 3D's shape, a first frame after a gap and a camera cut it refuses).
    // The flow and synthesis passes below then run only for a frame it did not make (the luma above always runs: the
    // built-in flow of a later frame reads it).
    int by_amd = 0;
    // AMD's passes take an SDR picture, so a PQ history (the game's own HDR picture) goes to the built-in interpolation.
    S.hudless_ok[cur] = 0;   // set below when this frame's picture without its HUD is made
    if (!frames && any_made && !pq && vk_fsr3fi_wanted()) {
        if (fi && fi->valid) {
            // The frame without its HUD (shaders/fg_hudless.comp), from the upscaled picture copied before the game
            // drew its HUD into it (vk_fsr3fi_note_picture). The game's last draw scales that picture to the whole
            // window texture, so it is made only for an uncropped frame of the picture's shape. With it for this frame
            // and the one before, AMD's passes interpolate the scene alone and put the HUD back as it is shown;
            // without, they take the frames as shown.
            VkImageView hl = VK_NULL_HANDLE, hl_prev = VK_NULL_HANDLE;
            if (fi->picture && cx == 0 && cy == 0 && cw == w && ch == h &&
                fabs((double)fi->picture_w * ch - (double)fi->picture_h * cw) <= 0.01 * (double)fi->picture_w * ch &&
                hudless_images(c, cb)) {
                const float tol = 3.0f / 255.0f;   // the last draw's own rounding against this pass's
                in[0] = (VkDescriptorImageInfo){S.linear, fi->picture, G};
                in[1] = (VkDescriptorImageInfo){S.nearest, S.hist[cur].view, G};
                pass(cb, &S.hudless, in, S.hudless_img[cur].view, uw, uh, &tol, sizeof tol);
                S.hudless_ok[cur] = 1;
                if (S.hudless_ok[prev]) hl = S.hudless_img[cur].view, hl_prev = S.hudless_img[prev].view;
            }
            by_amd = vk_fsr3fi_generate(cb, fi, S.hist[cur].view, S.hist[prev].view, hl, hl_prev, S.made.view, uw, uh);
            if (by_amd && hl) S.hudless_used++;
        } else
            say(SAY_NOFI, "fg: frame_gen_by = fsr3 needs the depth and motion of a temporal upscaler (upscaler = dlss, "
                          "fsr3 or fsr4, with the 3D drawn below the output size): the built-in interpolation makes this "
                          "frame", 0);
    }
    if (by_amd) S.made_n++, S.by_fsr++;
    if (any_made && !by_amd) {
        in[0] = (VkDescriptorImageInfo){S.nearest, S.luma_img[prev].view, G};
        in[1] = (VkDescriptorImageInfo){S.nearest, S.luma_img[cur].view, G};
        pass(cb, &S.flow, in, S.flow_img[0].view, fw, fh, NULL, 0);
        in[0] = (VkDescriptorImageInfo){S.nearest, S.flow_img[0].view, G};
        pass(cb, &S.median, in, S.flow_img[1].view, fw, fh, NULL, 0);
        in[0] = (VkDescriptorImageInfo){S.linear, S.hist[prev].view, G};
        in[1] = (VkDescriptorImageInfo){S.linear, S.hist[cur].view, G};
        in[2] = (VkDescriptorImageInfo){S.nearest, S.flow_img[1].view, G};
        if (!frames) {
            const struct { float t; int32_t pq; float white; } half = {0.5f, pq_push, white};
            pass(cb, &S.synth, in, S.made.view, uw, uh, &half, sizeof half);
            S.made_n++;
        }
        for (int k = 0; k < n; k++)
            if (!st->game[k]) {
                const struct { float t; int32_t pq; float white; } sp = {t[k], pq_push, white};
                pass(cb, &S.synth, in, st->frame[k].view, uw, uh, &sp, sizeof sp);
                S.made_n++;
            }
    }
    // the game's own frame goes into the set too: the history image is written again two frames on, while the set may
    // still be waiting its turn
    for (int k = 0; k < n; k++)
        if (st->game[k]) {
            const VkImageCopy cp = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                    {0, 0, 0}, {uw, uh, 1}};
            vkCmdCopyImage(cb, S.hist[cur].img, G, st->frame[k].img, G, 1, &cp);
        }
    // the last barrier orders later writes to the game's texture (the next submissions) after our blit's read of it, and
    // the display thread's blits of the set after its writes
    if (frames) full_barrier(cb);
    vkEndCommandBuffer(cb);
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi};
    vkResetFences(c->device, 1, &S.fence);
    vk_queue_lock();
    VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, S.fence);
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        if (st) {
            plat_mutex_lock(&D.mu);
            st->busy = 0;
            plat_mutex_unlock(&D.mu);
        }
        char msg[64];
        snprintf(msg, sizeof msg, "submit failed (%d)", (int)r);
        return fail(c, msg);
    }
    S.shown = cur;
    S.cur = prev;
    S.have_prev = 1;
    if (!frames) return blend;
    // Auto and 3x to 8x: to the display thread, which starts with the first set
    plat_mutex_lock(&D.mu);
    if (!D.started) {
        if (!plat_thread_spawn(display_main, NULL, "vk fg display", 0)) {
            st->busy = 0;
            plat_mutex_unlock(&D.mu);   // fail waits on the display queue under this lock
            return fail(c, "cannot start the display thread");
        }
        D.started = 1;
    }
    const int tail = (D.at_head + D.count) % SETS;
    D.jobs[tail] = (int)(st - S.sets), D.job_at[tail] = (int64_t)now, D.count++;
    plat_cond_broadcast(&D.cv);
    plat_mutex_unlock(&D.mu);
    return 2;   // Auto: also a set of no frames, when the game outruns the display and this frame has no refresh
}

int vk_fg_record(VkImage src, VkFormat fmt, int w, int h, const int crop[4], const VkFiInputs *fi, uint64_t frame_no,
                 uint64_t antilag) {
    plat_mutex_lock(&fgl);
    int r = record(src, fmt, w, h, crop, fi, frame_no, antilag);
    // a frame the display thread does not get is presented by the caller (or by vk_fg_show): the sets before it go out first
    if (r != 2) display_idle();
    plat_mutex_unlock(&fgl);
    return r;
}

void vk_fg_show(int (*pending)(void), uint64_t frame, uint64_t antilag) {
    plat_mutex_lock(&fgl);
    VkImage made = S.made.img, real = S.hist[S.shown].img;
    int w = (int)S.w, h = (int)S.h;
    const VkFormat fmt = S.img_fmt;
    int64_t interval = S.interval;
    plat_mutex_unlock(&fgl);
    if (!made || !real) return;   // released meanwhile
    vk_present_image(made, VK_IMAGE_LAYOUT_GENERAL, fmt, w, h, NULL);
    uint64_t made_at = kern_now_ns();
    if (frame) vk_thumbs_frame(frame);   // the thumbnail and the dump name the game's frame, not the made one
    vk_present_blit(real, VK_IMAGE_LAYOUT_GENERAL, fmt, w, h, NULL);
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
    vk_antilag_present(antilag);   // AMD Anti-Lag 2: only the game's own frame has a present stage, not the made one
    vk_present_flip();
    plat_mutex_lock(&fgl);
    S.cut += cut;
    S.late += now > due + 2000000;
    plat_mutex_unlock(&fgl);
}

void vk_fg_release(void) {   // under the present lock, device idle
    plat_mutex_lock(&D.mu);
    D.gen++;
    D.dead = 1;   // first: the display thread presents nothing more, and nothing waits on it
    for (int s = 0; s < SETS; s++) S.sets[s].busy = 0;
    D.count = 0;
    plat_cond_broadcast(&D.cv);
    plat_mutex_unlock(&D.mu);
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&fgl);
    if (c && S.tried) {
        free_all(c);
        memset(&S, 0, sizeof S);
    }
    plat_mutex_unlock(&fgl);
}
