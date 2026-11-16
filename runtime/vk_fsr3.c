// SPDX-License-Identifier: GPL-3.0-or-later
// AMD FSR 3.1 temporal upscaling in place of the game's upscaler (upscaler = fsr3), next to DLSS (vk_dlss.c). The
// game's upscale dispatch is replaced by AMD's FSR 3.1.4 upscaler passes (FidelityFX SDK v1.1.4, MIT, vendored under
// runtime/third_party/fidelityfx and compiled by tools/build_shaders.py into fsr3_*_spv.h), recorded here as dispatches
// of our own with push descriptors; nothing of AMD's is linked or loaded at run time. The inputs are DLSS's: the
// scene's jitter, depth and camera motion vectors come from vk_dlss.c (vk_temporal_inputs); the projection gives FSR its
// near and far distances (vk_fsr3_camera_factors). A frame without a scene at the input size is left to the game's
// upscaler.
//
// Derived from Hookshot's FSR 3.1 integration (gpu/vk/vk_fsr3.cpp and fsr3_math.h, https://github.com/NotRexed/Hookshot
// @ fab8cb37641714bd839ae83c2a7bc582bedc4d70, Copyright 2026 Hookshot contributors, GPL-3.0-or-later), rewritten in C
// for this runtime: the resource table, the constant block and the per-pass binding lists follow the SDK's
// ffx_fsr3upscaler.cpp and each pass file's FSR3UPSCALER_BIND_* numbering, as Hookshot writes them out. The differences:
// the passes are recorded through the recorder (rec_push), the constants go through a small host-visible ring checked
// against the submission's fence, the depth is the game's own (not reversed: the passes are built with
// INVERTED_DEPTH=0, so Hookshot's reversing copy is not needed), and RCAS is not used.
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"

#include "fsr3_accumulate_spv.h"
#include "fsr3_luma_instability_spv.h"
#include "fsr3_luma_pyramid_nowave_spv.h"
#include "fsr3_luma_pyramid_spv.h"
#include "fsr3_out_spv.h"
#include "fsr3_prepare_inputs_spv.h"
#include "fsr3_prepare_reactivity_spv.h"
#include "fsr3_shading_change_pyramid_nowave_spv.h"
#include "fsr3_shading_change_pyramid_spv.h"
#include "fsr3_shading_change_spv.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// A descriptor's kind, in the binding order each pass file declares. Every pass also takes the two samplers at bindings
// 1000 (point clamp) and 1001 (linear clamp).
enum { T, I, U };   // sampled texture, storage image, uniform buffer

typedef struct {
    const char *name;
    const uint32_t *spv;
    size_t bytes;
    const uint8_t *bind;
    unsigned count;
} Pass;

static const uint8_t PREPARE_INPUTS[] = {T, T, T, I, I, I, I, I, U};
static const uint8_t LUMA_PYRAMID[] = {T, T, I, I, I, I, I, I, I, I, I, U, U};
static const uint8_t SHADING_CHANGE_PYRAMID[] = {T, T, T, T, I, I, I, I, I, I, I, U, U};
static const uint8_t SHADING_CHANGE[] = {T, I, U};
static const uint8_t PREPARE_REACTIVITY[] = {T, T, T, T, T, T, T, T, T, I, I, I, U};
static const uint8_t LUMA_INSTABILITY[] = {T, T, T, T, T, T, T, I, I, U};
static const uint8_t ACCUMULATE[] = {T, T, T, T, T, T, T, T, T, I, I, I, U};

#define PASS(n, spv, b) {n, spv, sizeof spv, b, sizeof b}
// In the order of the enum below. The two _nowave pyramids are alternatives to the two before them, for GPUs without the
// subgroup operations the single-pass downsampler wants.
static const Pass PASSES[] = {
    PASS("prepare_inputs", fsr3_prepare_inputs_spv, PREPARE_INPUTS),
    PASS("luma_pyramid", fsr3_luma_pyramid_spv, LUMA_PYRAMID),
    PASS("luma_pyramid_nowave", fsr3_luma_pyramid_nowave_spv, LUMA_PYRAMID),
    PASS("shading_change_pyramid", fsr3_shading_change_pyramid_spv, SHADING_CHANGE_PYRAMID),
    PASS("shading_change_pyramid_nowave", fsr3_shading_change_pyramid_nowave_spv, SHADING_CHANGE_PYRAMID),
    PASS("shading_change", fsr3_shading_change_spv, SHADING_CHANGE),
    PASS("prepare_reactivity", fsr3_prepare_reactivity_spv, PREPARE_REACTIVITY),
    PASS("luma_instability", fsr3_luma_instability_spv, LUMA_INSTABILITY),
    PASS("accumulate", fsr3_accumulate_spv, ACCUMULATE),
};
#undef PASS
enum { P_PREPARE_INPUTS, P_LUMA_PYRAMID, P_LUMA_PYRAMID_NOWAVE, P_SHADING_CHANGE_PYRAMID, P_SHADING_CHANGE_PYRAMID_NOWAVE,
       P_SHADING_CHANGE, P_PREPARE_REACTIVITY, P_LUMA_INSTABILITY, P_ACCUMULATE, PASS_COUNT };
_Static_assert(sizeof PASSES / sizeof PASSES[0] == PASS_COUNT, "the enum names every pass");
#define SPD_STORED_MIPS 6
#define MAX_LEVELS 16
#define RING 8

typedef struct {
    VkDescriptorSetLayout set;
    VkPipelineLayout layout;
    VkPipeline pipe;
} Pipe;

typedef struct {
    VkImage image;
    VmaAllocation mem;
    VkImageView view;                 // every level
    VkImageView levels[MAX_LEVELS];   // one per level, for the mipped ones
    uint32_t w, h, nlevels;
} Img;

// cbFSR3Upscaler, as ffx_fsr3upscaler.cpp lays it out. The size is asserted because every pass reads it by offset: a
// field out of place is a wrong picture, not a compile error.
typedef struct {
    int32_t render_size[2], previous_render_size[2], upscale_size[2], previous_upscale_size[2], max_render_size[2],
        max_upscale_size[2];
    float device_to_view_depth[4];
    float jitter[2], previous_jitter[2], motion_vector_scale[2], downscale_factor[2], motion_vector_jitter_cancellation[2];
    float tan_half_fov, jitter_phase_count, delta_time, delta_pre_exposure, view_space_to_meters, frame_index,
        velocity_factor, reactiveness_scale, shading_change_scale, accumulation_added_per_frame, min_disocclusion_accumulation;
} Constants;
_Static_assert(sizeof(Constants) == 148, "cbFSR3Upscaler must match the SDK's layout");

typedef struct {
    uint32_t mips, num_work_groups, work_group_offset[2], render_size[2];
} SpdConstants;

typedef struct {   // one descriptor of a dispatch: an image view, or a range of the constants buffer
    VkImageView view;
    VkDeviceSize offset, size;
} Res;

static struct {
    int tried, ok, wave, failed;
    Pipe pass[PASS_COUNT];
    Pipe out;
    VkSampler point, linear;
    // The SDK's internal images. The pairs ping-pong on frame parity, as its resource tables do.
    Img accumulation[2], luma[2], luma_history[2], internal_upscaled[2];
    Img farthest_depth, shading_change, new_locks, spd_mips, farthest_depth_mip1, dilated_motion_vectors, dilated_depth,
        previous_nearest_depth, dilated_reactive_masks, frame_info, spd_atomic, lanczos_lut, default_reactive, output;
    uint32_t rw, rh, uw, uh;
    // The constants: a ring of slots in one host-visible buffer. A slot is free once the submission that used it is done.
    VkBuffer ubo;
    VmaAllocation ubo_mem;
    uint8_t *ubo_ptr;
    VkDeviceSize slot_fsr, slot_spd, stride;
    uint64_t ring_fence[RING];
    unsigned ring_next;
    // History, as the SDK's dispatch carries it.
    int32_t prev_render[2], prev_upscale[2];
    float prev_jitter[2], frame_index;
    uint32_t parity;
    int first, lut_uploaded;
    // Counters for the log.
    uint64_t frames, temporal, spatial, resets, told_fail;
    uint64_t last_qpc;
} G = {.wave = 1};

static const VkImageSubresourceRange ALL = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};

static uint64_t frames_now(void) { return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED); }

// The setting is read once a frame (the draws ask for it thousands of times): a change from the menu takes effect at the
// next frame.
int vk_fsr3_wanted(void) {
    static uint64_t seen = ~0ull;
    static int chosen;
    uint64_t f = frames_now();
    if (f != __atomic_load_n(&seen, __ATOMIC_RELAXED)) {
        // (upscaler = fsr4 without AMD's DLLs, or off Windows, runs this path)
        const char *up = cfg_value("TOTK_UPSCALER");
        __atomic_store_n(&chosen, !strcmp(up, "fsr3") || (!strcmp(up, "fsr4") && !vk_fsr4_wanted()), __ATOMIC_RELAXED);
        __atomic_store_n(&seen, f, __ATOMIC_RELAXED);
    }
    return __atomic_load_n(&chosen, __ATOMIC_RELAXED) && !__atomic_load_n(&G.failed, __ATOMIC_RELAXED);
}

static int fail(const char *why) {
    if (!__atomic_load_n(&G.failed, __ATOMIC_RELAXED)) rt_log("fsr3: %s; the game's upscaler runs", why);
    __atomic_store_n(&G.failed, 1, __ATOMIC_RELAXED);
    return 0;
}

// FSR's camera from the scene's projection (row-major, clip = p * view position, its w row 0 0 -1 0, near 1 far 5000 and
// OpenGL style: the depth buffer holds (ndc z + 1) / 2). With a view distance v in front of the camera ndc z =
// -p22 + p23 / v, so the stored depth is d = a + b / v with a = (1 - p22) / 2, b = p23 / 2, and v = b / (d - a): the
// passes' device_to_view_depth ([1] / (depth - [0])). [2] and [3] scale ndc x and y to view x and y per unit distance.
// 0 for a matrix that is not a perspective projection. (Hookshot fsr3_math.h camera_from_projection, which also flips a
// forward depth to FSR's reversed one: the passes here are built for the forward depth.)
int vk_fsr3_camera_factors(const float p[16], float f[4]) {
    if (p[12] != 0.0f || p[13] != 0.0f || p[14] != -1.0f || p[15] != 0.0f) return 0;
    if (!(p[0] > 0.0f) || !(p[5] > 0.0f) || !isfinite(p[0]) || !isfinite(p[5])) return 0;
    double a = (1.0 - (double)p[10]) * 0.5, b = (double)p[11] * 0.5;
    if (!isfinite(a) || !isfinite(b) || fabs(b) < 1e-12 || b > 0.0) return 0;   // b < 0: depth 0 at the near plane
    f[0] = (float)a, f[1] = (float)b, f[2] = 1.0f / p[0], f[3] = 1.0f / p[5];
    return 1;
}

static void barrier_mem(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                        VkAccessFlags2 da) {
    VkMemoryBarrier2 m = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                          .dstStageMask = ds, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m};
    vkCmdPipelineBarrier2(cb, &di);
}

static void destroy_now(Img *t) {
    VkCtx *c = vk_ctx();
    for (uint32_t i = 0; i < t->nlevels && t->nlevels > 1; i++)
        if (t->levels[i]) vkDestroyImageView(c->device, t->levels[i], NULL);
    if (t->view) vkDestroyImageView(c->device, t->view, NULL);
    if (t->image) vmaDestroyImage(c->vma, t->image, t->mem);
    memset(t, 0, sizeof *t);
}

// The old image goes when the recorded work that may still use it is done.
static void drop_image(Img *t) {
    if (!t->image) return;
    for (uint32_t i = 0; i < t->nlevels && t->nlevels > 1; i++) vk_defer_view(t->levels[i]);
    vk_defer_view(t->view);
    vk_defer_image(t->image, t->mem);
    memset(t, 0, sizeof *t);
}

static Img *const *all_images(size_t *n) {
    static Img *list[14 + 8];
    size_t k = 0;
    Img *one[] = {&G.farthest_depth, &G.shading_change, &G.new_locks, &G.spd_mips, &G.farthest_depth_mip1,
                  &G.dilated_motion_vectors, &G.dilated_depth, &G.previous_nearest_depth, &G.dilated_reactive_masks,
                  &G.frame_info, &G.spd_atomic, &G.lanczos_lut, &G.default_reactive, &G.output};
    for (size_t i = 0; i < sizeof one / sizeof one[0]; i++) list[k++] = one[i];
    for (int i = 0; i < 2; i++)
        list[k++] = &G.accumulation[i], list[k++] = &G.luma[i], list[k++] = &G.luma_history[i], list[k++] = &G.internal_upscaled[i];
    *n = k;
    return list;
}

static uint32_t max_u(uint32_t a, uint32_t b) { return a > b ? a : b; }

// One of the SDK's internal images: storage and sampled, GENERAL throughout, with a view per level when it has more
// than one (the pyramids bind their levels separately).
static int make_image(Img *t, VkFormat fmt, uint32_t w, uint32_t h, uint32_t levels) {
    VkCtx *c = vk_ctx();
    drop_image(t);
    t->w = max_u(1, w), t->h = max_u(1, h), t->nlevels = levels < 1 ? 1 : levels > MAX_LEVELS ? MAX_LEVELS : levels;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {t->w, t->h, 1}, .mipLevels = t->nlevels, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &t->image, &t->mem, NULL) != VK_SUCCESS) {
        rt_log("fsr3: a %ux%u image (format %d) was not made", t->w, t->h, (int)fmt);
        memset(t, 0, sizeof *t);
        return 0;
    }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t->nlevels, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &t->view) != VK_SUCCESS) return t->view = VK_NULL_HANDLE, 0;
    if (t->nlevels > 1)
        for (uint32_t i = 0; i < t->nlevels; i++) {
            vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1};
            if (vkCreateImageView(c->device, &vi, NULL, &t->levels[i]) != VK_SUCCESS) return 0;
        }
    return 1;
}

static int make_pipe(Pipe *p, const uint32_t *spv, size_t bytes, const VkDescriptorSetLayoutBinding *b, uint32_t nb,
                     uint32_t push_bytes, const char *name) {
    VkDevice d = vk_ctx()->device;
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                          .bindingCount = nb, .pBindings = b};
    if (vkCreateDescriptorSetLayout(d, &li, NULL, &p->set) != VK_SUCCESS) return 0;
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &p->set, .pushConstantRangeCount = push_bytes ? 1 : 0,
                                     .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(d, &pi, NULL, &p->layout) != VK_SUCCESS) return 0;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(d, &mi, NULL, &m) != VK_SUCCESS) return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = p->layout};
    VkResult r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipe);
    vkDestroyShaderModule(d, m, NULL);
    if (r != VK_SUCCESS) {
        rt_log("fsr3: the %s pass pipeline was not made (%d)", name, (int)r);
        p->pipe = VK_NULL_HANDLE;
        return 0;
    }
    return 1;
}

static int make_pass(int i) {
    const Pass *info = &PASSES[i];
    VkDescriptorSetLayoutBinding b[16];
    unsigned n = 0;
    for (; n < info->count; n++) {
        VkDescriptorType type = info->bind[n] == T ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                : info->bind[n] == I ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                     : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[n] = (VkDescriptorSetLayoutBinding){n, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    }
    // (the two samplers every pass declares, at the bindings the SDK's GLSL fixes them to)
    b[n++] = (VkDescriptorSetLayoutBinding){1000, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    b[n++] = (VkDescriptorSetLayoutBinding){1001, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    return make_pipe(&G.pass[i], info->spv, info->bytes, b, n, 0, info->name);
}

static int make_out(void) {
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
                                         {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    return make_pipe(&G.out, fsr3_out_spv, sizeof fsr3_out_spv, b, 2, 8, "output");
}

static int make_samplers(void) {
    VkDevice d = vk_ctx()->device;
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .minLod = 0.0f,
                              .maxLod = VK_LOD_CLAMP_NONE};
    if (vkCreateSampler(d, &si, NULL, &G.point) != VK_SUCCESS) return 0;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    return vkCreateSampler(d, &si, NULL, &G.linear) == VK_SUCCESS;
}

// Every storage format the internal images need, sampled and as storage. A GPU missing one cannot run the passes, and
// finding that out here beats a failed image in the middle of a frame.
static int formats_supported(VkPhysicalDevice phys) {
    static const VkFormat needed[] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM,
                                      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32_SFLOAT,
                                      VK_FORMAT_R32_UINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R16_SNORM};
    for (size_t i = 0; i < sizeof needed / sizeof needed[0]; i++) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(phys, needed[i], &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ||
            !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
            rt_log("fsr3: not on this GPU: format %d cannot be a storage and sampled image here", (int)needed[i]);
            return 0;
        }
    }
    return 1;
}

// What the passes need of the device, asked once: push descriptors enough for the widest pass (thirteen and the two
// samplers), the extended storage formats and storage writes without a format, every storage format of the internal
// images. Without subgroup quad operations in compute the two pyramids run as their wave-less variants.
static int supported(void) {
    static int answer = -1;
    if (answer >= 0) return answer;
    answer = 0;
    VkCtx *c = vk_ctx();
    VkPhysicalDevicePushDescriptorPropertiesKHR pd = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
    VkPhysicalDeviceSubgroupProperties sg = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext = &pd};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sg};
    vkGetPhysicalDeviceProperties2(c->phys, &p2);
    VkPhysicalDeviceFeatures ft;
    vkGetPhysicalDeviceFeatures(c->phys, &ft);
    if (!vk_pipeline_push_fn() || pd.maxPushDescriptors < 15)
        rt_log("fsr3: not on this GPU: no push descriptors, or fewer than 15 (%u)", pd.maxPushDescriptors);
    else if (!ft.shaderStorageImageExtendedFormats || !ft.shaderStorageImageWriteWithoutFormat)
        rt_log("fsr3: not on this GPU: no extended storage image formats, or no storage writes without a format");
    else if (formats_supported(c->phys)) {
        const VkSubgroupFeatureFlags want = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT;
        G.wave = (sg.supportedOperations & want) == want && (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT);
        answer = 1;
    }
    return answer;
}

// The constants' ring: slots of the two constant blocks and the Lanczos table, at the device's offset alignment.
static int make_ring(void) {
    VkCtx *c = vk_ctx();
    VkDeviceSize align = c->props.limits.minUniformBufferOffsetAlignment;
    if (align < 16) align = 16;
#define ROUND(n) (((n) + align - 1) / align * align)
    G.slot_fsr = ROUND(sizeof(Constants));
    G.slot_spd = ROUND(sizeof(SpdConstants));
    G.stride = G.slot_fsr + G.slot_spd + ROUND(128 * sizeof(int16_t));
#undef ROUND
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = G.stride * RING,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &G.ubo, &G.ubo_mem, &info) != VK_SUCCESS || !info.pMappedData) return 0;
    G.ubo_ptr = info.pMappedData;
    return 1;
}

static int make_pipelines(void) {
    if (!supported()) return 0;
    if (!G.tried) {
        G.tried = 1;
        G.ok = make_samplers() && make_ring() && make_out();
        for (int i = 0; i < PASS_COUNT && G.ok; i++) {
            int needed = G.wave ? (i != P_LUMA_PYRAMID_NOWAVE && i != P_SHADING_CHANGE_PYRAMID_NOWAVE)
                                : (i != P_LUMA_PYRAMID && i != P_SHADING_CHANGE_PYRAMID);
            if (needed) G.ok = make_pass(i);
        }
        if (G.ok) rt_log("fsr3: AMD FSR 3.1 passes are ready (%s subgroup operations)", G.wave ? "with" : "without");
    }
    return G.ok;
}

// The internal images for an evaluation at these sizes, made again when the sizes change.
static int make_images(uint32_t rw, uint32_t rh, uint32_t uw, uint32_t uh) {
    if (G.output.image && G.rw == rw && G.rh == rh && G.uw == uw && G.uh == uh) return 1;
    rw = max_u(1, rw), rh = max_u(1, rh), uw = max_u(1, uw), uh = max_u(1, uh);
    int ok = 1;
    for (int i = 0; i < 2 && ok; i++) {
        ok = ok && make_image(&G.accumulation[i], VK_FORMAT_R8_UNORM, rw, rh, 1);
        ok = ok && make_image(&G.luma[i], VK_FORMAT_R16_SFLOAT, rw, rh, 1);
        // (the pass files declare the luma history as rgba8; the SDK allocates RGBA16F, which a storage format
        // qualifier does not match, so this allocates what the shaders store)
        ok = ok && make_image(&G.luma_history[i], VK_FORMAT_R8G8B8A8_UNORM, rw, rh, 1);
        ok = ok && make_image(&G.internal_upscaled[i], VK_FORMAT_R16G16B16A16_SFLOAT, uw, uh, 1);
    }
    uint32_t hw = max_u(1, rw / 2), hh = max_u(1, rh / 2);
    uint32_t spd_levels = (uint32_t)floor(log2((double)max_u(hw, hh))) + 1;
    ok = ok && make_image(&G.farthest_depth, VK_FORMAT_R16_SFLOAT, rw, rh, 1);
    ok = ok && make_image(&G.shading_change, VK_FORMAT_R8_UNORM, hw, hh, 1);
    ok = ok && make_image(&G.new_locks, VK_FORMAT_R8_UNORM, uw, uh, 1);
    ok = ok && make_image(&G.spd_mips, VK_FORMAT_R16G16_SFLOAT, hw, hh, max_u(spd_levels, SPD_STORED_MIPS));
    ok = ok && make_image(&G.farthest_depth_mip1, VK_FORMAT_R16_SFLOAT, hw, hh, 1);
    ok = ok && make_image(&G.dilated_motion_vectors, VK_FORMAT_R16G16_SFLOAT, rw, rh, 1);
    ok = ok && make_image(&G.dilated_depth, VK_FORMAT_R32_SFLOAT, rw, rh, 1);
    ok = ok && make_image(&G.previous_nearest_depth, VK_FORMAT_R32_UINT, rw, rh, 1);
    ok = ok && make_image(&G.dilated_reactive_masks, VK_FORMAT_R8G8B8A8_UNORM, rw, rh, 1);
    ok = ok && make_image(&G.frame_info, VK_FORMAT_R32G32B32A32_SFLOAT, 1, 1, 1);
    ok = ok && make_image(&G.spd_atomic, VK_FORMAT_R32_UINT, 1, 1, 1);
    ok = ok && make_image(&G.lanczos_lut, VK_FORMAT_R16_SNORM, 128, 1, 1);
    ok = ok && make_image(&G.default_reactive, VK_FORMAT_R8_UNORM, 1, 1, 1);
    ok = ok && make_image(&G.output, VK_FORMAT_R16G16B16A16_SFLOAT, uw, uh, 1);
    G.first = 1;   // new images: their first layout, cleared, the history from nothing
    G.lut_uploaded = 0;
    G.parity = 0;
    if (!ok) {
        size_t n;
        Img *const *all = all_images(&n);
        for (size_t i = 0; i < n; i++) drop_image(all[i]);
        G.rw = G.rh = G.uw = G.uh = 0;
        return 0;
    }
    G.rw = rw, G.rh = rh, G.uw = uw, G.uh = uh;
    rt_log("fsr3: %ux%u -> %ux%u", rw, rh, uw, uh);
    return 1;
}

// One pass: its descriptors pushed in the order the pass file numbers them, then the two samplers it declares at 1000 and
// 1001, then the dispatch, and a barrier after it (every pass reads what the one before it wrote). The pushes go through
// the recorder like the motion pass's (vk_dlss.c mv_pass).
static void run(VkCommandBuffer cb, int pass, const Res *res, uint32_t gx, uint32_t gy) {
    const Pass *info = &PASSES[pass];
    VkDescriptorImageInfo images[16] = {0};
    VkDescriptorBufferInfo buffers[16] = {0};
    VkWriteDescriptorSet w[16] = {0};
    unsigned n = 0;
    for (; n < info->count; n++) {
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = n, .descriptorCount = 1};
        if (info->bind[n] == U) {
            buffers[n] = (VkDescriptorBufferInfo){G.ubo, res[n].offset, res[n].size};
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, w[n].pBufferInfo = &buffers[n];
        } else {
            images[n] = (VkDescriptorImageInfo){VK_NULL_HANDLE, res[n].view, VK_IMAGE_LAYOUT_GENERAL};
            w[n].descriptorType = info->bind[n] == T ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[n].pImageInfo = &images[n];
        }
    }
    VkDescriptorImageInfo point = {G.point, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorImageInfo linear = {G.linear, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1000, .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &point};
    n++;
    w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1001, .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &linear};
    n++;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pass[pass].pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pass[pass].layout, 0, n, w);
    vkCmdDispatch(cb, gx ? gx : 1, gy ? gy : 1, 1);
    barrier_mem(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
}

static uint32_t groups(int32_t size, uint32_t tile) { return ((uint32_t)(size < 1 ? 1 : size) + tile - 1) / tile; }

// The seconds since the previous evaluation, for the passes' time-based terms (16.7 ms when unknown).
static float frame_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    uint64_t prev = G.last_qpc;
    G.last_qpc = (uint64_t)t.QuadPart;
    if (prev) {
        float s = (float)((double)(G.last_qpc - prev) / (double)f.QuadPart);
        return s < 0.0f ? 0.0f : s > 1.0f ? 1.0f : s;
    }
#endif
    return 1.0f / 60.0f;
}

// Records one evaluation into G.output: colour, the scene depth and the motion vectors at the render size in, the picture
// at the upscale size out. t: this frame's inputs (vk_temporal_inputs), slot: the constants' ring slot.
static void evaluate(VkCommandBuffer cb, VkImageView color, const VkTemporalIn *t, const float dtv[4], unsigned slot) {
    const int32_t rw = (int32_t)G.rw, rh = (int32_t)G.rh, uw = (int32_t)G.uw, uh = (int32_t)G.uh;
    const int first = G.first, reset = t->reset || first;
    G.first = 0;

    Constants k = {0};
    k.render_size[0] = rw, k.render_size[1] = rh;
    k.upscale_size[0] = uw, k.upscale_size[1] = uh;
    k.max_render_size[0] = rw, k.max_render_size[1] = rh;
    k.max_upscale_size[0] = uw, k.max_upscale_size[1] = uh;
    const float jit[2] = {t->jx, t->jy};
    for (int i = 0; i < 2; i++) {
        k.previous_render_size[i] = first ? k.render_size[i] : G.prev_render[i];
        k.previous_upscale_size[i] = first ? k.upscale_size[i] : G.prev_upscale[i];
        k.jitter[i] = jit[i];
        k.previous_jitter[i] = first ? jit[i] : G.prev_jitter[i];
        G.prev_render[i] = k.render_size[i], G.prev_upscale[i] = k.upscale_size[i], G.prev_jitter[i] = jit[i];
    }
    memcpy(k.device_to_view_depth, dtv, sizeof k.device_to_view_depth);
    k.tan_half_fov = dtv[2];   // (not read by the 3.1.4 upscaler's passes; set as the SDK sets it)
    k.motion_vector_scale[0] = 1.0f / (float)rw;   // the vectors are in render pixels
    k.motion_vector_scale[1] = 1.0f / (float)rh;
    k.downscale_factor[0] = (float)rw / (float)uw;
    k.downscale_factor[1] = (float)rh / (float)uh;
    k.jitter_phase_count = (float)(t->phases ? t->phases : 8);
    k.delta_time = frame_seconds();
    k.delta_pre_exposure = 1.0f;
    k.view_space_to_meters = 1.0f;
    G.frame_index = reset ? 0.0f : G.frame_index + 1.0f;
    k.frame_index = G.frame_index;
    k.velocity_factor = 1.0f;   // (the SDK's FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_* defaults)
    k.reactiveness_scale = 1.0f;
    k.shading_change_scale = 1.0f;
    k.accumulation_added_per_frame = 1.0f / 3.0f;
    k.min_disocclusion_accumulation = -1.0f / 3.0f;

    const uint32_t spd_x = (G.rw - 1) / 64 + 1, spd_y = (G.rh - 1) / 64 + 1;
    SpdConstants spd = {0};
    double lg = floor(log2((double)max_u(G.rw, G.rh)));
    spd.mips = (uint32_t)(lg < 12.0 ? lg : 12.0);
    spd.num_work_groups = spd_x * spd_y;
    spd.render_size[0] = G.rw, spd.render_size[1] = G.rh;

    // This evaluation owns its slot until its submission is done.
    const VkDeviceSize base = slot * G.stride, lut_at = G.slot_fsr + G.slot_spd;
    uint8_t *mapped = G.ubo_ptr + base;
    memcpy(mapped, &k, sizeof k);
    memcpy(mapped + G.slot_fsr, &spd, sizeof spd);
    if (!G.lut_uploaded) {   // the accumulate pass's Lanczos weights, 128 taps as the SDK builds them
        int16_t weights[128];
        for (int i = 0; i < 128; i++) {
            float x = 2.0f * (float)i / 127.0f, pix = 3.14159265358979f * x;
            float y = fabsf(x) < FLT_EPSILON ? 1.0f : (sinf(pix) / pix) * (sinf(0.5f * pix) / (0.5f * pix));
            weights[i] = (int16_t)lroundf(y * 32767.0f);
        }
        memcpy(mapped + lut_at, weights, sizeof weights);
    }
    vmaFlushAllocation(vk_ctx()->vma, G.ubo_mem, base, G.stride);

    const uint32_t p = G.parity, q = p ^ 1u;

    // What was written before (the game's colour, the depth, the vector pass) has to be visible, and new images need
    // their first layout.
    barrier_mem(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    size_t nall;
    Img *const *all = all_images(&nall);
    if (first) {
        VkImageMemoryBarrier2 b[24];
        for (size_t i = 0; i < nall; i++)
            b[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                           .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                           .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                           .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                           .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                           .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = all[i]->image,
                                           .subresourceRange = ALL};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = (uint32_t)nall,
                               .pImageMemoryBarriers = b};
        vkCmdPipelineBarrier2(cb, &di);
        VkClearColorValue zero = {{0}};
        for (size_t i = 0; i < nall; i++) vkCmdClearColorImage(cb, all[i]->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &ALL);
    }
    if (!G.lut_uploaded) {
        G.lut_uploaded = 1;
        VkBufferImageCopy copy = {.bufferOffset = base + lut_at,
                                  .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {128, 1, 1}};
        vkCmdCopyBufferToImage(cb, G.ubo, G.lanczos_lut.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    }
    // Per frame: the nearest-depth scratch (the passes take the minimum of a forward depth, so it starts at 1.0, not 0),
    // the SPD counter and its mips; the history on a reset.
    VkClearColorValue zero = {{0}}, one_bits = {.uint32 = {0x3f800000u, 0, 0, 0}};
    vkCmdClearColorImage(cb, G.previous_nearest_depth.image, VK_IMAGE_LAYOUT_GENERAL, &one_bits, 1, &ALL);
    vkCmdClearColorImage(cb, G.spd_atomic.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &ALL);
    vkCmdClearColorImage(cb, G.spd_mips.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &ALL);
    if (reset) {
        vkCmdClearColorImage(cb, G.accumulation[p].image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &ALL);
        VkClearColorValue info = {.float32 = {-1.0f, 1.0f, 0.0f, 0.0f}};
        vkCmdClearColorImage(cb, G.frame_info.image, VK_IMAGE_LAYOUT_GENERAL, &info, 1, &ALL);
    }
    barrier_mem(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

#define V(img) ((Res){.view = (img).view})
#define MIP(l) ((Res){.view = G.spd_mips.levels[(l) < G.spd_mips.nlevels ? (l) : G.spd_mips.nlevels - 1]})
    const Res col = {.view = color}, depth = {.view = t->depth}, motion = {.view = t->mv};
    const Res exposure = V(G.frame_info);   // FSR computes its own exposure
    const Res cur_luma = V(G.luma[p]);
    const Res cb_fsr = {.offset = base, .size = G.slot_fsr};
    const Res cb_spd = {.offset = base + G.slot_fsr, .size = G.slot_spd};

    run(cb, P_PREPARE_INPUTS,
        (Res[]){motion, depth, col, V(G.dilated_motion_vectors), V(G.dilated_depth), V(G.previous_nearest_depth),
                V(G.farthest_depth), cur_luma, cb_fsr},
        groups(rw, 8), groups(rh, 8));
    run(cb, G.wave ? P_LUMA_PYRAMID : P_LUMA_PYRAMID_NOWAVE,
        (Res[]){cur_luma, V(G.farthest_depth), V(G.spd_atomic), V(G.frame_info), MIP(0), MIP(1), MIP(2), MIP(3), MIP(4),
                MIP(5), V(G.farthest_depth_mip1), cb_fsr, cb_spd},
        spd_x, spd_y);
    // The two pyramid passes share one atomic counter, and the single-pass downsampler uses it to elect the workgroup
    // that finishes the chain: it counts up to num_work_groups and the last one through does the tail. Cleared only once
    // a frame, the second pass would start at the total, elect no workgroup, and the ones waiting on the tail would spin.
    barrier_mem(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdClearColorImage(cb, G.spd_atomic.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &ALL);
    barrier_mem(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    run(cb, G.wave ? P_SHADING_CHANGE_PYRAMID : P_SHADING_CHANGE_PYRAMID_NOWAVE,
        (Res[]){cur_luma, V(G.luma[q]), V(G.dilated_motion_vectors), exposure, V(G.spd_atomic), MIP(0), MIP(1), MIP(2),
                MIP(3), MIP(4), MIP(5), cb_fsr, cb_spd},
        spd_x, spd_y);
    run(cb, P_SHADING_CHANGE, (Res[]){V(G.spd_mips), V(G.shading_change), cb_fsr}, groups(rw / 2, 8), groups(rh / 2, 8));
    run(cb, P_PREPARE_REACTIVITY,
        (Res[]){V(G.previous_nearest_depth), V(G.dilated_motion_vectors), V(G.dilated_depth), V(G.default_reactive),
                V(G.default_reactive), V(G.accumulation[p]), V(G.shading_change), cur_luma, exposure,
                V(G.dilated_reactive_masks), V(G.new_locks), V(G.accumulation[q]), cb_fsr},
        groups(rw, 8), groups(rh, 8));
    run(cb, P_LUMA_INSTABILITY,
        (Res[]){exposure, V(G.dilated_reactive_masks), V(G.dilated_motion_vectors), V(G.frame_info), V(G.luma_history[p]),
                V(G.farthest_depth_mip1), cur_luma, V(G.luma_history[q]), V(G.farthest_depth), cb_fsr},
        groups(rw, 8), groups(rh, 8));
    run(cb, P_ACCUMULATE,
        (Res[]){exposure, V(G.dilated_reactive_masks), V(G.dilated_motion_vectors), V(G.internal_upscaled[p]),
                V(G.lanczos_lut), V(G.farthest_depth_mip1), cur_luma, V(G.farthest_depth), col, V(G.internal_upscaled[q]),
                V(G.output), V(G.new_locks), cb_fsr},
        groups(uw, 8), groups(uh, 8));
#undef V
#undef MIP
    G.parity = q;
}

static VkImageView color_view(VkImage image, VkFormat fmt, VkImageUsageFlags usage) {
    VkImageViewUsageCreateInfo u = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = usage};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &u, .image = image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView v = VK_NULL_HANDLE;
    return vkCreateImageView(vk_ctx()->device, &vi, NULL, &v) == VK_SUCCESS ? v : VK_NULL_HANDLE;
}

// A free slot of the constants' ring (the submission that used it is done), or -1.
static int ring_take(uint64_t fence) {
    uint64_t done = vk_sync_fence_done();
    for (int i = 0; i < RING; i++) {
        unsigned s = (G.ring_next + (unsigned)i) % RING;
        if (G.ring_fence[s] <= done) {
            G.ring_fence[s] = fence;
            G.ring_next = s + 1;
            return (int)s;
        }
    }
    return -1;
}

int vk_fsr3_record(VkCommandBuffer cb, uint64_t fence, const VkTexImage *in, const VkTexImage *out) {
    if (!vk_fsr3_wanted()) return 0;
    uint32_t w = in->width, h = in->height, ow = out->width, oh = out->height;
    if (w < 32 || h < 32 || w > ow || h > oh) {   // this pair of images only: the game's upscaler runs for it
        static int logged;
        if (!logged++) rt_log("fsr3: unsupported size %ux%u -> %ux%u: the game's upscaler runs for it", w, h, ow, oh);
        return 0;
    }
    if (!make_pipelines()) return fail("this GPU lacks what the FSR 3.1 passes need, or a pass could not be made");
    // Everything that can fail without a scene comes before vk_temporal_inputs, so a 0 never follows recorded commands
    // (the motion pass binds its own pipeline, and on 0 the game's upscaler would dispatch on it).
    int slot = ring_take(fence);   // all in flight: the game's upscaler this frame
    if (slot < 0) {
        VkTemporalIn none = {0};   // records nothing; clears the jitter and the noted scene
        vk_temporal_done(&none, w, h, ow);
        G.spatial++;
        return 0;
    }
    if (!make_images(w, h, ow, oh)) {
        G.ring_fence[slot] = 0;
        return fail("its internal images could not be made");
    }
    VkImageView iv = color_view(in->image, in->format, VK_IMAGE_USAGE_SAMPLED_BIT);
    VkImageView ov = color_view(out->image, out->format, VK_IMAGE_USAGE_STORAGE_BIT);
    if (!iv || !ov) {
        vk_defer_view(iv), vk_defer_view(ov);
        G.ring_fence[slot] = 0;
        return fail("no sampled view of the input or storage view of the output");
    }
    VkTemporalIn t;
    if (!vk_temporal_inputs(cb, w, h, ow, oh, 0, &t)) {
        vk_defer_view(iv), vk_defer_view(ov);
        G.ring_fence[slot] = 0;
        return fail("its motion-vector image could not be made");
    }
    float dtv[4];
    // Without a scene at the input size (a menu, a cutscene's own passes) the game's upscaler runs this frame, and the
    // next temporal one starts from a reset. A scene whose projection FSR cannot use is the same, but the motion pass
    // may have bound its own pipeline: -1, and the caller binds the game's again.
    if (!t.temporal || !vk_fsr3_camera_factors(t.proj, dtv)) {
        int bound = t.temporal;
        t.temporal = 0;   // no jitter for the next frame either
        vk_temporal_done(&t, w, h, ow);
        vk_defer_view(iv), vk_defer_view(ov);
        G.ring_fence[slot] = 0;
        G.spatial++;
        return bound ? -1 : 0;
    }
    evaluate(cb, iv, &t, dtv, (unsigned)slot);
    // FSR's picture into the game's output image.
    int32_t size[2] = {(int32_t)ow, (int32_t)oh};
    VkDescriptorImageInfo ii[2] = {{VK_NULL_HANDLE, G.output.view, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, ov, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii[1]}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.out.pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.out.layout, 0, 2, wr);
    vkCmdPushConstants(cb, G.out.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof size, size);
    vkCmdDispatch(cb, (ow + 7) / 8, (oh + 7) / 8, 1);
    vk_defer_view(iv), vk_defer_view(ov);
    vk_temporal_done(&t, w, h, ow);
    G.frames++, G.temporal++, G.resets += t.reset;
    if (G.frames == 1 || G.frames % 1800 == 0)
        rt_log("fsr3: %llu frames, %llu temporal, %llu resets; %ux%u -> %ux%u, jitter %+.3f %+.3f px of %u phases",
               (unsigned long long)G.frames, (unsigned long long)G.temporal, (unsigned long long)G.resets, w, h, ow, oh,
               t.jx, t.jy, t.phases);
    return 1;
}

void vk_fsr3_release(void) {   // device idle
    if (!G.tried) return;
    VkCtx *c = vk_ctx();
    rt_log("fsr3: %llu frames, %llu temporal, %llu resets (%llu left to the game's upscaler)", (unsigned long long)G.frames,
           (unsigned long long)G.temporal, (unsigned long long)G.resets, (unsigned long long)G.spatial);
    size_t n;
    Img *const *all = all_images(&n);
    for (size_t i = 0; i < n; i++) destroy_now(all[i]);
    for (int i = 0; i <= PASS_COUNT; i++) {
        Pipe *p = i < PASS_COUNT ? &G.pass[i] : &G.out;
        if (p->pipe) vkDestroyPipeline(c->device, p->pipe, NULL);
        if (p->layout) vkDestroyPipelineLayout(c->device, p->layout, NULL);
        if (p->set) vkDestroyDescriptorSetLayout(c->device, p->set, NULL);
    }
    if (G.point) vkDestroySampler(c->device, G.point, NULL);
    if (G.linear) vkDestroySampler(c->device, G.linear, NULL);
    if (G.ubo) vmaDestroyBuffer(c->vma, G.ubo, G.ubo_mem);
    memset(&G, 0, sizeof G);
    G.wave = 1;
}
