// AMD FSR 3.1 frame interpolation, and the optical flow it runs on, recorded as our own Vulkan compute passes: the made
// frame between two of the game's when Frame generation is 2x and frame_gen_by = fsr3 (TOTK_FRAME_GEN_BY; opt-in,
// experimental). A C rewrite of Hookshot's gpu/vk/vk_fsr3fi.cpp
// (https://git.notrexed.dev/Hookshot/Hookshot @ 7191250b7b0f8a4becd79ee086229a2620872fc7, GPL-3.0).
//
// vk_framegen.c makes frames from colour alone. This answers the same question with AMD's passes, which use the scene's
// depth and motion as well: seventeen compute passes, seven of optical flow over two frames' luma and ten of
// interpolation over the game's motion and depth, all of them our own dispatches over AMD's FidelityFX SDK v1.1.4
// (runtime/third_party/fidelityfx, compiled by tools/build_shaders.py into runtime/fsr3_fi_*_spv.h and
// runtime/fsr3_of_*_spv.h), with no FidelityFX host library linked and nothing of AMD's loaded at run time.
//
// The depth and the motion are the ones a temporal upscaler was given this frame: vk_dlss.c vk_temporal_inputs makes
// them for DLSS, FSR 3.1 and FSR 4 alike, and vk_temporal_done, which each of them calls after its evaluation, hands
// them here (vk_fsr3fi_note), so this runs only while one of those upscalers draws. vk_backend.c present_image takes
// them for the frame it presents (vk_fsr3fi_take) and vk_framegen.c hands them to vk_fsr3fi_generate with the frame's
// colour, the frame before it and the image to write. The present drains before the replay submits the next frame
// (vk_backend.c submit_sync), so the depth and motion images still hold this frame's when that command buffer runs.
// Where it cannot run (not asked for, a device without the formats, features or subgroup operations the passes need, a
// frame without its inputs, a reset) the built-in interpolation runs exactly as before.
//
// FSR 3.1's interpolation has no phase input (ffx_frameinterpolation.h fixes t at 0.5): it makes the midpoint only, so
// it is used for the one made frame a game frame at 2x. The game draws its HUD into the picture the passes read, so the
// HUD is interpolated with the scene, as it is by the built-in interpolation. The SDK can be given the picture without
// the HUD as well (HUDLessColor); this runtime does not make it, so those inputs are left out of the bindings.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"

#include "fsr3_fi_disocclusion_mask_spv.h"
#include "fsr3_fi_game_motion_vector_field_spv.h"
#include "fsr3_fi_game_vector_field_inpainting_pyramid_spv.h"
#include "fsr3_fi_inpainting_pyramid_spv.h"
#include "fsr3_fi_inpainting_spv.h"
#include "fsr3_fi_interpolate_spv.h"
#include "fsr3_fi_optical_flow_vector_field_spv.h"
#include "fsr3_fi_reconstruct_and_dilate_spv.h"
#include "fsr3_fi_reconstruct_previous_depth_spv.h"
#include "fsr3_fi_setup_spv.h"
#include "fsr3_of_compute_optical_flow_v5_spv.h"
#include "fsr3_of_compute_scd_divergence_spv.h"
#include "fsr3_of_filter_optical_flow_v5_spv.h"
#include "fsr3_of_generate_scd_histogram_spv.h"
#include "fsr3_of_luminance_pyramid_spv.h"
#include "fsr3_of_prepare_luma_spv.h"
#include "fsr3_of_scale_optical_flow_v5_spv.h"

// ---- the passes and what each one binds

// A descriptor's kind. The optical-flow passes declare no sampler at all (every load there is a texelFetch); the
// interpolation passes declare one, a linear clamp sampler at binding 1000.
enum { T, I, U, B };   // sampled texture, storage image, uniform buffer, storage buffer

// One descriptor, at the binding number its pass's GLSL fixes it to. The numbers are not an index into the list: three
// of the seventeen passes leave gaps where the SDK's HLSL root signature has a slot the GLSL does not declare.
typedef struct { uint16_t binding; uint8_t kind; } Bind;

typedef struct {
    const char *name;
    const uint32_t *spv;
    size_t spv_bytes;
    const Bind *binds;
    uint32_t bind_count;
    int sampler;   // declares the linear clamp sampler at binding 1000
} PassInfo;

#define MAX_BIND 18   // the game vector field's pyramid

// Optical flow. Seven passes, no samplers.
static const Bind OF_PREPARE_LUMA[] = {{0, T}, {1, I}, {2, U}};
static const Bind OF_LUMINANCE_PYRAMID[] = {{0, I}, {1, I}, {2, I}, {3, I}, {4, I}, {5, I}, {6, I}, {7, U}, {8, U}};
static const Bind OF_SCD_HISTOGRAM[] = {{0, T}, {1, I}, {2, U}};
static const Bind OF_SCD_DIVERGENCE[] = {{0, I}, {1, I}, {2, I}, {3, I}, {4, U}};
static const Bind OF_SEARCH[] = {{0, T}, {1, T}, {2, I}, {3, I}, {4, U}};
static const Bind OF_FILTER[] = {{0, T}, {1, I}, {2, U}};
static const Bind OF_SCALE[] = {{0, T}, {1, T}, {2, T}, {3, I}, {4, I}, {5, U}};

// Frame interpolation. Ten passes, each with the linear clamp sampler.
static const Bind FI_RECONSTRUCT_AND_DILATE[] = {{0, T}, {1, T}, {2, I}, {3, I}, {4, I}, {5, U}};
static const Bind FI_SETUP[] = {{0, T}, {1, I}, {2, I}, {3, I}, {4, I}, {5, I}, {6, B}, {7, U}};
static const Bind FI_RECONSTRUCT_PREVIOUS_DEPTH[] = {{0, T}, {1, T}, {2, T}, {3, T}, {4, I}, {5, U}};
static const Bind FI_GAME_MOTION_VECTOR_FIELD[] = {{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {5, I}, {6, I}, {7, U}};
// The game vector field's inpainting pyramid: two textures, the counter buffer, thirteen mip levels, then both constant
// blocks. The pyramid is a single-pass downsampler, so it wants every level bound at once.
static const Bind FI_GAME_VECTOR_FIELD_PYRAMID[] = {
    {0, T}, {1, T}, {2, B}, {3, I}, {4, I}, {5, I}, {6, I}, {7, I}, {8, I},
    {9, I}, {10, I}, {11, I}, {12, I}, {13, I}, {14, I}, {15, I}, {16, U}, {17, U}};
// Binding 5 is not declared by this pass (an optical-flow debug view in the SDK's HLSL), so the list jumps from 4 to 6.
static const Bind FI_OPTICAL_FLOW_VECTOR_FIELD[] = {{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {6, I}, {7, I}, {8, U}};
static const Bind FI_DISOCCLUSION_MASK[] = {{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {5, T}, {6, T}, {7, I}, {8, U}};
static const Bind FI_INTERPOLATE[] = {{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {5, T}, {6, T}, {7, T}, {8, B}, {9, I}, {10, U}};
static const Bind FI_INPAINTING_PYRAMID[] = {{0, T}, {1, B}, {2, I}, {3, I}, {4, I}, {5, I}, {6, I}, {7, I}, {8, I},
                                             {9, I}, {10, I}, {11, I}, {12, I}, {13, I}, {14, I}, {15, U}, {16, U}};
// Bindings 4 to 6 are the HUD-less and UI inputs, which the GLSL does not declare without the option for them.
static const Bind FI_INPAINTING[] = {{0, T}, {1, T}, {2, T}, {3, T}, {7, I}, {8, U}};

#define PASS(n, spv, b, s) {n, spv, sizeof spv, b, sizeof b / sizeof b[0], s}

enum {
    P_OF_PREPARE_LUMA, P_OF_LUMINANCE_PYRAMID, P_OF_SCD_HISTOGRAM, P_OF_SCD_DIVERGENCE, P_OF_SEARCH, P_OF_FILTER,
    P_OF_SCALE, P_FI_RECONSTRUCT_AND_DILATE, P_FI_SETUP, P_FI_RECONSTRUCT_PREVIOUS_DEPTH, P_FI_GAME_MOTION_VECTOR_FIELD,
    P_FI_GAME_VECTOR_FIELD_PYRAMID, P_FI_OPTICAL_FLOW_VECTOR_FIELD, P_FI_DISOCCLUSION_MASK, P_FI_INTERPOLATE,
    P_FI_INPAINTING_PYRAMID, P_FI_INPAINTING, P_COUNT,
};

// The passes, in the order of the enum above.
static const PassInfo PASSES[] = {
    PASS("of prepare_luma", fsr3_of_prepare_luma_spv, OF_PREPARE_LUMA, 0),
    PASS("of luminance_pyramid", fsr3_of_luminance_pyramid_spv, OF_LUMINANCE_PYRAMID, 0),
    PASS("of scd_histogram", fsr3_of_generate_scd_histogram_spv, OF_SCD_HISTOGRAM, 0),
    PASS("of scd_divergence", fsr3_of_compute_scd_divergence_spv, OF_SCD_DIVERGENCE, 0),
    PASS("of search", fsr3_of_compute_optical_flow_v5_spv, OF_SEARCH, 0),
    PASS("of filter", fsr3_of_filter_optical_flow_v5_spv, OF_FILTER, 0),
    PASS("of scale", fsr3_of_scale_optical_flow_v5_spv, OF_SCALE, 0),
    PASS("fi reconstruct_and_dilate", fsr3_fi_reconstruct_and_dilate_spv, FI_RECONSTRUCT_AND_DILATE, 1),
    PASS("fi setup", fsr3_fi_setup_spv, FI_SETUP, 1),
    PASS("fi reconstruct_previous_depth", fsr3_fi_reconstruct_previous_depth_spv, FI_RECONSTRUCT_PREVIOUS_DEPTH, 1),
    PASS("fi game_motion_vector_field", fsr3_fi_game_motion_vector_field_spv, FI_GAME_MOTION_VECTOR_FIELD, 1),
    PASS("fi game_vector_field_pyramid", fsr3_fi_game_vector_field_inpainting_pyramid_spv, FI_GAME_VECTOR_FIELD_PYRAMID, 1),
    PASS("fi optical_flow_vector_field", fsr3_fi_optical_flow_vector_field_spv, FI_OPTICAL_FLOW_VECTOR_FIELD, 1),
    PASS("fi disocclusion_mask", fsr3_fi_disocclusion_mask_spv, FI_DISOCCLUSION_MASK, 1),
    PASS("fi interpolate", fsr3_fi_interpolate_spv, FI_INTERPOLATE, 1),
    PASS("fi inpainting_pyramid", fsr3_fi_inpainting_pyramid_spv, FI_INPAINTING_PYRAMID, 1),
    PASS("fi inpainting", fsr3_fi_inpainting_spv, FI_INPAINTING, 1),
};
#undef PASS
_Static_assert(sizeof PASSES / sizeof PASSES[0] == P_COUNT, "the enum must name every pass");

// ---- the constant blocks

// cbFI, exactly as ffx_frameinterpolation_private.h lays out FrameInterpolationConstants and as the std140 block in
// ffx_frameinterpolation_callbacks_glsl.h reads it. Every pass reads it by offset, so a field out of place is a wrong
// picture and not a compile error; the size is the one part of the layout a compiler can check.
typedef struct {
    int32_t render_size[2];
    int32_t display_size[2];
    float display_size_rcp[2];
    float camera_near;
    float camera_far;
    int32_t upscaler_target_size[2];
    int32_t mode;
    int32_t reset;
    float device_to_view_depth[4];
    float delta_time;
    int32_t hudless_attached_factor;   // 0: no picture without the HUD is given
    int32_t distortion_field_size[2];
    float optical_flow_scale[2];
    int32_t optical_flow_block_size;
    uint32_t dispatch_flags;
    int32_t max_render_size[2];
    int32_t optical_flow_half_res_mode;
    int32_t num_instances;
    int32_t interpolation_rect_base[2];
    int32_t interpolation_rect_size[2];
    float debug_bar_color[3];
    uint32_t back_buffer_transfer_function;
    float min_max_luminance[2];
    float tan_half_fov;
    int32_t pad1;
    float jitter[2];
    float motion_vector_scale[2];
} Constants;
_Static_assert(sizeof(Constants) == 176, "cbFI must match the SDK's layout");

// cbInpaintingPyramid: the single-pass downsampler's setup, staged twice a frame because the two pyramid passes
// downsample rectangles of different sizes.
typedef struct { uint32_t mips, num_work_groups, work_group_offset[2]; } PyramidConstants;
_Static_assert(sizeof(PyramidConstants) == 16, "cbInpaintingPyramid must match the SDK's layout");

// cbOF, as ffx_opticalflow_private.h lays out OpticalflowConstants. The pyramid level in it changes between dispatches,
// so there is a block per level rather than one a frame.
typedef struct {
    int32_t input_luma_resolution[2];
    uint32_t pyramid_level;
    uint32_t pyramid_level_count;
    int32_t frame_index;
    uint32_t back_buffer_transfer_function;
    float min_max_luminance[2];
} OfConstants;
_Static_assert(sizeof(OfConstants) == 32, "cbOF must match the SDK's layout");

// cbOF_SPD. Only numWorkGroupOpticalFlowInputPyramid is read by the luma pyramid (its mip count is fixed at six in the
// shader itself), but the whole block is written because the GLSL declares it.
typedef struct { uint32_t mips, num_work_groups, work_group_offset[2], num_work_groups_pyramid, pad0, pad1, pad2; } OfSpdConstants;
_Static_assert(sizeof(OfSpdConstants) == 32, "cbOF_SPD must match the SDK's layout");

// ---- state

#define OF_LEVELS 7         // advancedAlgorithmIterations in ffx_opticalflow.cpp
#define OF_BLOCK 8          // opticalFlowBlockSize, likewise
#define PYRAMID_MIPS 13     // the mip images the two inpainting pyramid passes declare
#define SCD_HISTOGRAM_W (256 * (3 * 3))   // HistogramBins * HistogramsPerDim^2
#define COUNTERS_BYTES 16   // two uints, rounded up: COUNTER_SPD and the reset index

typedef struct { VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkPipeline pipeline; } Pipe;

typedef struct {
    VkImage image;
    VmaAllocation alloc;
    VkImageView view;                    // every level
    VkImageView levels[PYRAMID_MIPS];    // one per level, for the mipped ones
    uint32_t width, height, level_count;
} Img;

// One descriptor of a dispatch: an image view, or a range of a buffer.
typedef struct { VkImageView view; VkBuffer buffer; VkDeviceSize offset, size; } Res;

static struct {
    int capable;            // 0 not asked, 1 yes, -1 no
    int tried, ok, failed;  // failed: an allocation failed once; the images are not made again every frame
    PFN_vkCmdPushDescriptorSetKHR push;
    Pipe passes[P_COUNT];
    VkSampler linear;
    // Optical flow's internal images. The luma pyramids ping-pong on frame parity: one holds this frame's luma and the
    // other the frame before it, which is the pair the search works on.
    Img of_luma[2];                 // R8_UINT at display size, seven levels
    Img of_flow[2][OF_LEVELS];      // R16G16_SINT, one image a level (their sizes are halved rounding up)
    Img of_vector;                  // the flow the interpolation reads, at level 0's size
    Img of_confidence;              // zero: this flow produces none, and the SDK binds none
    Img of_scd_histogram, of_scd_previous_histogram, of_scd_temp, of_scd_output;
    // Frame interpolation's internal images. The first three are the SDK's "shared" resources, which its own FSR 3
    // pairing fills from the upscaler; here the reconstruct-and-dilate pass fills them.
    Img fi_dilated_depth, fi_dilated_motion_vectors, fi_reconstructed_depth_previous;
    Img fi_reconstructed_depth_interpolated;
    Img fi_game_mv_x, fi_game_mv_y, fi_of_mv_x, fi_of_mv_y;
    Img fi_disocclusion_mask, fi_inpainting_pyramid;
    Img fi_distortion_field;        // 1x1 and zero: this game has no lens distortion
    // The counters the two pyramid passes and the setup pass share: a storage buffer, as the GLSL declares them.
    VkBuffer counters;
    VmaAllocation counters_alloc;
    // The constant blocks of one evaluation. One set is enough: vk_framegen.c waits for its command buffer's fence
    // before it records the next one.
    VkBuffer ubo;
    VmaAllocation ubo_alloc;
    uint8_t *ubo_map;
    VkDeviceSize slot_fi, slot_pyramid, slot_of, slot_of_spd, stride;
    uint32_t width, height, render_width, render_height;
    int ready;
    uint32_t flow_width[OF_LEVELS], flow_height[OF_LEVELS];
    uint32_t parity;
    int32_t of_frame_index;
    int first_frame;
    uint64_t sequence;              // VkFiInputs.sequence of the frame the luma history holds
    int in_general;                 // the UNDEFINED -> GENERAL barriers have been recorded
    uint64_t made;
} G;

static Img *all_images(uint32_t i) {
    static Img *const fixed[] = {
        &G.of_luma[0], &G.of_luma[1], &G.of_vector, &G.of_confidence, &G.of_scd_histogram, &G.of_scd_previous_histogram,
        &G.of_scd_temp, &G.of_scd_output, &G.fi_dilated_depth, &G.fi_dilated_motion_vectors,
        &G.fi_reconstructed_depth_previous, &G.fi_reconstructed_depth_interpolated, &G.fi_game_mv_x, &G.fi_game_mv_y,
        &G.fi_of_mv_x, &G.fi_of_mv_y, &G.fi_disocclusion_mask, &G.fi_inpainting_pyramid, &G.fi_distortion_field};
    const uint32_t n = sizeof fixed / sizeof fixed[0];
    if (i < n) return fixed[i];
    i -= n;
    return i < 2 * OF_LEVELS ? &G.of_flow[i / OF_LEVELS][i % OF_LEVELS] : NULL;
}
#define N_ALL (19 + 2 * OF_LEVELS)

static const VkImageSubresourceRange ALL_LEVELS = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};

// The inputs of the frame being presented, noted by the upscaler and taken at the present (see the top).
static PlatMutex nl = PLAT_MUTEX_INIT;
static VkFiInputs noted;
static uint64_t note_count;

static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src,
                           VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst) {
    VkMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src_stage, .srcAccessMask = src,
                          .dstStageMask = dst_stage, .dstAccessMask = dst};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
}

static void destroy_image(VkCtx *c, Img *t) {   // device idle
    for (uint32_t i = 0; i < PYRAMID_MIPS; i++)
        if (t->levels[i]) vkDestroyImageView(c->device, t->levels[i], NULL);
    if (t->view) vkDestroyImageView(c->device, t->view, NULL);
    if (t->image) vmaDestroyImage(c->vma, t->image, t->alloc);
    memset(t, 0, sizeof *t);
}

// One of the SDK's internal images: storage and sampled, GENERAL throughout, with a view per level when it has more
// than one (both pyramids bind their levels separately while the passes that read a pyramid read it as one mipped
// texture).
static int make_image(Img *t, VkFormat format, uint32_t width, uint32_t height, uint32_t levels) {
    VkCtx *c = vk_ctx();
    t->width = width ? width : 1, t->height = height ? height : 1;
    t->level_count = levels < 1 ? 1 : levels > PYRAMID_MIPS ? PYRAMID_MIPS : levels;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
                            .extent = {t->width, t->height, 1}, .mipLevels = t->level_count, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &ai, &t->image, &t->alloc, NULL) != VK_SUCCESS) {
        t->image = VK_NULL_HANDLE;
        rt_log("fsr3fi: a %ux%u image (format %d) was not made", t->width, t->height, (int)format);
        return 0;
    }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t->level_count, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &t->view) != VK_SUCCESS) {
        t->view = VK_NULL_HANDLE;
        return 0;
    }
    for (uint32_t level = 0; t->level_count > 1 && level < t->level_count; level++) {
        vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1};
        if (vkCreateImageView(c->device, &vi, NULL, &t->levels[level]) != VK_SUCCESS) {
            t->levels[level] = VK_NULL_HANDLE;
            return 0;
        }
    }
    return 1;
}

static int make_pipeline(Pipe *p, const PassInfo *info) {
    VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutBinding bindings[MAX_BIND + 1];
    uint32_t n = 0;
    for (; n < info->bind_count; n++) {
        static const VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        bindings[n] = (VkDescriptorSetLayoutBinding){info->binds[n].binding, types[info->binds[n].kind], 1,
                                                     VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    }
    if (info->sampler)
        bindings[n++] = (VkDescriptorSetLayoutBinding){1000, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                          .bindingCount = n, .pBindings = bindings};
    if (vkCreateDescriptorSetLayout(c->device, &li, NULL, &p->dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &p->dsl};
    if (vkCreatePipelineLayout(c->device, &pi, NULL, &p->layout) != VK_SUCCESS) return 0;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = info->spv_bytes,
                                   .pCode = info->spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = p->layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipeline);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) {
        rt_log("fsr3fi: the %s pass pipeline was not made (%d)", info->name, (int)r);
        p->pipeline = VK_NULL_HANDLE;
        return 0;
    }
    return 1;
}

// What this device can do, which cannot change while it runs: asked once and kept. The interpolation and inpainting
// passes write (and the inpainting pass also reads) the output through an image declared with no format qualifier,
// because the SDK lets the application choose the backbuffer's format: StorageImageRead/WriteWithoutFormat. Most of
// the internal images are in formats outside Vulkan's mandatory storage set. Both single-pass downsamplers and the
// flow search use subgroup arithmetic and quad operations. vk_device.c enables every feature the device reports.
static int capable(void) {
    if (G.capable) return G.capable > 0;
    VkCtx *c = vk_ctx();
    if (!c || !c->device) return 0;   // asked before the device is made: not an answer to keep
    G.capable = -1;
    G.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkPhysicalDevicePushDescriptorPropertiesKHR pd = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
    VkPhysicalDeviceSubgroupProperties sg = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext = &pd};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sg};
    vkGetPhysicalDeviceProperties2(c->phys, &p2);
    if (!G.push || pd.maxPushDescriptors < MAX_BIND + 1) {   // the widest pass pushes 18 descriptors and the sampler
        rt_log("fsr3fi: this GPU has no push descriptors, or fewer than %d in a set: the built-in interpolation runs",
               MAX_BIND + 1);
        return 0;
    }
    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceFeatures(c->phys, &feats);
    if (!feats.shaderStorageImageWriteWithoutFormat || !feats.shaderStorageImageReadWithoutFormat ||
        !feats.shaderStorageImageExtendedFormats) {
        rt_log("fsr3fi: this GPU cannot read or write a storage image without a format, or has no extended storage "
               "formats: the built-in interpolation runs");
        return 0;
    }
    static const VkFormat needed[] = {VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_SFLOAT,
                                      VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
    for (size_t i = 0; i < sizeof needed / sizeof needed[0]; i++) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(c->phys, needed[i], &props);
        const VkFormatFeatureFlags want = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((props.optimalTilingFeatures & want) != want) {
            rt_log("fsr3fi: format %d cannot be a storage and sampled image on this GPU: the built-in interpolation runs",
                   (int)needed[i]);
            return 0;
        }
    }
    const VkSubgroupFeatureFlags want = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                        VK_SUBGROUP_FEATURE_QUAD_BIT;
    if ((sg.supportedOperations & want) != want || !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        rt_log("fsr3fi: this GPU's compute stage lacks the subgroup arithmetic or quad operations the pyramid and flow "
               "search passes were compiled with (%#x): the built-in interpolation runs", (unsigned)sg.supportedOperations);
        return 0;
    }
    G.capable = 1;
    rt_log("fsr3fi: AMD FSR 3.1 frame interpolation can run on this GPU");
    return 1;
}

int vk_fsr3fi_wanted(void) {
    // the setting first: a device is not asked anything, and nothing is logged, while it is not chosen
    return !strcmp(cfg_value("TOTK_FRAME_GEN_BY"), "fsr3") && capable() && !G.failed;
}

// The sampler, the counters, the constants buffer and the pipelines, once per device.
static int setup(void) {
    if (G.tried) return G.ok;
    G.tried = 1;
    VkCtx *c = vk_ctx();
    // The one sampler the interpolation passes declare: linear, clamped to the edge.
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                              .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
                              .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = VK_LOD_CLAMP_NONE};
    if (vkCreateSampler(c->device, &si, NULL, &G.linear) != VK_SUCCESS) {
        G.linear = VK_NULL_HANDLE;
        return 0;
    }
    // TRANSFER_DST: the counters are zeroed with vkCmdFillBuffer before the first frame
    VkBufferCreateInfo cbi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = COUNTERS_BYTES,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo cai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(c->vma, &cbi, &cai, &G.counters, &G.counters_alloc, NULL) != VK_SUCCESS) {
        G.counters = VK_NULL_HANDLE;
        return 0;
    }
    // Optical flow keeps a block per pyramid level, plus the one for the passes before the pyramid.
    VkDeviceSize align = c->props.limits.minUniformBufferOffsetAlignment;
    if (align < 16) align = 16;
#define ROUND_UP(n) (((VkDeviceSize)(n) + align - 1) / align * align)
    G.slot_fi = ROUND_UP(sizeof(Constants));
    G.slot_pyramid = ROUND_UP(sizeof(PyramidConstants));
    G.slot_of = ROUND_UP(sizeof(OfConstants));
    G.slot_of_spd = ROUND_UP(sizeof(OfSpdConstants));
#undef ROUND_UP
    G.stride = G.slot_fi + 2 * G.slot_pyramid + (OF_LEVELS + 1) * G.slot_of + G.slot_of_spd;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = G.stride,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &G.ubo, &G.ubo_alloc, &info) != VK_SUCCESS) {
        G.ubo = VK_NULL_HANDLE;
        return 0;
    }
    G.ubo_map = info.pMappedData;
    for (int i = 0; i < P_COUNT; i++)
        if (!make_pipeline(&G.passes[i], &PASSES[i])) return 0;
    rt_log("fsr3fi: %d pass pipelines ready", (int)P_COUNT);
    return G.ok = 1;
}

static uint32_t log2_floor(uint32_t v) {
    uint32_t n = 0;
    while (v >>= 1) n++;
    return n;
}

static void free_images(VkCtx *c) {   // device idle
    for (uint32_t i = 0; i < N_ALL; i++) destroy_image(c, all_images(i));
    G.ready = 0;
    G.width = G.height = G.render_width = G.render_height = 0;
}

// The images for a display size and a render size. A new size waits for the device, as vk_framegen.c does for its own:
// an earlier evaluation may still read the old ones.
static int prepare(uint32_t uw, uint32_t uh, uint32_t rw, uint32_t rh) {
    if (!capable() || G.failed || !uw || !uh || !rw || !rh) return 0;
    if (!setup()) {
        if (!G.failed) rt_log("fsr3fi: its sampler, buffers or pipelines could not be made: the built-in interpolation runs");
        G.failed = 1;
        return 0;
    }
    if (G.ready && G.width == uw && G.height == uh && G.render_width == rw && G.render_height == rh) return 1;
    VkCtx *c = vk_ctx();
    if (G.ready) {
        vk_queue_lock();
        vkDeviceWaitIdle(c->device);
        vk_queue_unlock();
    }
    free_images(c);
    // The flow pyramid's levels are halved rounding up, which is not what a mip chain does: an image a level, as the
    // SDK allocates them. The luma pyramid's levels are halved rounding down, so that one is a single mipped image.
    G.flow_width[0] = (uw + OF_BLOCK - 1) / OF_BLOCK;
    G.flow_height[0] = (uh + OF_BLOCK - 1) / OF_BLOCK;
    for (uint32_t l = 1; l < OF_LEVELS; l++) {
        G.flow_width[l] = (G.flow_width[l - 1] + 1) / 2;
        G.flow_height[l] = (G.flow_height[l - 1] + 1) / 2;
    }
    const uint32_t pyramid_w = uw / 2 ? uw / 2 : 1, pyramid_h = uh / 2 ? uh / 2 : 1;
    uint32_t pyramid_levels = log2_floor(pyramid_w > pyramid_h ? pyramid_w : pyramid_h) + 1;
    if (pyramid_levels > PYRAMID_MIPS) pyramid_levels = PYRAMID_MIPS;
    // Seven luma levels is what the flow asks for, but a window smaller than 64 pixels has fewer mips than that; the
    // level index is clamped, so a short chain costs accuracy at the coarsest levels rather than an invalid image.
    uint32_t luma_levels = log2_floor(uw > uh ? uw : uh) + 1;
    if (luma_levels > OF_LEVELS) luma_levels = OF_LEVELS;

    int made = 1;
    for (uint32_t i = 0; i < 2 && made; i++) {
        made = make_image(&G.of_luma[i], VK_FORMAT_R8_UINT, uw, uh, luma_levels);
        for (uint32_t l = 0; l < OF_LEVELS && made; l++)
            made = make_image(&G.of_flow[i][l], VK_FORMAT_R16G16_SINT, G.flow_width[l], G.flow_height[l], 1);
    }
    made = made && make_image(&G.of_vector, VK_FORMAT_R16G16_SINT, G.flow_width[0], G.flow_height[0], 1);
    // The flow carries no per-block confidence, and neither does the SDK's: its own dispatch leaves the binding null
    // and the pass reads zero through it. A cleared image of the flow's size is the same answer with every fetch in
    // bounds.
    made = made && make_image(&G.of_confidence, VK_FORMAT_R32_UINT, G.flow_width[0], G.flow_height[0], 1);
    made = made && make_image(&G.of_scd_histogram, VK_FORMAT_R32_UINT, SCD_HISTOGRAM_W, 1, 1);
    made = made && make_image(&G.of_scd_previous_histogram, VK_FORMAT_R32_SFLOAT, SCD_HISTOGRAM_W, 1, 1);
    made = made && make_image(&G.of_scd_temp, VK_FORMAT_R32_UINT, 3, 1, 1);
    made = made && make_image(&G.of_scd_output, VK_FORMAT_R32_UINT, 3, 1, 1);
    made = made && make_image(&G.fi_dilated_depth, VK_FORMAT_R32_SFLOAT, rw, rh, 1);
    made = made && make_image(&G.fi_dilated_motion_vectors, VK_FORMAT_R16G16_SFLOAT, rw, rh, 1);
    made = made && make_image(&G.fi_reconstructed_depth_previous, VK_FORMAT_R32_UINT, rw, rh, 1);
    made = made && make_image(&G.fi_reconstructed_depth_interpolated, VK_FORMAT_R32_UINT, rw, rh, 1);
    made = made && make_image(&G.fi_game_mv_x, VK_FORMAT_R32_UINT, rw, rh, 1);
    made = made && make_image(&G.fi_game_mv_y, VK_FORMAT_R32_UINT, rw, rh, 1);
    made = made && make_image(&G.fi_of_mv_x, VK_FORMAT_R32_UINT, rw, rh, 1);
    made = made && make_image(&G.fi_of_mv_y, VK_FORMAT_R32_UINT, rw, rh, 1);
    made = made && make_image(&G.fi_disocclusion_mask, VK_FORMAT_R8G8_UNORM, rw, rh, 1);
    made = made && make_image(&G.fi_inpainting_pyramid, VK_FORMAT_R16G16B16A16_SFLOAT, pyramid_w, pyramid_h, pyramid_levels);
    made = made && make_image(&G.fi_distortion_field, VK_FORMAT_R8G8_UNORM, 1, 1, 1);
    if (!made) {
        G.failed = 1;
        free_images(c);   // none of them was used
        rt_log("fsr3fi: its images could not be made: the built-in interpolation runs");
        return 0;
    }
    G.width = uw, G.height = uh, G.render_width = rw, G.render_height = rh;
    G.ready = 1;
    G.in_general = 0;
    G.first_frame = 1;
    G.parity = 0;
    G.of_frame_index = 0;
    rt_log("fsr3fi: %ux%u from %ux%u inputs, flow %ux%u, pyramid %ux%u with %u levels", uw, uh, rw, rh, G.flow_width[0],
           G.flow_height[0], pyramid_w, pyramid_h, pyramid_levels);
    return 1;
}

// Records one pass: its descriptors pushed at the binding numbers its GLSL fixes them to, then the sampler it declares
// (when it declares one), then the dispatch.
static void record_pass(VkCommandBuffer cb, int pass, const Res *res, uint32_t gx, uint32_t gy, uint32_t gz) {
    const PassInfo *info = &PASSES[pass];
    const Pipe *p = &G.passes[pass];
    VkDescriptorImageInfo images[MAX_BIND + 1];
    VkDescriptorBufferInfo buffers[MAX_BIND];
    VkWriteDescriptorSet writes[MAX_BIND + 1];
    uint32_t n = 0;
    for (; n < info->bind_count; n++) {
        const uint8_t kind = info->binds[n].kind;
        writes[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                           .dstBinding = info->binds[n].binding, .descriptorCount = 1};
        if (kind == U || kind == B) {
            buffers[n] = (VkDescriptorBufferInfo){res[n].buffer, res[n].offset, res[n].size};
            writes[n].descriptorType = kind == U ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[n].pBufferInfo = &buffers[n];
        } else {
            images[n] = (VkDescriptorImageInfo){VK_NULL_HANDLE, res[n].view, VK_IMAGE_LAYOUT_GENERAL};
            writes[n].descriptorType = kind == T ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[n].pImageInfo = &images[n];
        }
    }
    if (info->sampler) {
        images[n] = (VkDescriptorImageInfo){G.linear, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        writes[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1000,
                                           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                                           .pImageInfo = &images[n]};
        n++;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    G.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, n, writes);
    vkCmdDispatch(cb, gx ? gx : 1, gy ? gy : 1, gz ? gz : 1);
    // every pass reads what the one before it wrote, and the two pyramids read back their own levels across workgroups
    memory_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
}

// ffxSpdSetup from ffx_spd.h: the thread group count over a rectangle in 64x64 tiles, the offset of the first tile, and
// how many mips the downsampler is to write. fixed_mips below zero asks for the count the rectangle's size implies,
// which is what both inpainting pyramids pass.
static void spd_setup(uint32_t width, uint32_t height, uint32_t *groups, uint32_t *offset, uint32_t *mips, int fixed_mips) {
    if (!width) width = 1;
    if (!height) height = 1;
    offset[0] = offset[1] = 0;
    groups[0] = (width - 1) / 64 + 1;
    groups[1] = (height - 1) / 64 + 1;
    if (fixed_mips >= 0) {
        *mips = (uint32_t)fixed_mips;
        return;
    }
    uint32_t m = log2_floor(width > height ? width : height);
    *mips = m > 12 ? 12 : m;
}

static void clear_image(VkCommandBuffer cb, const Img *t) {
    const VkClearColorValue zero = {{0}};
    if (t->image) vkCmdClearColorImage(cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &ALL_LEVELS);
}

static Res tex(const Img *t) { return (Res){.view = t->view}; }
static Res level(const Img *t, uint32_t l) {
    if (t->level_count <= 1) return (Res){.view = t->view};
    return (Res){.view = t->levels[l < t->level_count ? l : t->level_count - 1]};
}
static uint32_t groups(uint32_t size, uint32_t tile) { return ((size ? size : 1) + tile - 1) / tile; }

static uint64_t last_ns;   // when the previous frame was noted (nl), for the passes' delta time

// Called by vk_temporal_done (vk_dlss.c) after a temporal upscaler's evaluation, with the inputs it was given and the
// render size: notes them for the frame's present when frame_gen_by = fsr3 and Frame generation is 2x; otherwise, and
// for a frame without a scene or whose projection the passes cannot use, the frame has no inputs. The camera factors
// the passes need are the upscaler's own (vk_fsr3.c vk_fsr3_camera_factors), for the same depth (not reversed).
void vk_fsr3fi_note(const VkTemporalIn *t, uint32_t w, uint32_t h) {
    plat_mutex_lock(&nl);
    noted.valid = 0;   // a frame this does not take is not interpolated with the one before it
    if (t && t->temporal && t->depth && t->mv && w && h && !strcmp(cfg_value("TOTK_FRAME_GEN"), "2x") &&
        vk_fsr3fi_wanted()) {
        float dtv[4];
        if (vk_fsr3_camera_factors(t->proj, dtv)) {
            const uint64_t now = kern_now_ns();
            noted = (VkFiInputs){.valid = 1, .depth = t->depth, .motion = t->mv, .render_width = w, .render_height = h,
                                 .tan_half_fov_x = dtv[2], .jitter = {t->jx, t->jy},
                                 .frame_ms = last_ns && now - last_ns < 250000000ull ? (float)(now - last_ns) / 1e6f : 16.667f,
                                 .reset = t->reset, .sequence = ++note_count};
            memcpy(noted.device_to_view_depth, dtv, sizeof dtv);
            last_ns = now;
        }
    }
    plat_mutex_unlock(&nl);
}

int vk_fsr3fi_take(VkFiInputs *out) {
    plat_mutex_lock(&nl);
    *out = noted;
    noted.valid = 0;   // once: a present never interpolates with another frame's depth and motion
    plat_mutex_unlock(&nl);
    return out->valid;
}

int vk_fsr3fi_generate(VkCommandBuffer cb, const VkFiInputs *in, VkImageView color, VkImageView previous,
                       VkImageView out, uint32_t width, uint32_t height) {
    if (!vk_fsr3fi_wanted()) return 0;
    if (!in || !in->valid || !in->depth || !in->motion || !color || !previous || !out) return 0;
    if (!width || !height || !in->render_width || !in->render_height) return 0;
    // no camera: vk_fsr3_camera_factors gives the depth's factor as negative (depth 0 at the near plane), never 0
    if (!(in->device_to_view_depth[1] < 0.0f)) return 0;
    // The passes scale the motion and the depth between the render size and the picture's: a picture of another shape
    // than the 3D (a window with bars the game drew itself) is left to the built-in interpolation.
    {
        const long long lhs = (long long)in->render_width * height, rhs = (long long)in->render_height * width;
        if (lhs - rhs > lhs / 50 || rhs - lhs > lhs / 50) return 0;
    }
    if (!prepare(width, height, in->render_width, in->render_height)) return 0;

    // Everything this file owns starts UNDEFINED and lives in GENERAL from here on.
    if (!G.in_general) {
        G.in_general = 1;
        VkImageMemoryBarrier2 b[N_ALL];
        uint32_t n = 0;
        for (uint32_t i = 0; i < N_ALL; i++) {
            const Img *t = all_images(i);
            if (!t->image) continue;
            b[n++] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                             .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                             .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                             .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = t->image,
                                             .subresourceRange = ALL_LEVELS};
        }
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = n,
                               .pImageMemoryBarriers = b};
        vkCmdPipelineBarrier2(cb, &di);
        for (uint32_t i = 0; i < N_ALL; i++) clear_image(cb, all_images(i));
        vkCmdFillBuffer(cb, G.counters, 0, COUNTERS_BYTES, 0);
        memory_barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    }

    // A cut, a load, a resize, or a frame this file did not see: every history it carries belongs to another frame.
    const int reset = in->reset || G.first_frame || in->sequence != G.sequence + 1;
    G.first_frame = 0;
    G.sequence = in->sequence;
    const int32_t rw = (int32_t)G.render_width, rh = (int32_t)G.render_height;
    const int32_t uw = (int32_t)G.width, uh = (int32_t)G.height;

    // ---- the constant blocks

    Constants k = {0};
    k.render_size[0] = rw, k.render_size[1] = rh;
    k.display_size[0] = uw, k.display_size[1] = uh;
    k.display_size_rcp[0] = 1.0f / (float)uw, k.display_size_rcp[1] = 1.0f / (float)uh;
    k.camera_near = 0.0f;   // not read by the passes (the SDK sets them from the application's camera)
    k.camera_far = 0.0f;
    k.upscaler_target_size[0] = uw, k.upscaler_target_size[1] = uh;
    k.mode = 0;
    k.reset = reset;
    memcpy(k.device_to_view_depth, in->device_to_view_depth, sizeof k.device_to_view_depth);
    k.delta_time = in->frame_ms;
    // No picture without the HUD (the SDK's HUDLessColor): the flow and the interpolation work on the picture as it is
    // shown, and the inpainting pass takes it for its own source.
    k.hudless_attached_factor = 0;
    k.distortion_field_size[0] = 1, k.distortion_field_size[1] = 1;
    // GetOpticalFlowSize() in the GLSL reads this back as (1 / scale) / block size, so the scale is the reciprocal of
    // the display size; it is also what turns a flow vector in luma pixels into the UV motion the vector field packs.
    k.optical_flow_scale[0] = 1.0f / (float)uw, k.optical_flow_scale[1] = 1.0f / (float)uh;
    k.optical_flow_block_size = OF_BLOCK;
    k.dispatch_flags = 0;            // no debug view, no debug bar
    k.max_render_size[0] = rw, k.max_render_size[1] = rh;
    k.optical_flow_half_res_mode = 0;
    k.num_instances = 1;
    k.interpolation_rect_size[0] = uw, k.interpolation_rect_size[1] = uh;
    k.back_buffer_transfer_function = 0;   // the SDR picture (vk_framegen.c makes no frames from the HDR one)
    k.min_max_luminance[0] = 0.0f, k.min_max_luminance[1] = 1.0f;
    // fTanHalfFOV is the horizontal half angle's tangent, as ffxFrameInterpolationDispatch builds it from the vertical
    // field of view and the render aspect: 1 / p[0][0] of the scene's projection
    k.tan_half_fov = in->tan_half_fov_x;
    k.jitter[0] = in->jitter[0], k.jitter[1] = in->jitter[1];
    k.motion_vector_scale[0] = 1.0f / (float)rw;   // the vectors are in render pixels (vk_dlss.c vk_temporal_inputs)
    k.motion_vector_scale[1] = 1.0f / (float)rh;

    // The two inpainting pyramids downsample different rectangles (the game vector field's is at the render size and
    // the frame's at the display size), so each gets its own block.
    uint32_t game_groups[2], game_offset[2], game_mips = 0;
    spd_setup((uint32_t)rw, (uint32_t)rh, game_groups, game_offset, &game_mips, -1);
    const PyramidConstants game_pyramid = {game_mips, game_groups[0] * game_groups[1], {game_offset[0], game_offset[1]}};
    uint32_t frame_groups[2], frame_offset[2], frame_mips = 0;
    spd_setup((uint32_t)uw, (uint32_t)uh, frame_groups, frame_offset, &frame_mips, -1);
    const PyramidConstants frame_pyramid = {frame_mips, frame_groups[0] * frame_groups[1], {frame_offset[0], frame_offset[1]}};

    G.of_frame_index = reset ? 0 : G.of_frame_index + 1;
    OfConstants of = {.input_luma_resolution = {uw, uh}, .pyramid_level = 0, .pyramid_level_count = OF_LEVELS,
                      .frame_index = G.of_frame_index, .back_buffer_transfer_function = 0, .min_max_luminance = {0.0f, 1.0f}};
    // The luma pyramid's own shader hard-codes six mips, so the four the SDK asks ffxSpdSetup for here are never read;
    // the work group count is.
    uint32_t luma_groups[2], luma_offset[2], luma_mips = 0;
    spd_setup((uint32_t)uw, (uint32_t)uh, luma_groups, luma_offset, &luma_mips, 4);
    const OfSpdConstants of_spd = {luma_mips, luma_groups[0] * luma_groups[1], {luma_offset[0], luma_offset[1]},
                                   luma_groups[0] * luma_groups[1], 0, 0, 0};

    const VkDeviceSize at_fi = 0, at_game_pyramid = at_fi + G.slot_fi, at_frame_pyramid = at_game_pyramid + G.slot_pyramid,
                       at_of = at_frame_pyramid + G.slot_pyramid, at_of_spd = at_of + (OF_LEVELS + 1) * G.slot_of;
    memcpy(G.ubo_map + at_fi, &k, sizeof k);
    memcpy(G.ubo_map + at_game_pyramid, &game_pyramid, sizeof game_pyramid);
    memcpy(G.ubo_map + at_frame_pyramid, &frame_pyramid, sizeof frame_pyramid);
    memcpy(G.ubo_map + at_of, &of, sizeof of);   // slot 0: the passes before the flow's own pyramid
    for (uint32_t l = 0; l < OF_LEVELS; l++) {
        OfConstants lv = of;
        lv.pyramid_level = l;
        memcpy(G.ubo_map + at_of + (VkDeviceSize)(l + 1) * G.slot_of, &lv, sizeof lv);
    }
    memcpy(G.ubo_map + at_of_spd, &of_spd, sizeof of_spd);
    vmaFlushAllocation(vk_ctx()->vma, G.ubo_alloc, 0, G.stride);

    const Res cb_fi = {.buffer = G.ubo, .offset = at_fi, .size = G.slot_fi};
    const Res cb_game_pyramid = {.buffer = G.ubo, .offset = at_game_pyramid, .size = G.slot_pyramid};
    const Res cb_frame_pyramid = {.buffer = G.ubo, .offset = at_frame_pyramid, .size = G.slot_pyramid};
    const Res cb_of_spd = {.buffer = G.ubo, .offset = at_of_spd, .size = G.slot_of_spd};
#define CB_OF(slot) ((Res){.buffer = G.ubo, .offset = at_of + (VkDeviceSize)(slot) * G.slot_of, .size = G.slot_of})
    const Res counters = {.buffer = G.counters, .offset = 0, .size = COUNTERS_BYTES};

    // ---- the clears the SDK's dispatch records

    // what the caller's passes wrote (the colour's copy) and the upscaler's inputs have to be visible to the first pass
    memory_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    // The nearest-depth scratch is cleared every frame, to the far depth: the passes take the minimum of the game's
    // depth, which is not reversed (the passes are built with INVERTED_DEPTH=0), so it starts at 1.0.
    {
        const VkClearColorValue far = {.uint32 = {0x3f800000u, 0, 0, 0}};
        vkCmdClearColorImage(cb, G.fi_reconstructed_depth_previous.image, VK_IMAGE_LAYOUT_GENERAL, &far, 1, &ALL_LEVELS);
        if (!reset)
            vkCmdClearColorImage(cb, G.fi_reconstructed_depth_interpolated.image, VK_IMAGE_LAYOUT_GENERAL, &far, 1,
                                 &ALL_LEVELS);
    }
    if (reset) {   // the histories the flow and the scene-change detector carry, as the SDK clears them
        clear_image(cb, &G.of_scd_temp), clear_image(cb, &G.of_scd_output), clear_image(cb, &G.of_scd_histogram);
        clear_image(cb, &G.of_scd_previous_histogram), clear_image(cb, &G.of_luma[0]), clear_image(cb, &G.of_luma[1]);
    }
    memory_barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // ---- the optical flow, over this frame's luma and the frame before it

    const uint32_t p = G.parity, q = p ^ 1u;
    // col and prev: the interpolation sources of this frame and the one before; shown: the picture as it is presented
    const Res shown = {.view = color}, dst = {.view = out};
    const Res col = {.view = color}, prev = {.view = previous};
    {   // two pixels across and two down a thread, which is where the double rounding comes from
        const Res res[] = {col, level(&G.of_luma[p], 0), CB_OF(0)};
        record_pass(cb, P_OF_PREPARE_LUMA, res, groups(groups((uint32_t)uw, 2), 16), groups(groups((uint32_t)uh, 2), 16), 1);
    }
    {
        const Res res[] = {level(&G.of_luma[p], 0), level(&G.of_luma[p], 1), level(&G.of_luma[p], 2),
                           level(&G.of_luma[p], 3), level(&G.of_luma[p], 4), level(&G.of_luma[p], 5),
                           level(&G.of_luma[p], 6), CB_OF(0), cb_of_spd};
        record_pass(cb, P_OF_LUMINANCE_PYRAMID, res, luma_groups[0], luma_groups[1], 1);
    }
    {   // scene change detection: a 3x3 grid of strata, each with its own 256-bin histogram, then one dispatch that
        // compares this frame's histograms with the last frame's
        const uint32_t strata_width = (uint32_t)uw / 4 / 3;
        const Res res[] = {tex(&G.of_luma[p]), tex(&G.of_scd_histogram), CB_OF(0)};
        record_pass(cb, P_OF_SCD_HISTOGRAM, res, groups(strata_width, 32), 16, 3 * 3);
    }
    {
        const Res res[] = {tex(&G.of_scd_histogram), tex(&G.of_scd_previous_histogram), tex(&G.of_scd_temp),
                           tex(&G.of_scd_output), CB_OF(0)};
        record_pass(cb, P_OF_SCD_DIVERGENCE, res, 3 * 3, 3, 1);
    }
    // The search walks the pyramid from the coarsest level down, and the two flow images swap roles at every level as
    // well as at every frame: the search writes A, the filter reads A and writes B, and the scale reads B and seeds
    // the next level down. The index arithmetic is the SDK's (isOddFrame != isOddLevel), as one exclusive or.
    for (int32_t l = OF_LEVELS - 1; l >= 0; l--) {
        const uint32_t a = (p ^ (uint32_t)l) & 1u, b = a ^ 1u;
        uint32_t luma_w = (uint32_t)uw >> l, luma_h = (uint32_t)uh >> l;
        if (!luma_w) luma_w = 1;
        if (!luma_h) luma_h = 1;
        {   // 64 threads a group, each covering four luma pixels across and sixteen rows down
            const Res res[] = {level(&G.of_luma[p], (uint32_t)l), level(&G.of_luma[q], (uint32_t)l), tex(&G.of_flow[a][l]),
                               tex(&G.of_scd_output), CB_OF((uint32_t)l + 1)};
            record_pass(cb, P_OF_SEARCH, res, (groups(luma_w, 4) * 16 + 63) / 64, groups(luma_h, 16), 1);
        }
        {   // at the bottom the filter writes the flow the interpolation reads
            const Res res[] = {tex(&G.of_flow[a][l]), l == 0 ? tex(&G.of_vector) : tex(&G.of_flow[b][l]),
                               CB_OF((uint32_t)l + 1)};
            record_pass(cb, P_OF_FILTER, res, groups(G.flow_width[l], 16), groups(G.flow_height[l], 4), 1);
        }
        if (l > 0) {
            const Res res[] = {level(&G.of_luma[p], (uint32_t)l), level(&G.of_luma[q], (uint32_t)l), tex(&G.of_flow[b][l]),
                               tex(&G.of_flow[b][l - 1]), tex(&G.of_scd_output), CB_OF((uint32_t)l + 1)};
            record_pass(cb, P_OF_SCALE, res, groups(G.flow_width[l - 1], 4), groups(G.flow_height[l - 1], 4), 1);
        }
    }

    // ---- frame interpolation

    const uint32_t render_x = groups((uint32_t)rw, 8), render_y = groups((uint32_t)rh, 8);
    const uint32_t display_x = groups((uint32_t)uw, 8), display_y = groups((uint32_t)uh, 8);
    // The flow's own grid, eight blocks a group, the last partial block rounded up so the vector field is written over
    // the whole flow image.
    const uint32_t flow_x = groups(G.flow_width[0], 8), flow_y = groups(G.flow_height[0], 8);
#define PYR(l) level(&G.fi_inpainting_pyramid, l)
    {   // The SDK records this one from its own prepare entry point, alongside the upscaler; here it opens the
        // interpolation. It fills the three resources the SDK calls shared.
        const Res res[] = {{.view = in->motion}, {.view = in->depth}, tex(&G.fi_reconstructed_depth_previous),
                           tex(&G.fi_dilated_motion_vectors), tex(&G.fi_dilated_depth), cb_fi};
        record_pass(cb, P_FI_RECONSTRUCT_AND_DILATE, res, render_x, render_y, 1);
    }
    {   // zeroes both vector fields and the disocclusion mask, and moves the frame counter the interpolation reads
        // along, which is why it runs even on a reset
        const Res res[] = {tex(&G.of_scd_output), tex(&G.fi_game_mv_x), tex(&G.fi_game_mv_y), tex(&G.fi_of_mv_x),
                           tex(&G.fi_of_mv_y), tex(&G.fi_disocclusion_mask), counters, cb_fi};
        record_pass(cb, P_FI_SETUP, res, render_x, render_y, 1);
    }
    if (!reset) {
        {
            const Res res[] = {tex(&G.fi_dilated_motion_vectors), tex(&G.fi_dilated_depth), col, tex(&G.fi_distortion_field),
                               tex(&G.fi_reconstructed_depth_interpolated), cb_fi};
            record_pass(cb, P_FI_RECONSTRUCT_PREVIOUS_DEPTH, res, render_x, render_y, 1);
        }
        {
            const Res res[] = {tex(&G.fi_dilated_motion_vectors), tex(&G.fi_dilated_depth), prev, col,
                               tex(&G.fi_distortion_field), tex(&G.fi_game_mv_x), tex(&G.fi_game_mv_y), cb_fi};
            record_pass(cb, P_FI_GAME_MOTION_VECTOR_FIELD, res, render_x, render_y, 1);
        }
        {
            const Res res[] = {tex(&G.fi_game_mv_x), tex(&G.fi_game_mv_y), counters, PYR(0), PYR(1), PYR(2), PYR(3), PYR(4),
                               PYR(5), PYR(6), PYR(7), PYR(8), PYR(9), PYR(10), PYR(11), PYR(12), cb_fi, cb_game_pyramid};
            record_pass(cb, P_FI_GAME_VECTOR_FIELD_PYRAMID, res, game_groups[0], game_groups[1], 1);
        }
        {
            const Res res[] = {tex(&G.of_vector), tex(&G.of_confidence), tex(&G.fi_dilated_depth), prev, col,
                               tex(&G.fi_of_mv_x), tex(&G.fi_of_mv_y), cb_fi};
            record_pass(cb, P_FI_OPTICAL_FLOW_VECTOR_FIELD, res, flow_x, flow_y, 1);
        }
        {
            const Res res[] = {tex(&G.fi_game_mv_x), tex(&G.fi_game_mv_y), tex(&G.fi_reconstructed_depth_previous),
                               tex(&G.fi_dilated_depth), tex(&G.fi_reconstructed_depth_interpolated),
                               tex(&G.fi_inpainting_pyramid), tex(&G.fi_distortion_field), tex(&G.fi_disocclusion_mask),
                               cb_fi};
            record_pass(cb, P_FI_DISOCCLUSION_MASK, res, render_x, render_y, 1);
        }
    }
    {   // The frame itself. Where it could not decide a colour it leaves the alpha low, and the two passes after it
        // fill those pixels in from the pyramid they build over this very output.
        const Res res[] = {tex(&G.fi_game_mv_x), tex(&G.fi_game_mv_y), tex(&G.fi_of_mv_x), tex(&G.fi_of_mv_y), prev, col,
                           tex(&G.fi_disocclusion_mask), tex(&G.fi_inpainting_pyramid), counters, dst, cb_fi};
        record_pass(cb, P_FI_INTERPOLATE, res, display_x, display_y, 1);
    }
    {
        const Res res[] = {dst, counters, PYR(0), PYR(1), PYR(2), PYR(3), PYR(4), PYR(5), PYR(6), PYR(7), PYR(8), PYR(9),
                           PYR(10), PYR(11), PYR(12), cb_fi, cb_frame_pyramid};
        record_pass(cb, P_FI_INPAINTING_PYRAMID, res, frame_groups[0], frame_groups[1], 1);
    }
    {   // The present backbuffer, then the interpolation source: the same image without a HUD-less colour, so there is
        // nothing to take the HUD from.
        const Res res[] = {tex(&G.of_scd_output), tex(&G.fi_inpainting_pyramid), shown, col, dst, cb_fi};
        record_pass(cb, P_FI_INPAINTING, res, display_x, display_y, 1);
    }
#undef PYR
#undef CB_OF
    G.parity = q;   // the luma pyramids swap, so the next flow search has the pair it needs
    // A reset frame has no history to interpolate from: the preparation passes were skipped, so what the interpolation
    // wrote is not a frame between two. Everything above still ran, which gives the next frame its luma pair; this one
    // is refused, and the caller's own interpolation takes it.
    if (reset) return 0;
    if (!G.made++) rt_log("fsr3fi: making the 2x frames at %dx%d from %dx%d depth and motion", uw, uh, rw, rh);
    return 1;
}

void vk_fsr3fi_release(void) {   // device idle
    VkCtx *c = vk_ctx();
    if (c && c->device) {
        for (uint32_t i = 0; i < N_ALL; i++) destroy_image(c, all_images(i));
        for (int i = 0; i < P_COUNT; i++) {
            if (G.passes[i].pipeline) vkDestroyPipeline(c->device, G.passes[i].pipeline, NULL);
            if (G.passes[i].layout) vkDestroyPipelineLayout(c->device, G.passes[i].layout, NULL);
            if (G.passes[i].dsl) vkDestroyDescriptorSetLayout(c->device, G.passes[i].dsl, NULL);
        }
        if (G.linear) vkDestroySampler(c->device, G.linear, NULL);
        if (G.counters) vmaDestroyBuffer(c->vma, G.counters, G.counters_alloc);
        if (G.ubo) vmaDestroyBuffer(c->vma, G.ubo, G.ubo_alloc);
    }
    memset(&G, 0, sizeof G);
    plat_mutex_lock(&nl);
    noted.valid = 0;   // the device is going: no upscaler's inputs stay
    plat_mutex_unlock(&nl);
}
