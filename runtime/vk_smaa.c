// SMAA 1x on the present (TOTK_SMAA = low|medium|high; off by default): Jimenez et al.'s morphological anti-aliasing
// on the finished picture at the swapchain size, after FSR or the bilinear blit has scaled it and before the menu overlay
// is drawn, so the HUD of the game is smoothed and the overlay is not. vk_present.c present_image calls it with the
// swapchain image in TRANSFER_DST. The picture's rectangle is copied into an RGBA16F image and runs through the three
// passes of runtime/shaders/smaa_*.comp (colour edge detection, blending weights from the area and search lookup
// textures, neighbourhood blending) as compute shaders; the result is blitted back 1:1. The quality picks the preset
// (SMAA.hlsl SMAA_PRESET_*) of the first two shaders. The setting applies at the next frame: only the pipelines differ.
#include "vk.h"

#include <string.h>

#include "cfg.h"
#include "host.h"
#include "shaders/smaa/AreaTex.h"
#include "shaders/smaa/SearchTex.h"
#include "smaa_blend_spv.h"
#include "smaa_edges_high_spv.h"
#include "smaa_edges_low_spv.h"
#include "smaa_edges_medium_spv.h"
#include "smaa_weights_high_spv.h"
#include "smaa_weights_low_spv.h"
#include "smaa_weights_medium_spv.h"

// the swapchain is 8-bit UNORM, so RGBA8 would halve the memory of these targets; not done
#define COLOR_FMT VK_FORMAT_R16G16B16A16_SFLOAT
#define MASK_FMT VK_FORMAT_R8G8B8A8_UNORM

enum { COLOR, EDGES, WEIGHTS, RESULT, TARGETS };
enum { LOW, MEDIUM, HIGH, QUALITIES };

typedef struct {
    VkImage img;
    VmaAllocation alloc;
    VkImageView view;
} Image;

static struct {
    int tried, ok, uploaded, logged;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline edges[QUALITIES], weights[QUALITIES], blend;
    VkSampler linear, point;
    Image target[TARGETS], area, search;
    VkBuffer staging;
    VmaAllocation staging_alloc;
    uint32_t w, h;
    uint32_t failed_w, failed_h;   // the size the targets could not be allocated at; not retried until it changes
} S;

static VkPipeline make_pipeline(VkCtx *c, const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = S.layout};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
    vkDestroyShaderModule(c->device, m, NULL);
    return p;
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

static int make_lookups(VkCtx *c) {
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!make_image(c, &S.area, VK_FORMAT_R8G8_UNORM, AREATEX_WIDTH, AREATEX_HEIGHT, usage) ||
        !make_image(c, &S.search, VK_FORMAT_R8_UNORM, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, usage))
        return 0;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = AREATEX_SIZE + SEARCHTEX_SIZE,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
    if (vmaCreateBuffer(c->vma, &bi, &ai, &S.staging, &S.staging_alloc, NULL) != VK_SUCCESS) return 0;
    void *p;
    if (vmaMapMemory(c->vma, S.staging_alloc, &p) != VK_SUCCESS) return 0;
    memcpy(p, areaTexBytes, AREATEX_SIZE);
    memcpy((uint8_t *)p + AREATEX_SIZE, searchTexBytes, SEARCHTEX_SIZE);
    vmaUnmapMemory(c->vma, S.staging_alloc);
    return vmaFlushAllocation(c->vma, S.staging_alloc, 0, VK_WHOLE_SIZE) == VK_SUCCESS;
}

static int init(VkCtx *c) {
    S.tried = 1;
    S.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    if (!S.push) {
        rt_log("vk: SMAA off: no VK_KHR_push_descriptor");
        return 0;
    }
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < 4; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 4, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &S.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
                              .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkSamplerCreateInfo pi = si;
    pi.magFilter = pi.minFilter = VK_FILTER_NEAREST;
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &S.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &S.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &S.linear) != VK_SUCCESS ||
        vkCreateSampler(c->device, &pi, NULL, &S.point) != VK_SUCCESS ||
        !(S.edges[LOW] = make_pipeline(c, smaa_edges_low_spv, sizeof smaa_edges_low_spv)) ||
        !(S.edges[MEDIUM] = make_pipeline(c, smaa_edges_medium_spv, sizeof smaa_edges_medium_spv)) ||
        !(S.edges[HIGH] = make_pipeline(c, smaa_edges_high_spv, sizeof smaa_edges_high_spv)) ||
        !(S.weights[LOW] = make_pipeline(c, smaa_weights_low_spv, sizeof smaa_weights_low_spv)) ||
        !(S.weights[MEDIUM] = make_pipeline(c, smaa_weights_medium_spv, sizeof smaa_weights_medium_spv)) ||
        !(S.weights[HIGH] = make_pipeline(c, smaa_weights_high_spv, sizeof smaa_weights_high_spv)) ||
        !(S.blend = make_pipeline(c, smaa_blend_spv, sizeof smaa_blend_spv)) || !make_lookups(c)) {
        rt_log("vk: SMAA off: its resources could not be created");
        return 0;
    }
    return S.ok = 1;
}

static void free_targets(VkCtx *c) {
    for (int i = 0; i < TARGETS; i++) free_image(c, &S.target[i]);
    S.w = S.h = 0;
}

static void wait_idle(VkCtx *c) {
    vk_queue_lock();   // vkDeviceWaitIdle needs every queue left alone, and the replay submits from its own thread
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
}

static int ensure_targets(VkCtx *c, int tw, int th) {
    if (tw <= 0 || th <= 0) return 0;
    uint32_t w = (uint32_t)tw, h = (uint32_t)th;
    if (S.target[COLOR].img && S.w == w && S.h == h) return 1;
    if (S.failed_w == w && S.failed_h == h) return 0;
    wait_idle(c);
    free_targets(c);
    const VkImageUsageFlags work = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!make_image(c, &S.target[COLOR], COLOR_FMT, w, h, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) ||
        !make_image(c, &S.target[EDGES], MASK_FMT, w, h, work) ||
        !make_image(c, &S.target[WEIGHTS], MASK_FMT, w, h, work) ||
        !make_image(c, &S.target[RESULT], COLOR_FMT, w, h, work | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        free_targets(c);
        S.failed_w = w;
        S.failed_h = h;
        rt_log("vk: SMAA off: no memory for its %ux%u targets", w, h);
        return 0;
    }
    S.failed_w = S.failed_h = 0;
    S.w = w;
    S.h = h;
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss,
                    VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                               .dstStageMask = ds, .dstAccessMask = da, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

static void upload_lookup(VkCommandBuffer cb, VkImage img, VkDeviceSize offset, uint32_t w, uint32_t h) {
    const VkPipelineStageFlags2 XF = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier(cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, XF, 0, XF,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region = {.bufferOffset = offset, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {w, h, 1}};
    vkCmdCopyBufferToImage(cb, S.staging, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, XF, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

static void run_pass(VkCommandBuffer cb, VkPipeline p, const VkDescriptorImageInfo *in, uint32_t inputs, VkImageView out) {
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[4];
    for (uint32_t i = 0; i < inputs; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1,
                                      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &in[i]};
    w[inputs] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 3,
                                       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .pImageInfo = &oi};
    float metrics[4] = {1.0f / (float)S.w, 1.0f / (float)S.h, (float)S.w, (float)S.h};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    rec_push(S.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.layout, 0, inputs + 1, w);
    vkCmdPushConstants(cb, S.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof metrics, metrics);
    vkCmdDispatch(cb, (S.w + 7) / 8, (S.h + 7) / 8, 1);
}

int vk_smaa_record(VkCommandBuffer cb, VkImage dst, const int to[4]) {
    const char *q = cfg_value("TOTK_SMAA");
    int quality = !strcmp(q, "low") ? LOW : !strcmp(q, "medium") ? MEDIUM : !strcmp(q, "high") ? HIGH : -1;
    VkCtx *c = vk_ctx();
    if (quality < 0) {   // turned off: hand the targets (about 200 MB at 4K) back until it is turned on again
        if (S.target[COLOR].img) {
            wait_idle(c);
            free_targets(c);
        }
        return 0;
    }
    if (!S.tried) init(c);
    int tx = to[0], ty = to[1], tw = to[2], th = to[3];
    if (!S.ok || !ensure_targets(c, tw, th)) return 0;
    if (!S.logged) {
        S.logged = 1;
        rt_log("vk: SMAA active on the %dx%d picture (%s)", tw, th, q);
    }
    if (!S.uploaded) {
        S.uploaded = 1;
        upload_lookup(cb, S.area.img, 0, AREATEX_WIDTH, AREATEX_HEIGHT);
        upload_lookup(cb, S.search.img, AREATEX_SIZE, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT);
    }
    const VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, XF = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                ALL = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    const VkAccessFlags2 WRITE = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, READ = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL, DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // the swapchain image holds the scaled picture (the caller's last write is a transfer); copy its rectangle out
    barrier(cb, dst, DST, SRC, XF, VK_ACCESS_2_TRANSFER_WRITE_BIT, XF, VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < TARGETS; i++)   // the earlier frame's reads of the targets finish before this frame writes them
        barrier(cb, S.target[i].img, VK_IMAGE_LAYOUT_UNDEFINED, G, ALL, 0, CS | XF,
                VK_ACCESS_2_TRANSFER_WRITE_BIT | WRITE | READ);
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .srcOffsets = {{tx, ty, 0}, {tx + tw, ty + th, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstOffsets = {{0, 0, 0}, {tw, th, 1}}};
    vkCmdBlitImage(cb, dst, SRC, S.target[COLOR].img, G, 1, &region, VK_FILTER_NEAREST);
    barrier(cb, S.target[COLOR].img, G, G, XF, VK_ACCESS_2_TRANSFER_WRITE_BIT, CS, READ);

    VkDescriptorImageInfo color = {S.linear, S.target[COLOR].view, G};
    run_pass(cb, S.edges[quality], &color, 1, S.target[EDGES].view);
    barrier(cb, S.target[EDGES].img, G, G, CS, WRITE, CS, READ);

    VkDescriptorImageInfo weights_in[3] = {{S.linear, S.target[EDGES].view, G}, {S.linear, S.area.view, G},
                                           {S.point, S.search.view, G}};
    run_pass(cb, S.weights[quality], weights_in, 3, S.target[WEIGHTS].view);
    barrier(cb, S.target[WEIGHTS].img, G, G, CS, WRITE, CS, READ);

    VkDescriptorImageInfo blend_in[2] = {color, {S.linear, S.target[WEIGHTS].view, G}};
    run_pass(cb, S.blend, blend_in, 2, S.target[RESULT].view);

    // back into the swapchain image 1:1, which the caller finds in TRANSFER_DST as before
    barrier(cb, S.target[RESULT].img, G, G, CS, WRITE, XF, VK_ACCESS_2_TRANSFER_READ_BIT);
    barrier(cb, dst, SRC, DST, XF, 0, XF, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit back = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .srcOffsets = {{0, 0, 0}, {tw, th, 1}},
                        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .dstOffsets = {{tx, ty, 0}, {tx + tw, ty + th, 1}}};
    vkCmdBlitImage(cb, S.target[RESULT].img, G, dst, DST, 1, &back, VK_FILTER_NEAREST);
    return 1;
}

void vk_smaa_release(void) {
    VkCtx *c = vk_ctx();
    if (!c || !S.tried) return;
    free_targets(c);
    free_image(c, &S.area);
    free_image(c, &S.search);
    if (S.staging) vmaDestroyBuffer(c->vma, S.staging, S.staging_alloc);
    for (int i = 0; i < QUALITIES; i++) {
        if (S.edges[i]) vkDestroyPipeline(c->device, S.edges[i], NULL);
        if (S.weights[i]) vkDestroyPipeline(c->device, S.weights[i], NULL);
    }
    if (S.blend) vkDestroyPipeline(c->device, S.blend, NULL);
    if (S.linear) vkDestroySampler(c->device, S.linear, NULL);
    if (S.point) vkDestroySampler(c->device, S.point, NULL);
    if (S.layout) vkDestroyPipelineLayout(c->device, S.layout, NULL);
    if (S.dsl) vkDestroyDescriptorSetLayout(c->device, S.dsl, NULL);
    memset(&S, 0, sizeof S);
}
