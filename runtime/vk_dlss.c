// SPDX-License-Identifier: GPL-2.0-or-later
// DLSS Super Resolution in place of the game's own upscaler (upscaler = dlss, experimental, NVIDIA RTX only). The
// game scales its graded frame to the output size with a compute dispatch (vk_exec_draw.c dlss_upscale picks it by
// the hash of its SPIR-V); here NVIDIA NGX evaluates DLSS on the same input (texture slot 0, the render resolution)
// into the same output (image slot 1) instead. The game has no velocity target or history to borrow (bananza-plan.md,
// "Later"), so the temporal inputs are made here: the scene's draws get a sub-pixel jitter in their viewport
// (vk_dlss_jitter), the scene depth and the camera block are noted at its draws (vk_dlss_scene_draw), and a compute
// pass turns depth and the camera's view-projection of this and the last frame into camera motion vectors
// (shaders/dlss_mv.comp). A frame without them (no scene noted, another size) runs spatially as before: zero motion,
// a zero depth image, Reset = 1. Any NGX failure logs one "dlss: <reason>" line and the game's dispatch runs from then on.
//
// NGX itself: the core (_nvngx.dll) comes with the NVIDIA driver and is found through the registry keys NVIDIA's
// header-only loader reads; the DLSS model (nvngx_dlss.dll) is the user's and sits next to dkb.exe (docs/dkb-dlss.md).
// The core is called through its C exports. Its parameter object is a C++ class built by MSVC, so its virtual methods
// go through the MSVC vtable layout (overloads in reverse declaration order: Set void* 0, int 3, unsigned 4, float 6;
// Get int* 11). No NVIDIA header or binary is in this repository: the few declarations below carry the values of
// NVIDIA's DLSS SDK 310.9 (nvsdk_ngx_defs.h, nvsdk_ngx_defs_vk.h, nvsdk_ngx_params.h).
//
// Derived from the NGX-direct DLSS path of the Shadow of the Colossus PC fork (namespace Ngx in
// gpu/shadps4/video_core/renderer_vulkan/vk_dlssg.cpp, https://github.com/Lucas-blip-png/ShadowOfTheColossus-PC
// @ 7184fb84e341b24fff824b6fc16d3341fda41828, GPL-2.0-or-later), rewritten in C for this runtime. halton, next_jitter
// and inv4 are taken nearly verbatim, and the camera reprojection follows mv_pass, from TotK Recomp's runtime/vk_dlaa.c
// (https://github.com/valentderah/totk-pc-recomp @ cc4df7d140762a0f45d1776e841d8d9b94371746, Copyright (c) 2026
// Valiantsin Dzerakh, MIT: LICENSE-MIT).
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_exec.h"

#ifdef _WIN32
#include "dlss_mv_spv.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef uint32_t NgxResult;
#define NGX_FAILED(r) (((r) & 0xFFF00000u) == 0xBAD00000u)
enum { NGX_VERSION_API = 0x15, NGX_FEATURE_SUPER_SAMPLING = 1 };
enum { Q_MAX_PERF, Q_BALANCED, Q_MAX_QUALITY, Q_ULTRA_PERF, Q_ULTRA_QUALITY, Q_DLAA };
enum { F_MV_LOW_RES = 1 << 1, F_AUTO_EXPOSURE = 1 << 6 };
enum { PRESET_L = 12, PRESET_M = 13 };   // NGX render preset hints
typedef struct NgxParam NgxParam;
typedef struct NgxHandle NgxHandle;
typedef struct {   // NVSDK_NGX_Resource_VK holding an image view (the larger member of its union)
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    uint32_t width, height;
    uint32_t type;   // 0: image view
    uint8_t read_write;
} NgxResourceVk;

typedef NgxResult (*PfnRequired)(unsigned *, const char ***, unsigned *, const char ***);
typedef NgxResult (*PfnInit)(unsigned long long, const wchar_t *, VkInstance, VkPhysicalDevice, VkDevice,
                             PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, uint32_t, const void *);
typedef NgxResult (*PfnParams)(NgxParam **);
typedef NgxResult (*PfnCreate)(VkDevice, VkCommandBuffer, uint32_t, NgxParam *, NgxHandle **);
typedef NgxResult (*PfnEvaluate)(VkCommandBuffer, const NgxHandle *, const NgxParam *, void *);
typedef NgxResult (*PfnRelease)(NgxHandle *);
typedef NgxResult (*PfnShutdown)(VkDevice);
typedef NgxResult (*PfnDestroy)(NgxParam *);

static HMODULE dll;
static PfnInit ngx_init;
static PfnParams ngx_caps, ngx_alloc;
static PfnCreate ngx_create;
static PfnEvaluate ngx_eval;
static PfnRelease ngx_release;
static PfnShutdown ngx_shutdown;
static PfnDestroy ngx_destroy;
static const char **inst_ext, **dev_ext;
static unsigned n_inst_ext, n_dev_ext;
static int state;   // 0 not tried, 1 ready, -1 failed (the game's upscaler from then on)
static NgxParam *params;

// The feature and its own inputs, made for one input and output size: the motion vectors (written by the motion pass,
// or cleared to zero) and a zero depth image for frames without the scene's, both in GENERAL.
static struct {
    NgxHandle *h;
    uint32_t w, h_, ow, oh;
    VkImage mv, depth;
    VmaAllocation mv_a, depth_a;
    VkImageView mv_v, depth_v;
    int fresh;   // created since the last evaluation: no history
} F;

// The scene of the frame being recorded, noted at its draws (vk_dlss_scene_draw) and used by the next evaluation.
// The camera block is the one Bananza's 3D draws read as vertex constant bank c3, 0x980 bytes, found in a frame dump
// (TOTK_DUMP with DKB_DUMP_UBO=1, 1.0.0 in the field): rows 0-2 the view matrix (3x4), rows 3-6 the view-projection,
// rows 7-10 the projection (OpenGL-style, near 1 far 5000, not reversed), row-major with one matrix row per vec4. The
// shadow and probe passes read blocks of the same size; their depth is not at the upscaler's input size, which tells
// them apart. Update 3.2.0 gives the same layout in play (checked by printing the rows: an orthonormal view, the
// view-projection as their product, a projection of near 1 and far 5000).
#define CAM_BLOCK 0x980u
static struct {
    int have;               // a scene draw since the last evaluation
    uint32_t serial;        // VkExec.pass_serial of the pass it was noted in (one note per pass)
    VkTexImage z;           // the depth image's fields vk_exec_target_view reads (a copy: the texture's own may go)
    VkFormat zfmt;
    uint32_t zlevel, zlayer;
    float cam[44];          // camera block rows 0-10: view, view-projection, projection
} S;

// The jitter of the scene's draws and the camera of the last evaluation. Draws into the depth image the last temporal
// evaluation used are jittered by a Halton(2,3) sub-pixel offset of the viewport; the offset moves on at each
// evaluation, and the jitter stops when DLSS has not run for a few frames (a jittered picture nobody resolves shakes).
static struct {
    VkImage depth;          // scene depth of the last temporal evaluation (VK_NULL_HANDLE: no jitter)
    uint32_t w, h;          // the upscaler's last input size: only a depth of that size is the scene's
    uint64_t frame;         // gfx_counters.frames at the last evaluation
    float jx, jy;           // this frame's offset
    int used;               // a draw took it since the last evaluation
    uint32_t n;             // evaluations so far (the Halton index)
    int prev_ok;            // prev_* hold the last temporal evaluation's camera
    double prev_vp[16];
    float prev_view[12];
    uint64_t temporal, spatial, resets;   // evaluations with motion vectors, without, and with Reset = 1
} J;

// The motion-vector compute pass.
static struct { int ok; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; VkSampler samp; } MV = {.ok = -1};
// Features replaced while the GPU may still run them: released once their submission's fence is done.
#define GRAVE 8
static struct { NgxHandle *h; uint64_t fence; } grave[GRAVE];

static void *slot(const NgxParam *p, int i) { return (*(void *const *const *)p)[i]; }
static void set_ptr(NgxParam *p, const char *n, void *v) { ((void (*)(NgxParam *, const char *, void *))slot(p, 0))(p, n, v); }
static void set_i(NgxParam *p, const char *n, int v) { ((void (*)(NgxParam *, const char *, int))slot(p, 3))(p, n, v); }
static void set_u(NgxParam *p, const char *n, unsigned v) { ((void (*)(NgxParam *, const char *, unsigned))slot(p, 4))(p, n, v); }
static void set_f(NgxParam *p, const char *n, float v) { ((void (*)(NgxParam *, const char *, float))slot(p, 6))(p, n, v); }
static int get_i(NgxParam *p, const char *n) {
    int v = 0;
    ((NgxResult (*)(NgxParam *, const char *, int *))slot(p, 11))(p, n, &v);
    return v;
}

static int fail(const char *why) {
    if (state >= 0) rt_log("dlss: %s; the game's upscaler runs", why);
    state = -1;
    return 0;
}

int vk_dlss_wanted(void) {
    static int w = -1;
    if (w < 0) w = !strcmp(cfg_value("TOTK_UPSCALER"), "dlss");
    return w && state >= 0;
}

// Before the instance: finds _nvngx.dll and asks it for its extensions.
static int load(void) {
    static int tried;
    if (tried) return dll != NULL;
    tried = 1;
    wchar_t path[MAX_PATH + 16];
    static const wchar_t *const keys[][2] = {
        {L"System\\CurrentControlSet\\Services\\nvlddmkm\\Parameters\\NGXCore", L"NGXPath"},
        {L"System\\CurrentControlSet\\Services\\nvlddmkm\\NGXCore", L"NGXPath"},
        {L"SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", L"FullPath"},
    };
    int found = 0;
    for (int i = 0; i < 3 && !found; i++) {
        DWORD size = MAX_PATH * sizeof(wchar_t);
        found = RegGetValueW(HKEY_LOCAL_MACHINE, keys[i][0], keys[i][1], RRF_RT_REG_SZ, NULL, path, &size) == ERROR_SUCCESS;
    }
    if (!found) return fail("no NVIDIA NGX core in the registry (no NVIDIA driver?)");
    wcscat(path, L"\\_nvngx.dll");
    dll = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!dll) return fail("the driver's _nvngx.dll does not load");
#define GET(v, name) (v = (__typeof__(v))(void *)GetProcAddress(dll, name)) != NULL
    PfnRequired required;
    if (!(GET(required, "NVSDK_NGX_VULKAN_RequiredExtensions") && GET(ngx_init, "NVSDK_NGX_VULKAN_Init_Ext2") &&
          GET(ngx_caps, "NVSDK_NGX_VULKAN_GetCapabilityParameters") && GET(ngx_alloc, "NVSDK_NGX_VULKAN_AllocateParameters") &&
          GET(ngx_create, "NVSDK_NGX_VULKAN_CreateFeature1") && GET(ngx_eval, "NVSDK_NGX_VULKAN_EvaluateFeature") &&
          GET(ngx_release, "NVSDK_NGX_VULKAN_ReleaseFeature"))) {
        FreeLibrary(dll);
        dll = NULL;
        return fail("_nvngx.dll lacks an export DLSS needs");
    }
    GET(ngx_shutdown, "NVSDK_NGX_VULKAN_Shutdown1");
    GET(ngx_destroy, "NVSDK_NGX_VULKAN_DestroyParameters");
#undef GET
    if (NGX_FAILED(required(&n_inst_ext, &inst_ext, &n_dev_ext, &dev_ext))) n_inst_ext = n_dev_ext = 0;
    rt_log("dlss: NGX core loaded, it asks for %u instance and %u device extensions", n_inst_ext, n_dev_ext);
    return 1;
}

int vk_dlss_instance_exts(const char **out, int max) {
    if (!vk_dlss_wanted() || !load()) return 0;
    int n = 0;
    for (unsigned i = 0; i < n_inst_ext && n < max; i++) out[n++] = inst_ext[i];
    return n;
}

int vk_dlss_device_exts(const char **out, int max) {
    if (!vk_dlss_wanted() || !dll) return 0;
    int n = 0;
    for (unsigned i = 0; i < n_dev_ext && n < max; i++) out[n++] = dev_ext[i];
    return n;
}

// After the device, at the first upscale: NGX initialised and DLSS Super Resolution available.
static int init(void) {
    if (state) return state > 0;
    if (!dll) return fail("the NGX core was not loaded before the device");
    VkCtx *c = vk_ctx();
    if (c->props.vendorID != 0x10de) return fail("the Vulkan device is not an NVIDIA GPU");
    wchar_t dir[MAX_PATH + 32];
    DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    wchar_t *cut = n ? wcsrchr(dir, L'\\') : NULL;
    if (!cut) return fail("the executable's folder is unknown");
    wcscpy(cut + 1, L"nvngx_dlss.dll");
    if (GetFileAttributesW(dir) == INVALID_FILE_ATTRIBUTES) return fail("nvngx_dlss.dll is not next to dkb.exe");
    *cut = 0;
    // 231313132: the application id of NVIDIA's DLSS samples, for applications without their own. The data path is
    // where NGX looks for nvngx_dlss.dll first and writes its logs.
    NgxResult r = ngx_init(231313132ull, dir, c->instance, c->phys, c->device, vkGetInstanceProcAddr,
                           vkGetDeviceProcAddr, NGX_VERSION_API, NULL);
    char why[96];
    if (NGX_FAILED(r)) {
        snprintf(why, sizeof why, "NGX initialisation failed (%#x)", r);
        return fail(why);
    }
    // From here a failure shuts NGX down again (its capability and parameter objects first).
    NgxParam *caps = NULL;
    r = ngx_caps(&caps);
    if (NGX_FAILED(r) || !caps) snprintf(why, sizeof why, "NGX has no capability parameters (%#x)", r);
    else if (!get_i(caps, "SuperSampling.Available"))
        snprintf(why, sizeof why, "DLSS is not available on this GPU (newer driver needed: %d)",
                 get_i(caps, "SuperSampling.NeedsUpdatedDriver"));
    else if (NGX_FAILED(ngx_alloc(&params)) || !params) snprintf(why, sizeof why, "NGX gave no parameter object");
    else why[0] = 0;
    if (caps && ngx_destroy) ngx_destroy(caps);
    if (why[0]) {
        if (params && ngx_destroy) ngx_destroy(params);
        params = NULL;
        if (ngx_shutdown) ngx_shutdown(c->device);
        return fail(why);
    }
    state = 1;
    return 1;
}

static void reap(void) {
    uint64_t done = vk_sync_fence_done();
    for (int i = 0; i < GRAVE; i++)
        if (grave[i].h && grave[i].fence <= done) ngx_release(grave[i].h), grave[i].h = NULL;
}

static void drop_inputs(void) {
    vk_defer_view(F.mv_v), vk_defer_view(F.depth_v);
    vk_defer_image(F.mv, F.mv_a), vk_defer_image(F.depth, F.depth_a);
    F.mv = F.depth = VK_NULL_HANDLE, F.mv_v = F.depth_v = VK_NULL_HANDLE, F.mv_a = F.depth_a = NULL;
}

static VkImageView view(VkImage img, VkFormat fmt, VkImageUsageFlags usage) {
    VkImageViewUsageCreateInfo u = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = usage};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &u, .image = img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView v = VK_NULL_HANDLE;
    return vkCreateImageView(vk_ctx()->device, &vi, NULL, &v) == VK_SUCCESS ? v : VK_NULL_HANDLE;
}

// A w x h image of fmt in GENERAL, cleared to zero by commands recorded into cb (through the recorder, before it
// drains). extra: usage besides sampling and the clear (storage for the motion vectors).
static int zero_image(VkCommandBuffer cb, VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags extra, VkImage *img,
                      VmaAllocation *a, VkImageView *v) {
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ii, &ai, img, a, NULL) != VK_SUCCESS) return 0;
    *v = view(*img, fmt, VK_IMAGE_USAGE_SAMPLED_BIT | extra);
    if (!*v) return 0;
    VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = *img, .subresourceRange = rg};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
    VkClearColorValue zero = {{0}};
    vkCmdClearColorImage(cb, *img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
    b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT, b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier2(cb, &di);
    return 1;
}

static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 m = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                          .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                          .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m};
    vkCmdPipelineBarrier2(cb, &di);
}

// The motion pass's pipeline (push descriptors: the depth, the vectors), made at the first temporal frame.
static int mv_init(void) {
    if (MV.ok >= 0) return MV.ok;
    MV.ok = 0;
    VkDevice d = vk_ctx()->device;
    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 2, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 80};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlss_mv_spv,
                                   .pCode = dlss_mv_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!vk_pipeline_push_fn() || vkCreateDescriptorSetLayout(d, &dci, NULL, &MV.dsl) != VK_SUCCESS) goto fail;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &MV.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(d, &li, NULL, &MV.pl) != VK_SUCCESS || vkCreateSampler(d, &si, NULL, &MV.samp) != VK_SUCCESS ||
        vkCreateShaderModule(d, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = MV.pl};
    VkResult r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &ci, NULL, &MV.pipe);
    vkDestroyShaderModule(d, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    return MV.ok = 1;
fail:
    rt_log("dlss: the motion-vector pass could not be set up: DLSS stays spatial (no jitter or motion vectors)");
    return 0;
}

static void mv_release(VkDevice d) {
    if (MV.pipe) vkDestroyPipeline(d, MV.pipe, NULL);
    if (MV.pl) vkDestroyPipelineLayout(d, MV.pl, NULL);
    if (MV.dsl) vkDestroyDescriptorSetLayout(d, MV.dsl, NULL);
    if (MV.samp) vkDestroySampler(d, MV.samp, NULL);
    memset(&MV, 0, sizeof MV);
    MV.ok = -1;
}

// Inverse of a row-major 4x4 matrix in double precision (Gauss-Jordan with partial pivoting); 0 when singular. Double:
// the game's world coordinates are in the hundreds and its far plane at 5000, where a float inverse leaves noise of a
// fraction of a pixel that DLSS would follow.
static int inv4(const double m[16], double o[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 8; k++) a[r][k] = k < 4 ? m[r * 4 + k] : (k - 4 == r);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (fabs(a[piv][col]) < 1e-300) return 0;
        for (int k = 0; k < 8; k++) { double t = a[col][k]; a[col][k] = a[piv][k]; a[piv][k] = t; }
        double dv = a[col][col];
        for (int k = 0; k < 8; k++) a[col][k] /= dv;
        for (int r = 0; r < 4; r++)
            if (r != col) {
                double f = a[r][col];
                for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
            }
    }
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) o[r * 4 + k] = a[r][k + 4];
    return 1;
}

// A camera cut (a cutscene's new shot, a respawn): the camera moved more than 20 units or turned more than about 25
// degrees since the last frame, where the history shows another place. The camera's position is -R^T t of the view
// matrix [R | t], its forward axis the third row of R.
// ponytail: fixed thresholds; per-scene tuning (or the game's own cut flag) if fast camera moves reset too often.
static int camera_cut(const float v[12], const float p[12]) {
    double pos[3], ppos[3], fwd = 0;
    for (int i = 0; i < 3; i++) {
        pos[i] = -(v[i] * v[3] + v[4 + i] * v[7] + v[8 + i] * v[11]);
        ppos[i] = -(p[i] * p[3] + p[4 + i] * p[7] + p[8 + i] * p[11]);
        fwd += (double)v[8 + i] * p[8 + i];
    }
    double dx = pos[0] - ppos[0], dy = pos[1] - ppos[1], dz = pos[2] - ppos[2];
    return dx * dx + dy * dy + dz * dz > 400.0 || fwd < 0.9;
}

// The motion vectors of this frame into F.mv, recorded through the recorder: m (row-major) takes this frame's clip
// space to the last frame's; j is the jitter this frame's scene was drawn with.
static void mv_pass(VkCommandBuffer cb, VkImageView zv, const double m[16], uint32_t w, uint32_t h, float jx, float jy) {
    float pc[20];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) pc[c * 4 + r] = (float)m[r * 4 + c];   // GLSL mat4: column-major
    pc[16] = (float)w, pc[17] = (float)h, pc[18] = jx, pc[19] = jy;
    VkDescriptorImageInfo zi = {MV.samp, zv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, F.mv_v, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &zi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, MV.pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, MV.pl, 0, 2, wr);
    vkCmdPushConstants(cb, MV.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
}

static void mv_zero(VkCommandBuffer cb) {
    VkClearColorValue zero = {{0}};
    VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, F.mv, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
    full_barrier(cb);
}

static float halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    for (; i; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}

// The next frame's offset. The sequence is as long as NVIDIA's DLSS guide asks for the scale, 8 x (output / input)^2
// phases, 16 at least (as TotK Recomp's vk_dlaa.c next_jitter).
static void next_jitter(uint32_t w, uint32_t ow) {
    double r = w ? (double)ow / w : 1.0;
    uint32_t phases = (uint32_t)ceil(8.0 * r * r - 1e-6);
    phases = phases < 16 ? 16 : phases > 72 ? 72 : phases;
    uint32_t i = J.n++ % phases + 1;
    J.jx = halton(i, 2) - 0.5f, J.jy = halton(i, 3) - 0.5f;
}

static uint64_t frames_now(void) { return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED); }

void vk_dlss_jitter(VkImage depth, float j[2]) {
    if (!J.depth || depth != J.depth || state <= 0) return;
    if (frames_now() - J.frame > 2) {   // DLSS has stopped running (a menu without the upscaler, a failure)
        J.depth = VK_NULL_HANDLE;
        return;
    }
    j[0] = J.jx, j[1] = J.jy;
    J.used = 1;
}

void vk_dlss_scene_draw(const VkExec *ex) {
    if (state < 0) return;
    const VkExecTarget *z = &ex->depth;
    const p_bind_buf *b = &ex->ubo[0][0];   // vertex bank c3: the replay keeps the user banks from c3 on
    if ((S.have && S.serial == ex->pass_serial) || b->size != CAM_BLOCK || !b->addr || !z->img || z->img->image != z->image ||
        z->is3d || z->samples > 1 || !ex->color[0].image || !(z->aspect & VK_IMAGE_ASPECT_DEPTH_BIT))
        return;
    if (J.w && (z->width != J.w || z->height != J.h)) return;   // a shadow map, a probe, a half-size pass
    // The block as of the submit call (vk_usnap.c): the game rewrites its uniform blocks soon after submitting, and a
    // GPU-side update never shows in guest memory. Not snapshotted (or dropped by such an update): no scene noted, so
    // this frame runs spatially.
    const void *cam = vk_usnap_bytes(b->addr, sizeof S.cam);
    if (!cam) return;
    vk_texture_lock();   // against vk_dlss_image_gone (textures are destroyed on other threads)
    memcpy(S.cam, cam, sizeof S.cam);
    S.z = *z->img;
    S.zfmt = z->format, S.zlevel = z->level, S.zlayer = z->base_layer;
    S.z.width = z->width, S.z.height = z->height;   // of the level drawn
    S.have = 1, S.serial = ex->pass_serial;
    vk_texture_unlock();
}

void vk_dlss_image_gone(VkImage image) {
    if (image == S.z.image) S.have = 0;
    if (image == J.depth) J.depth = VK_NULL_HANDLE, J.prev_ok = 0;
}

// The mode follows the scale: 1.5x quality, 1.7x balanced, 2x performance, 3x ultra performance, 1x DLAA.
static int quality(uint32_t w, uint32_t ow) {
    float s = (float)ow / (float)w;
    return s < 1.1f ? Q_DLAA : s < 1.6f ? Q_MAX_QUALITY : s < 1.85f ? Q_BALANCED : s < 2.5f ? Q_MAX_PERF : Q_ULTRA_PERF;
}

// A feature for w x h -> ow x oh, recorded into cb. A dynamic resolution change makes the game render into a new
// texture of another size, and every new size recreates the feature here: a hitch of a few milliseconds each time.
// ponytail: one feature per size; create it once at the largest render size and pass the size per frame through
// DLSS.Render.Subrect.Dimensions if those hitches show.
// 1 ready; 0 nothing recorded by NGX; -1 NGX failed after it may have recorded into cb.
static int feature(VkCommandBuffer cb, uint64_t fence, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    if (F.h && F.w == w && F.h_ == h && F.ow == ow && F.oh == oh) return 1;
    reap();
    if (F.h) {
        int i = 0;
        while (i < GRAVE && grave[i].h) i++;
        if (i == GRAVE) return 0;   // every slot still in flight: the game's upscaler this frame, a new try next frame
        grave[i].h = F.h, grave[i].fence = fence;
        F.h = NULL;
        drop_inputs();
    }
    int q = quality(w, ow);
    set_u(params, "CreationNodeMask", 1);
    set_u(params, "VisibilityNodeMask", 1);
    set_u(params, "Width", w);
    set_u(params, "Height", h);
    set_u(params, "OutWidth", ow);
    set_u(params, "OutHeight", oh);
    set_i(params, "PerfQualityValue", q);
    set_i(params, "DLSS.Feature.Create.Flags", F_MV_LOW_RES | F_AUTO_EXPOSURE);
    set_i(params, "DLSS.Enable.Output.Subrects", 0);
    // DLSS 4.5's second transformer model (preset M) in every mode but Ultra Performance, which has its own (L).
    set_u(params, "DLSS.Hint.Render.Preset.DLAA", PRESET_M);
    set_u(params, "DLSS.Hint.Render.Preset.Quality", PRESET_M);
    set_u(params, "DLSS.Hint.Render.Preset.Balanced", PRESET_M);
    set_u(params, "DLSS.Hint.Render.Preset.Performance", PRESET_M);
    set_u(params, "DLSS.Hint.Render.Preset.UltraPerformance", PRESET_L);
    if (!zero_image(cb, VK_FORMAT_R16G16_SFLOAT, w, h, VK_IMAGE_USAGE_STORAGE_BIT, &F.mv, &F.mv_a, &F.mv_v) ||
        !zero_image(cb, VK_FORMAT_R32_SFLOAT, w, h, 0, &F.depth, &F.depth_a, &F.depth_v)) {
        drop_inputs();
        return fail("its zero motion-vector and depth images could not be made");
    }
    rec_drain();   // NGX records into cb on this thread: everything queued for the recorder first
    NgxResult r = ngx_create(vk_ctx()->device, cb, NGX_FEATURE_SUPER_SAMPLING, params, &F.h);
    if (NGX_FAILED(r) || !F.h) {
        char why[96];
        snprintf(why, sizeof why, "creating DLSS %ux%u -> %ux%u failed (%#x)", w, h, ow, oh, r);
        F.h = NULL;
        drop_inputs();
        fail(why);
        return -1;
    }
    F.w = w, F.h_ = h, F.ow = ow, F.oh = oh, F.fresh = 1;
    static const char *const names[] = {"Performance", "Balanced", "Quality", "Ultra Performance", "Ultra Quality", "DLAA"};
    rt_log("dlss: %s %ux%u -> %ux%u, preset %s", names[q], w, h, ow, oh, q == Q_ULTRA_PERF ? "L" : "M");
    return 1;
}

int vk_dlss_record(VkCommandBuffer cb, uint64_t fence, const VkTexImage *in, const VkTexImage *out) {
    // The setting is live: fsr or bilinear chosen while the game runs turns DLSS off here (the extensions stay).
    if (!vk_dlss_wanted() || strcmp(cfg_value("TOTK_UPSCALER"), "dlss") || !init()) return 0;
    uint32_t w = in->width, h = in->height, ow = out->width, oh = out->height;
    if (w < 32 || h < 32 || w > ow || h > oh) {   // this pair of images only: the game's upscaler runs for it
        static int logged;
        if (!logged++) rt_log("dlss: unsupported size %ux%u -> %ux%u: the game's upscaler runs for it", w, h, ow, oh);
        return 0;
    }
    // Views of the game's images for this frame only: they go when the submission is done.
    VkImageView iv = view(in->image, in->format, VK_IMAGE_USAGE_SAMPLED_BIT);
    VkImageView ov = view(out->image, out->format, VK_IMAGE_USAGE_STORAGE_BIT);
    if (!iv || !ov) {
        vk_defer_view(iv), vk_defer_view(ov);
        return fail("no sampled view of the input or storage view of the output");
    }
    int made = feature(cb, fence, w, h, ow, oh);
    if (made <= 0) {
        vk_defer_view(iv), vk_defer_view(ov);
        return made;
    }
    VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    NgxResourceVk color = {iv, in->image, rg, in->format, w, h, 0, 0};
    NgxResourceVk output = {ov, out->image, rg, out->format, ow, oh, 0, 1};
    NgxResourceVk depth = {F.depth_v, F.depth, rg, VK_FORMAT_R32_SFLOAT, w, h, 0, 0};
    NgxResourceVk mv = {F.mv_v, F.mv, rg, VK_FORMAT_R16G16_SFLOAT, w, h, 0, 0};
    // Temporal when this frame's scene was noted at the input size: its depth and camera motion go to DLSS. Otherwise
    // the zero inputs and a new history, as a spatial upscaler.
    float jx = J.used ? J.jx : 0.0f, jy = J.used ? J.jy : 0.0f;
    // The texture lock keeps the noted depth image from going (vk_dlss_image_gone) between the checks and its view;
    // once the view exists the image's destruction waits for this submission (vk_defer_image).
    vk_texture_lock();
    int temporal = S.have && S.z.width == w && S.z.height == h && (S.z.usage & VK_IMAGE_USAGE_SAMPLED_BIT) && mv_init();
    int reset = 1;
    double vp[16];
    VkImageView zv = temporal ? vk_exec_target_view(&S.z, S.zfmt, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                    S.zlevel, S.zlayer, 1, 0)
                              : VK_NULL_HANDLE;
    // The last frame's camera only when it is the frame just before (a gap: a menu, a load, frames at another size).
    int prev = J.prev_ok && J.depth == S.z.image && frames_now() - J.frame <= 2;
    vk_texture_unlock();
    if (temporal) {
        for (int i = 0; i < 16; i++) vp[i] = S.cam[12 + i];
        double ci[16], m[16];
        if (prev && !F.fresh && !camera_cut(S.cam, J.prev_view) && inv4(vp, ci)) {
            for (int r = 0; r < 4; r++)
                for (int k = 0; k < 4; k++) {
                    double v = 0;
                    for (int i = 0; i < 4; i++) v += J.prev_vp[r * 4 + i] * ci[i * 4 + k];
                    m[r * 4 + k] = v;
                }
            mv_pass(cb, zv, m, w, h, jx, jy);
            reset = 0;
        } else {
            mv_zero(cb);
        }
        depth = (NgxResourceVk){zv, S.z.image, {VK_IMAGE_ASPECT_DEPTH_BIT, S.zlevel, 1, S.zlayer, 1}, S.zfmt, w, h, 0, 0};
    } else {
        mv_zero(cb);
    }
    set_ptr(params, "Color", &color);
    set_ptr(params, "Output", &output);
    set_ptr(params, "Depth", &depth);
    set_ptr(params, "MotionVectors", &mv);
    set_f(params, "Jitter.Offset.X", jx);
    set_f(params, "Jitter.Offset.Y", jy);
    set_f(params, "Sharpness", 0.0f);
    set_i(params, "Reset", reset);
    set_f(params, "MV.Scale.X", 1.0f);
    set_f(params, "MV.Scale.Y", 1.0f);
    set_u(params, "DLSS.Render.Subrect.Dimensions.Width", w);
    set_u(params, "DLSS.Render.Subrect.Dimensions.Height", h);
    set_f(params, "DLSS.Pre.Exposure", 1.0f);
    set_f(params, "DLSS.Exposure.Scale", 1.0f);
    rec_drain();   // the game's images are in GENERAL (vk_texture.c); the caller's full barrier comes before this
    NgxResult r = ngx_eval(cb, F.h, params, NULL);
    vk_defer_view(iv), vk_defer_view(ov);
    if (NGX_FAILED(r)) {
        char why[64];
        snprintf(why, sizeof why, "evaluating DLSS failed (%#x)", r);
        J.depth = VK_NULL_HANDLE;
        fail(why);
        return -1;
    }
    // This frame's camera is the next one's last; the next frame's scene is jittered when this one was temporal.
    F.fresh = 0, S.have = 0, J.used = 0;
    J.w = w, J.h = h, J.frame = frames_now();
    J.temporal += temporal, J.spatial += !temporal, J.resets += reset;
    if (temporal) {
        memcpy(J.prev_vp, vp, sizeof vp);
        memcpy(J.prev_view, S.cam, sizeof J.prev_view);
        J.prev_ok = 1, J.depth = S.z.image;
    } else {
        J.prev_ok = 0, J.depth = VK_NULL_HANDLE;
    }
    next_jitter(w, ow);
    // A summary at the first evaluation and then every 1800 (half a minute of DLSS at 60 fps; the game skips its
    // upscaler, and DLSS with it, while it renders at the output size). The projection's z column shows an off-centre
    // or jittered frustum of the game's own.
    uint64_t n = J.temporal + J.spatial;
    if (n == 1 || n % 1800 == 0) {
        rt_log("dlss: %llu frames upscaled by present %llu, %llu with depth and camera motion, %llu spatial (no scene "
               "depth at the input size), %llu history resets; jitter %.3f %.3f, projection z column %.5f %.5f",
               (unsigned long long)(J.temporal + J.spatial), (unsigned long long)J.frame, (unsigned long long)J.temporal,
               (unsigned long long)J.spatial, (unsigned long long)J.resets, jx, jy, S.cam[30], S.cam[34]);
    }
    return 1;
}

void vk_dlss_release(void) {   // device idle
    if (!params) return;   // NGX never initialised
    for (int i = 0; i < GRAVE; i++)
        if (grave[i].h) ngx_release(grave[i].h), grave[i].h = NULL;
    if (F.h) ngx_release(F.h), F.h = NULL;
    VkCtx *c = vk_ctx();
    if (c) {
        if (F.mv_v) vkDestroyImageView(c->device, F.mv_v, NULL);
        if (F.depth_v) vkDestroyImageView(c->device, F.depth_v, NULL);
        if (F.mv) vmaDestroyImage(c->vma, F.mv, F.mv_a);
        if (F.depth) vmaDestroyImage(c->vma, F.depth, F.depth_a);
        mv_release(c->device);
        if (ngx_destroy) ngx_destroy(params);
        if (ngx_shutdown) ngx_shutdown(c->device);
    }
    memset(&F, 0, sizeof F);
    memset(&S, 0, sizeof S);
    memset(&J, 0, sizeof J);
    params = NULL;
    if (state > 0) state = 0;   // a new device initialises NGX again; a failure stays a failure
}

#else   // NGX is Windows-only here

int vk_dlss_wanted(void) { return 0; }
int vk_dlss_instance_exts(const char **out, int max) { (void)out, (void)max; return 0; }
int vk_dlss_device_exts(const char **out, int max) { (void)out, (void)max; return 0; }
int vk_dlss_record(VkCommandBuffer cb, uint64_t fence, const VkTexImage *in, const VkTexImage *out) {
    (void)cb, (void)fence, (void)in, (void)out;
    return 0;
}
void vk_dlss_release(void) {}
void vk_dlss_jitter(VkImage depth, float j[2]) { (void)depth, (void)j; }
void vk_dlss_scene_draw(const VkExec *ex) { (void)ex; }
void vk_dlss_image_gone(VkImage image) { (void)image; }

#endif
