// SPDX-License-Identifier: GPL-3.0-or-later
// AMD FSR 4 temporal upscaling in place of the game's upscaler (upscaler = fsr4), next to FSR 3.1 (vk_fsr3.c) and DLSS
// (vk_dlss.c). Unlike FSR 3.1, nothing of AMD's is built in: the user's signed amd_fidelityfx_loader_vk.dll (with
// amd_fidelityfx_upscaler_vk.dll beside it, from the FidelityFX SDK v2.4.0 release, Kits/FidelityFX/signedbin) is loaded
// from the executable's folder at run time, the way vk_dlss.c loads NGX, and driven through the FidelityFX API
// (runtime/third_party/ffx_api, MIT). The loader lists the upscalers this GPU can run; the newest 4.x is taken, and
// without the DLLs or a 4.x provider (as on NVIDIA) upscaler = fsr4 logs one line and vk_fsr3.c runs instead
// (vk_fsr3_wanted). The inputs are DLSS's, from vk_dlss.c (vk_temporal_inputs). Windows only: elsewhere this file is
// stubs.
//
// Derived from Hookshot's FSR 4 engine (gpu/vk/vk_fsr4.cpp, https://github.com/NotRexed/Hookshot
// @ fab8cb37641714bd839ae83c2a7bc582bedc4d70, Copyright 2026 Hookshot contributors, GPL-3.0-or-later), rewritten in C
// for this runtime. Its host side follows the API guide of FidelityFX SDK 2.4 (Kits/FidelityFX/docs/getting-started/
// ffx-api.md): a Vulkan backend description chained to the upscale context description, the inputs read in
// SHADER_READ_ONLY_OPTIMAL, the output written in GENERAL. Hookshot's was first written for Lichii's OoT branch
// (madebyjordan, vk_fsr4_upscaler.cpp, GPL-3.0-or-later). The differences: the dispatch is recorded after rec_drain
// (vk_rec.h) like NGX's, an old context is destroyed once the submission that may use it is done, and the camera's near
// and far distances and field of view come from the scene's projection.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "third_party/ffx_api/api/include/ffx_api.h"
#include "third_party/ffx_api/api/include/ffx_api_types.h"
#include "third_party/ffx_api/api/include/vk/ffx_api_vk.h"
#include "third_party/ffx_api/upscalers/include/ffx_upscale.h"

#define GRAVE 4

// An upscale context and the descriptions it was made from: AMD's library keeps pointers into them until the context is
// destroyed, so they live in the same allocation.
typedef struct {
    struct ffxCreateBackendVKDesc backend;
    struct ffxOverrideVersion version;
    struct ffxCreateContextDescUpscale upscale;
    ffxContext context;
    uint32_t width, height;   // its output, which is also the most the render size can be
    int non_linear;           // the colour is perceptual (not an _SRGB or float format)
} Context;

static struct {
    int state;   // 0 not tried, 1 loaded, -1 not available
    HMODULE lib;
    PfnFfxCreateContext create;
    PfnFfxDestroyContext destroy;
    PfnFfxQuery query;
    PfnFfxDispatch dispatch;
    uint64_t version_id;
    VkDeviceContext device_context;
    Context *context;
    struct { Context *c; uint64_t fence; } grave[GRAVE];   // old contexts, until the submission that may use them is done
    int failed;
    uint64_t frames, temporal, spatial, resets, last_qpc;
} G;

static SRWLOCK load_lock = SRWLOCK_INIT;

static uint64_t frames_now(void) { return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED); }

static void on_message(uint32_t type, const wchar_t *message) {
    char text[512] = {0};
    if (message) WideCharToMultiByte(CP_UTF8, 0, message, -1, text, (int)sizeof text - 1, NULL, NULL);
    rt_log("fsr4: AMD's %s: %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", text);
}

// The loader and the entry points, and the newest 4.x upscaler it lists for this device (the first of equal versions, so
// a provider's Auto, which picks the GPU's model, comes before INT8 and FP8). 1 when FSR 4 can run here; the log says why
// not otherwise. Once; the device must exist.
static int load(void) {
    VkCtx *c = vk_ctx();
    wchar_t path[MAX_PATH + 40];
    DWORD n = GetModuleFileNameW(NULL, path, MAX_PATH);
    wchar_t *cut = n ? wcsrchr(path, L'\\') : NULL;
    if (!cut) return rt_log("fsr4: the executable's folder is unknown; FSR 3.1 runs instead"), 0;
    wcscpy(cut + 1, L"amd_fidelityfx_loader_vk.dll");
    if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES)
        return rt_log("fsr4: amd_fidelityfx_loader_vk.dll is not next to dkb.exe (docs/dkb-dlss.md); FSR 3.1 runs instead"), 0;
    // (the upscaler DLL the loader picks lies beside it: the loader's folder first in the search)
    G.lib = LoadLibraryExW(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!G.lib)
        return rt_log("fsr4: amd_fidelityfx_loader_vk.dll was not loaded (Windows error %lu); FSR 3.1 runs instead",
                      (unsigned long)GetLastError()), 0;
    G.create = (PfnFfxCreateContext)(void *)GetProcAddress(G.lib, "ffxCreateContext");
    G.destroy = (PfnFfxDestroyContext)(void *)GetProcAddress(G.lib, "ffxDestroyContext");
    G.query = (PfnFfxQuery)(void *)GetProcAddress(G.lib, "ffxQuery");
    G.dispatch = (PfnFfxDispatch)(void *)GetProcAddress(G.lib, "ffxDispatch");
    if (!G.create || !G.destroy || !G.query || !G.dispatch)
        return rt_log("fsr4: the loader lacks the FidelityFX API's entry points; FSR 3.1 runs instead"), 0;
    G.device_context = (VkDeviceContext){c->device, c->phys, vkGetDeviceProcAddr};
    struct ffxQueryDescGetVersions v = {0};
    v.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    v.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    v.device = &G.device_context;
    uint64_t count = 0;
    v.outputCount = &count;
    if (G.query(NULL, &v.header) != FFX_API_RETURN_OK || !count)
        return rt_log("fsr4: AMD's library lists no upscaler for this GPU; FSR 3.1 runs instead"), 0;
    uint64_t *ids = calloc(count, sizeof *ids);
    const char **names = calloc(count, sizeof *names);
    int ok = 0;
    if (ids && names) {
        v.versionIds = ids;
        v.versionNames = names;
        if (G.query(NULL, &v.header) != FFX_API_RETURN_OK) {
            rt_log("fsr4: AMD's upscalers could not be listed; FSR 3.1 runs instead");
        } else {
            char listed[256] = "";
            unsigned best[3] = {0, 0, 0};
            for (uint64_t i = 0; i < count; i++) {
                const char *name = names[i] ? names[i] : "";
                size_t used = strlen(listed);
                snprintf(listed + used, sizeof listed - used, "%s%s", used ? ", " : "", name);
                unsigned u[3] = {0, 0, 0};
                sscanf(name, "%u.%u.%u", &u[0], &u[1], &u[2]);
                if (u[0] == 4 && (!G.version_id || u[1] > best[1] || (u[1] == best[1] && u[2] > best[2]))) {
                    memcpy(best, u, sizeof best);
                    G.version_id = ids[i];
                }
            }
            if (!G.version_id)
                rt_log("fsr4: no FSR 4 for this GPU (AMD's library has: %s); FSR 3.1 runs instead", listed);
            else
                ok = 1, rt_log("fsr4: AMD's upscalers: %s; using %u.%u.%u", listed, best[0], best[1], best[2]);
        }
    }
    free(ids), free(names);
    return ok;
}

// FSR 4 can run here (asked once, the first time the setting names it; the device exists by then).
static int usable(void) {
    if (__atomic_load_n(&G.state, __ATOMIC_ACQUIRE)) return G.state > 0;
    if (!vk_ctx()) return 0;
    AcquireSRWLockExclusive(&load_lock);
    if (!G.state) __atomic_store_n(&G.state, load() ? 1 : -1, __ATOMIC_RELEASE);
    ReleaseSRWLockExclusive(&load_lock);
    return G.state > 0;
}

// The setting is read once a frame (the draws ask for it thousands of times).
int vk_fsr4_wanted(void) {
    static uint64_t seen = ~0ull;
    static int chosen;
    uint64_t f = frames_now();
    if (f != __atomic_load_n(&seen, __ATOMIC_RELAXED)) {
        __atomic_store_n(&chosen, !strcmp(cfg_value("TOTK_UPSCALER"), "fsr4"), __ATOMIC_RELAXED);
        __atomic_store_n(&seen, f, __ATOMIC_RELAXED);
    }
    return __atomic_load_n(&chosen, __ATOMIC_RELAXED) && !__atomic_load_n(&G.failed, __ATOMIC_RELAXED) && usable();
}

static int fail(const char *why) {
    if (!__atomic_load_n(&G.failed, __ATOMIC_RELAXED)) rt_log("fsr4: %s; FSR 3.1 runs", why);
    __atomic_store_n(&G.failed, 1, __ATOMIC_RELAXED);
    return 0;
}

static void destroy_context(Context *c) {
    if (!c) return;
    if (c->context) G.destroy(&c->context, NULL);
    free(c);
}

// Old contexts that no submission can still use go.
static void sweep(void) {
    uint64_t done = vk_sync_fence_done();
    for (int i = 0; i < GRAVE; i++)
        if (G.grave[i].c && G.grave[i].fence <= done) destroy_context(G.grave[i].c), G.grave[i].c = NULL;
}

// The current context is let go: destroyed once the submission being recorded is done. A full graveyard (sizes changing
// every frame) destroys its oldest entry early. ponytail: a stale context is destroyed under an in-flight use then; the
// output size changes only with the window, so it is not expected.
static void retire(uint64_t fence) {
    Context *old = G.context;
    G.context = NULL;
    if (!old) return;
    int slot = 0;
    for (int i = 0; i < GRAVE; i++) {
        if (!G.grave[i].c) {
            slot = i;
            break;
        }
        if (G.grave[i].fence < G.grave[slot].fence) slot = i;
    }
    destroy_context(G.grave[slot].c);
    G.grave[slot].c = old, G.grave[slot].fence = fence;
}

// A context for outputs of w x h, any render size up to that (the game's dynamic resolution changes it every so often: one
// context takes them all).
// Float and _SRGB colour is not perceptual data the upscaler should treat as gamma coded.
static int perceptual(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R8_SRGB: case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R8G8B8_SRGB: case VK_FORMAT_B8G8R8_SRGB: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return 0;
    default:
        return 1;
    }
}

static Context *make_context(uint32_t w, uint32_t h, int non_linear) {
    Context *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    c->backend.deviceContext = G.device_context;
    c->version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    c->version.versionId = G.version_id;
    c->backend.header.pNext = &c->version.header;
    c->upscale.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    c->upscale.header.pNext = &c->backend.header;
    // The scene's colour is the game's LDR, perceptual image (no auto exposure); the depth is not inverted.
    c->non_linear = non_linear;
    c->upscale.flags = (non_linear ? FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE : 0) | FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
    c->upscale.maxRenderSize = (struct FfxApiDimensions2D){w, h};
    c->upscale.maxUpscaleSize = (struct FfxApiDimensions2D){w, h};
    c->upscale.fpMessage = &on_message;
    ffxReturnCode_t r = G.create(&c->context, &c->upscale.header, NULL);
    if (r != FFX_API_RETURN_OK || !c->context) {
        rt_log("fsr4: AMD's upscaler was not made (%d) for %ux%u", (int)r, w, h);
        c->context = NULL;
        free(c);
        return NULL;
    }
    c->width = w, c->height = h;
    struct ffxQueryGetProviderVersion pv = {0};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    int known = G.query(&c->context, &pv.header) == FFX_API_RETURN_OK && pv.versionName;
    rt_log("fsr4: FSR %s made for outputs of %ux%u", known ? pv.versionName : "?", w, h);
    return c;
}

// The FidelityFX surface format of a Vulkan one (ffxApiGetSurfaceFormatVK is C++ only in the SDK's header); UNKNOWN for
// the formats the upscaler's inputs and outputs do not come in.
static uint32_t surface_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R32G32B32A32_SFLOAT: return FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_R32G32_SFLOAT: return FFX_API_SURFACE_FORMAT_R32G32_FLOAT;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_X8_D24_UNORM_PACK32: return FFX_API_SURFACE_FORMAT_R32_UINT;
    case VK_FORMAT_R8G8B8A8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM: return FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
    case VK_FORMAT_R16G16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16G16_FLOAT;
    case VK_FORMAT_R16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16_FLOAT;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT: return FFX_API_SURFACE_FORMAT_R16_UNORM;
    case VK_FORMAT_R8_UNORM: return FFX_API_SURFACE_FORMAT_R8_UNORM;
    case VK_FORMAT_R8G8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8_UNORM;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    default: return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}

static int is_depth(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

static int has_stencil(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

// An image as the dispatch names it (the SDK's ffxApiGetImageResourceDescriptionVK and ffxApiGetResourceVK).
static struct FfxApiResource resource(VkImage image, VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage,
                                      uint32_t state) {
    struct FfxApiResource r = {0};
    r.resource = (void *)image;
    r.state = state;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = surface_format(fmt);
    r.description.width = w, r.description.height = h, r.description.depth = 1, r.description.mipCount = 1;
    r.description.usage = FFX_API_RESOURCE_USAGE_READ_ONLY | (is_depth(fmt) ? FFX_API_RESOURCE_USAGE_DEPTHTARGET : 0) |
                          (has_stencil(fmt) ? FFX_API_RESOURCE_USAGE_STENCILTARGET : 0) |
                          ((usage & VK_IMAGE_USAGE_STORAGE_BIT) ? FFX_API_RESOURCE_USAGE_UAV : 0);
    return r;
}

static VkImageMemoryBarrier2 barrier(VkImage image, VkImageAspectFlags aspect, VkAccessFlags2 src, VkAccessFlags2 dst,
                                     VkImageLayout from, VkImageLayout to) {
    return (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = src,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstAccessMask = dst,
                                   .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
                                   .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
}

static void barriers(VkCommandBuffer cb, const VkImageMemoryBarrier2 *b, uint32_t n) {
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = n, .pImageMemoryBarriers = b};
    vkCmdPipelineBarrier2(cb, &di);
}

// The scene's camera from its projection (row-major, OpenGL style: the depth buffer holds (ndc z + 1) / 2, as in
// vk_fsr3.c vk_fsr3_camera_factors): the near and far distances and the vertical field of view. 0 for a matrix that is
// not a perspective projection.
static int camera(const float p[16], float *near_z, float *far_z, float *fov_y) {
    if (p[12] != 0.0f || p[13] != 0.0f || p[14] != -1.0f || p[15] != 0.0f) return 0;
    if (!(p[0] > 0.0f) || !(p[5] > 0.0f) || !isfinite(p[0]) || !isfinite(p[5])) return 0;
    *fov_y = 2.0f * atanf(1.0f / p[5]);
    *near_z = 1.0f, *far_z = 5000.0f;   // (the game's own, when the matrix gives no sane ones)
    if (p[11] > 0.0f) return 0;   // as vk_fsr3.c vk_fsr3_camera_factors: depth 0 is at the near plane
    double n = (double)p[11] / ((double)p[10] - 1.0), f = (double)p[11] / ((double)p[10] + 1.0);
    if (isfinite(n) && n > 0.0) {
        *near_z = (float)n;
        if (isfinite(f) && f > n) *far_z = (float)f;   // else (an infinite far plane): the near one only
        else if (*far_z <= *near_z) *far_z = *near_z * 5000.0f;
    }
    return 1;
}

static float frame_ms(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    uint64_t prev = G.last_qpc;
    G.last_qpc = (uint64_t)t.QuadPart;
    if (!prev) return 1000.0f / 30.0f;
    float ms = (float)(1000.0 * (double)(G.last_qpc - prev) / (double)f.QuadPart);
    return ms < 0.0f ? 0.0f : ms > 1000.0f ? 1000.0f : ms;
}

int vk_fsr4_record(VkCommandBuffer cb, uint64_t fence, const VkTexImage *in, const VkTexImage *out) {
    if (!vk_fsr4_wanted()) return 0;
    uint32_t w = in->width, h = in->height, ow = out->width, oh = out->height;
    if (w < 32 || h < 32 || w > ow || h > oh) {   // this pair of images only: the game's upscaler runs for it
        static int logged;
        if (!logged++) rt_log("fsr4: unsupported size %ux%u -> %ux%u: the game's upscaler runs for it", w, h, ow, oh);
        return 0;
    }
    sweep();
    int reset_ctx = 0;
    const int nl = perceptual(in->format);
    static int fmt_logged;
    if (!fmt_logged++) rt_log("fsr4: colour format %d, %s", (int)in->format, nl ? "perceptual" : "linear or sRGB-decoded");
    if (!G.context || G.context->width != ow || G.context->height != oh || G.context->non_linear != nl) {
        retire(fence);
        G.context = make_context(ow, oh, nl);
        if (!G.context) return fail("its upscaler could not be made");
        reset_ctx = 1;
    }
    VkTemporalIn t;
    if (!vk_temporal_inputs(cb, w, h, ow, oh, reset_ctx, &t)) return fail("its motion-vector image could not be made");
    float near_z, far_z, fov_y;
    // Without a scene at the input size (a menu, a cutscene's own passes), a projection FSR cannot use or a depth view of a
    // level or layer other than the first, the game's upscaler runs this frame (-1 when the motion pass bound its own
    // pipeline: the caller binds the game's again).
    if (!t.temporal || !camera(t.proj, &near_z, &far_z, &fov_y) || t.depth_level || t.depth_layer) {
        int bound = t.temporal;
        t.temporal = 0;   // no jitter for the next frame either
        vk_temporal_done(&t, w, h, ow);
        G.spatial++;
        return bound ? -1 : 0;
    }
    const VkImageAspectFlags depth_aspect = VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil(t.depth_format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    const VkImageLayout GEN = VK_IMAGE_LAYOUT_GENERAL, RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkAccessFlags2 any = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    const VkImageMemoryBarrier2 before[4] = {
        barrier(in->image, VK_IMAGE_ASPECT_COLOR_BIT, any, VK_ACCESS_2_SHADER_READ_BIT, GEN, RO),
        barrier(t.depth_image, depth_aspect, any, VK_ACCESS_2_SHADER_READ_BIT, GEN, RO),
        barrier(t.mv_image, VK_IMAGE_ASPECT_COLOR_BIT, any, VK_ACCESS_2_SHADER_READ_BIT, GEN, RO),
        barrier(out->image, VK_IMAGE_ASPECT_COLOR_BIT, any, any, GEN, GEN)};
    barriers(cb, before, 4);

    struct ffxDispatchDescUpscale d = {0};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = cb;
    d.color = resource(in->image, in->format, w, h, in->usage, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = resource(t.depth_image, t.depth_format, w, h, VK_IMAGE_USAGE_SAMPLED_BIT, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = resource(t.mv_image, VK_FORMAT_R16G16_SFLOAT, ow, oh, VK_IMAGE_USAGE_SAMPLED_BIT,
                               FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = resource(out->image, out->format, ow, oh, VK_IMAGE_USAGE_STORAGE_BIT, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = (struct FfxApiFloatCoords2D){t.jx, t.jy};   // render pixels, as the projection was offset: DLSS's
    d.motionVectorScale = (struct FfxApiFloatCoords2D){1.0f, 1.0f};   // render pixels already, towards the frame before
    d.renderSize = (struct FfxApiDimensions2D){w, h};
    d.upscaleSize = (struct FfxApiDimensions2D){ow, oh};
    d.enableSharpening = false;   // the Sharpness row sharpens at the window, after the HUD
    d.frameTimeDelta = frame_ms();
    d.preExposure = 1.0f;
    d.reset = t.reset || reset_ctx;
    d.cameraNear = near_z, d.cameraFar = far_z, d.cameraFovAngleVertical = fov_y;
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags = nl ? FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB : 0;
    rec_drain();   // AMD records into cb on this thread: everything queued for the recorder first
    ffxReturnCode_t r = G.dispatch(&G.context->context, &d.header);

    const VkImageMemoryBarrier2 after[4] = {
        barrier(in->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_2_SHADER_READ_BIT, any, RO, GEN),
        barrier(t.depth_image, depth_aspect, VK_ACCESS_2_SHADER_READ_BIT, any, RO, GEN),
        barrier(t.mv_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_2_SHADER_READ_BIT, any, RO, GEN),
        barrier(out->image, VK_IMAGE_ASPECT_COLOR_BIT, any, any, GEN, GEN)};
    barriers(cb, after, 4);
    if (r != FFX_API_RETURN_OK) {
        char why[64];
        snprintf(why, sizeof why, "its dispatch failed (%d)", (int)r);
        fail(why);
        t.temporal = 0;
        vk_temporal_done(&t, w, h, ow);
        return -1;
    }
    vk_temporal_done(&t, w, h, ow);
    G.frames++, G.temporal++, G.resets += (uint64_t)d.reset;
    if (G.frames == 1 || G.frames % 1800 == 0)
        rt_log("fsr4: %llu frames, %llu temporal, %llu resets; %ux%u -> %ux%u, jitter %+.3f %+.3f px of %u phases",
               (unsigned long long)G.frames, (unsigned long long)G.temporal, (unsigned long long)G.resets, w, h, ow, oh,
               t.jx, t.jy, t.phases);
    return 1;
}

void vk_fsr4_release(void) {   // device idle
    if (G.state <= 0) return;
    rt_log("fsr4: %llu frames, %llu temporal, %llu resets (%llu left to the game's upscaler)", (unsigned long long)G.frames,
           (unsigned long long)G.temporal, (unsigned long long)G.resets, (unsigned long long)G.spatial);
    for (int i = 0; i < GRAVE; i++) destroy_context(G.grave[i].c), G.grave[i].c = NULL;
    destroy_context(G.context);
    G.context = NULL;
}

#else   // not Windows: FSR 4 needs AMD's DLLs, which are Windows only

int vk_fsr4_wanted(void) { return 0; }
int vk_fsr4_record(VkCommandBuffer cb, uint64_t fence, const VkTexImage *in, const VkTexImage *out) {
    (void)cb, (void)fence, (void)in, (void)out;
    return 0;
}
void vk_fsr4_release(void) {}

#endif
