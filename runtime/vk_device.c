// Vulkan instance, physical device choice, logical device, VMA, timeline semaphore and the validation messenger.
// Ported from LAS Recomp, runtime/nvn/vulkan/device.c:33-330
// (https://github.com/peiikai/LASRecomp @ da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai),
// without SDL. The VMA implementation is runtime/vk_vma.cpp (VMA is C++ only).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "platform/platform.h"

static VkCtx g_vk;
static int g_ready;
static PlatMutex g_lock = PLAT_MUTEX_INIT;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type,
                                               const VkDebugUtilsMessengerCallbackDataEXT *d, void *user) {
    (void)type, (void)user;
    int err = (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
    if (err) plat_inc32(&g_vk.validation_errors);
    static volatile int32_t logged;
    if (plat_inc32(&logged) <= 500) rt_log("vk validation %s: %s", err ? "error" : "warning", d->pMessage);
    return VK_FALSE;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static int listed(const char *const *names, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(names[i], name)) return 1;
    return 0;
}

static int layer_present(const char *name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, NULL);
    VkLayerProperties *l = calloc(n ? n : 1, sizeof *l);
    vkEnumerateInstanceLayerProperties(&n, l);
    int found = 0;
    for (uint32_t i = 0; i < n; i++) if (!strcmp(l[i].layerName, name)) found = 1;
    free(l);
    return found;
}

static int instance_ext_present(const char *name) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    VkExtensionProperties *e = calloc(n ? n : 1, sizeof *e);
    if (!e) return 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, e);
    int found = has_ext(e, n, name);
    free(e);
    return found;
}

// Mesa's drivers (RADV on Linux and the Steam Deck) keep a shader cache of their own beside our pipeline cache, by
// default in ~/.cache and up to 1 GB: about 35 MB in a minute of the title screen. A Deck's /home is small and the game
// usually sits on the SD card, so unless MESA_SHADER_CACHE_DIR names a folder, it goes to mesa-cache beside the exe.
static void mesa_cache_beside_exe(void) {
    char dir[PLAT_PATH_MAX * 2];
    if (plat_getenv("MESA_SHADER_CACHE_DIR", dir, sizeof dir)) return;
    int len = plat_exe_path(dir, sizeof dir - 16);
    char *cut = len ? plat_last_sep(dir) : NULL;
    if (!cut) return;
    strcpy(cut + 1, "mesa-cache");
    plat_setenv("MESA_SHADER_CACHE_DIR", dir);
}

#ifdef __APPLE__
// An instance extension the loader offers. Enabling one it does not offer fails vkCreateInstance outright, so the
// portability enumeration extension below is checked first (macOS only, create_instance).
static int instance_ext_present(const char *name) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    VkExtensionProperties *e = calloc(n ? n : 1, sizeof *e);
    vkEnumerateInstanceExtensionProperties(NULL, &n, e);
    int found = has_ext(e, n, name);
    free(e);
    return found;
}
#endif

static int create_instance(void) {
    mesa_cache_beside_exe();
    const char *ext[28];
    uint32_t n_ext = (uint32_t)gfx_window_vk_extensions(ext, 8);
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    int validate = cfg_bool("TOTK_VK_VALIDATION");
    if (validate && !layer_present(layers[0])) {
        rt_log("vk: TOTK_VK_VALIDATION set but %s is not installed; continuing without", layers[0]);
        validate = 0;
    }
    // The checkpoints also want debug utils, for the device address binding messenger (vk_crumbs.c).
    int utils = validate;
    const char *ck = cfg_get("TOTK_GPU_CHECKPOINTS");
    if (!utils && ck && !strcmp(ck, "1")) {
        uint32_t ni = 0;
        vkEnumerateInstanceExtensionProperties(NULL, &ni, NULL);
        VkExtensionProperties *ip = calloc(ni ? ni : 1, sizeof *ip);
        if (ip) {
            vkEnumerateInstanceExtensionProperties(NULL, &ni, ip);
            utils = has_ext(ip, ni, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            free(ip);
        } else utils = 0;
    }
    if (utils) ext[n_ext++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    g_vk.debug_utils = utils;
    if (vk_hdr_wanted() && instance_ext_present(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) {   // HDR10 surface formats
        g_vk.has_swapchain_colorspace = 1;
        ext[n_ext++] = VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME;
    }
    if (vk_dlss_wanted()) {   // upscaler = dlss: what NVIDIA NGX asks for (vk_dlss.c), where the loader offers it
        const char *dx[16];
        int nd = vk_dlss_instance_exts(dx, 16);
        for (int i = 0; i < nd && n_ext < 26; i++)
            if (!listed(ext, n_ext, dx[i]) && instance_ext_present(dx[i])) ext[n_ext++] = dx[i];
    }
    // MoltenVK is a portability implementation: the loader only lists it for an instance that asks for the portability
    // enumeration extension, and the logical device then enables VK_KHR_portability_subset (create_device). Only on
    // macOS: the Khronos loader offers the extension on Windows and Linux too, and asking there would let a portability
    // device in, without the geometryShader requirement, where none was accepted before.
    VkInstanceCreateFlags flags = 0;
#ifdef __APPLE__
    if (n_ext < 12 && instance_ext_present(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        ext[n_ext++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "TotK Recomp",
                             .pEngineName = "TotK Recomp", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
                               .flags = flags,
                               .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext,
                               .enabledLayerCount = validate ? 1 : 0, .ppEnabledLayerNames = layers};
    rdoc_init();
    VkResult r = vkCreateInstance(&ci, NULL, &g_vk.instance);
    if (r != VK_SUCCESS) { rt_log("vk: vkCreateInstance failed (%d); is a Vulkan 1.3 driver installed?", (int)r); return 0; }
    g_vk.validation = validate;
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT f =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g_vk.instance, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_cb};
        if (f) f(g_vk.instance, &mi, NULL, &g_vk.messenger);
    }
    return 1;
}

static const char *const k_required_ext[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};

// VK_KHR_portability_subset is in <vulkan/vulkan_beta.h>, which only the VK_ENABLE_BETA_EXTENSIONS define pulls in;
// the subset itself is final and MoltenVK requires it on every device it exposes.
#define VK_KHR_PORTABILITY_SUBSET_NAME "VK_KHR_portability_subset"

// Name of the first missing required extension, or NULL.
static const char *missing_required(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
    VkExtensionProperties *e = calloc(n ? n : 1, sizeof *e);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, e);
    const char *miss = NULL;
    for (size_t i = 0; i < sizeof k_required_ext / sizeof *k_required_ext && !miss; i++)
        if (!has_ext(e, n, k_required_ext[i])) miss = k_required_ext[i];
    free(e);
    return miss;
}

// Graphics+compute family that can present to the window.
static uint32_t find_gfx_family(VkPhysicalDevice pd, const VkQueueFamilyProperties *qp, uint32_t nq) {
    for (uint32_t i = 0; i < nq; i++)
        if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            gfx_window_vk_presentable(g_vk.instance, pd, i))
            return i;
    return UINT32_MAX;
}

static int name_has(const char *s, const char *sub) {
    for (size_t n = strlen(sub); *s; s++)
        if (!strncasecmp(s, sub, n)) return 1;
    return 0;
}

// First discrete GPU with Vulkan 1.3, the required extensions and a graphics+present queue; TOTK_GPU_INDEX (index) or
// TOTK_GPU_NAME (case-insensitive part of the name) overrides.
static int pick_device(void) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_vk.instance, &n, NULL);
    if (!n) { rt_log("vk: no Vulkan device"); return 0; }
    VkPhysicalDevice *pd = calloc(n, sizeof *pd);
    vkEnumeratePhysicalDevices(g_vk.instance, &n, pd);
    const char *e = cfg_get("TOTK_GPU_INDEX");   // no fixed default: the best device when unset
    int want = e && *e ? atoi(e) : -1;
    const char *name = cfg_get("TOTK_GPU_NAME");
    int best = -1, best_score = -1;
    VkPhysicalDeviceProperties best_p = {0};
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        rt_log("vk: gpu %u: %s (type %d, api %u.%u.%u)", i, p.deviceName, (int)p.deviceType, VK_API_VERSION_MAJOR(p.apiVersion),
               VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        if (want >= 0 && (int)i != want) continue;
        if (name && *name && !name_has(p.deviceName, name)) continue;
        if (p.apiVersion < VK_API_VERSION_1_3) { rt_log("vk: gpu %u skipped: Vulkan below 1.3", i); continue; }
        const char *miss = missing_required(pd[i]);
        if (miss) { rt_log("vk: gpu %u skipped: missing extension %s", i, miss); continue; }
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, NULL);
        VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, qp);
        uint32_t fam = find_gfx_family(pd[i], qp, nq);
        free(qp);
        if (fam == UINT32_MAX) { rt_log("vk: gpu %u skipped: no graphics+compute+present queue family", i); continue; }
        int score = (int)i == want ? 100 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
                  : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        // The loader lists the same GPU twice when an old driver's ICD json is left behind, and the old driver can come
        // first. On a tie in score between entries with the same vendor and device ID, the newer driverVersion wins.
        // Follows ocarina-recomp runtime/vk_device.c (Hookshot, ef4c2b0), at 14566bd, GPL-3.0.
        int dup = best >= 0 && score == best_score && p.vendorID == best_p.vendorID && p.deviceID == best_p.deviceID;
        if (dup) rt_log("vk: gpu %u and gpu %d are the same GPU (leftover driver); keeping the newer driver",
                        i, best);
        if (score > best_score || (dup && p.driverVersion > best_p.driverVersion)) {
            best_score = score; best = (int)i; best_p = p;
        }
    }
    if (best < 0) {
        if (want >= 0) rt_log("vk: TOTK_GPU_INDEX=%d is not usable (see above)", want);
        else if (name && *name) rt_log("vk: no usable gpu matches TOTK_GPU_NAME=%s (see above)", name);
        else rt_log("vk: no device with Vulkan 1.3 and the required extensions (see above)");
        free(pd);
        return 0;
    }
    g_vk.phys = pd[best];
    g_vk.device_index = best;
    free(pd);
    vkGetPhysicalDeviceProperties(g_vk.phys, &g_vk.props);
    vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &g_vk.mem_props);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, qp);
    g_vk.gfx_family = find_gfx_family(g_vk.phys, qp, nq);
    g_vk.xfer_family = g_vk.gfx_family;
    for (uint32_t i = 0; i < nq; i++)   // a family that does transfer only (async DMA engine)
        if ((qp[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
            g_vk.xfer_family = i;
            break;
        }
    free(qp);
    rt_log("vk: using gpu %d: %s", best, g_vk.props.deviceName);
    vk_mem_class_log(&g_vk.props, &g_vk.mem_props);
    return 1;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features f11;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDeviceVulkan13Features f13;
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc;
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds;
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT eds2;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3;
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vids;
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond;
    VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb;
    VkPhysicalDeviceProvokingVertexFeaturesEXT pv;
    VkPhysicalDeviceRobustness2FeaturesEXT rob2;
    VkPhysicalDeviceFaultFeaturesEXT fault;
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fsi;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
    VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT fbl;
    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lr;
    VkPhysicalDeviceAddressBindingReportFeaturesEXT abr;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pei;
    VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT lrs;
} FeatureChain;

static int create_device(void) {
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, ep);
    const char *ext[48];
    uint32_t n_ext = 0;
    for (size_t i = 0; i < sizeof k_required_ext / sizeof *k_required_ext; i++) ext[n_ext++] = k_required_ext[i];
    // A portability implementation (MoltenVK) requires this one on every device it exposes; no feature of it is needed
    // here, so its feature struct is left out and both sides stay at the defaults.
    int portable = has_ext(ep, ne, VK_KHR_PORTABILITY_SUBSET_NAME);
    if (portable) ext[n_ext++] = VK_KHR_PORTABILITY_SUBSET_NAME;

    FeatureChain *c = calloc(1, sizeof *c);
    void **tail = &c->f2.pNext;
#define CHAIN(member, stype) do { c->member.sType = stype; *tail = &c->member; tail = &c->member.pNext; } while (0)
    c->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    CHAIN(f11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    CHAIN(f12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    CHAIN(f13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
    CHAIN(clip, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT);
#define OPTIONAL_EXT(flag, name, member, stype) do { \
        if (has_ext(ep, ne, name)) { g_vk.flag = 1; ext[n_ext++] = name; CHAIN(member, stype); } } while (0)
    OPTIONAL_EXT(has_custom_border_color, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, cbc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT);
    OPTIONAL_EXT(has_eds, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, eds, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT);
    OPTIONAL_EXT(has_eds2, VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, eds2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT);
    OPTIONAL_EXT(has_eds3, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, eds3, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT);
    OPTIONAL_EXT(has_vertex_input_dynamic_state, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, vids, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT);
    OPTIONAL_EXT(has_conditional_rendering, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, cond, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
    OPTIONAL_EXT(has_transform_feedback, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME, xfb, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT);
    OPTIONAL_EXT(has_provoking_vertex, VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, pv, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT);
    OPTIONAL_EXT(has_robustness2, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, rob2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
    OPTIONAL_EXT(has_device_fault, VK_EXT_DEVICE_FAULT_EXTENSION_NAME, fault, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
    // Ryujinx's SPIR-V declares fragment shader interlock whenever the host reports it (vk_pipeline.c spirv_fixup).
    OPTIONAL_EXT(has_fragment_shader_interlock, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, fsi,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT);
    // Maxwell applies primitive restart to list draws too (docs/maxwell.md 5.2); core Vulkan only to strips and fans.
    OPTIONAL_EXT(has_list_restart, VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME, lrs,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT);
    // Feedback loops (vk.h VkCtx.feedback): the layout, and barriers inside dynamic rendering
    int has_fbl = 0, has_lr = 0;
    if (has_ext(ep, ne, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
        has_ext(ep, ne, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME)) {
        has_fbl = has_lr = 1;
        ext[n_ext++] = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
        ext[n_ext++] = VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME;
        CHAIN(fbl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT);
        CHAIN(lr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR);
    }
    if (has_ext(ep, ne, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) { g_vk.has_memory_budget = 1; ext[n_ext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME; }
    if (vk_hdr_wanted() && has_ext(ep, ne, VK_EXT_HDR_METADATA_EXTENSION_NAME)) {   // the display's peak (vk_hdr_metadata)
        g_vk.has_hdr_metadata = 1;
        ext[n_ext++] = VK_EXT_HDR_METADATA_EXTENSION_NAME;
    }
    // GPU breadcrumbs for a lost device (vk_crumbs.c): AMD's drivers and RADV report it.
    if (has_ext(ep, ne, VK_AMD_BUFFER_MARKER_EXTENSION_NAME)) { g_vk.has_buffer_marker = 1; ext[n_ext++] = VK_AMD_BUFFER_MARKER_EXTENSION_NAME; }
    // NVIDIA's counterpart, opt-in because every pipeline bind then carries a checkpoint command.
    {
        const char *e = cfg_get("TOTK_GPU_CHECKPOINTS");
        if (e && !strcmp(e, "1") && has_ext(ep, ne, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME)) {
            g_vk.has_nv_checkpoints = 1;
            ext[n_ext++] = VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME;
            // Which VAs the driver maps and unmaps, to name the memory a fault address falls in (vk_crumbs.c).
            if (g_vk.debug_utils && has_ext(ep, ne, VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
                g_vk.has_address_binding_report = 1;
                ext[n_ext++] = VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME;
                CHAIN(abr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT);
            }
            // The driver's compiled code size and ISA per pipeline, to map a faulting shader binary to our stages.
            OPTIONAL_EXT(has_pipeline_exec_info, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, pei,
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR);
        }
    }
    // Graphics pipeline library: chained last so it can be dropped again when vk_caps_gpl declines it.
    void **gpl_link = NULL;
    if (has_ext(ep, ne, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) && has_ext(ep, ne, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
        ext[n_ext++] = VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME;
        ext[n_ext++] = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME;
        gpl_link = tail;
        CHAIN(gpl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT);
    }

    // The driver fills the chain with what it supports; the same chain then enables all of it (robustBufferAccess
    // stays on: guest shaders index constant buffers freely and out-of-range reads must stay harmless).
    vkGetPhysicalDeviceFeatures2(g_vk.phys, &c->f2);
    vk_caps_driver(g_vk.phys);
    vk_caps_features(&c->f2.features);
    if (gpl_link) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gp = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &gp};
        vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
        if (!vk_caps_gpl(&c->gpl, &gp)) { *gpl_link = NULL; n_ext -= 2; }
    }
    // upscaler = dlss on an NVIDIA GPU: the device extensions NVIDIA NGX asks for (vk_dlss.c), where the device has
    // them, after the two above that n_ext -= 2 may take back. VK_EXT_buffer_device_address stays out when the core
    // bufferDeviceAddress feature is on: both together are invalid (VUID-VkDeviceCreateInfo-pNext-04748).
    if (vk_dlss_wanted() && g_vk.props.vendorID == 0x10de) {
        const char *dx[16];
        int nd = vk_dlss_device_exts(dx, 16);
        for (int i = 0; i < nd && n_ext < 48; i++)
            if (!listed(ext, n_ext, dx[i]) && has_ext(ep, ne, dx[i]) &&
                !(c->f12.bufferDeviceAddress && !strcmp(dx[i], VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)))
                ext[n_ext++] = dx[i];
    }
    free(ep);
    const VkPhysicalDeviceFeatures *h = &c->f2.features;
    const char *miss = NULL;
    if (!c->f12.timelineSemaphore) miss = "timelineSemaphore";
    else if (!c->f13.dynamicRendering) miss = "dynamicRendering";
    else if (!c->f13.synchronization2) miss = "synchronization2";
    else if (!h->shaderStorageImageWriteWithoutFormat) miss = "shaderStorageImageWriteWithoutFormat";
    else if (!c->clip.depthClipControl) miss = "depthClipControl";
    else if (!h->shaderInt64) miss = "shaderInt64";
    else if (!h->samplerAnisotropy) miss = "samplerAnisotropy";
    else if (!h->independentBlend) miss = "independentBlend";
    // Metal has no geometry stage, so MoltenVK reports no geometryShader. Only the pipelines that use one are lost
    // (with a log line when they are created), which is better than refusing every device on a portability driver.
    else if (!h->geometryShader && !portable) miss = "geometryShader";
    else if (!h->textureCompressionBC) miss = "textureCompressionBC";
    if (miss) { rt_log("vk: required feature missing on %s: %s", g_vk.props.deviceName, miss); free(c); return 0; }
    if (!h->geometryShader)   // vk_pipeline.c skips the pipelines that need it (P.f_geometry)
        rt_log("vk: %s has no geometry stage (portability subset): draws that need one are skipped", g_vk.props.deviceName);
    if (!c->fault.deviceFault) g_vk.has_device_fault = 0;   // vkGetDeviceFaultInfoEXT needs the feature, not just the extension
    if (g_vk.has_list_restart && !c->lrs.primitiveTopologyListRestart) g_vk.has_list_restart = 0;
    rt_log("vk: primitive restart on list draws: %s", g_vk.has_list_restart ? "yes (primitiveTopologyListRestart)"
           : "no: a list draw's restart index is fetched as a vertex");
    if (g_vk.has_address_binding_report && !c->abr.reportAddressBinding) g_vk.has_address_binding_report = 0;
    if (g_vk.has_pipeline_exec_info && !c->pei.pipelineExecutableInfo) g_vk.has_pipeline_exec_info = 0;
    g_vk.has_buffer_device_address = c->f12.bufferDeviceAddress;
    g_vk.has_astc_ldr = h->textureCompressionASTC_LDR;
    {
        const char *e = cfg_get("TOTK_FEEDBACK");
        int layout_ok = has_fbl && has_lr && c->fbl.attachmentFeedbackLoopLayout && c->lr.dynamicRenderingLocalRead;
        // Under RADV the feedback layout is not the default: an RX 6600 hung (ring gfx timeout, no page fault) on the
        // first frame that used it (issue #122, a guess until a tester confirms); TOTK_FEEDBACK=layout opts back in.
        int layout_wanted = e && !strcmp(e, "layout") ? 1 : !vk_caps.radv;
        g_vk.feedback = e && !strcmp(e, "0") ? VK_FEEDBACK_OFF
                      : layout_ok && layout_wanted && !(e && !strcmp(e, "restart")) ? VK_FEEDBACK_LAYOUT : VK_FEEDBACK_RESTART;
        rt_log("vk: feedback loops: %s (attachmentFeedbackLoopLayout %d, dynamicRenderingLocalRead %d)",
               g_vk.feedback == VK_FEEDBACK_LAYOUT ? "feedback layout + by-region barrier"
               : g_vk.feedback == VK_FEEDBACK_RESTART ? "pass restart + full barrier" : "off (TOTK_FEEDBACK=0)",
               has_fbl && c->fbl.attachmentFeedbackLoopLayout, has_lr && c->lr.dynamicRenderingLocalRead);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi[2] = {
        {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g_vk.gfx_family, .queueCount = 1,
         .pQueuePriorities = &prio}};
    uint32_t nqi = 1;
    if (g_vk.xfer_family != g_vk.gfx_family) {
        qi[1] = (VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                          .queueFamilyIndex = g_vk.xfer_family, .queueCount = 1, .pQueuePriorities = &prio};
        nqi = 2;
    }
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &c->f2, .queueCreateInfoCount = nqi,
                             .pQueueCreateInfos = qi, .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext};
    VkResult r = vkCreateDevice(g_vk.phys, &di, NULL, &g_vk.device);
    free(c);
    if (r != VK_SUCCESS) { rt_log("vk: vkCreateDevice failed (%d)", (int)r); return 0; }
    vkGetDeviceQueue(g_vk.device, g_vk.gfx_family, 0, &g_vk.gfx_queue);
    if (g_vk.xfer_family != g_vk.gfx_family) vkGetDeviceQueue(g_vk.device, g_vk.xfer_family, 0, &g_vk.xfer_queue);
    else { g_vk.xfer_queue = g_vk.gfx_queue; g_vk.xfer_shared = 1; }

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &emh};
    vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
    g_vk.min_imported_host_alignment = (uint32_t)emh.minImportedHostPointerAlignment;
    vk_caps_init(&g_vk);

    VmaAllocatorCreateInfo ai = {.physicalDevice = g_vk.phys, .device = g_vk.device, .instance = g_vk.instance,
                                 .vulkanApiVersion = VK_API_VERSION_1_3,
                                 .flags = (g_vk.has_buffer_device_address ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0) |
                                          (g_vk.has_memory_budget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0),
                                 .pDeviceMemoryCallbacks = vk_headroom_callbacks()};
    if ((r = vmaCreateAllocator(&ai, &g_vk.vma)) != VK_SUCCESS) { rt_log("vk: vmaCreateAllocator failed (%d)", (int)r); return 0; }

    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    if ((r = vkCreateSemaphore(g_vk.device, &si, NULL, &g_vk.timeline)) != VK_SUCCESS) {
        rt_log("vk: timeline semaphore creation failed (%d)", (int)r);
        return 0;
    }
    vk_crumbs_init(&g_vk);
    return 1;
}

static long g_errors_total;     // outlives the device so the probe can print it after vk_shutdown

static void destroy_all(void) {
    gfx_window_release_vk();
    if (g_vk.device) vkDeviceWaitIdle(g_vk.device);
    if (g_vk.device) vk_clear_shutdown(g_vk.device);
    if (g_vk.device) vk_dlss_release();
    if (g_vk.device) vk_astc_release();
    if (g_vk.device) vk_crumbs_release();
    if (g_vk.timeline) vkDestroySemaphore(g_vk.device, g_vk.timeline, NULL);
    vk_headroom_stop();
    if (g_vk.vma) vmaDestroyAllocator(g_vk.vma);
    if (g_vk.device) vkDestroyDevice(g_vk.device, NULL);
    if (g_vk.messenger) {
        PFN_vkDestroyDebugUtilsMessengerEXT f =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g_vk.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (f) f(g_vk.instance, g_vk.messenger, NULL);
    }
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, NULL);
    g_errors_total += g_vk.validation_errors;
    memset(&g_vk, 0, sizeof g_vk);
}

long vk_validation_errors_total(void) { return g_errors_total + (g_ready ? g_vk.validation_errors : 0); }

int vk_init(void) {
    plat_mutex_lock(&g_lock);
    if (!g_ready) {
        memset(&g_vk, 0, sizeof g_vk);
        if (create_instance() && pick_device() && create_device()) g_ready = 1, vk_headroom_start();
        else destroy_all();
    }
    int ok = g_ready;
    plat_mutex_unlock(&g_lock);
    return ok;
}

void vk_shutdown(void) {
    plat_mutex_lock(&g_lock);
    if (g_ready) { destroy_all(); g_ready = 0; }
    plat_mutex_unlock(&g_lock);
}

VkCtx *vk_ctx(void) { return g_ready ? &g_vk : NULL; }

static const char *fault_address_type(VkDeviceFaultAddressTypeEXT t) {
    switch (t) {
    case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "none";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "read invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "write invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "execute invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "instruction pointer (unknown)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "instruction pointer (invalid)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "instruction pointer (fault)";
    default: return "unknown type";
    }
}

// The query runs once per process. Threads that hit the loss at the same time wait on the lock until it is logged, so
// none of them reaches its rt_fatal and ends the process before the report is written.
void vk_device_fault_report(VkResult r) {
    static PlatMutex lock = PLAT_MUTEX_INIT;
    static int done;
    if (r != VK_ERROR_DEVICE_LOST) return;
    plat_mutex_lock(&lock);
    if (done) { plat_mutex_unlock(&lock); return; }
    done = 1;
    vk_crumbs_report();
    VkCtx *c = vk_ctx();
    if (c && c->vma) {   // how full each heap was: a fault under memory pressure reads differently
        VmaBudget b[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(c->vma, b);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(c->phys, &mp);
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
            rt_log("vk: device lost: heap %u usage %llu MiB of budget %llu MiB (VMA blocks %llu MiB)", i,
                   (unsigned long long)(b[i].usage >> 20), (unsigned long long)(b[i].budget >> 20),
                   (unsigned long long)(b[i].statistics.blockBytes >> 20));
    }
    PFN_vkGetDeviceFaultInfoEXT get =
        c && c->has_device_fault ? (PFN_vkGetDeviceFaultInfoEXT)vkGetDeviceProcAddr(c->device, "vkGetDeviceFaultInfoEXT")
                                 : NULL;
    if (!get) {
        rt_log("vk: device lost; no fault report (VK_EXT_device_fault %s)", c && c->has_device_fault ? "has no entry point"
                                                                                                    : "not enabled");
        plat_mutex_unlock(&lock);
        return;
    }
    VkDeviceFaultCountsEXT n = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    VkResult q = get(c->device, &n, NULL);
    VkDeviceFaultAddressInfoEXT *ai = calloc(n.addressInfoCount + 1, sizeof *ai);
    VkDeviceFaultVendorInfoEXT *vi = calloc(n.vendorInfoCount + 1, sizeof *vi);
    VkDeviceFaultInfoEXT info = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, .pAddressInfos = ai, .pVendorInfos = vi};
    uint64_t binary = n.vendorBinarySize;
    // The vendor binary is fetched only under DKB_GPU_CHECKPOINTS=1 and within a sane size; otherwise its size alone is
    // logged. Per the spec the first call returns sizes and the second fills the data pointers.
    void *bin = NULL;
    if (vk_checkpoints_on() && binary > 0 && binary <= (64ull << 20)) bin = malloc((size_t)binary);
    n.vendorBinarySize = bin ? binary : 0;
    info.pVendorBinaryData = bin;
    if (q == VK_SUCCESS && ai && vi) q = get(c->device, &n, &info);
    if (bin && (q == VK_SUCCESS || q == VK_INCOMPLETE)) {
        // Best effort: any failure here is a log line only. The dump can hold derived shader code, so it stays in the
        // local logs folder next to dkb.log and is never attached to reports.
        const VkDeviceFaultVendorBinaryHeaderVersionOneEXT *h = bin;
        if (n.vendorBinarySize >= sizeof *h)
            rt_log("vk: device fault binary header: vendor %#x device %#x driver %#x app name offset %u app version %#x "
                   "engine version %#x api version %#x", h->vendorID, h->deviceID, h->driverVersion, h->applicationNameOffset,
                   h->applicationVersion, h->engineVersion, h->apiVersion);
        const char *lp = rt_log_path();
        char path[PLAT_PATH_MAX * 3 + 48];
        const char *cut = lp ? strrchr(lp, '/') : NULL;
        if (cut) {
            snprintf(path, sizeof path, "%.*s/dkb-fault-%llu.bin", (int)(cut - lp), lp, (unsigned long long)time(NULL));
            if (plat_file_write_atomic(path, bin, (size_t)n.vendorBinarySize))
                rt_log("vk: device fault binary saved: %s (%llu bytes)", path, (unsigned long long)n.vendorBinarySize);
            else rt_log("vk: device fault binary: cannot write %s", path);
        } else rt_log("vk: device fault binary not saved (no log file)");
    }
    free(bin);
    if (q != VK_SUCCESS && q != VK_INCOMPLETE) {
        rt_log("vk: device lost; vkGetDeviceFaultInfoEXT failed (%d)", (int)q);
    } else {
        rt_log("vk: device fault: \"%s\" (%u address infos, %u vendor infos, vendor binary %llu bytes)", info.description,
               n.addressInfoCount, n.vendorInfoCount, (unsigned long long)binary);
        // The faulting access lies in the precision-aligned range around the reported address.
        for (uint32_t i = 0; i < n.addressInfoCount; i++)
        {
            rt_log("vk: device fault address %u: %s at %#llx, precision %#llx", i, fault_address_type(ai[i].addressType),
                   (unsigned long long)ai[i].reportedAddress, (unsigned long long)ai[i].addressPrecision);
            vk_crumbs_report_address(ai[i].reportedAddress);
        }
        for (uint32_t i = 0; i < n.vendorInfoCount; i++)
            rt_log("vk: device fault vendor info %u: \"%s\" code %#llx data %#llx", i, vi[i].description,
                   (unsigned long long)vi[i].vendorFaultCode, (unsigned long long)vi[i].vendorFaultData);
    }
    free(ai);
    free(vi);
    plat_mutex_unlock(&lock);
}

int vk_selftest(void) {
    if (!vk_init()) return 1;
    VkCtx *c = &g_vk;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 64 << 10,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
    VkBuffer buf;
    VmaAllocation alloc;
    VkResult r = vmaCreateBuffer(c->vma, &bi, &aci, &buf, &alloc, NULL);
    if (r != VK_SUCCESS) { rt_log("vk selftest: vmaCreateBuffer failed (%d)", (int)r); return 2; }
    vmaDestroyBuffer(c->vma, buf, alloc);

    VkSemaphoreSignalInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = c->timeline, .value = 1};
    if ((r = vkSignalSemaphore(c->device, &sig)) != VK_SUCCESS) { rt_log("vk selftest: vkSignalSemaphore failed (%d)", (int)r); return 3; }
    uint64_t value = 1;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &c->timeline,
                              .pValues = &value};
    if ((r = vkWaitSemaphores(c->device, &wi, 1000000000ull)) != VK_SUCCESS) { rt_log("vk selftest: vkWaitSemaphores failed (%d)", (int)r); return 4; }
    uint64_t now = 0;
    vkGetSemaphoreCounterValue(c->device, c->timeline, &now);
    if (now != 1) { rt_log("vk selftest: timeline value %llu, expected 1", (unsigned long long)now); return 5; }
    return 0;
}

// --probe-vkdevice (host.h)
int probe_vkdevice(void) {
    int rc = vk_selftest();
    VkCtx *c = vk_ctx();
    if (rc || !c) { printf("probe-vkdevice: FAILED (selftest %d)\n", rc ? rc : 1); vk_shutdown(); return 1; }
    uint32_t v = c->props.apiVersion;
    printf("probe-vkdevice: device %d \"%s\" api %u.%u.%u, gfx queue family %u, transfer family %u%s, host import alignment %#x\n",
           c->device_index, c->props.deviceName, VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v),
           c->gfx_family, c->xfer_family, c->xfer_shared ? " (shared queue)" : "", c->min_imported_host_alignment);
    printf("probe-vkdevice: optional: custom_border=%d eds=%d eds2=%d eds3=%d vertex_input=%d cond_render=%d xfb=%d provoking=%d "
           "robustness2=%d fault=%d budget=%d bda=%d interlock=%d validation=%d\n",
           c->has_custom_border_color, c->has_eds, c->has_eds2, c->has_eds3, c->has_vertex_input_dynamic_state,
           c->has_conditional_rendering, c->has_transform_feedback, c->has_provoking_vertex, c->has_robustness2,
           c->has_device_fault, c->has_memory_budget, c->has_buffer_device_address, c->has_fragment_shader_interlock,
           c->validation);
    vk_shutdown();
    printf("probe-vkdevice: %s (validation errors %ld)\n", g_errors_total ? "FAILED" : "passed", g_errors_total);
    return g_errors_total ? 1 : 0;
}
