// GPU breadcrumbs for a lost device (vk.h vk_crumb). VK_AMD_buffer_marker writes a 32-bit code into a buffer from
// the command stream: at TOP_OF_PIPE once the command processor reaches the marker, at BOTTOM_OF_PIPE once every
// command before it has finished. Each crumb writes both, into two words of one host-coherent buffer, so after a
// device loss word 0 names the newest point the GPU reached and word 1 the newest point it had finished: the hang lies
// between them. A code is the crumb kind in the top 4 bits and its number in the low 28. The markers cost two small
// writes per crumb, and crumbs mark submissions, upload batches, ASTC chunks and presents, not draws.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"

static PFN_vkCmdWriteBufferMarkerAMD write_marker;
static VkBuffer buf;
static VmaAllocation alloc;
static volatile uint32_t *words;   // 0 and 1: the newest crumb reached and finished; 2 and 3: the same for VK_CRUMB_BIND

// TOTK_CRUMB_BIND=1: ring of the newest pipeline binds, indexed by bind number & (BIND_RING - 1). Recording runs a frame
// or more ahead of the GPU, hence the size. Written by recording threads without a lock; a slot's seq holds the bind
// number plus BIND_VALID (bind number 0 occurs at the 28-bit wrap), so a slot being rewritten is skipped by the report.
enum { BIND_RING = 1 << 16, BIND_VALID = 1 << 28 };
static int bind_on;
static uint32_t bind_n;
static struct { uint32_t seq; VkPipeline pipe; uint64_t stage[3]; } bind_ring[BIND_RING];

// NVIDIA (VK_NV_device_diagnostic_checkpoints, DKB_GPU_CHECKPOINTS=1): every pipeline bind sets a checkpoint whose marker
// is the VkPipeline; after a device loss the queue reports the checkpoints the GPU reached. The registry maps a pipeline
// to the NVN 2 names of its stages (what the bind site knows), filled at bind time, so the report names the shaders.
static PFN_vkCmdSetCheckpointNV set_cp;
static PFN_vkGetQueueCheckpointDataNV get_cp;
static PFN_vkGetQueueCheckpointData2NV get_cp2;
// Handles cannot be reused while the registry is read, only because bound pipelines live until vk_pipeline_shutdown.
// When programs share one VkPipeline the last bind wins. Work after the last bind (clear, ASTC, FSR, copies) is
// attributed to that pipeline.
enum { CP_SLOTS = 8192, CP_PROBES = 64 };   // ponytail: open addressing, a full table drops new pipelines; enlarge if the report shows "unknown"
static struct { VkPipeline pipe; uint64_t stage[3]; } cp_reg[CP_SLOTS];   // vertex, geometry, fragment (or compute, 0, 0)
static PlatMutex cp_lock = PLAT_MUTEX_INIT;
static volatile uint32_t cp_tag_n;   // next tag slot, see vk_checkpoint_tag

// High bits of a Fibonacci multiply (pointers have constant low bits), as find_stage in vk_exec_draw.c.
static uint32_t cp_slot(const void *p) { return (uint32_t)(((uint64_t)(uintptr_t)p * 0x9E3779B97F4A7C15ull) >> 51); }

// Device address bindings (VK_EXT_device_address_binding_report, with the checkpoints on): the driver reports every
// range of GPU virtual addresses it binds or unbinds to a debug utils messenger, tens of thousands of events in a long
// run. A ring of the newest events loses the mapping of a long-lived allocation, so the bound ranges are kept as a live
// set (open addressing keyed by base address, linear probing, backward-shift delete: a bind inserts, an unbind removes)
// and only the newest AB_UNBINDS unbinds are kept, to name a range freed before a fault (use after free). The callback
// runs on driver threads, so one spinlock guards both; the device-lost report only try-locks it.
enum { AB_LIVE = 1 << 17, AB_UNBINDS = 8192 };
typedef struct { uint64_t seq, base, size, handle; uint32_t flags, type; } AbEvent;   // seq 0 marks a free slot
static AbEvent ab_live[AB_LIVE];
static AbEvent ab_unb[AB_UNBINDS];
static uint64_t ab_n, ab_nlive, ab_dropped, ab_nunb;   // events seen, live entries, binds not stored, unbinds seen
static volatile int ab_lock;
static VkDebugUtilsMessengerEXT ab_messenger;

static uint32_t ab_home(uint64_t base) { return (uint32_t)((base * 0x9E3779B97F4A7C15ull) >> 47); }   // 17 bits
static void ab_acquire(void) { while (__atomic_exchange_n(&ab_lock, 1, __ATOMIC_ACQUIRE)) { } }
static void ab_release(void) { __atomic_store_n(&ab_lock, 0, __ATOMIC_RELEASE); }

// Caller holds the lock.
static void ab_bind(const AbEvent *e) {
    if (ab_nlive >= AB_LIVE - 1) { ab_dropped++; return; }   // keep one slot free so probes terminate
    uint32_t i = ab_home(e->base);
    while (ab_live[i].seq) i = (i + 1) & (AB_LIVE - 1);
    ab_live[i] = *e;
    ab_nlive++;
}

static void ab_unbind(const AbEvent *e) {
    uint32_t i = ab_home(e->base), hit = AB_LIVE;
    for (; ab_live[i].seq; i = (i + 1) & (AB_LIVE - 1)) {
        if (ab_live[i].base != e->base) continue;
        if (ab_live[i].handle == e->handle) { hit = i; break; }
        if (hit == AB_LIVE) hit = i;   // same base, other handle: take it if nothing matches better
    }
    if (hit != AB_LIVE) {
        // Backward-shift delete: pull later entries of the probe run into the hole when their home allows it.
        uint32_t j = hit;
        for (;;) {
            j = (j + 1) & (AB_LIVE - 1);
            if (!ab_live[j].seq) break;
            uint32_t h = ab_home(ab_live[j].base);
            if (((j - h) & (AB_LIVE - 1)) >= ((j - hit) & (AB_LIVE - 1))) ab_live[hit] = ab_live[j], hit = j;
        }
        memset(&ab_live[hit], 0, sizeof ab_live[hit]);
        ab_nlive--;
    }
    ab_unb[ab_nunb++ % AB_UNBINDS] = *e;   // kept even if the bind was never seen or was dropped
}

static VKAPI_ATTR VkBool32 VKAPI_CALL ab_callback(VkDebugUtilsMessageSeverityFlagBitsEXT sev,
                                                  VkDebugUtilsMessageTypeFlagsEXT type,
                                                  const VkDebugUtilsMessengerCallbackDataEXT *d, void *user) {
    (void)sev, (void)type, (void)user;
    for (const VkBaseInStructure *s = d->pNext; s; s = s->pNext) {
        if (s->sType != VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT) continue;
        const VkDeviceAddressBindingCallbackDataEXT *b = (const void *)s;
        AbEvent e = {.base = b->baseAddress, .size = b->size, .flags = b->flags,
                     .type = d->objectCount ? (uint32_t)d->pObjects[0].objectType : 0,
                     .handle = d->objectCount ? d->pObjects[0].objectHandle : 0};
        ab_acquire();
        e.seq = ++ab_n;
        if (b->bindingType == VK_DEVICE_ADDRESS_BINDING_TYPE_UNBIND_EXT) ab_unbind(&e);
        else ab_bind(&e);
        ab_release();
    }
    return VK_FALSE;
}

static const char *ab_object(uint32_t t) {
    return t == VK_OBJECT_TYPE_DEVICE_MEMORY ? "memory" : t == VK_OBJECT_TYPE_BUFFER ? "buffer" : t == VK_OBJECT_TYPE_IMAGE ? "image" : "object";
}

static void ab_print(const char *why, const AbEvent *e, uint64_t a) {
    rt_log("vk: address binding #%llu %s: %#llx..%#llx (%llu bytes) %s %#llx%s, address is %#llx %s the start", (unsigned long long)e->seq, why,
           (unsigned long long)e->base, (unsigned long long)(e->base + e->size), (unsigned long long)e->size,
           ab_object(e->type), (unsigned long long)e->handle,
           e->flags & VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT ? " (driver internal)" : "",
           (unsigned long long)(a >= e->base ? a - e->base : e->base - a), a >= e->base ? "past" : "before");
}

void vk_crumbs_report_address(uint64_t a) {
    if (!ab_messenger) return;
    // The device is lost, but a driver thread may still hold the lock (or have died holding it): spin briefly, then read
    // without it. An unlocked read can show a torn entry, never crash: the tables are static.
    int locked = 0;
    for (uint32_t spin = 0; spin < 200000 && !locked; spin++) locked = !__atomic_exchange_n(&ab_lock, 1, __ATOMIC_ACQUIRE);
    uint64_t total = ab_n;
    rt_log("vk: address %#llx: %llu binding events seen, %llu live ranges, %llu binds dropped (live set full)%s",
           (unsigned long long)a, (unsigned long long)total, (unsigned long long)ab_nlive,
           (unsigned long long)ab_dropped, locked ? "" : "; lock busy, reading the tables without it (best effort)");
    uint32_t shown = 0;
    const AbEvent *below = NULL, *above = NULL;
    for (uint32_t i = 0; i < AB_LIVE; i++) {
        const AbEvent *e = &ab_live[i];
        if (!e->seq) continue;
        if (a >= e->base && a - e->base < e->size) {
            if (shown++ < 16) ab_print("is live and contains the address", e, a);
        } else if (e->base + e->size <= a) {
            if (!below || e->base + e->size > below->base + below->size) below = e;
        } else if (e->base > a && (!above || e->base < above->base)) {
            above = e;
        }
    }
    if (shown > 16) rt_log("vk: address %#llx: %u more live ranges contain it", (unsigned long long)a, shown - 16);
    if (!shown) {
        rt_log("vk: address %#llx: no live range contains it", (unsigned long long)a);
        if (below) ab_print("nearest live range below", below, a);
        if (above) ab_print("nearest live range above", above, a);
    }
    uint64_t nunb = ab_nunb, first = nunb > AB_UNBINDS ? nunb - AB_UNBINDS : 0;
    uint32_t freed = 0;
    for (uint64_t k = nunb; k > first && freed < 8; k--) {   // newest first
        const AbEvent *e = &ab_unb[(k - 1) % AB_UNBINDS];
        if (!e->seq || a < e->base || a - e->base >= e->size) continue;
        freed++;
        rt_log("vk: address %#llx: UNBOUND before the fault, %llu binding events ago:", (unsigned long long)a,
               (unsigned long long)(total - e->seq));
        ab_print("(unbind)", e, a);
    }
    if (!freed) rt_log("vk: address %#llx: no recent unbind (last %u kept) contains it", (unsigned long long)a, AB_UNBINDS);
    if (locked) ab_release();
}

void vk_crumbs_init(VkCtx *c) {
    if (c->has_address_binding_report) {
        PFN_vkCreateDebugUtilsMessengerEXT f =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(c->instance, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT,
            .pfnUserCallback = ab_callback};
        if (f && f(c->instance, &mi, NULL, &ab_messenger) == VK_SUCCESS)
            rt_log("vk: device address bindings recorded (VK_EXT_device_address_binding_report)");
        else rt_log("vk: device address binding report unavailable");
    } else if (c->has_nv_checkpoints) {
        rt_log("vk: device address binding report not available (VK_EXT_device_address_binding_report): fault addresses stay unmapped");
    }
    if (c->has_nv_checkpoints) {
        set_cp = (PFN_vkCmdSetCheckpointNV)vkGetDeviceProcAddr(c->device, "vkCmdSetCheckpointNV");
        get_cp = (PFN_vkGetQueueCheckpointDataNV)vkGetDeviceProcAddr(c->device, "vkGetQueueCheckpointDataNV");
        get_cp2 = (PFN_vkGetQueueCheckpointData2NV)vkGetDeviceProcAddr(c->device, "vkGetQueueCheckpointData2NV");
        if (!set_cp || !(get_cp || get_cp2)) set_cp = NULL;
        rt_log(set_cp ? "vk: GPU checkpoints on (VK_NV_device_diagnostic_checkpoints)" : "vk: GPU checkpoints unavailable");
    }
    if (!c->has_buffer_marker) return;
    write_marker = (PFN_vkCmdWriteBufferMarkerAMD)vkGetDeviceProcAddr(c->device, "vkCmdWriteBufferMarkerAMD");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 32,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    if (!write_marker || vmaCreateBuffer(c->vma, &bi, &ai, &buf, &alloc, &info) != VK_SUCCESS) {
        rt_log("vk: GPU breadcrumbs unavailable");
        write_marker = NULL;
        return;
    }
    words = info.pMappedData;
    words[0] = words[1] = words[2] = words[3] = 0;
    const char *bind = cfg_get("TOTK_CRUMB_BIND");
    bind_on = bind && !strcmp(bind, "1");
    rt_log("vk: GPU breadcrumbs on (VK_AMD_buffer_marker)%s", bind_on ? ", pipeline binds marked (TOTK_CRUMB_BIND=1)" : "");
}

// isa_dump state (below), reset on release.
static uint32_t pipes;
static uint64_t written;
static PFN_vkGetPipelineExecutablePropertiesKHR get_props;
static PFN_vkGetPipelineExecutableStatisticsKHR get_stats;
static PFN_vkGetPipelineExecutableInternalRepresentationsKHR get_irs;

void vk_crumbs_release(void) {
    VkCtx *c = vk_ctx();
    if (ab_messenger && c) {
        PFN_vkDestroyDebugUtilsMessengerEXT f =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(c->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (f) f(c->instance, ab_messenger, NULL);
    }
    ab_messenger = VK_NULL_HANDLE;
    ab_acquire();
    ab_n = ab_nlive = ab_dropped = ab_nunb = 0;
    memset(ab_live, 0, sizeof ab_live);
    memset(ab_unb, 0, sizeof ab_unb);
    ab_release();
    cp_tag_n = 0;
    if (buf && c) vmaDestroyBuffer(c->vma, buf, alloc);
    buf = VK_NULL_HANDLE, alloc = NULL, words = NULL, write_marker = NULL;
    bind_on = 0, bind_n = 0;
    memset(bind_ring, 0, sizeof bind_ring);
    set_cp = NULL, get_cp = NULL, get_cp2 = NULL;
    get_props = NULL, get_stats = NULL, get_irs = NULL, pipes = 0, written = 0;
    plat_mutex_lock(&cp_lock);
    memset(cp_reg, 0, sizeof cp_reg);
    plat_mutex_unlock(&cp_lock);
}

int vk_checkpoints_on(void) { return set_cp != NULL; }

void vk_checkpoint(VkCommandBuffer cb, const void *marker) {
    if (set_cp && cb) set_cp(cb, marker);
}

VkPipelineCreateFlags vk_pipeline_capture_flags(void) {
    VkCtx *c = vk_ctx();
    return c && c->has_pipeline_exec_info && set_cp
               ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR
               : 0;
}

// DKB_GPU_CHECKPOINTS=1 with VK_KHR_pipeline_executable_properties: one log line per executable of a pipeline (its
// statistics, which include the compiled binary size on NVIDIA) and its internal representations (the driver's ISA text
// or binary) saved next to dkb.log, so a faulting shader binary of a known size and code offset can be matched to the
// NVN 2 stages that built it. The files hold code derived from the game: local only, ignored by git.
enum { ISA_MAX_PIPES = 4096, ISA_MAX_BYTES = 512u << 20 };

static void isa_dump(VkPipeline p, uint64_t vs, uint64_t gs, uint64_t fs) {
    VkCtx *c = vk_ctx();
    if (!c || !c->has_pipeline_exec_info || __atomic_fetch_add(&pipes, 1, __ATOMIC_RELAXED) >= ISA_MAX_PIPES) return;
    plat_mutex_lock(&cp_lock);   // the entry points load once, the byte budget is shared
    if (!get_irs) {
        get_props = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(c->device, "vkGetPipelineExecutablePropertiesKHR");
        get_stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(c->device, "vkGetPipelineExecutableStatisticsKHR");
        get_irs = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(
            c->device, "vkGetPipelineExecutableInternalRepresentationsKHR");
    }
    plat_mutex_unlock(&cp_lock);
    if (!get_props || !get_stats || !get_irs) return;
    uint32_t ne = 0;
    VkPipelineInfoKHR pinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = p};
    if (get_props(c->device, &pinfo, &ne, NULL) != VK_SUCCESS || !ne) return;
    VkPipelineExecutablePropertiesKHR *pr = calloc(ne, sizeof *pr);
    if (!pr) return;
    for (uint32_t i = 0; i < ne; i++) pr[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    get_props(c->device, &pinfo, &ne, pr);
    const char *lp = rt_log_path();
    const char *cut = lp ? strrchr(lp, '/') : NULL;
    for (uint32_t i = 0; i < ne; i++) {
        VkPipelineExecutableInfoKHR ei = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = p, .executableIndex = i};
        const char *stage = pr[i].stages & VK_SHADER_STAGE_VERTEX_BIT ? "vertex" : pr[i].stages & VK_SHADER_STAGE_FRAGMENT_BIT ? "fragment"
                          : pr[i].stages & VK_SHADER_STAGE_GEOMETRY_BIT ? "geometry" : "other";
        char line[2048];
        int n = snprintf(line, sizeof line, "vk: pipeline %p (NVN 2 stages %016llx %016llx %016llx) executable %u %s \"%s\" \"%s\":", (void *)p,
                         (unsigned long long)vs, (unsigned long long)gs, (unsigned long long)fs, i, stage, pr[i].name, pr[i].description);
        uint32_t ns = 0;
        VkPipelineExecutableStatisticKHR *st = NULL;
        if (get_stats(c->device, &ei, &ns, NULL) == VK_SUCCESS && ns && (st = calloc(ns, sizeof *st))) {
            for (uint32_t j = 0; j < ns; j++) st[j].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            get_stats(c->device, &ei, &ns, st);
            for (uint32_t j = 0; j < ns && n > 0 && (size_t)n < sizeof line; j++) {
                const VkPipelineExecutableStatisticValueKHR *v = &st[j].value;
                switch (st[j].format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: n += snprintf(line + n, sizeof line - n, " %s=%u", st[j].name, v->b32); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: n += snprintf(line + n, sizeof line - n, " %s=%lld", st[j].name, (long long)v->i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: n += snprintf(line + n, sizeof line - n, " %s=%llu", st[j].name, (unsigned long long)v->u64); break;
                default: n += snprintf(line + n, sizeof line - n, " %s=%g", st[j].name, v->f64); break;
                }
            }
            free(st);
        }
        rt_log("%s", line);
        uint32_t nr = 0;
        if (get_irs(c->device, &ei, &nr, NULL) != VK_SUCCESS || !nr) continue;
        VkPipelineExecutableInternalRepresentationKHR *ir = calloc(nr, sizeof *ir);
        if (!ir) continue;
        for (uint32_t j = 0; j < nr; j++) ir[j].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        if (get_irs(c->device, &ei, &nr, ir) == VK_SUCCESS) {   // sizes only: pData is NULL
            for (uint32_t j = 0; j < nr; j++) {
                size_t sz = ir[j].dataSize;
                if (!sz || __atomic_fetch_add(&written, sz, __ATOMIC_RELAXED) + sz > ISA_MAX_BYTES) {   // reserve before malloc
                    if (sz) __atomic_fetch_sub(&written, sz, __ATOMIC_RELAXED);
                    rt_log("vk: pipeline %p executable %u representation \"%s\" (%zu bytes) not saved (budget)", (void *)p, i, ir[j].name, sz);
                    continue;
                }
                if (!(ir[j].pData = malloc(sz))) __atomic_fetch_sub(&written, sz, __ATOMIC_RELAXED);
            }
            get_irs(c->device, &ei, &nr, ir);   // fills the buffers just allocated
            {
                for (uint32_t j = 0; j < nr; j++) {
                    if (!ir[j].pData) continue;
                    char path[PLAT_PATH_MAX * 3 + 96], suffix[24] = "";
                    if (i || j) snprintf(suffix, sizeof suffix, "-%u-%u", i, j);
                    if (cut) snprintf(path, sizeof path, "%.*s/dkb-isa-%llx-%s%s.%s", (int)(cut - lp), lp, (unsigned long long)(uintptr_t)p, stage,
                                      suffix, ir[j].isText ? "txt" : "bin");
                    if (cut && plat_file_write_atomic(path, ir[j].pData, ir[j].dataSize)) {
                        rt_log("vk: pipeline %p executable %u representation \"%s\" saved: %s (%zu bytes)", (void *)p, i, ir[j].name, path, ir[j].dataSize);
                    } else __atomic_fetch_sub(&written, ir[j].dataSize, __ATOMIC_RELAXED), rt_log("vk: pipeline %p executable %u representation \"%s\": cannot save", (void *)p, i, ir[j].name);
                }
            }
            for (uint32_t j = 0; j < nr; j++) free(ir[j].pData);
        }
        free(ir);
    }
    free(pr);
}

void vk_checkpoint_note(VkPipeline p, uint64_t a, uint64_t b, uint64_t c) {
    if (!set_cp || !p) return;
    plat_mutex_lock(&cp_lock);
    uint32_t i = cp_slot(p);
    int n = 0;
    while (n < CP_PROBES && cp_reg[i].pipe && cp_reg[i].pipe != p) n++, i = (i + 1) % CP_SLOTS;
    int fresh = n < CP_PROBES && !cp_reg[i].pipe;
    if (n < CP_PROBES) cp_reg[i].pipe = p, cp_reg[i].stage[0] = a, cp_reg[i].stage[1] = b, cp_reg[i].stage[2] = c;
    plat_mutex_unlock(&cp_lock);
    if (fresh) isa_dump(p, a, b, c);   // first bind of this pipeline: its stage names are known now
}

// Non-pipeline checkpoints (dispatch, copies, blits, clears; vk_rec.c): the marker points into this ring and names the
// command, plus the compute pipeline last bound on the recording thread for a dispatch. The ring wraps: a checkpoint
// older than CP_TAGS tags would read a newer tag, far more than the frames in flight hold.
enum { CP_TAGS = 1 << 16 };
static struct { const char *kind; VkPipeline compute; } cp_tags[CP_TAGS];

static _Thread_local VkPipeline cp_compute;

// The last-bound pipeline is per recording thread, not per command buffer, so a dispatch may carry the pipeline another
// buffer bound last on that thread (diagnostic only).
void vk_checkpoint_compute(VkPipeline p) { cp_compute = p; }

const void *vk_checkpoint_tag(const char *kind) {
    if (!set_cp) return NULL;
    uint32_t i = __atomic_fetch_add(&cp_tag_n, 1, __ATOMIC_RELAXED) % CP_TAGS;
    cp_tags[i].kind = kind;
    cp_tags[i].compute = !strncmp(kind, "dispatch", 8) ? cp_compute : VK_NULL_HANDLE;
    return &cp_tags[i];
}

static void pipeline_name(const void *m, char *o, size_t size);

static void checkpoint_name(const void *m, char *o, size_t size) {
    if ((const char *)m >= (const char *)cp_tags && (const char *)m < (const char *)(cp_tags + CP_TAGS)) {
        int n = snprintf(o, size, "%s", ((const typeof(cp_tags[0]) *)m)->kind);
        if (((const typeof(cp_tags[0]) *)m)->compute && n > 0 && (size_t)n < size) {
            snprintf(o + n, size - (size_t)n, ", last bound compute ");
            n = (int)strlen(o);
            pipeline_name(((const typeof(cp_tags[0]) *)m)->compute, o + n, size - (size_t)n);
        }
        return;
    }
    pipeline_name(m, o, size);
}

static void pipeline_name(const void *m, char *o, size_t size) {
    snprintf(o, size, "marker %p, pipeline not registered", m);
    plat_mutex_lock(&cp_lock);
    uint32_t i = cp_slot(m);
    for (int n = 0; n < CP_PROBES && cp_reg[i].pipe; n++, i = (i + 1) % CP_SLOTS)
        if ((const void *)cp_reg[i].pipe == m) {
            snprintf(o, size, "pipeline %p, NVN 2 stages %016llx %016llx %016llx (vertex or compute, geometry, fragment)", m,
                     (unsigned long long)cp_reg[i].stage[0], (unsigned long long)cp_reg[i].stage[1],
                     (unsigned long long)cp_reg[i].stage[2]);
            break;
        }
    plat_mutex_unlock(&cp_lock);
}

static void checkpoints_report(void) {
    VkCheckpointData2NV d2[16];
    VkCheckpointDataNV d1[16];
    uint32_t n = 16;
    VkCtx *c = vk_ctx();
    if (!c) return;
    VkQueue q = c->gfx_queue;
    if (get_cp2) {
        for (int i = 0; i < 16; i++) d2[i] = (VkCheckpointData2NV){.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV};
        get_cp2(q, &n, d2);
    } else {
        for (int i = 0; i < 16; i++) d1[i] = (VkCheckpointDataNV){.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV};
        get_cp(q, &n, d1);
    }
    rt_log("vk: GPU checkpoints: %u reported on the graphics queue", n);
    for (uint32_t i = 0; i < n && i < 16; i++) {
        char name[256];
        const void *m = get_cp2 ? d2[i].pCheckpointMarker : d1[i].pCheckpointMarker;
        checkpoint_name(m, name, sizeof name);
        unsigned long long st = get_cp2 ? (unsigned long long)d2[i].stage : (unsigned long long)d1[i].stage;
        if (st == 0x1) rt_log("vk: GPU checkpoint %u: stage TOP_OF_PIPE, %s", i, name);
        else if (st == 0x2000) rt_log("vk: GPU checkpoint %u: stage BOTTOM_OF_PIPE, %s", i, name);
        else rt_log("vk: GPU checkpoint %u: stage %#llx, %s", i, st, name);
    }
}

void vk_crumb(VkCommandBuffer cb, int kind, uint64_t n) {
    if (!write_marker || !cb) return;
    uint32_t code = (uint32_t)kind << 28 | (uint32_t)(n & 0x0fffffff);
    VkDeviceSize at = kind == VK_CRUMB_BIND ? 8 : 0;   // binds have their own words so they do not hide the submit crumbs
    write_marker(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, buf, at, code);
    write_marker(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, buf, at + 4, code);
}

int vk_crumb_bind_on(void) { return bind_on; }

uint32_t vk_crumb_bind_note(VkPipeline p, uint64_t vs, uint64_t gs, uint64_t fs) {
    if (!bind_on) return 0;
    uint32_t seq = __atomic_add_fetch(&bind_n, 1, __ATOMIC_RELAXED) & 0x0fffffff;   // the crumb's 28 bits; 2^28 is a multiple of BIND_RING
    uint32_t i = seq & (BIND_RING - 1);
    __atomic_store_n(&bind_ring[i].seq, 0, __ATOMIC_RELAXED);
    bind_ring[i].pipe = p, bind_ring[i].stage[0] = vs, bind_ring[i].stage[1] = gs, bind_ring[i].stage[2] = fs;
    __atomic_store_n(&bind_ring[i].seq, seq | BIND_VALID, __ATOMIC_RELEASE);
    return seq;
}

static void describe(uint32_t code, char *o, size_t size) {
    static const char *const kinds[] = {"nothing", "the start of a part of submission", "the end of submission",
                                        "the start of upload batch", "the end of upload batch",
                                        "ASTC decode chunk", "the start of present",
                                        "the end of present", "pipeline bind"};
    unsigned k = code >> 28;
    if (!code) snprintf(o, size, "nothing yet");
    else if (k < sizeof kinds / sizeof *kinds) snprintf(o, size, "%s %u", kinds[k], code & 0x0fffffff);
    else snprintf(o, size, "unknown code %#x", code);
}

void vk_crumbs_report(void) {
    if (set_cp) checkpoints_report();
    if (!words) return;
    char reached[96], finished[96];
    describe(words[0], reached, sizeof reached);
    describe(words[1], finished, sizeof finished);
    rt_log("vk: GPU breadcrumbs: the GPU last reached %s; everything before %s had finished", reached, finished);
    if (!bind_on) return;
    describe(words[2], reached, sizeof reached);
    describe(words[3], finished, sizeof finished);
    rt_log("vk: GPU bind breadcrumbs: the GPU last reached %s; everything before %s had finished", reached, finished);
    // Recording runs ahead of the GPU, so the binds in question are looked up by number: the one reached, the one
    // finished, and the 3 before each. A slot another bind has reused since is skipped.
    uint32_t reach = words[2] & 0x0fffffff, fin = words[3] & 0x0fffffff;
    (void)__atomic_load_n(&bind_n, __ATOMIC_RELAXED);
    for (int w = 0; w < 2; w++) {
        uint32_t at = w ? fin : reach;
        for (uint32_t k = 0; k < 4; k++) {
            uint32_t s = (at - k) & 0x0fffffff, i = s & (BIND_RING - 1);
            if (__atomic_load_n(&bind_ring[i].seq, __ATOMIC_ACQUIRE) != (s | BIND_VALID)) continue;
            rt_log("vk: %s bind %u: pipeline %p, NVN 2 stages %016llx %016llx %016llx (vertex or compute, geometry, fragment)",
                   k ? (w ? "before finished" : "before reached") : (w ? "finished" : "reached"), s, (void *)bind_ring[i].pipe,
                   (unsigned long long)bind_ring[i].stage[0], (unsigned long long)bind_ring[i].stage[1],
                   (unsigned long long)bind_ring[i].stage[2]);
        }
    }
}
