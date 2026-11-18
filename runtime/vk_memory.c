// Vulkan memory of the graphics layer: NVN memory pools as Vulkan
// buffers, the virtual-pool page table, GPU address resolution, page write generations for re-uploads, and the
// upload batch (command buffer, staging buffers, deferred destruction) that vk_texture.c records into.
//
// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/totk_vpool.c (request checks, unmap
// semantics, 64 KiB pages) and nvn_dirty.c (generation stamps: a texture is stale when a covered page carries a
// generation newer than its upload), @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile); and from LAS
// Recomp runtime/nvn/vulkan/exec.c:2094-2131 (vk_pages_changed: every write-watch query resets the OS bits, so each
// query stamps the pages it saw with a new generation shared by all users) and textures.c:240-270 (deferred
// destruction) @ da35b83a (MIT, Copyright (c) 2026 Peiikai). MNG aliases host VM (mach_vm_remap) and protects pages
// (fault handler); here virtual pools are a GPU-side page table over device-local memory and dirty pages come from
// GetWriteWatch (mem_written_pages), so neither is needed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "mirror_sparse_spv.h"
#include "nvn_api.h"
#include "msort.h"
#include "nvn_pool_index.h"
#include "vk.h"
#include "vk_spec.h"
#include "platform/platform.h"

// Timestamps for the tracking cost, in ns; qpc_ms converts.
static uint64_t qpc(void) { return plat_now_ns(); }

static double qpc_ms(uint64_t ticks) { return (double)ticks / 1e6; }

// ---------------------------------------------------------------- pools
enum { POOL_IMPORT = 1, POOL_PHYS, POOL_VIRT };
enum { MS_NONE, MS_ON, MS_REFUSED };

typedef struct { struct PoolRec *phys; uint64_t off; } VPage;   // virtual page -> physical pool and offset

typedef struct PoolRec {
    const pool_t *key;
    uint64_t storage_serial;          // unique backing lifetime; virtual aliases use their physical record's serial
    int kind;
    uint32_t flags;
    uint64_t lo, size;                 // GPU address range of the pool
    uint64_t import_lo, import_hi;     // POOL_IMPORT: the imported host range (aligned outwards)
    VkBuffer buf;
    VkDeviceMemory mem;                // POOL_IMPORT
    VmaAllocation alloc;               // POOL_PHYS
    VPage *pt;                         // POOL_VIRT: size / 64 KiB entries
    uint64_t npages;
    volatile int32_t used;                // resolved at least once: the GPU may still read it at finalize
    volatile uint8_t lazy;                // POOL_IMPORT whose import is deferred to the first GPU use (TOTK_POOL_LAZY, lazy_import)
    uint64_t init_frame;               // the present count at nvnMemoryPoolInitialize (mem_frame), for the import log
    // TOTK_TRACE=perf dirty census (vk_mem_dirty_submit), replay thread under the submit lock only:
    uint32_t bound_sub;                // last submit that bound it as a buffer (dirty_sub), 0 never
    uint32_t synced_gen;               // page generation its lazy-mirror count covers; DIRTY_UNSYNCED until scanned
    uint8_t bufbound;                  // ever bound as a vertex, index, uniform, storage or indirect buffer
    uint8_t untracked;                 // the last dirty scan found no write watch over it
    // VRAM mirror (POOL_IMPORT, "VRAM mirror" below). mstate and mbuf change on the replay thread only, once each
    // (mirror_create); the helper threads read them meanwhile (vk_mem_use), so mstate goes through ms_get and ms_set.
    uint8_t mstate;                    // MS_NONE until the first buffer use, then MS_ON or MS_REFUSED
    VkBuffer mbuf;                     // MS_ON: device-local copy of [import_lo, import_hi), same offsets as buf
    VmaAllocation malloc_;
    uint32_t acc_sub;                  // replay: the last submission that synced the mirror (mirror_sync)
    uint32_t sub_hist;                 // mirror_lock: the submissions that synced it, bit i for acc_sub - i
    uint32_t snap_sub;                 // replay: the last submission vk_mem_mirror_begin pre-snapped it for
    uint32_t wb_sub;                   // replay: the last submission that wrote mirror bytes back to guest memory
    uint32_t mgen;                     // gen_lock: every page write stamped up to this generation is in the mirror
                                       // (or in a copy queued before its next use)
    struct KeptRange { uint64_t lo, hi; } *kept;   // mirror_lock: whole pages of GPU writes kept in the mirror (vk_mem_gpu_kept),
    int nkept, ckept;                  // as offsets from import_lo, sorted and disjoint; mirror_sync leaves them out of a flush
    uint64_t fl_lo, fl_hi;             // flush_lock: nvnMemoryPoolFlushMappedRange bytes not copied yet (fl_hi 0: none)
    PlatMutex sync_lock;               // mirror_sync of this pool, one thread at a time (zero is unlocked)
    // Shadow storage (POOL_IMPORT in shadow mode, "shadow storage" below): buf is a host-visible copy of the guest pages.
    VmaAllocation salloc;
    uint8_t *smap;                     // its mapping; offsets as in buf (from import_lo)
    uint32_t sgen;                     // gen_lock: every page write stamped up to this generation is in the shadow
    volatile uint8_t stouched;         // on the touched list: the GPU may use it in the submission being recorded
} PoolRec;

// A pool's mirror state. The replay thread stores it with release order after the mirror's buffer exists, and every
// reader loads it with acquire order: a helper thread that reads MS_ON in vk_mem_use reads the buffer that goes with it.
static inline int ms_get(const PoolRec *r) { return __atomic_load_n(&r->mstate, __ATOMIC_ACQUIRE); }
static inline void ms_set(PoolRec *r, int state) { __atomic_store_n(&r->mstate, (uint8_t)state, __ATOMIC_RELEASE); }

static void mirror_forget(PoolRec *r);   // pool_lock exclusive: off the mirrored list
static int shadow_on;                    // shadow mode: imported pools get shadow storage instead (vk_mem_class_log)
static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size);
static void shadow_touch(PoolRec *r);
static void shadow_forget(PoolRec *r);
static void mirror_retiring(PoolRec *r);  // vk_pool_fini: kept until mirror_retire (or vk_mem_shutdown)
static void mirror_retire(void *arg);    // completion thread: destroys a finalized mirrored pool's record
static void grow(void **arr, int *cap, size_t elem);

static PlatMutex pool_lock = PLAT_MUTEX_INIT;
// Shared acquires try first and time a contended wait (TOTK_TRACE=perf, perf_lock_wait): a pool imported or mapped on
// another thread holds the lock exclusively across driver calls, and every draw resolves its buffers under it.
static void pool_lock_shared(void) {
    if (plat_mutex_trylock_shared(&pool_lock)) return;
    uint64_t t0 = kern_now_ns();
    plat_mutex_lock_shared(&pool_lock);
    if (rt_trace & RT_TRACE_PERF) perf_lock_wait(PERF_LOCK_POOL, kern_now_ns() - t0);
}
// Bumped under pool_lock (exclusive) by every change to the index or a page table: vk_mem_resolve's per-thread hits
// stay valid while it is unchanged.
static uint64_t pool_gen = 1;
static uint64_t storage_serial;
// Bumped when a buffer use may get another buffer for the same span (a mirror created, refused or forgotten) and at
// every submission's start (mirror syncs are per submission). vk_mem_gen: both together.
static uint64_t mirror_gen;
static pidx pool_index;                // POOL_IMPORT, POOL_PHYS and POOL_VIRT records by GPU address
static PoolRec **phys_list;            // physical pools by key, for MapVirtual requests
static int nphys, cap_phys;
static int32_t n_imports, n_import_failures;
static volatile uint64_t import_alloc_ns;   // all vkAllocateMemory calls that imported guest memory (vk_mem_census logs it)
static volatile uint64_t import_touch_ns, import_total_ns;   // mem_touch and all of import_pool, same census line
static PlatMutex lazy_lock = PLAT_MUTEX_INIT;   // one deferred import at a time (the driver serializes them anyway)
static volatile int32_t n_lazy_imports;         // pools imported on their first GPU use instead of at nvnMemoryPoolInitialize
static PFN_vkGetMemoryHostPointerPropertiesEXT p_host_props;
static volatile uint64_t mem_frame;             // the frame count of the last present (vk_mem_census)
// Imported memory released (destroy_rec: vkDestroyBuffer and vkFreeMemory of a host-pointer import): count and time in
// all, and since the last frame line (lazy_frame_log).
static volatile int64_t rel_n, rel_ns, rel_frame_n, rel_frame_ns;

static VkBufferUsageFlags pool_usage(const VkCtx *c) {
    VkBufferUsageFlags u = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (c->has_transform_feedback)
        u |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT | VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    if (c->has_conditional_rendering) u |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    return u;
}

static int memory_type(const VkCtx *c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if ((bits >> i & 1) && (c->mem_props.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if (bits >> i & 1) return (int)i;
    return -1;
}

// CPU-visible storage: the guest pages themselves become the buffer's memory (VK_EXT_external_memory_host).
// The phases of one import, in ns (import_pool fills it for its log line): mem_thaw, mem_touch, the host pointer query,
// vkCreateBuffer, vkAllocateMemory (the driver pins the pages there) and vkBindBufferMemory; total is all of
// import_pool, wait the time lazy_import waited for lazy_lock before it (filled by lazy_import).
typedef struct { uint64_t thaw, touch, props, create, alloc, bind, total, wait; } ImportTimes;
static int import_pool_timed(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size, ImportTimes *t);
// where: when it runs, for the log line ("on first GPU use", "at initialize"). out, when not
// NULL, gets the times of the phases.
static int import_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size, const char *where, ImportTimes *out) {
    ImportTimes t = {0};
    uint64_t t0 = qpc();
    mem_thaw(addr, size);    // the import may need write access to the pages
    uint64_t t1 = qpc();
    mem_touch(addr, size);   // the driver pins the pages: reserved ones of a lazy heap (TOTK_HEAP_LAZY) would fail
    uint64_t t2 = qpc();
    int ok = import_pool_timed(c, r, addr, size, &t);
    uint64_t total = qpc() - t0;
    t.thaw = t1 - t0, t.touch = t2 - t1, t.total = total;
    if (out) *out = t;
    __atomic_add_fetch(&import_touch_ns, t2 - t1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&import_total_ns, total, __ATOMIC_RELAXED);
    // A slow vkAllocateMemory, or any import of 64 MiB or more (the terrain arena of a layer is 450 MiB, #397), gets a
    // line with every phase (the first 32): the stall of a first GPU use was more than the allocation alone.
    static int32_t n_slow_alloc;
    if ((t.alloc > 10000000ull || size >= (64ull << 20)) && plat_inc32(&n_slow_alloc) <= 32)
        rt_log("vk: importing %llu MiB of guest memory took %.1f ms in vkAllocateMemory (%.1f ms in all, %s: thaw %.1f, "
               "touch %.1f, host pointer query %.1f, vkCreateBuffer %.1f, vkBindBufferMemory %.1f)%s",
               (unsigned long long)(size >> 20), qpc_ms(t.alloc), qpc_ms(total), where, qpc_ms(t.thaw), qpc_ms(t.touch),
               qpc_ms(t.props), qpc_ms(t.create), qpc_ms(t.bind), ok ? "" : "; failed");
    return ok;
}

static int import_pool_timed(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size, ImportTimes *t) {
    if (shadow_on) return shadow_pool(c, r, addr, size);
    if (!p_host_props)
        p_host_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(c->device, "vkGetMemoryHostPointerPropertiesEXT");
    uint64_t al = c->min_imported_host_alignment ? c->min_imported_host_alignment : 0x1000;
    r->import_lo = addr & ~(al - 1);
    r->import_hi = (addr + size + al - 1) & ~(al - 1);
    uint64_t n = r->import_hi - r->import_lo;
    VkMemoryHostPointerPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    // The Vulkan spec ties the memory types to the pointer given, but every guest pool here is plain committed
    // read/write memory (platform/win32.c plat_mem_reserve/plat_mem_commit: no write-combined, uncached or file-view
    // pages), so the answer is assumed to be the same for all of them. NVIDIA's driver took long in this query for the big guest pools of a level change (five guest threads in turn, 1.8 s in a TOTK_PROFILE_SLOW dump:
    // the callq before dkb.exe+9331ce7 is this call, handle type 0x80). The first answer is kept for the other pools;
    // TOTK_HOST_PROPS_CACHE=0 asks every time. A type that would not do fails in vkAllocateMemory below, like before.
    static volatile uint32_t host_bits;
    static int cache_on = -1;
    static int32_t n_slow_props;
    if (cache_on < 0) {
        const char *e = cfg_get("TOTK_HOST_PROPS_CACHE");
        cache_on = !(e && !strcmp(e, "0"));
    }
    VkResult res;
    uint32_t known = cache_on ? __atomic_load_n(&host_bits, __ATOMIC_ACQUIRE) : 0;
    uint64_t tp = qpc();
    if (known) {
        hp.memoryTypeBits = known, res = VK_SUCCESS;
    } else {
        uint64_t t0 = qpc();
        res = p_host_props ? p_host_props(c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                          (void *)(uintptr_t)r->import_lo, &hp)
                           : VK_ERROR_EXTENSION_NOT_PRESENT;
        double ms = (double)(qpc() - t0) / 1e6;
        if (ms > 10.0 && plat_inc32(&n_slow_props) <= 16)
            rt_log("vk: vkGetMemoryHostPointerPropertiesEXT of %llu MiB took %.1f ms", (unsigned long long)(n >> 20), ms);
        if (res == VK_SUCCESS && hp.memoryTypeBits && cache_on) __atomic_store_n(&host_bits, hp.memoryTypeBits, __ATOMIC_RELEASE);
    }
    t->props = qpc() - tp;
    if (res != VK_SUCCESS || !hp.memoryTypeBits) {
        rt_log("vk: pool %#llx+%#llx: host pointer not importable (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        return 0;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = n,
                             .usage = pool_usage(c), .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    uint64_t tc = qpc();
    res = vkCreateBuffer(c->device, &bi, NULL, &r->buf);
    t->create = qpc() - tc;
    if (res != VK_SUCCESS) {
        rt_log("vk: pool %#llx: vkCreateBuffer for the import failed (%d)", (unsigned long long)addr, (int)res);
        return 0;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, r->buf, &req);
    int type = memory_type(c, req.memoryTypeBits & hp.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type < 0 || req.size > n) {
        rt_log("vk: pool %#llx: no memory type for the import (buffer %#x, host %#x, size %#llx of %#llx)",
               (unsigned long long)addr, req.memoryTypeBits, hp.memoryTypeBits, (unsigned long long)req.size,
               (unsigned long long)n);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        return 0;
    }
    VkImportMemoryHostPointerInfoEXT im = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                           .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                           .pHostPointer = (void *)(uintptr_t)r->import_lo};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &im, .allocationSize = n,
                               .memoryTypeIndex = (uint32_t)type};
    uint64_t ta = qpc();
    int op = plat_op_begin(PLAT_OP_IMPORT, r->import_lo, n);
    res = vkAllocateMemory(c->device, &ai, NULL, &r->mem);
    if (res != VK_SUCCESS && known && p_host_props) {
        r->mem = VK_NULL_HANDLE;
        VkMemoryHostPointerPropertiesEXT hp2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        int t2 = -1;
        if (p_host_props(c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)(uintptr_t)r->import_lo, &hp2) == VK_SUCCESS)
            t2 = memory_type(c, req.memoryTypeBits & hp2.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t2 >= 0) {
            ai.memoryTypeIndex = (uint32_t)t2;
            res = vkAllocateMemory(c->device, &ai, NULL, &r->mem);
        }
    }
    if (res != VK_SUCCESS) r->mem = VK_NULL_HANDLE;
    plat_op_end(op, 0);
    t->alloc = qpc() - ta;
    __atomic_add_fetch(&import_alloc_ns, t->alloc, __ATOMIC_RELAXED);
    uint64_t tb = qpc();
    if (res == VK_SUCCESS) res = vkBindBufferMemory(c->device, r->buf, r->mem, 0);
    t->bind = qpc() - tb;
    if (res != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: host memory import failed (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        if (r->mem) vkFreeMemory(c->device, r->mem, NULL);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->mem = VK_NULL_HANDLE;
        return 0;
    }
    plat_inc32(&n_imports);
    return 1;
}

// nvnMemoryPoolInitialize only records an imported pool, and the import waits for the first time the GPU resolves an
// address in it (TOTK_POOL_LAZY=0 imports at once, as before). A profile of a level change showed hundreds of guest-memory
// imports, each about 20 ms of NVIDIA driver time whatever its size, one at a time, with the guest threads blocked in
// them; the memory census of the same runs had the GPU resolve only 6% of the imported pools at frame 600 and 27% at
// frame 5400. On a scripted 72 s route (RTX 5060 Ti) it cut the imports from 3219 to 901 and their total from 27 s to
// 1.2 s, and the stalls from four of 360 to 950 ms to one of 993 ms.
// Shadow storage copies the guest pages when it is made, so it always imports at once.
static int lazy_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("TOTK_POOL_LAZY");
        on = !(e && !strcmp(e, "0")) && !shadow_on;
        if (on) rt_log("vk: imported pools are imported on their first GPU use (TOTK_POOL_LAZY)");
    }
    return on;
}

// The deferred imports since the last present (lazy_frame_log): count, bytes, time with the wait for lazy_lock, the
// largest one, and the time by phase in ImportTimes order (thaw, touch, host pointer query, vkCreateBuffer,
// vkAllocateMemory, vkBindBufferMemory, the wait for lazy_lock). The first frame of a layer imports hundreds of pools on
// the replay thread. The imports of pools under 1 MiB are counted apart (count, bytes, vkAllocateMemory time), so the
// line tells a fixed cost per import from one per byte: the large imports pinned about 0.16 ms per MiB on NVIDIA
// (72 to 693 MiB, 11 to 122 ms, the TOTK_POOL_EAGER runs of build/bench/pe4-on-*.log).
static volatile int64_t lz_n, lz_bytes, lz_ns, lz_max_bytes, lz_max_ns, lz_phase[7], lz_small[3];

static void lz_note(const ImportTimes *t, uint64_t size, uint64_t ns) {
    plat_inc64(&lz_n);
    plat_add64(&lz_bytes, (int64_t)size);
    plat_add64(&lz_ns, (int64_t)ns);
    const uint64_t ph[7] = {t->thaw, t->touch, t->props, t->create, t->alloc, t->bind, t->wait};
    for (int i = 0; i < 7; i++) plat_add64(&lz_phase[i], (int64_t)ph[i]);
    if (size < (1u << 20)) plat_inc64(&lz_small[0]), plat_add64(&lz_small[1], (int64_t)size), plat_add64(&lz_small[2], (int64_t)t->alloc);
    if ((int64_t)size > lz_max_bytes) lz_max_bytes = (int64_t)size, lz_max_ns = (int64_t)ns;   // lazy_lock
}

// TOTK_POOL_IMPORT_LOG=N: the first N imports made on a pool's first GPU use get a line each
// with the pool's address, size and flags, the frame of its initialize, the caller of vk_mem_resolve that needed it (an
// exe offset for the link map), the time of each phase, and its neighbours in the pool index (the gap to each, and
// whether that pool is imported, waiting for its first use or used). For deciding what an import ahead of the first use
// could predict, and whether neighbouring pools could share one import. 0 (default) logs none.
static int import_log_on(void) {
    static int max = -1;
    static volatile int32_t n;
    if (max < 0) {
        const char *e = cfg_get("TOTK_POOL_IMPORT_LOG");
        max = e && *e ? atoi(e) : 0;
        if (max < 0) max = 0;
    }
    return max > 0 && plat_inc32(&n) <= max;
}

// vk_mem_resolve's caller on this thread, for the import log (set on its locked path only).
static _Thread_local uintptr_t resolve_ra;

// Index entry at's neighbour on side dir (-1 below, +1 above) for the import log. pool_lock held.
static void neighbour(int at, int dir, const PoolRec *r, char *buf, size_t n) {
    int j = at + dir;
    if (at < 0 || j < 0 || j >= pool_index.n) {
        snprintf(buf, n, "none");
        return;
    }
    const PoolRec *q = pool_index.e[j].buf;
    long long gap = dir < 0 ? (long long)r->lo - (long long)(q->lo + q->size) : (long long)q->lo - (long long)(r->lo + r->size);
    snprintf(buf, n, "%lld bytes away, %#x %#llx bytes, %s%s", gap, q->flags, (unsigned long long)q->size,
             q->kind != POOL_IMPORT ? "not imported" : q->lazy ? "waiting" : q->buf ? "imported" : "without a buffer",
             q->used ? ", used" : "");
}

static void import_log(const char *how, const PoolRec *r, int at, const ImportTimes *t, int ok) {
    char below[96], above[96], via[48];
    neighbour(at, -1, r, below, sizeof below);
    neighbour(at, +1, r, above, sizeof above);
    snprintf(via, sizeof via, "exe+%#llx", (unsigned long long)(resolve_ra ? resolve_ra - plat_exe_base() : 0));
    rt_log("vk: import %s at frame %llu: pool %#llx+%#llx flags %#x, initialized at frame %llu, for %s: %.3f ms (lazy_lock "
           "wait %.3f, thaw %.3f, touch %.3f, host pointer query %.3f, vkCreateBuffer %.3f, vkAllocateMemory %.3f, "
           "vkBindBufferMemory %.3f)%s; below: %s; above: %s", how, (unsigned long long)mem_frame, (unsigned long long)r->lo,
           (unsigned long long)r->size, r->flags, (unsigned long long)r->init_frame, via, (t->total + t->wait) / 1e6,
           t->wait / 1e6, t->thaw / 1e6, t->touch / 1e6, t->props / 1e6, t->create / 1e6, t->alloc / 1e6, t->bind / 1e6,
           ok ? "" : ", failed", below, above);
}

// The deferred import. The caller holds pool_lock, shared or exclusive: a finalize needs it exclusively, so the record
// stays alive here. at: r's entry in pool_index, for the import log. Returns whether r has a GPU buffer afterwards.
static int lazy_import(PoolRec *r, int at) {
    if (!__atomic_load_n(&r->lazy, __ATOMIC_ACQUIRE)) return r->buf != VK_NULL_HANDLE;
    uint64_t ph = perf_ph_begin_cpu();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
    uint64_t t0 = qpc();
    plat_mutex_lock(&lazy_lock);
    if (r->lazy) {
        ImportTimes t = {0};
        uint64_t t1 = qpc();
        int ok = import_pool(vk_ctx(), r, r->lo, r->size, "on first GPU use", &t);
        t.wait = t1 - t0;
        if (!ok) {
            long n = plat_inc32(&n_import_failures);
            if (n <= 8) rt_log("vk: pool %#llx (flags %#x, %#llx bytes) has no GPU buffer (%ld so far)", (unsigned long long)r->lo,
                               r->flags, (unsigned long long)r->size, n);
            r->buf = VK_NULL_HANDLE;
        } else {
            plat_inc32(&n_lazy_imports);
        }
        __atomic_store_n(&r->lazy, 0, __ATOMIC_RELEASE);
        lz_note(&t, r->size, qpc() - t0);
        if (import_log_on()) import_log("on first GPU use", r, at, &t, ok);
    }
    plat_mutex_unlock(&lazy_lock);
    perf_ph_end(PERF_PH_IMPORT, ph);
    return r->buf != VK_NULL_HANDLE;
}

// Every present (vk_mem_census): a frame whose first-use imports or releases of imported memory took 10 ms or more gets
// a line (the first 128), with the first-use imports' time by phase, the mem_touch chunks the commit map answered and
// those it queried (all callers of mem_touch, file reads included), and the releases.
static void lazy_frame_log(uint64_t frame) {
    static uint64_t touch_last[4];
    uint64_t touch[4];
    mem_touch_stats(touch);
    int64_t known = (int64_t)(touch[1] - touch_last[1]), queried = (int64_t)(touch[2] - touch_last[2]);
    memcpy(touch_last, touch, sizeof touch);
    int64_t rn = plat_xchg64(&rel_frame_n, 0), rns = plat_xchg64(&rel_frame_ns, 0);
    // Not under lazy_lock, which an import holds for up to 100 ms: an import that ends meanwhile may land in either
    // frame's line, and its largest-pool figures in neither.
    int64_t n = plat_xchg64(&lz_n, 0), bytes = plat_xchg64(&lz_bytes, 0), ns = plat_xchg64(&lz_ns, 0);
    int64_t mb = plat_xchg64(&lz_max_bytes, 0), mn = plat_xchg64(&lz_max_ns, 0);
    int64_t ph[7], sm[3];
    for (int i = 0; i < 7; i++) ph[i] = plat_xchg64(&lz_phase[i], 0);
    for (int i = 0; i < 3; i++) sm[i] = plat_xchg64(&lz_small[i], 0);
    static int lines;
    if ((ns < 10000000 && rns < 10000000) || lines >= 128) return;
    lines++;
    rt_log("vk: frame %llu: %lld pools (%.1f MiB) imported on their first GPU use took %.1f ms; the largest, %.1f MiB, "
           "%.1f ms; by phase: lazy_lock wait %.1f, thaw %.1f, touch %.1f (%lld chunks known committed, %lld queried), "
           "host pointer query %.1f, vkCreateBuffer %.1f, vkAllocateMemory %.1f (%lld pools under 1 MiB, %.1f MiB, %.1f "
           "ms of it), vkBindBufferMemory %.1f; imported memory released %lld times, %.1f ms", (unsigned long long)frame, (long long)n, bytes / 1048576.0, ns / 1e6,
           mb / 1048576.0, mn / 1e6, ph[6] / 1e6, ph[0] / 1e6, ph[1] / 1e6, (long long)known, (long long)queried,
           ph[2] / 1e6, ph[3] / 1e6, ph[4] / 1e6, (long long)sm[0], sm[1] / 1048576.0, sm[2] / 1e6, ph[5] / 1e6,
           (long long)rn, rns / 1e6);
}

// Physical pool: device-local memory of the pool size that both buffers and optimal-tiling images can be bound to.
static int alloc_physical(VkCtx *c, PoolRec *r) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = r->size, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult res = vkCreateBuffer(c->device, &bi, NULL, &r->buf);
    if (res != VK_SUCCESS) { rt_log("vk: physical pool: vkCreateBuffer failed (%d)", (int)res); return 0; }
    VkMemoryRequirements rb;
    vkGetBufferMemoryRequirements(c->device, r->buf, &rb);
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {256, 256, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    VkDeviceImageMemoryRequirements di = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &ii};
    VkMemoryRequirements2 ri = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(c->device, &di, &ri);
    uint32_t bits = rb.memoryTypeBits & ri.memoryRequirements.memoryTypeBits;
    if (!bits) bits = rb.memoryTypeBits;
    VkDeviceSize align = NVN_VPOOL_PAGE;
    if (rb.alignment > align) align = rb.alignment;
    if (ri.memoryRequirements.alignment > align) align = ri.memoryRequirements.alignment;
    VkMemoryRequirements req = {.size = (rb.size + align - 1) & ~(align - 1), .alignment = align, .memoryTypeBits = bits};
    VmaAllocationCreateInfo aci = {.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                   .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    if ((res = vmaAllocateMemory(c->vma, &req, &aci, &r->alloc, NULL)) != VK_SUCCESS ||
        (res = vmaBindBufferMemory(c->vma, r->alloc, r->buf)) != VK_SUCCESS) {
        rt_log("vk: physical pool of %#llx bytes: device-local allocation failed (%d)", (unsigned long long)r->size, (int)res);
        if (r->alloc) vmaFreeMemory(c->vma, r->alloc);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->alloc = NULL;
        return 0;
    }
    if (vk_fix_zero()) {   // Switch pool memory starts zeroed; once per pool, not per image placed in it
        VkCommandBuffer cb = vk_upload_begin();
        vkCmdFillBuffer(cb, r->buf, 0, VK_WHOLE_SIZE, 0);
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(cb, &di);
        vk_upload_end();
    }
    return 1;
}

int vk_fix_zero(void) {
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("TOTK_FIX_ZERO");
    }
    return on;
}

static void index_put(PoolRec *r) {
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    pidx_ent e = {.lo = r->lo, .hi = r->lo + r->size, .base = r->lo, .buf = r, .gpu = r->lo, .key = r->key};
    pidx_remove(&pool_index, r->key, NULL);   // an object initialized again without Finalize
    if (!pidx_insert(&pool_index, &e)) rt_fatal("vk: out of memory for the pool index");
}

static PoolRec *phys_find(const pool_t *key) {
    for (int i = 0; i < nphys; i++) if (phys_list[i]->key == key) return phys_list[i];
    return NULL;
}

static void destroy_rec(VkCtx *c, PoolRec *r) {
    if (r->mbuf) vmaDestroyBuffer(c->vma, r->mbuf, r->malloc_);
    if (r->salloc) vmaDestroyBuffer(c->vma, r->buf, r->salloc), r->buf = VK_NULL_HANDLE;
    uint64_t t0 = r->mem ? qpc() : 0;
    if (r->buf) vkDestroyBuffer(c->device, r->buf, NULL);
    if (r->mem) {   // a host-pointer import: the driver unpins the pages (rel_n)
        int op = plat_op_begin(PLAT_OP_IMPORT_FREE, r->import_lo, r->import_hi - r->import_lo);
        vkFreeMemory(c->device, r->mem, NULL);
        plat_op_end(op, 0);
        int64_t ns = (int64_t)(qpc() - t0);
        plat_inc64(&rel_n), plat_add64(&rel_ns, ns), plat_inc64(&rel_frame_n), plat_add64(&rel_frame_ns, ns);
    }
    if (r->alloc) vmaFreeMemory(c->vma, r->alloc);
    free(r->pt);
    free(r->kept);
    free(r);
}

static int pool_init(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) { rt_log("vk: MemoryPoolInitialize before vk_init"); return 0; }
    // Reinitializing the same object retires its old backing before installing the new mapping. In particular,
    // descriptors at a reused address must not inherit registrations from a previous storage lifetime.
    int reinit = 0;
    pool_lock_shared();
    for (int i = 0; i < pool_index.n; i++) if (pool_index.e[i].key == p) { reinit = 1; break; }
    plat_mutex_unlock_shared(&pool_lock);
    if (reinit) vk_pool_fini(p);
    // The GPU never sees a GPU_NO_ACCESS pool, except a physical one (0x189; 1.4.3 also 0x18c, CPU-visible) through the
    // virtual pools that map it. Either gets device-local memory: 1.4.3's CPU never writes its 0x18c pool's storage
    // (unchanged over 3600 frames of the intro), so the guest storage and the GPU copy need no sync.
    if ((p->flags & NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS) && !(p->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL)) return 1;
    PoolRec *r = calloc(1, sizeof *r);
    if (!r) rt_fatal("vk: out of memory for a pool record");
    r->key = p;
    r->flags = p->flags;
    r->lo = (uintptr_t)p->mem;
    r->size = p->size;
    r->init_frame = mem_frame;
    int ok;
    if (p->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) {
        r->kind = POOL_VIRT;
        r->npages = p->size / NVN_VPOOL_PAGE;
        r->pt = calloc(r->npages ? r->npages : 1, sizeof *r->pt);
        ok = r->pt != NULL;
    } else if (p->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) {
        r->kind = POOL_PHYS;
        ok = alloc_physical(c, r);
    } else {
        r->kind = POOL_IMPORT;
        if (lazy_enabled()) {
            // Deferred (lazy_import): the host range is the one import_pool would compute, for code that reads it early.
            uint64_t al = c->min_imported_host_alignment ? c->min_imported_host_alignment : 0x1000;
            r->import_lo = r->lo & ~(al - 1);
            r->import_hi = (r->lo + r->size + al - 1) & ~(al - 1);
            r->lazy = 1;
            ok = 1;
        } else {
            ok = import_pool(c, r, (uintptr_t)p->mem, p->size, "at initialize", NULL);
        }
        if (!ok) {
            // Not fatal for the game: the pool works for the CPU, and vk_mem_resolve reports its addresses as unbacked.
            long n = plat_inc32(&n_import_failures);
            if (n <= 8) rt_log("vk: pool %p (flags %#x, %#llx bytes) has no GPU buffer (%ld so far)", (void *)p->mem,
                               p->flags, (unsigned long long)p->size, n);
            r->buf = VK_NULL_HANDLE;
            ok = 1;
        }
    }
    if (!ok) { free(r->pt); free(r); return 0; }
    plat_mutex_lock(&pool_lock);
    if (!++storage_serial) ++storage_serial;
    r->storage_serial = storage_serial;
    index_put(r);
    if (r->kind == POOL_PHYS) {
        if (nphys == cap_phys) {
            cap_phys = cap_phys ? cap_phys * 2 : 16;
            phys_list = realloc(phys_list, (size_t)cap_phys * sizeof *phys_list);
            if (!phys_list) rt_fatal("vk: out of memory for the physical pool list");
        }
        phys_list[nphys++] = r;
    }
    plat_mutex_unlock(&pool_lock);
    return 1;
}

int vk_pool_init(pool_t *p) {
    int op = plat_op_begin(PLAT_OP_POOL_INIT, (uintptr_t)p->mem, p->size);
    int ok = pool_init(p);
    plat_op_end(op, 0);
    return ok;
}

// What the release of a used pool waits for before the guest gets its storage back. TOTK_POOL_FINI_WAIT=idle (the
// default) is the whole graphics queue idle, with the queue lock held, which costs a driver round trip per pool even when
// the GPU has nothing left: a level change frees many pools one after another (a 1.6 s frame in a TOTK_PROFILE_SLOW dump,
// five guest threads in turn inside this wait). fence waits for the timeline value of the newest finished submission
// instead and does not enter the driver when that value is already done. It covers the work the game has synchronized
// on, which is what a release is allowed to follow; it does not cover the parts that vk_sync_split already submitted
// for the submission still being recorded (they signal only at its end), which the idle wait did.
static void pool_fini_wait_(VkCtx *c) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_get("TOTK_POOL_FINI_WAIT");
        mode = e && !strcmp(e, "fence");
        if (mode) rt_log("vk: a pool release waits for the newest submission on the timeline (TOTK_POOL_FINI_WAIT=fence)");
        else if (e && *e && strcmp(e, "idle")) rt_log("vk: TOTK_POOL_FINI_WAIT=%s is not idle or fence: waiting for the queue to idle", e);
    }
    if (mode) {
        uint64_t v = vk_sync_issued();
        if (v && vk_sync_fence_done() < v) vk_sync_fence_wait(v, UINT64_MAX);
        return;
    }
    vk_queue_lock();
    vkQueueWaitIdle(c->gfx_queue);
    vk_queue_unlock();
}

static void pool_fini_wait(VkCtx *c, const PoolRec *r) {
    int op = plat_op_begin(PLAT_OP_POOL_WAIT, r->lo, r->size);
    pool_fini_wait_(c);
    plat_op_end(op, 0);
}

// Releases of used pools that waited over 10 ms are logged (the first 64 of them), with the running total of all waits.
static volatile int32_t pf_slow;
static volatile uint64_t pf_flush_ns, pf_wait_ns;
static void pool_fini_note(uint64_t size, uint64_t flush_ns, uint64_t wait_ns) {
    __atomic_add_fetch(&pf_flush_ns, flush_ns, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pf_wait_ns, wait_ns, __ATOMIC_RELAXED);
    if (flush_ns + wait_ns <= 10000000ull) return;
    long n = plat_inc32(&pf_slow);
    if (n <= 64)
        rt_log("vk: used pool of %llu MiB released: upload flush %.1f ms, wait for the GPU %.1f ms (slow release %ld; %.0f ms in all releases so far)",
               (unsigned long long)(size >> 20), flush_ns / 1e6, wait_ns / 1e6, n, (pf_flush_ns + pf_wait_ns) / 1e6);
}

static void pool_fini(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    pidx_ent e;
    plat_mutex_lock(&pool_lock);
    int had = pidx_remove(&pool_index, p, &e);
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    PoolRec *r = had ? e.buf : NULL;
    if (r && ms_get(r) == MS_ON) mirror_forget(r);
    if (r && r->kind == POOL_PHYS) {
        for (int i = 0; i < nphys; i++)
            if (phys_list[i] == r) { phys_list[i] = phys_list[--nphys]; break; }
        for (int i = 0; i < pool_index.n; i++) {   // virtual pages that still map it become unmapped
            PoolRec *v = pool_index.e[i].buf;
            if (v->kind != POOL_VIRT) continue;
            for (uint64_t k = 0; k < v->npages; k++)
                if (v->pt[k].phys == r) v->pt[k].phys = NULL;
        }
    }
    plat_mutex_unlock(&pool_lock);
    if (!r) return;
    // Texture lock is acquired only after releasing pool_lock. Descriptor lookup takes them in the other direction
    // (texture, then pool), just as normal image creation does. Imported aliases still alive preserve shared bytes.
    if (r->kind == POOL_IMPORT) vk_texture_descriptors_retire(r->lo, r->size, 0);
    else if (r->kind == POOL_PHYS) vk_texture_descriptors_retire(0, 0, r->storage_serial);
    if (r->smap) shadow_forget(r);
    if (r->kind == POOL_IMPORT) mem_thaw(r->lo, r->size);   // the storage goes back to the guest writable
    // Images cut from this range go first; a used buffer or memory waits until the GPU is done with it, since the
    // guest gets the storage back when this returns.
    vk_texture_range_gone(r->lo, r->size, r->alloc);
    if (r->used && r->kind != POOL_VIRT) {
        uint64_t t0 = qpc();
        vk_upload_flush(1);
        uint64_t t1 = qpc();
        pool_fini_wait(c, r);
        pool_fini_note(r->size, t1 - t0, qpc() - t1);
    }
    vk_mem_collect();
    // A mirrored pool's record and buffers outlive the submission being recorded, whose sync copies and write-backs may
    // name them (the game frees a pool only once the GPU is done with it, but its CPU writes may still be queued).
    if (r->mbuf) mirror_retiring(r), vk_completion_add(vk_sync_issued() + 1, mirror_retire, r);
    else destroy_rec(c, r);
}

void vk_pool_fini(pool_t *p) {
    int op = plat_op_begin(PLAT_OP_POOL_FINI, (uintptr_t)p->mem, p->size);
    pool_fini(p);
    plat_op_end(op, 0);
}

// Requests were checked by the front end (nvn_api_core.c, MNG totk_vpool.c); apply them in order.
int vk_pool_map_virtual(pool_t *v, int n, const void *requests) {
    const nvn_mapping_t *rq = requests;
    plat_mutex_lock(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)v->mem, 0);
    PoolRec *r = e && e->key == v ? e->buf : NULL;
    int ok = r && r->kind == POOL_VIRT;
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; ok && i < n; i++) {
        PoolRec *ph = rq[i].physical_pool ? phys_find((const pool_t *)(const void *)rq[i].physical_pool) : NULL;
        if (rq[i].physical_pool && !ph) {
            rt_log("vk: MapVirtual request %d names physical pool %p that has no Vulkan memory", i,
                   (const void *)rq[i].physical_pool);
            ok = 0;
            break;
        }
        uint64_t first = (uint64_t)rq[i].virtual_offset / NVN_VPOOL_PAGE, count = rq[i].size / NVN_VPOOL_PAGE;
        if (first + count > r->npages) { ok = 0; break; }
        for (uint64_t k = 0; k < count; k++)
            r->pt[first + k] = (VPage){ph, ph ? (uint64_t)rq[i].physical_offset + k * NVN_VPOOL_PAGE : 0};
    }
    plat_mutex_unlock(&pool_lock);
    if (!ok) rt_log("vk: MapVirtual on pool %p failed", (void *)v->mem);
    return ok;
}

// Per-thread copies of resolved import and physical pools that overlap no other index entry (so pidx_find picks them for
// any range inside), valid while pool_gen is unchanged: the replay resolves a few dozen pools thousands of times per
// frame (16 vertex streams and every buffer binding of each draw). A thread keeps up to RPOOL_N of them sorted by
// address (rpool, all of one pool_gen, emptied when it moves), and a byte map from the 64 KiB block of an address to
// the copy that held it last (rmap): a hit is one compare, and a block the map does not answer is found by a binary
// search of the copies before pool_lock is taken. With a table of 512 copies indexed by the 4 KiB page, as before, 40%
// of the resolves in Bananza's Lake Level (3.2.0) missed: a pool filled one slot per page it was read at, the vertex
// data of a frame is spread over more pages than the table held, and each miss took pool_lock shared, an atomic write
// to one cache line for every thread of a sliced pass. Now 2.4% miss, from 20 KiB per thread (the old table took 36).
// TOTK_RESOLVE_CHECK=1 compares every hit with the locked lookup and stops on a difference.
static int rec_class(const PoolRec *r) {
    if (r->kind != POOL_IMPORT) return VK_MCLS_DEVICE;
    return r->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? VK_MCLS_HOST_NOACC
         : r->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED ? VK_MCLS_HOST_CACHED : VK_MCLS_HOST_UNC;
}

typedef struct { uint64_t lo, hi, base, end; VkBuffer buf; VmaAllocation alloc; int cls; void *pool; } ResolveHit;
#define RPOOL_N 255   // rmap holds an index
#define RMAP_BLOCKS 4096
static _Thread_local ResolveHit rpool[RPOOL_N];
static _Thread_local int rpool_n;
static _Thread_local uint64_t rpool_gen;
static _Thread_local uint8_t rmap[RMAP_BLOCKS];   // the rpool index that held an address of the block last

static uint8_t *rmap_slot(uint64_t addr) { return &rmap[(addr >> 16 ^ addr >> 28) & (RMAP_BLOCKS - 1)]; }

// The copy holding [addr, end) at pool_gen g, NULL for none.
static const ResolveHit *rhit_find(uint64_t g, uint64_t addr, uint64_t end) {
    if (rpool_gen != g) return NULL;
    uint8_t *m = rmap_slot(addr);
    const ResolveHit *h = &rpool[*m];
    if (*m < rpool_n && addr >= h->lo && end <= h->hi) return h;
    int lo = 0, hi = rpool_n;   // the last copy with lo <= addr
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (rpool[mid].lo <= addr) lo = mid + 1;
        else hi = mid;
    }
    if (!lo || !(addr >= rpool[lo - 1].lo && end <= rpool[lo - 1].hi)) return NULL;
    *m = (uint8_t)(lo - 1);
    return &rpool[lo - 1];
}

// A copy for a pool's index entry [lo, hi) at pool_gen g, kept in order: the entry overlaps no other one.
static ResolveHit *rhit_add(uint64_t g, uint64_t lo, uint64_t hi) {
    if (rpool_gen != g || rpool_n == RPOOL_N) rpool_gen = g, rpool_n = 0;
    int k = 0;
    while (k < rpool_n && rpool[k].lo < lo) k++;
    if (k == rpool_n || rpool[k].lo != lo) {
        memmove(&rpool[k + 1], &rpool[k], sizeof *rpool * (size_t)(rpool_n - k));
        rpool_n++;
    }
    rpool[k].lo = lo, rpool[k].hi = hi;
    return &rpool[k];
}

static int resolve_check_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("TOTK_RESOLVE_CHECK");
        on = e && *e && strcmp(e, "0");
    }
    return on;
}

static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out);

int vk_mem_inspect(uint64_t addr, uint64_t size, VkMemInspect *out) {
    if (!out) return 0;
    memset(out, 0, sizeof *out);
    if (!addr || !size || addr > UINT64_MAX - size) return 0;
    int ok = 0;
    VkMemInspect result = {0};
    pool_lock_shared();
    // Match bind_range's one-byte resolver candidate, then check the whole request against that same backing.
    const pidx_ent *e = pidx_find(&pool_index, addr, 1);
    const PoolRec *r = e ? e->buf : NULL;
    if (!r || !r->size || r->lo > UINT64_MAX - r->size || addr < r->lo) goto done;
    uint64_t rel = addr - r->lo;
    if (rel >= r->size || size > r->size - rel) goto done;
    uint64_t remaining = r->size - rel;
    result.pool_flags = r->flags;
    if (r->kind == POOL_IMPORT) {
        // A pool still waiting for its first GPU use has no buffer yet: the callers are passive diagnostics, so they
        // report it as not backed instead of paying a ~20 ms import for a log line.
        if (!r->buf || addr < r->import_lo || addr + size > r->import_hi) goto done;
        result.pool_kind = VK_MEM_INSPECT_POOL_IMPORT;
        result.offset = addr - r->import_lo;
        result.avail = r->import_hi - addr;
        if (result.avail > remaining) result.avail = remaining;
        result.cls = rec_class(r);
        _Static_assert(MS_NONE == VK_MEM_INSPECT_MIRROR_NONE && MS_ON == VK_MEM_INSPECT_MIRROR_ON &&
                       MS_REFUSED == VK_MEM_INSPECT_MIRROR_REFUSED, "inspection mirror-state schema");
        result.mirror_state = ms_get(r);
        result.mirror_present = r->mbuf != VK_NULL_HANDLE;
        result.shadow_storage = r->smap != NULL;
    } else if (r->kind == POOL_PHYS) {
        if (!r->buf) goto done;
        result.pool_kind = VK_MEM_INSPECT_POOL_PHYSICAL;
        result.offset = rel;
        result.avail = remaining;
        result.cls = VK_MCLS_DEVICE;
    } else if (r->kind == POOL_VIRT) {
        uint64_t k = rel / NVN_VPOOL_PAGE, in_page = rel % NVN_VPOOL_PAGE;
        uint64_t pages = r->size / NVN_VPOOL_PAGE;
        if (pages > r->npages) pages = r->npages;
        if (!r->pt || k >= pages) goto done;
        const PoolRec *ph = r->pt[k].phys;
        uint64_t start = r->pt[k].off;
        if (!ph || ph->kind != POOL_PHYS || !ph->buf || start % NVN_VPOOL_PAGE ||
            start > ph->size || NVN_VPOOL_PAGE > ph->size - start) goto done;
        uint64_t end = start + NVN_VPOOL_PAGE;
        for (uint64_t j = k + 1; j < pages && r->pt[j].phys == ph && r->pt[j].off == end; j++) {
            if (NVN_VPOOL_PAGE > ph->size - end) break;
            end += NVN_VPOOL_PAGE;
        }
        result.pool_kind = VK_MEM_INSPECT_POOL_VIRTUAL;
        result.offset = start + in_page;
        result.avail = end - result.offset;
        if (result.avail > remaining) result.avail = remaining;
        result.cls = VK_MCLS_DEVICE;
    } else {
        goto done;
    }
    if (result.avail < size) goto done;
    result.buffer_present = 1;
    *out = result;
    ok = 1;
done:
    plat_mutex_unlock_shared(&pool_lock);
    return ok;
}

int vk_mem_descriptor_key(uint64_t addr, uint64_t size, VkDescKey *out) {
    memset(out, 0, sizeof *out);
    if (!addr || !size || addr > UINT64_MAX - size) return 0;
    int ok = 0;
    pool_lock_shared();
    const pidx_ent *e = pidx_find(&pool_index, addr, size);
    const PoolRec *r = e ? e->buf : NULL;
    if (r && r->kind == POOL_IMPORT) {
        *out = (VkDescKey){0, addr};
        ok = 1;
    } else if (r && r->kind == POOL_PHYS) {
        *out = (VkDescKey){r->storage_serial, addr - r->lo};
        ok = 1;
    } else if (r && r->kind == POOL_VIRT) {
        uint64_t rel = addr - r->lo, k = rel / NVN_VPOOL_PAGE;
        const PoolRec *ph = k < r->npages ? r->pt[k].phys : NULL;
        if (ph) {
            uint64_t off = r->pt[k].off + rel % NVN_VPOOL_PAGE;
            uint64_t end = r->pt[k].off + NVN_VPOOL_PAGE;
            for (uint64_t j = k + 1; end - off < size && j < r->npages && r->pt[j].phys == ph &&
                 r->pt[j].off == end; j++) end += NVN_VPOOL_PAGE;
            if (end - off >= size && off <= ph->size && size <= ph->size - off) {
                *out = (VkDescKey){ph->storage_serial, off};
                ok = 1;
            }
        }
    }
    plat_mutex_unlock_shared(&pool_lock);
    return ok;
}

int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out) {
    memset(out, 0, sizeof *out);
    uint64_t g = __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE), end = addr + size;
    const ResolveHit *h = end >= addr ? rhit_find(g, addr, end) : NULL;
    if (h) {
        out->buffer = h->buf;
        out->offset = addr - h->base;
        out->avail = h->end - addr;
        out->alloc = h->alloc;
        out->cls = h->cls;
        out->pool = h->pool;
        if (h->alloc) out->alloc_offset = out->offset;
        if (shadow_on && h->pool && !((PoolRec *)h->pool)->stouched) shadow_touch(h->pool);
        if (resolve_check_on()) {
            VkMemSpan s;
            if (!resolve_locked(addr, size, &s) || memcmp(&s, out, sizeof s))
                rt_fatal("vk: resolve cache of 0x%llx differs from the pool index", (unsigned long long)addr);
        }
        return 1;
    }
    resolve_ra = (uintptr_t)__builtin_return_address(0);   // who needed a first-use import, for the import log
    return resolve_locked(addr, size, out);
}

// The pool index lookup under the lock; a non-overlapped import or physical pool goes into the address's cache slot.
static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out) {
    memset(out, 0, sizeof *out);
    pool_lock_shared();
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)addr, size);
    PoolRec *r = e ? e->buf : NULL;
    int ok = 0;
    if (r && r->kind == POOL_IMPORT && __atomic_load_n(&r->lazy, __ATOMIC_ACQUIRE))
        lazy_import(r, (int)(e - pool_index.e));   // first GPU use
    if (r && (r->kind == POOL_IMPORT ? r->buf != VK_NULL_HANDLE : r->kind == POOL_PHYS)) {
        int i = (int)(e - pool_index.e);
        if ((i == 0 || pool_index.maxhi[i - 1] <= e->lo) && (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= e->hi)) {
            // pool_gen is stable here: it changes only under the exclusive lock.
            ResolveHit *h = rhit_add(__atomic_load_n(&pool_gen, __ATOMIC_RELAXED), e->lo, e->hi);
            h->buf = r->buf, h->cls = rec_class(r);
            if (r->kind == POOL_IMPORT) h->base = r->import_lo, h->end = r->import_hi, h->alloc = NULL, h->pool = r;
            else h->base = r->lo, h->end = r->lo + r->size, h->alloc = r->alloc, h->pool = NULL;
        }
    }
    if (r && r->kind == POOL_IMPORT && r->buf) {
        out->cls = rec_class(r);
        out->pool = r;
        out->buffer = r->buf;
        out->offset = addr - r->import_lo;
        out->avail = r->import_hi - addr;
        if (r->smap && !r->stouched) shadow_touch(r);
        ok = 1;
    } else if (r && r->kind == POOL_PHYS) {
        out->buffer = r->buf;
        out->offset = out->alloc_offset = addr - r->lo;
        out->avail = r->size - out->offset;
        out->alloc = r->alloc;
        out->cls = VK_MCLS_DEVICE;
        ok = 1;
    } else if (r && r->kind == POOL_VIRT) {
        uint64_t rel = addr - r->lo, k = rel / NVN_VPOOL_PAGE;
        if (k < r->npages && r->pt[k].phys) {
            PoolRec *ph = r->pt[k].phys;
            uint64_t off = r->pt[k].off + rel % NVN_VPOOL_PAGE, end = r->pt[k].off + NVN_VPOOL_PAGE;
            for (uint64_t j = k + 1; j < r->npages && r->pt[j].phys == ph && r->pt[j].off == end; j++) end += NVN_VPOOL_PAGE;
            out->buffer = ph->buf;
            out->offset = out->alloc_offset = off;
            out->avail = end - off;
            out->alloc = ph->alloc;
            out->cls = VK_MCLS_DEVICE;
            ok = out->avail >= size;
            if (ok && !ph->used) plat_xchg32(&ph->used, 1);   // read first: the flag is only ever set
        }
    }
    if (ok && !r->used) plat_xchg32(&r->used, 1);
    plat_mutex_unlock_shared(&pool_lock);
    if (!ok) memset(out, 0, sizeof *out);
    return ok;
}

uint64_t vk_mem_gen(void) {
    return __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE) + __atomic_load_n(&mirror_gen, __ATOMIC_ACQUIRE);
}

uint64_t vk_mem_pool_gen(void) { return __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE); }

int vk_mem_class_of(uint64_t addr) {
    VkMemSpan m;
    return vk_mem_resolve(addr, 1, &m) ? m.cls : VK_MCLS_NONE;
}

// ---------------------------------------------------------------- census
static volatile int64_t traffic[VK_MCLS_N][VK_MUSE_N];

// Every caller is the replay (vk_exec*, under vk_sync_submit's lock), about ten adds per draw: one writer at a time, so
// a plain add, not a locked one. An add racing the census's reset may be lost.
static _Thread_local int count_quiet;   // a slice of a parallel pass: several threads would fight over the counters' lines

void vk_mem_count_quiet(int on) { count_quiet = on; }

void vk_mem_count(int cls, int use, uint64_t bytes) {
    if (count_quiet) return;
    if ((unsigned)cls < VK_MCLS_N && (unsigned)use < VK_MUSE_N)
        __atomic_store_n(&traffic[cls][use], __atomic_load_n(&traffic[cls][use], __ATOMIC_RELAXED) + (int64_t)bytes,
                         __ATOMIC_RELAXED);
}

static void flag_names(uint32_t f, char *buf, size_t n) {
    static const char *const names[] = {"CPU_NO_ACCESS", "CPU_UNCACHED", "CPU_CACHED", "GPU_NO_ACCESS", "GPU_UNCACHED",
                                        "GPU_CACHED", "SHADER_CODE", "COMPRESSIBLE", "PHYSICAL", "VIRTUAL"};
    size_t k = 0;
    buf[0] = 0;
    for (int i = 0; i < 10; i++)
        if (f >> i & 1) k += (size_t)snprintf(buf + k, k < n ? n - k : 0, "%s%s", k ? "|" : "", names[i]);
    if (!k) snprintf(buf, n, "none");
}

// One line per (kind, flags) group of the live pools, then the bytes the replay bound or copied per frame by pool
// class since the previous census (the first one: since frame 1200). Logged at frame 1800 (in the world from the
// run-game saves), at frame 600, and with TOTK_TRACE=perf every 1800 frames.
static void dirty_frame(uint64_t frame);

static void mirror_log(uint64_t frame);
static void shadow_log(uint64_t frame);
static int dead_pending(void);

static void vram_log(uint64_t frame);

void vk_mem_census(uint64_t frame) {
    static uint64_t window_start;
    static int report_every = -1;   // TOTK_MEM_REPORT, read on the first present
    if (report_every < 0) report_every = cfg_int("TOTK_MEM_REPORT");
    __atomic_store_n(&mem_frame, frame, __ATOMIC_RELAXED);
    lazy_frame_log(frame);
    if (report_every > 0 && frame % (uint64_t)report_every == 0) {
        char when[32];
        snprintf(when, sizeof when, "frame %llu", (unsigned long long)frame);
        mem_commit_report(when);
    }
    if ((rt_trace & RT_TRACE_PERF) && frame % 600 == 0) vram_log(frame);
    if ((rt_trace & RT_TRACE_PERF) && !vk_mirror_on) dirty_frame(frame);
    if ((rt_trace & RT_TRACE_PERF) ? frame % 60 == 0 : frame == 600 || frame == 1800)
        mirror_log(frame), shadow_log(frame);   // mirror_log also logs the write-watch line
    if (frame == 1200) {   // loading is over: the first window starts here
        for (int c = 0; c < VK_MCLS_N; c++)
            for (int u = 0; u < VK_MUSE_N; u++) plat_xchg64(&traffic[c][u], 0);
        window_start = frame;
        return;
    }
    // Frame 600 as well: runs on slow machines end before 1800.
    if (frame != 600 && (frame % 1800 || !(frame == 1800 || (rt_trace & RT_TRACE_PERF)))) return;
    typedef struct { int kind; uint32_t flags; uint64_t n, bytes; } Group;
    Group g[64];
    int ng = 0;
    uint64_t total[4] = {0}, count[4] = {0}, used_n = 0, used_bytes = 0, prev_hi = 0, near_n = 0, gap_bytes = 0;
    uint64_t adj_n = 0;
    pool_lock_shared();
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT && r->used) {
            used_n++, used_bytes += r->size;
            // Neighbours: a used imported pool that starts within 1 MiB of the previous one's import end (the index is
            // sorted by address), so one larger import could have covered both; adj_n those with no gap at all.
            if (prev_hi && r->import_lo >= prev_hi && r->import_lo - prev_hi <= (1u << 20)) near_n++, gap_bytes += r->import_lo - prev_hi;
            if (prev_hi && r->import_lo == prev_hi) adj_n++;
            if (r->import_hi) prev_hi = r->import_hi;
        }
        int j = 0;
        while (j < ng && (g[j].kind != r->kind || g[j].flags != r->flags)) j++;
        if (j == ng && ng < 64) g[ng++] = (Group){r->kind, r->flags, 0, 0};
        if (j < ng) g[j].n++, g[j].bytes += r->size;
        total[r->kind] += r->size, count[r->kind]++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    rt_log("mem: census at frame %llu: import %llu pools %.1f MiB (%d without a GPU buffer), physical %llu pools %.1f MiB, "
           "virtual %llu pools %.1f MiB of address space",
           (unsigned long long)frame, (unsigned long long)count[POOL_IMPORT], total[POOL_IMPORT] / 1048576.0,
           n_import_failures, (unsigned long long)count[POOL_PHYS], total[POOL_PHYS] / 1048576.0,
           (unsigned long long)count[POOL_VIRT], total[POOL_VIRT] / 1048576.0);
    // How many imported pools the GPU has resolved at least once: the rest paid the import (about 20 ms of driver time
    // each on NVIDIA, whatever the size) for memory that only the CPU touched.
    rt_log("mem: of the %llu imported pools, %llu (%.1f MiB) were resolved by the GPU so far; %.0f ms in all imports (%d, %d of them on first GPU use)",
           (unsigned long long)count[POOL_IMPORT], (unsigned long long)used_n, used_bytes / 1048576.0,
           __atomic_load_n(&import_alloc_ns, __ATOMIC_RELAXED) / 1e6, n_imports, n_lazy_imports);
    rt_log("mem: imports took %.0f ms in all, %.0f ms of it committing and touching the pages; %llu used pools start within "
           "1 MiB of the previous one (%.1f MiB of gaps), %llu right after it", __atomic_load_n(&import_total_ns, __ATOMIC_RELAXED) / 1e6,
           __atomic_load_n(&import_touch_ns, __ATOMIC_RELAXED) / 1e6, (unsigned long long)near_n, gap_bytes / 1048576.0,
           (unsigned long long)adj_n);
    uint64_t touch[4];
    mem_touch_stats(touch);
    rt_log("mem: mem_touch: %llu calls, %llu chunks of 64 KiB known committed, %llu queried, %llu found reserved by "
           "TOTK_TOUCH_MAP=verify; imported memory released %lld times in %.0f ms",
           (unsigned long long)touch[0], (unsigned long long)touch[1], (unsigned long long)touch[2],
           (unsigned long long)touch[3], (long long)__atomic_load_n(&rel_n, __ATOMIC_RELAXED),
           __atomic_load_n(&rel_ns, __ATOMIC_RELAXED) / 1e6);
    static const char *const kinds[] = {"", "import", "physical", "virtual"};
    for (int j = 0; j < ng; j++) {
        char f[160];
        flag_names(g[j].flags, f, sizeof f);
        rt_log("mem:   %s %#x %s: %llu pools, %.1f MiB", kinds[g[j].kind], g[j].flags, f, (unsigned long long)g[j].n,
               g[j].bytes / 1048576.0);
    }
    uint64_t frames = window_start && frame > window_start ? frame - window_start : frame;
    static const char *const cls[] = {"", "host CPU_NO_ACCESS", "host CPU_UNCACHED", "host CPU_CACHED", "device", "scratch"};
    rt_log("mem: traffic per frame over %llu frames, MiB (bound ranges, not bytes read): class: vertex index uniform "
           "storage indirect copy-src gpu-write cpu-read", (unsigned long long)frames);
    for (int c = 1; c < VK_MCLS_N; c++) {
        double v[VK_MUSE_N];
        for (int u = 0; u < VK_MUSE_N; u++) v[u] = plat_xchg64(&traffic[c][u], 0) / 1048576.0 / (double)(frames ? frames : 1);
        rt_log("mem:   %s: %.2f %.2f %.2f %.2f %.3f %.2f %.2f %.3f", cls[c], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    }
    window_start = frame;
}

// TOTK_VRAM_LIMIT=<MiB> (test knob, unset = off): caps the device-local budget every VRAM consumer reads here (texture
// pressure, headroom, the perf line), to see how the runtime behaves on a card with less video memory. The driver still
// lets allocations go past it, so it shows the runtime's reaction, not the driver paging a real small card does.
static void vram_limit(uint64_t *budget) {
    static int64_t lim = -1;
    if (lim < 0) { const char *s = cfg_get("TOTK_VRAM_LIMIT"); lim = s ? (int64_t)strtoull(s, NULL, 10) << 20 : 0; }
    if (lim && *budget > (uint64_t)lim) *budget = (uint64_t)lim;
}

int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact) {
    VkCtx *c = vk_ctx();
    *used = *budget = 0, *exact = 0;
    if (!c) return 0;
    const VkPhysicalDeviceMemoryProperties *mp = &c->mem_props;
    if (c->has_memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
        uint64_t ph = perf_ph_begin();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame); NVIDIA asks the kernel
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
        perf_ph_end(PERF_PH_BUDGET, ph);
        for (uint32_t h = 0; h < p2.memoryProperties.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
            if (p2.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                *used += b.heapUsage[h], *budget += b.heapBudget[h];
        *exact = 1;
        vram_limit(budget);
        return *budget != 0;
    }
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    for (uint32_t h = 0; h < mp->memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (mp->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            *used += vb[h].statistics.blockBytes, *budget += mp->memoryHeaps[h].size;
    vram_limit(budget);
    return *budget != 0;
}

// ---------------------------------------------------------------- page write generations
// Two-level table over the guest address space (below 2^39): 256 MiB chunks of 65,536 page generations, allocated
// when a page of the chunk is first stamped. 0 = never stamped.
#define GEN_CHUNK_SHIFT 28
#define GEN_PAGE_SHIFT 12
#define GEN_CHUNKS (RT_ADDRESS_LIMIT >> GEN_CHUNK_SHIFT)
#define GEN_PER_CHUNK (1u << (GEN_CHUNK_SHIFT - GEN_PAGE_SHIFT))
static uint32_t *gen_chunk[GEN_CHUNKS];
static uint32_t gen_now;                 // last generation handed out
static PlatMutex gen_lock = PLAT_MUTEX_INIT;
// The exclusive takes that run write-watch queries: TOTK_TRACE=perf counts the waits over 100 us (perf_lock_wait), to see
// whether a query held under another lock moved its wait here (#317).
static void gen_lock_x(void) {
    if (plat_mutex_trylock(&gen_lock)) return;
    uint64_t t0 = kern_now_ns();
    plat_mutex_lock(&gen_lock);
    if (rt_trace & RT_TRACE_PERF) perf_lock_wait(PERF_LOCK_GEN, kern_now_ns() - t0);
}
static uint64_t gen_pages[1024];
// The newest generation stamped on any page of each 2 MiB region (the write-watch snapshot regions below): the mirror
// sync skips regions with nothing newer than its pool's mirror.
#define REG_SHIFT 21
static uint32_t reg_max[RT_ADDRESS_LIMIT >> REG_SHIFT];
// Per page, the generation of the last stamp that reported a GPU write the page's VRAM mirror already holds
// (vk_gpu_written_into), allocated per chunk on the first such stamp. While a page's generation is still that stamp,
// mirror_sync leaves the page out of its copies.
static uint32_t *own_chunk[GEN_CHUNKS];   // gen_lock

// Every write-watch query of the generation table (gen_lock held exclusively): calls, bytes and time since the last
// mirror summary, all consumers together; the queries that found written pages (they also flush TLBs) apart.
static uint64_t ww_calls, ww_bytes, ww_ticks, ww_dirty_calls, ww_dirty_ticks, ww_pages;
// TOTK_TRACE=perf (#317): the same queries' time on the CPU (plat_self_cpu_ns), so that a query that waits (in the
// kernel, for another thread's VirtualQuery or VirtualProtect over the same reservation) shows apart from one that
// works; the time by consumer; and the slowest query of the window with its range and answer.
enum { WW_CHECK, WW_SNAP, WW_SHADOW, WW_DIRTY, WW_FREEZE, WW_WHO_N };
static const char *const ww_who_name[WW_WHO_N] = {"texture check", "snapshot", "shadow sync", "dirty census", "freeze"};
static uint64_t ww_cpu_ns, ww_who_ns[WW_WHO_N], ww_slow_ns, ww_slow_cpu, ww_slow_addr, ww_slow_size, ww_slow_pages;
static int ww_slow_who;

// TOTK_TRACE=perf: a query that took TOTK_WW_SLOW_MS (5 by default) or more gets a line with the operations other
// threads were inside meanwhile (platform.h plat_op_overlap), the first 64 such queries. They are grouped by thread
// and kind: count, the time of theirs inside the query's interval, the longest, the longest run a VirtualQuery
// reported, and the range they covered; those still running are listed one by one with their nesting. gen_lock is held
// (the line costs tens of microseconds, after a query that waited milliseconds).
#define WW_OPS 4096
typedef struct {
    uint32_t tid;
    int role, kind, n;
    uint64_t in_ns, max_ns, max_run, lo, hi;
} WwOpGroup;
static PlatOp ww_ops[WW_OPS];
static WwOpGroup ww_groups[64];

static void ww_slow_report(uint64_t addr, uint64_t size, int who, uint64_t t0, uint64_t t1, uint64_t cpu, uint64_t pages) {
    static uint64_t min_ns;
    static int lines;
    if (!min_ns) {
        const char *e = cfg_get("TOTK_WW_SLOW_MS");
        min_ns = (uint64_t)(e && atof(e) > 0 ? atof(e) * 1e6 : 5e6);
    }
    if (t1 - t0 < min_ns || lines >= 64) return;
    lines++;
    int lost, n = plat_op_overlap(t0, t1, plat_thread_id(), ww_ops, WW_OPS, &lost), ng = 0;
    char run_line[1400];
    size_t ro = 0;
    run_line[0] = 0;
    for (int i = 0; i < n; i++) {
        const PlatOp *o = &ww_ops[i];
        if (!o->done) {   // still running: one by one, nested ones after the outer ones of the same thread
            if (ro < sizeof run_line - 160)
                ro += (size_t)snprintf(run_line + ro, sizeof run_line - ro, "%s[%u %s] %s%s at %#llx+%#llx for %.2f ms",
                                       ro ? "; " : "", o->tid, plat_op_role_name(o->role), o->depth ? "> " : "",
                                       plat_op_name(o->kind), (unsigned long long)o->addr, (unsigned long long)o->size,
                                       (o->t1 - o->t0) / 1e6);
            continue;
        }
        int g = 0;
        while (g < ng && !(ww_groups[g].tid == o->tid && ww_groups[g].kind == o->kind)) g++;
        if (g == ng) {
            if (ng == (int)(sizeof ww_groups / sizeof *ww_groups)) continue;
            ww_groups[ng++] = (WwOpGroup){o->tid, o->role, o->kind, 0, 0, 0, 0, UINT64_MAX, 0};
        }
        WwOpGroup *G = &ww_groups[g];
        uint64_t a = o->t0 > t0 ? o->t0 : t0, b = o->t1 < t1 ? o->t1 : t1, d = o->t1 - o->t0;
        G->n++, G->in_ns += b - a;
        if (d > G->max_ns) G->max_ns = d;
        if (o->run > G->max_run) G->max_run = o->run;
        if (o->addr < G->lo) G->lo = o->addr;
        if (o->addr + o->size > G->hi) G->hi = o->addr + o->size;
    }
    for (int i = 1; i < ng; i++)   // the most time inside the query first
        for (int j = i; j > 0 && ww_groups[j].in_ns > ww_groups[j - 1].in_ns; j--) {
            WwOpGroup x = ww_groups[j];
            ww_groups[j] = ww_groups[j - 1], ww_groups[j - 1] = x;
        }
    char line[2400];
    size_t o = 0;
    line[0] = 0;
    for (int g = 0; g < ng && g < 16 && o < sizeof line - 200; g++) {
        const WwOpGroup *G = &ww_groups[g];
        char run[32] = "";
        if (G->max_run) snprintf(run, sizeof run, ", run %.0f MiB", G->max_run / 1048576.0);
        o += (size_t)snprintf(line + o, sizeof line - o, "%s[%u %s] %s %d (%.2f ms inside, longest %.2f ms%s, %#llx-%#llx)",
                              o ? "; " : "", G->tid, plat_op_role_name(G->role), plat_op_name(G->kind), G->n,
                              G->in_ns / 1e6, G->max_ns / 1e6, run, (unsigned long long)G->lo,
                              (unsigned long long)G->hi);
    }
    rt_log("mem: slow write-watch query: %.3f ms (%.3f on the CPU) over %.1f MiB at %#llx, %llu pages, by %s; ended "
           "operations of other threads inside it (%d groups%s): %s; still running: %s", (t1 - t0) / 1e6, cpu / 1e6,
           size / 1048576.0, (unsigned long long)addr, (unsigned long long)pages, ww_who_name[who], ng,
           lost ? ", the ring was overwritten since its start" : "", ng ? line : "none", ro ? run_line : "none");
}

static uint64_t ww_query(uint64_t addr, uint64_t size, int who) {
    int perf = (rt_trace & RT_TRACE_PERF) != 0;
    uint64_t c0 = perf ? plat_self_cpu_ns() : 0, t0 = qpc();
    uint64_t n = mem_written_pages(addr, size, gen_pages, sizeof gen_pages / sizeof *gen_pages);
    uint64_t t1 = qpc(), dt = t1 - t0;
    ww_ticks += dt, ww_calls++, ww_bytes += size;
    if (n && n != UINT64_MAX) ww_dirty_calls++, ww_dirty_ticks += dt, ww_pages += n;
    if (perf) {
        uint64_t cpu = plat_self_cpu_ns() - c0;
        ww_cpu_ns += cpu, ww_who_ns[who] += dt;
        if (dt > ww_slow_ns)
            ww_slow_ns = dt, ww_slow_cpu = cpu, ww_slow_addr = addr, ww_slow_size = size,
            ww_slow_pages = n == UINT64_MAX ? 0 : n, ww_slow_who = who;
        ww_slow_report(addr, size, who, t0, t1, cpu, n == UINT64_MAX ? 0 : n);
    }
    return n;
}

static void stamp_page(uint64_t a, uint32_t g);

// gen_lock held exclusively. Stamps every page of [lo, hi) the CPU wrote with g, in as many queries as gen_pages needs;
// 0 when the range is not write-watched. A full answer resets only the pages it holds, and the platforms report pages
// in address order (gfx_abi_probe.c check_write_watch), so the next query starts after the last page reported. Each
// query used to start at lo again: a range with many written pages cost one scan of all of it per 1024 pages, and the
// mirror snapshot at the first use of the 450 MiB terrain arena of a layer (115,000 written pages, 113 queries) took
// 14 to 21 ms of the replay (#397); resumed, it takes 3 to 8 ms. who is the consumer (WW_CHECK and the others) that
// ww_query books each query's time under; each resumed query is a ww_query of its own, so the slowest query of a
// TOTK_TRACE=perf window is one of them, with the range from where it started to hi.
static int ww_stamp(uint64_t lo, uint64_t hi, uint32_t g, int who) {
    const uint64_t max = sizeof gen_pages / sizeof *gen_pages;
    for (uint64_t a = lo; a < hi;) {
        uint64_t n = ww_query(a, hi - a, who);
        if (n == UINT64_MAX) return 0;
        for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g);
        if (n < max) break;
        a = (gen_pages[n - 1] & ~0xfffull) + (1ull << GEN_PAGE_SHIFT);
    }
    return 1;
}

// Frozen regions: a 2 MiB region whose pages nothing wrote for freeze_idle submissions is made read-only
// (memory.c mem_freeze), and write-watch queries skip it until a write or a file read thaws it. Most of the pool memory
// the mirror and the texture checks cover is static (meshes, textures), and a query costs about 1 us per MiB (4-5 on an
// 8th-gen i7), so this keeps the queries to the regions the CPU writes. The region need not lie wholly in pools: NVN
// pools are small and not 2 MiB aligned, so most regions also hold other guest memory, and a rule that refused those
// left 340 of 436 queried MiB a frame unfrozen in town although nothing had written them for 300 submissions. The
// other memory is safe to freeze for the same reason pools are: the only kernel writes into guest memory are file
// reads, which thaw their buffer first (fs_host.c, romfs.c), and every user-mode write faults and thaws.
#define FREEZE_PER_SUB 32
static int freeze_idle;                       // 0: off
static uint32_t ww_ep = 1;                    // gen_lock: the submission being recorded (vk_mem_mirror_begin)
static uint32_t reg_last[RT_ADDRESS_LIMIT >> REG_SHIFT];    // gen_lock: ww_ep of the region's last stamp
static uint32_t reg_tried[RT_ADDRESS_LIMIT >> REG_SHIFT];   // gen_lock: ww_ep of its last freeze candidacy
static uint32_t *fcand;                       // gen_lock: regions to try at the next freeze pass
static int nfcand, cfcand;
static uint64_t n_frz_refused;                // gen_lock: candidates mem_freeze refused

static uint32_t *gen_slot(uint64_t page_addr, int create) {
    uint64_t ci = page_addr >> GEN_CHUNK_SHIFT;
    if (ci >= GEN_CHUNKS) return NULL;
    if (!gen_chunk[ci]) {
        if (!create) return NULL;
        gen_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!gen_chunk[ci]) rt_fatal("vk: out of memory for page generations");
    }
    return &gen_chunk[ci][(page_addr >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
}

// gen_lock held exclusively; g is the newest generation handed out.
static void stamp_page(uint64_t a, uint32_t g) {
    uint32_t *s = gen_slot(a, 1);
    if (!s) return;
    *s = g;
    reg_max[a >> REG_SHIFT] = g;
    reg_last[a >> REG_SHIFT] = ww_ep;
}

static void stamp_range(uint64_t lo, uint64_t hi, uint32_t g) {
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) stamp_page(a, g);
}

void vk_guest_written(uint64_t addr, uint64_t size) {
    if (!size || nvn_gpu_only(addr)) return;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    stamp_range(lo, hi, ++gen_now);
    plat_mutex_unlock(&gen_lock);
}

static void wb_open(uint64_t a, uint64_t n);

int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    wb_open(addr, size);   // the caller reads the range next: the lazy write-back's pages get their bytes first
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    gen_lock_x();
    uint32_t g = ++gen_now;
    int tracked = 1, frozen = 1;
    for (uint64_t a = lo & ~((1ull << REG_SHIFT) - 1); a < hi && frozen; a += 1ull << REG_SHIFT) frozen = mem_frozen(a);
    if (!frozen) tracked = ww_stamp(lo, hi, g, WW_CHECK);
    int changed = 0;
    for (uint64_t a = lo; tracked && a < hi && !changed; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 0);
        if (s && *s > since) changed = 1;
    }
    plat_mutex_unlock(&gen_lock);
    *now = g;
    return tracked ? changed : -1;
}

// Write-watch snapshots per submission: the guest address space in 2 MiB regions, each queried at most once per epoch
// (a guest submission), at the first check that covers it. A query costs about 1.3 us plus 0.55 us per MiB, so one per
// region instead of one per texture. The snapshot only adds observations to the generation table: a page written
// before it carries a generation newer than any check before it, and a page written after it keeps its OS bit until
// the next query over it (at the latest the next epoch's snapshot).
#define SNAP_SHIFT 21
#define SNAP_REGIONS (RT_ADDRESS_LIMIT >> SNAP_SHIFT)
static uint32_t snap_at[SNAP_REGIONS];      // epoch of the region's last snapshot; 0 never
static uint8_t snap_untracked[SNAP_REGIONS];
static uint64_t n_snap_checks, n_snap_calls, n_snap_untracked;

// gen_lock held exclusively. Queries the regions r0..r1 that were not snapped in epoch, one query per run of them;
// returns the number of queries.
static uint64_t snap_regions(uint64_t r0, uint64_t r1, uint32_t epoch) {
    uint32_t g = 0;
    uint64_t calls = 0;
    for (uint64_t r = r0; r <= r1;) {
        if (snap_at[r] == epoch) { r++; continue; }
        if (mem_frozen(r << SNAP_SHIFT)) {   // nothing in it changed since its freeze
            snap_at[r] = epoch, snap_untracked[r] = 0;
            r++;
            continue;
        }
        uint64_t k = r;   // one query over a run of regions not seen in this epoch
        while (k + 1 <= r1 && snap_at[k + 1] != epoch && !mem_frozen((k + 1) << SNAP_SHIFT)) k++;
        uint64_t a = r << SNAP_SHIFT, b = (k + 1) << SNAP_SHIFT;
        if (!g) g = ++gen_now;
        int tracked = ww_stamp(a, b, g, WW_SNAP);
        calls++;
        for (uint64_t j = r; j <= k; j++) {
            snap_at[j] = epoch, snap_untracked[j] = (uint8_t)!tracked;
            if (tracked && freeze_idle && ww_ep - reg_last[j] >= (uint32_t)freeze_idle &&
                ww_ep - reg_tried[j] >= (uint32_t)freeze_idle) {
                reg_tried[j] = ww_ep;
                if (nfcand == cfcand) grow((void **)&fcand, &cfcand, sizeof *fcand);
                fcand[nfcand++] = (uint32_t)j;
            }
        }
        r = k + 1;
    }
    return calls;
}

int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    wb_open(addr, size);
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (!epoch || r1 >= SNAP_REGIONS) return vk_mem_changed(addr, size, since, now);
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t r = r0; r <= r1; r++)
        if (snap_at[r] != epoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        gen_lock_x();
        n_snap_calls += snap_regions(r0, r1, epoch);
    }
    int untracked = 0, changed = 0;
    for (uint64_t r = r0; r <= r1; r++) untracked |= snap_untracked[r];
    // Pages only in regions stamped after since (reg_max; REG_SHIFT is SNAP_SHIFT): an unchanged texture of 4 MiB was
    // 1,024 page reads, at about 1,100 texture checks a frame.
    for (uint64_t r = r0; !untracked && r <= r1 && !changed; r++) {
        if (reg_max[r] <= since) continue;
        uint64_t a = r << REG_SHIFT > lo ? r << REG_SHIFT : lo, b = (r + 1) << REG_SHIFT < hi ? (r + 1) << REG_SHIFT : hi;
        for (; a < b && !changed; a += 1ull << GEN_PAGE_SHIFT) {
            uint32_t *s = gen_slot(a, 0);
            if (s && *s > since) changed = 1;
        }
    }
    *now = gen_now;   // every observation so far is in the table
    __atomic_add_fetch(&n_snap_checks, 1, __ATOMIC_RELAXED);
    if (untracked) __atomic_add_fetch(&n_snap_untracked, 1, __ATOMIC_RELAXED);
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    return untracked ? -1 : changed;
}

void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked) {
    *checks = __atomic_exchange_n(&n_snap_checks, 0, __ATOMIC_RELAXED);
    *calls = __atomic_exchange_n(&n_snap_calls, 0, __ATOMIC_RELAXED);
    *untracked = __atomic_exchange_n(&n_snap_untracked, 0, __ATOMIC_RELAXED);
}

// ---------------------------------------------------------------- dirty census (TOTK_TRACE=perf)
// What a device-local mirror of the buffer-bound pools would upload. The replay notes every
// imported pool it binds as a vertex, index, uniform, storage or indirect buffer (vk_mem_bound_note). At the start of
// each submit one write-watch query per run of such pools stamps the pages written since with a new generation, in the
// shared generation table (no second consumer of the OS bits: texture checks see the stamps), and the pages of those
// pools are counted:
// - eager: pages stamped after the previous submit's scan, in every buffer-bound pool (upload at every submit);
// - lazy: pages stamped after the pool's previous count, in pools the previous submit bound (upload when bound);
// - first use: the size of a pool the first time it is bound (a mirror copies the whole pool then).
// Classes: CPU_CACHED, CPU_UNCACHED, other (CPU_NO_ACCESS). Logged as `mem: dirty` every 60 frames.
#define DIRTY_UNSYNCED UINT32_MAX
#define DIRTY_GAP 0x100000ull   // pools closer than this share one query (the gap's pages are stamped too)
#define DIRTY_WIN 60
enum { DC_CACHED, DC_UNC, DC_OTHER, DC_N };
typedef struct {
    uint64_t eager[DC_N], lazy[DC_N], first[DC_N];
    uint64_t submits, queries, scan_ns, sub_max, frames;
} DirtyAcc;
static uint32_t dirty_sub = 1, dirty_gen;   // submit lock: the current submit number, generation of the last scan
static uint64_t first_pending[DC_N], ever_n[DC_N], ever_bytes[DC_N];   // submit lock
static PoolRec **dlist;                     // submit lock: the buffer-bound pools of a scan
static int dcap;
static PlatMutex dirty_lock = PLAT_MUTEX_INIT;   // everything below
static DirtyAcc dfr, dwin;                  // this frame, this window
static uint32_t dwin_kib[DIRTY_WIN][DC_N];  // eager KiB of each frame of the window
static uint64_t live_n[DC_N], live_bytes[DC_N], live_untracked, ever_snap_n[DC_N], ever_snap_bytes[DC_N];

static int dc_class(const PoolRec *r) {
    int c = rec_class(r);
    return c == VK_MCLS_HOST_CACHED ? DC_CACHED : c == VK_MCLS_HOST_UNC ? DC_UNC : DC_OTHER;
}

void vk_mem_bound_note(void *pool) {   // TOTK_TRACE=perf; parallel slices call it too (dirty_lock)
    PoolRec *r = pool;
    if (__atomic_load_n(&r->bound_sub, __ATOMIC_RELAXED) == dirty_sub) return;
    plat_mutex_lock(&dirty_lock);
    if (r->bound_sub != dirty_sub) {
        r->bound_sub = dirty_sub;
        if (!r->bufbound) {
            r->bufbound = 1, r->synced_gen = DIRTY_UNSYNCED;
            int k = dc_class(r);
            ever_n[k]++, ever_bytes[k] += r->size, first_pending[k] += r->size;
        }
    }
    plat_mutex_unlock(&dirty_lock);
}

static uint64_t page_lo(const PoolRec *r) { return r->lo & ~0xfffull; }
static uint64_t page_hi(const PoolRec *r) { return (r->lo + r->size + 0xfff) & ~0xfffull; }

// ---------------------------------------------------------------- shadow storage
// Where importing guest memory costs too much (Linux: re-protecting pages that amdgpu imported as a userptr
// invalidates the buffer, about 1.3 ms per 16 MiB at the next submit, and the import's write pinning clears the
// write watch), a CPU-visible pool gets a host-visible buffer of its own instead, at the offsets the import would have:
// - CPU -> shadow: right before each submit (vk_mem_shadow_sync), the pools the replay resolved since the last one are
//   queried for written pages; pages stamped after a pool's sgen are copied in by the CPU.
// - GPU -> guest: the replay notes each GPU write into CPU-visible memory at record time (vk_mem_gpu_pending) and
//   reports it at the fence (vk_gpu_written), which copies those bytes back into guest memory. Until then the CPU
//   copies skip them: a page the CPU and the GPU both write keeps the GPU's bytes.
// TOTK_SHADOW=1/0 forces the mode on or off; the default is on for Linux, off where importing is cheap (Windows, and
// MoltenVK, which imports host memory without Linux's write-watch and page re-protection).
typedef struct { uint64_t lo, hi; } SRange;
static PlatMutex shadow_lock = PLAT_MUTEX_INIT;   // after pool_lock; before gen_lock
static PoolRec **stouch;                          // shadow_lock: pools resolved since the last sync
static int nstouch, cstouch;
static SRange *spend;                             // shadow_lock: GPU writes recorded and not copied back yet
static int nspend, cspend;
static SRange *sblocked;                          // shadow_lock: sorted union of spend, rebuilt when it changes
static int nsblocked, csblocked, spend_dirty;
static int64_t s_live_n, s_live_bytes;            // shadow_lock
static uint64_t s_syncs, s_copied, s_queries, s_untracked, s_wb_bytes, s_ticks;   // shadow_lock: since the last log

static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    r->import_lo = addr & ~0xfffull;
    r->import_hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t n = r->import_hi - r->import_lo;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    // Cached where the device has it: the fence copies read it back on the CPU.
    VmaAllocationCreateInfo ai = {.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  .preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
    VmaAllocationInfo info;
    VkResult res = vmaCreateBuffer(c->vma, &bi, &ai, &r->buf, &r->salloc, &info);
    if (res != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: shadow allocation failed (%d)", (unsigned long long)addr, (unsigned long long)size,
               (int)res);
        r->buf = VK_NULL_HANDLE, r->salloc = NULL;
        return 0;
    }
    r->smap = info.pMappedData;
    plat_mutex_lock(&gen_lock);
    r->sgen = gen_now;   // the copy below holds every write stamped so far; later stamps are copied again
    plat_mutex_unlock(&gen_lock);
    memcpy(r->smap, (const void *)(uintptr_t)r->import_lo, n);
    plat_mutex_lock(&shadow_lock);
    s_live_n++, s_live_bytes += (int64_t)n;
    plat_mutex_unlock(&shadow_lock);
    plat_inc32(&n_imports);
    return 1;
}

static void shadow_touch(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    if (!r->stouched) {
        if (nstouch == cstouch) grow((void **)&stouch, &cstouch, sizeof *stouch);
        stouch[nstouch++] = r;
        r->stouched = 1;
    }
    plat_mutex_unlock(&shadow_lock);
}

// vk_pool_fini, after the pool left the index: off the touched list before its record goes.
static void wb_open(uint64_t a, uint64_t n);

static void shadow_forget(PoolRec *r) {
    wb_open(r->import_lo, r->import_hi - r->import_lo);   // what its sealed pages are owed, while the shadow is there
    plat_mutex_lock(&shadow_lock);
    for (int i = 0; i < nstouch; i++)
        if (stouch[i] == r) { stouch[i] = stouch[--nstouch]; break; }
    s_live_n--, s_live_bytes -= (int64_t)(r->import_hi - r->import_lo);
    plat_mutex_unlock(&shadow_lock);
}

static int shadow_range_cmp(const void *x, const void *y) {
    const SRange *a = x, *b = y;
    return a->lo < b->lo ? -1 : a->lo > b->lo ? 1 : a->hi < b->hi ? -1 : a->hi > b->hi;
}

// shadow_lock held: the writeback list changes independently of the CPU's dirty-page queries. Build its union once
// per change, so each copied range searches by address rather than scanning thousands of pending writes.
static void shadow_blocked_update(void) {
    if (!spend_dirty) return;
    while (csblocked < nspend) grow((void **)&sblocked, &csblocked, sizeof *sblocked);
    if (nspend) {
        memcpy(sblocked, spend, (size_t)nspend * sizeof *spend);
        qsort(sblocked, (size_t)nspend, sizeof *sblocked, shadow_range_cmp);
    }
    nsblocked = 0;
    for (int i = 0; i < nspend; i++) {
        SRange r = sblocked[i];
        if (nsblocked && r.lo <= sblocked[nsblocked - 1].hi) {
            if (r.hi > sblocked[nsblocked - 1].hi) sblocked[nsblocked - 1].hi = r.hi;
        } else sblocked[nsblocked++] = r;
    }
    spend_dirty = 0;
}

// shadow_lock held: copy guest [a, b) except for bytes a pending GPU write will replace at its fence.
static void shadow_copy(PoolRec *r, uint64_t a, uint64_t b) {
    int lo = 0, hi = nsblocked;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (sblocked[mid].hi <= a) lo = mid + 1;
        else hi = mid;
    }
    for (int i = lo; i < nsblocked && sblocked[i].lo < b; i++) {
        if (sblocked[i].lo > a) {
            uint64_t end = sblocked[i].lo < b ? sblocked[i].lo : b;
            memcpy(r->smap + (a - r->import_lo), (const void *)(uintptr_t)a, end - a);
            s_copied += end - a;
        }
        if (sblocked[i].hi >= b) return;
        if (sblocked[i].hi > a) a = sblocked[i].hi;
    }
    if (a < b) {
        memcpy(r->smap + (a - r->import_lo), (const void *)(uintptr_t)a, b - a);
        s_copied += b - a;
    }
}

// shadow_lock and gen_lock held: stamps the pool's written pages and copies those newer than its shadow.
static void shadow_sync_pool(PoolRec *r) {
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint32_t g = ++gen_now;
    int tracked = 1;
    for (uint64_t a = lo; a < hi;) {   // one query per run of regions that are not frozen
        uint64_t b = (a | ((1ull << REG_SHIFT) - 1)) + 1;
        if (b > hi) b = hi;
        if (mem_frozen(a)) { a = b; continue; }
        while (b < hi && !mem_frozen(b)) b = b + (1ull << REG_SHIFT) < hi ? b + (1ull << REG_SHIFT) : hi;
        s_queries++;
        if (!ww_stamp(a, b, g, WW_SHADOW)) tracked = 0;
        a = b;
    }
    uint32_t since = r->sgen;
    r->sgen = gen_now;
    if (!tracked) {   // nothing tells which pages changed: all of them
        s_untracked++;
        shadow_copy(r, lo, hi);
        return;
    }
    uint64_t run = 0;
    for (uint64_t k = lo >> REG_SHIFT; k <= (hi - 1) >> REG_SHIFT; k++) {
        uint64_t a = k << REG_SHIFT, b = a + (1ull << REG_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (reg_max[k] <= since || !ch) {
            if (run) shadow_copy(r, run, a), run = 0;
            continue;
        }
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT) {
            int dirty = ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since;
            if (dirty && !run) run = x;
            else if (!dirty && run) shadow_copy(r, run, x), run = 0;
        }
    }
    if (run) shadow_copy(r, run, hi);
}

void vk_mem_shadow_sync(void) {
    if (!shadow_on) return;
    uint64_t t0 = qpc();
    plat_mutex_lock(&shadow_lock);
    if (nstouch) {
        shadow_blocked_update();
        plat_mutex_lock(&gen_lock);
        for (int i = 0; i < nstouch; i++) shadow_sync_pool(stouch[i]), stouch[i]->stouched = 0;
        plat_mutex_unlock(&gen_lock);
        s_syncs += (uint64_t)nstouch;
        nstouch = 0;
    }
    s_ticks += qpc() - t0;
    plat_mutex_unlock(&shadow_lock);
}

void vk_mem_gpu_pending(uint64_t addr, uint64_t size) {
    if (!shadow_on || !size) return;
    plat_mutex_lock(&shadow_lock);
    if (nspend == cspend) grow((void **)&spend, &cspend, sizeof *spend);
    spend[nspend++] = (SRange){addr, addr + size};
    spend_dirty = 1;
    plat_mutex_unlock(&shadow_lock);
}

void vk_gpu_written(uint64_t addr, uint64_t size) {
    if (shadow_on && size) {
        pool_lock_shared();
        const pidx_ent *e = pidx_find(&pool_index, addr, size);
        PoolRec *r = e ? e->buf : NULL;
        plat_mutex_lock(&shadow_lock);
        if (r && r->smap && addr >= r->import_lo && addr + size <= r->import_hi) {
            mem_thaw(addr, size);
            memcpy((void *)(uintptr_t)addr, r->smap + (addr - r->import_lo), size);
            s_wb_bytes += size;
        }
        for (int i = 0; i < nspend; i++)
            if (spend[i].lo == addr && spend[i].hi == addr + size) { spend[i] = spend[--nspend], spend_dirty = 1; break; }
        plat_mutex_unlock(&shadow_lock);
        plat_mutex_unlock_shared(&pool_lock);
    }
    vk_guest_written(addr, size);
}

// vk_gpu_written_many without the shadow storage. The GPU writes of a command buffer, about 14,800 ranges a frame in
// Hilltop, reach it on the completion thread before the fence counts as done (vk_exec_draw.c cb_done), so the guest
// thread waiting in nvnSyncWait waits for them. Through vk_gpu_written_into one by one, each range took the pool lock
// and gen_lock and a generation of its own: about 0.37 ms a frame. Here a batch of ranges takes both locks once (BATCH
// bounds how long other threads wait for gen_lock), and the ranges through a pool's mirror share one generation: two
// such stamps of one page give it the same own_chunk note as two generations in a row would (the second keeps a note
// the first made and keeps 0 after a 0), and nothing reads the pages between them while gen_lock is held. Any other
// stamp (vk_guest_written's) takes a generation of its own and ends the shared one, so an own_chunk note never survives
// a newer write the mirror does not hold. A page the previous range of the same pool stamped with the same generation
// is not stamped again. TOTK_STAMP_BATCH=0 stamps one by one as before; "ab" for an in-run A/B. Each size is read
// without VK_WB_LAZY (vk.h): with that bit set the end of the range wraps past the top of the address space, so
// pidx_find finds no pool and the stamp's walk covers no page at all.
static void written_batch(const VkGpuWrite *w, int n) {
    enum { BATCH = 1024 };
    static int mode = -1;
    if (mode < 0) mode = rt_knob("STAMP_BATCH", "1");
    if (!RT_KNOB_ON(mode) || !vk_mirror_on) {
        for (int i = 0; i < n; i++) vk_gpu_written_into(w[i].addr, w[i].size & ~VK_WB_LAZY, w[i].buf);
        return;
    }
    for (int i = 0; i < n;) {
        int end = n - i > BATCH ? i + BATCH : n;
        pool_lock_shared();
        plat_mutex_lock(&gen_lock);
        uint32_t g = 0;                          // the shared generation of mirror stamps, 0 after another stamp
        const PoolRec *prev = NULL;
        uint64_t prev_lo = 0, prev_hi = 0;       // pages the previous mirror range stamped with g
        for (; i < end; i++) {
            uint64_t addr = w[i].addr, size = w[i].size & ~VK_WB_LAZY;
            if (!size || nvn_gpu_only(addr)) continue;   // vk_guest_written stamps nothing for those
            uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
            const pidx_ent *e = w[i].buf ? pidx_find(&pool_index, addr, size) : NULL;
            const PoolRec *r = e ? e->buf : NULL;
            if (!r || ms_get(r) != MS_ON || r->mbuf != w[i].buf || addr < r->import_lo || addr + size > r->import_hi) {
                stamp_range(lo, hi, ++gen_now);   // as vk_guest_written
                g = 0, prev = NULL;
                continue;
            }
            if (!g) g = ++gen_now;
            uint64_t from = lo;
            if (r == prev && lo >= prev_lo && lo <= prev_hi) {   // continues the span already stamped with g
                if (prev_hi > from) from = prev_hi;
                if (hi > prev_hi) prev_hi = hi;
            } else {
                prev_lo = lo, prev_hi = hi;
            }
            prev = r;
            uint32_t mgen = r->mgen;
            for (uint64_t a = from; a < hi; a += 1ull << GEN_PAGE_SHIFT) {
                uint32_t *s = gen_slot(a, 1);
                if (!s) continue;
                uint64_t ci = a >> GEN_CHUNK_SHIFT;
                if (!own_chunk[ci] && !(own_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t))))
                    rt_fatal("vk: out of memory for page generations");
                uint32_t *o = &own_chunk[ci][(a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)], prior = *s;
                stamp_page(a, g);
                *o = prior <= mgen || prior == *o ? g : 0;
            }
        }
        plat_mutex_unlock(&gen_lock);
        plat_mutex_unlock_shared(&pool_lock);
    }
}

// ---- lazy write-back (TOTK_WB_LAZY=1, off by default; Linux)
// Most of what a frame's fences copy back to guest memory is never read there: of the 15 MiB a frame in the cave after
// Bananza's tutorial, 14 are a thousand storage buffers that compute shaders write and later GPU work reads. With the
// lazy write-back on, a range the replay marked VK_WB_LAZY is not copied at its fence. Its pages are sealed instead
// (plat_mem_seal: no access), with the byte ranges owed to each page kept here, and the first access to a sealed page,
// a read or a write from any thread, copies that page's ranges from the shadow and opens it (wb_fault, from the
// platform's fault handler). So the guest sees the same bytes as with the copy at the fence, later GPU writes to a
// page still sealed cost nothing, and a page somebody touches is not sealed again for a while: one second after the
// first touch, doubling with each one up to about a minute, so a buffer the game reads every frame goes back to the
// copy at the fence after one fault.
// - What a sealed page owes is read from the shadow at the touch, not at the fence: the bytes may be those of a later
//   GPU write than the one whose fence sealed the page, as on the console, where the two share the memory.
// - A page's own lock (wb_lock) is a leaf: the fault handler takes nothing else, and nothing touches guest memory
//   under it except the copy into a page it has just opened. Regions with sealed pages are never frozen
//   (mem_freeze_exclude), so that copy cannot fault.
// - A kernel write into guest memory fails on a sealed page instead of faulting; file reads thaw their buffer first
//   (mem_thaw), and mem_thaw opens the sealed pages of its range (wb_open, through mem_thaw_hook). So do the checks
//   that ask whether a range changed (vk_mem_changed, vk_mem_changed_snap): their caller reads the range next.
// - A sealed page is not stamped as written at the fence: its guest bytes did not change. The copy at the touch writes
//   them, which the write watch reports like any other write.
// - Not covered: a kernel read from a sealed page (the game passing such a buffer to a file write), which fails. Only
//   storage buffers written by shaders are ever sealed.
#define WB_SEGS 6
enum { WB_EMPTY, WB_SEALED, WB_OPEN };
typedef struct {
    uint64_t page;
    const uint8_t *src;          // the shadow's bytes of this page, while sealed
    uint32_t open_until;         // WB_OPEN: plat_now_ms before which the page is not sealed again
    uint16_t seg[WB_SEGS][2];    // WB_SEALED: byte ranges [lo, hi) of the page owed to it, merged and sorted
    uint8_t nseg, state, backoff, retries;
} WbPage;
static PlatMutex wb_lock = PLAT_MUTEX_INIT;   // leaf
static WbPage *wbp;                           // wb_lock: open addressing by page
static uint32_t wbp_cap, wbp_used;            // wb_lock
static volatile int32_t wb_nsealed;           // pages sealed now
static int wb_on = -1;
static uint64_t wb_seals, wb_touches, wb_skipped, wb_touch_bytes;   // wb_lock: since the last log

static uint32_t wb_slot(uint64_t page) { return (uint32_t)(((page >> 12) * 0x9E3779B97F4A7C15ull) >> 40); }

// wb_lock held. NULL when the page has no entry and insert is 0.
static WbPage *wb_find(uint64_t page, int insert) {
    if (!wbp_cap) {
        if (!insert) return NULL;
        wbp_cap = 1u << 14;
        wbp = calloc(wbp_cap, sizeof *wbp);
        if (!wbp) rt_fatal("vk: out of memory for the lazy write-back's pages");
    }
    for (uint32_t at = wb_slot(page) & (wbp_cap - 1);; at = (at + 1) & (wbp_cap - 1)) {
        if (wbp[at].state != WB_EMPTY && wbp[at].page == page) return &wbp[at];
        if (wbp[at].state == WB_EMPTY) {
            if (!insert) return NULL;
            if (wbp_used * 2 >= wbp_cap) break;
            wbp_used++;
            wbp[at] = (WbPage){.page = page};
            return &wbp[at];
        }
    }
    WbPage *old = wbp;   // half full: twice the table (not in the fault handler, which never inserts)
    uint32_t cap = wbp_cap;
    wbp_cap *= 2, wbp_used = 0;
    wbp = calloc(wbp_cap, sizeof *wbp);
    if (!wbp) rt_fatal("vk: out of memory for the lazy write-back's pages");
    for (uint32_t i = 0; i < cap; i++)
        if (old[i].state != WB_EMPTY) {
            WbPage *e = wb_find(old[i].page, 1);
            uint64_t pg = e->page;
            *e = old[i], e->page = pg;
        }
    free(old);
    return wb_find(page, insert);
}

// wb_lock held: the page gets what it is owed and is open for a while.
static void wb_open_page(WbPage *e) {
    if (e->state != WB_SEALED) return;
    plat_mem_seal(e->page, 0x1000, 0);
    for (int i = 0; i < e->nseg; i++) {
        memcpy((void *)(uintptr_t)(e->page + e->seg[i][0]), e->src + e->seg[i][0],
               (size_t)(e->seg[i][1] - e->seg[i][0]));
        wb_touch_bytes += (uint64_t)(e->seg[i][1] - e->seg[i][0]);
    }
    e->state = WB_OPEN, e->nseg = 0, e->retries = 0;
    e->open_until = (uint32_t)plat_now_ms() + (1000u << e->backoff);
    if (e->backoff < 6) e->backoff++;
    wb_touches++;
    plat_dec32(&wb_nsealed);
}

// The platform's access hook: any thread, in its fault handler. 1 retries the access.
static int wb_fault(uint64_t addr) {
    if (!__atomic_load_n(&wb_nsealed, __ATOMIC_ACQUIRE) && !wbp_cap) return 0;
    int handled = 0;
    plat_mutex_lock(&wb_lock);
    WbPage *e = wb_find(addr & ~0xfffull, 0);
    if (e && e->state == WB_SEALED) wb_open_page(e), handled = 1;
    // Another thread opened the page between this one's fault and its turn here: the access only needs its retry. A
    // fault of another kind on a page that was sealed once gets a few retries, then goes on to the crash handler.
    // The page is opened again on the way: an access must never find a page of ours closed with nothing owed.
    else if (e && e->state == WB_OPEN && e->retries < 8) plat_mem_seal(e->page, 0x1000, 0), e->retries++, handled = 1;
    plat_mutex_unlock(&wb_lock);
    return handled;
}

// Opens the sealed pages of [a, a+n): before a kernel write there, before a caller reads the range for a comparison,
// and before a pool's shadow goes away.
static void wb_open(uint64_t a, uint64_t n) {
    if (!n || !__atomic_load_n(&wb_nsealed, __ATOMIC_ACQUIRE)) return;
    uint64_t lo = a & ~0xfffull, hi = (a + n + 0xfff) & ~0xfffull;
    plat_mutex_lock(&wb_lock);
    if ((hi - lo) >> 12 > wbp_cap) {   // a large range: the table is the shorter walk
        for (uint32_t i = 0; i < wbp_cap; i++)
            if (wbp[i].state == WB_SEALED && wbp[i].page >= lo && wbp[i].page < hi) wb_open_page(&wbp[i]);
    } else {
        for (uint64_t pg = lo; pg < hi; pg += 0x1000) {
            WbPage *e = wb_find(pg, 0);
            if (e) wb_open_page(e);
        }
    }
    plat_mutex_unlock(&wb_lock);
}

static int wb_lazy_on(void) {
    if (wb_on < 0) {
        wb_on = shadow_on && cfg_bool("TOTK_WB_LAZY") && plat_mem_access_hook(wb_fault);
        if (wb_on) mem_thaw_hook(wb_open);
        if (cfg_bool("TOTK_WB_LAZY"))
            rt_log("mem: lazy write-back %s (TOTK_WB_LAZY=1)",
                   wb_on ? "on: a storage buffer a shader wrote is copied back to guest memory at the first CPU access "
                           "to its pages"
                         : "not available here: every GPU write is copied back at its fence");
    }
    return wb_on;
}

// wb_lock held: [lo, hi) of the guest page e->page joins what the page is owed. 0 when the page has no room for it.
static int wb_owe(WbPage *e, uint32_t lo, uint32_t hi) {
    int i = 0;
    while (i < e->nseg && e->seg[i][1] < lo) i++;
    if (i < e->nseg && e->seg[i][0] <= hi) {   // touches or overlaps seg i, and maybe the ones after it
        if (lo < e->seg[i][0]) e->seg[i][0] = (uint16_t)lo;
        int j = i;
        while (j + 1 < e->nseg && e->seg[j + 1][0] <= hi) j++;
        if (e->seg[j][1] > hi) hi = e->seg[j][1];
        e->seg[i][1] = (uint16_t)hi;
        memmove(&e->seg[i + 1], &e->seg[j + 1], (size_t)(e->nseg - 1 - j) * sizeof e->seg[0]);
        e->nseg = (uint8_t)(e->nseg - (j - i));
        return 1;
    }
    if (e->nseg == WB_SEGS) return 0;
    memmove(&e->seg[i + 1], &e->seg[i], (size_t)(e->nseg - i) * sizeof e->seg[0]);
    e->seg[i][0] = (uint16_t)lo, e->seg[i][1] = (uint16_t)hi, e->nseg++;
    return 1;
}

// One submission's GPU writes. A heavy Bananza frame reports about 4,400 of them (a thousand 15,000-byte storage
// buffers among them, 15 MiB in all), and one vk_gpu_written per range cost the completion thread 6.7 ms a frame in the
// cave after the tutorial: every call searched the whole pending list, took four locks, and its copy faulted once per
// write-watched page. The game's threads wait for these fences, so that time was frame time. Here the ranges are
// sorted once, their pages are thawed and marked written in runs (plat_mem_prewrite), the copies run under one hold of
// the locks, and one pass over the pending list removes them. With the lazy write-back, the pages of a VK_WB_LAZY
// range that nobody touched lately are sealed in place of their copy, and a batch of fewer than 4 ranges goes
// through vk_gpu_written_into one by one. Without the shadow storage the ranges are stamped in batches.
typedef struct { uint64_t lo, hi; const uint8_t *src; } WbCopy;   // src: the shadow's byte for guest address lo

static int wb_cmp(const void *x, const void *y) {
    const uint64_t *a = x, *b = y;
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : a[1] > b[1];
}

void vk_gpu_written_many(const VkGpuWrite *in, int n) {
    if (!shadow_on) {
        written_batch(in, n);
        return;
    }
    if (n < 4) {
        for (int i = 0; i < n; i++) vk_gpu_written_into(in[i].addr, in[i].size & ~VK_WB_LAZY, in[i].buf);
        return;
    }
    int lazy = wb_lazy_on();
    uint64_t (*w)[3] = malloc((size_t)n * sizeof *w);   // {lo, hi, lazy}
    uint8_t *gone = calloc((size_t)n, 1);
    WbCopy *cp = NULL;
    uint64_t *seal = NULL;
    int ncp = 0, ccp = 0, nseal = 0, cseal = 0;
    if (!w || !gone) rt_fatal("vk: out of memory for a write-back batch");
    int m = 0;
    for (int i = 0; i < n; i++) {
        uint64_t size = in[i].size & ~VK_WB_LAZY;
        if (!size) continue;
        w[m][0] = in[i].addr, w[m][1] = in[i].addr + size, w[m][2] = lazy && (in[i].size & VK_WB_LAZY);
        if (w[m][2]) mem_freeze_exclude(w[m][0], size);   // before any lock here: it takes memory.c's
        m++;
    }
    qsort(w, (size_t)m, sizeof *w, wb_cmp);
    pool_lock_shared();
    plat_mutex_lock(&shadow_lock);
    // Which bytes are copied now (cp) and which pages are sealed (seal, ascending as w is).
    PoolRec *r = NULL;
    uint32_t now = lazy ? (uint32_t)plat_now_ms() : 0;
    if (lazy) plat_mutex_lock(&wb_lock);
    for (int i = 0; i < m; i++) {
        uint64_t a = w[i][0], b = w[i][1];
        if (!r || a < r->import_lo || b > r->import_hi) {   // sorted: the last pool usually holds the next range
            const pidx_ent *e = pidx_find(&pool_index, a, b - a);
            r = e ? e->buf : NULL;
        }
        if (!r || !r->smap || a < r->import_lo || b > r->import_hi) continue;
        const uint8_t *src = r->smap + (a - r->import_lo);
        if (!w[i][2]) {
            if (ncp == ccp) grow((void **)&cp, &ccp, sizeof *cp);
            cp[ncp++] = (WbCopy){a, b, src};
            continue;
        }
        for (uint64_t pg = a & ~0xfffull; pg < b; pg += 0x1000) {
            uint64_t lo = pg > a ? pg : a, hi = pg + 0x1000 < b ? pg + 0x1000 : b;
            WbPage *e = wb_find(pg, 1);
            const uint8_t *psrc = r->smap + (pg - r->import_lo);
            if (e->state == WB_SEALED && e->src != psrc) wb_open_page(e);   // the page's pool changed under it
            int sealed = e->state == WB_SEALED && wb_owe(e, (uint32_t)(lo - pg), (uint32_t)(hi - pg));
            // too many pieces: copied from now on
            if (e->state == WB_SEALED && !sealed) wb_open_page(e), e->backoff = 6;
            if (!sealed && (e->state == WB_EMPTY || (int32_t)(now - e->open_until) >= 0)) {
                e->state = WB_SEALED, e->src = psrc, e->nseg = 0, e->retries = 0;
                plat_inc32(&wb_nsealed);
                wb_owe(e, (uint32_t)(lo - pg), (uint32_t)(hi - pg));
                if (!nseal || seal[nseal - 1] != pg) {
                    if (nseal == cseal) grow((void **)&seal, &cseal, sizeof *seal);
                    seal[nseal++] = pg;
                }
                sealed = 1;
            }
            if (sealed) { wb_skipped += hi - lo; continue; }
            if (ncp == ccp) grow((void **)&cp, &ccp, sizeof *cp);
            cp[ncp++] = (WbCopy){lo, hi, src + (lo - a)};
        }
    }
    if (lazy) {
        // A page sealed above and opened again by a later range of this batch (too many pieces) stays open.
        int live = 0;
        for (int i = 0; i < nseal; i++) {
            WbPage *e = wb_find(seal[i], 0);
            if (e && e->state == WB_SEALED) seal[live++] = seal[i];
        }
        nseal = live;
        for (int i = 0; i < nseal;) {   // one protection change per run of new pages
            int j = i + 1;
            while (j < nseal && seal[j] == seal[j - 1] + 0x1000) j++;
            if (plat_mem_seal(seal[i], (uint64_t)(j - i) << 12, 1)) wb_seals += (uint64_t)(j - i);
            else   // refused (the kernel's limit of mappings, say): these pages are copied now and left alone for good
                for (int k = i; k < j; k++) {
                    WbPage *e = wb_find(seal[k], 0);
                    wb_open_page(e), e->backoff = 6, e->open_until = now + (1u << 30);
                }
            i = j;
        }
        plat_mutex_unlock(&wb_lock);
    }
    // The copies: neighbours and pieces on one page become one thaw and one prewrite. A sealed page among them (a
    // piece that is not lazy on a page another range sealed) is opened by mem_thaw's hook.
    for (int i = 0; i < ncp;) {
        uint64_t lo = cp[i].lo & ~0xfffull, hi = (cp[i].hi + 0xfff) & ~0xfffull;
        int j = i + 1;
        for (; j < ncp && (cp[j].lo & ~0xfffull) <= hi; j++)
            if (((cp[j].hi + 0xfff) & ~0xfffull) > hi) hi = (cp[j].hi + 0xfff) & ~0xfffull;
        mem_thaw(lo, hi - lo);
        plat_mem_prewrite(lo, hi - lo);
        i = j;
    }
    for (int i = 0; i < ncp; i++) {
        memcpy((void *)(uintptr_t)cp[i].lo, cp[i].src, (size_t)(cp[i].hi - cp[i].lo));
        s_wb_bytes += cp[i].hi - cp[i].lo;
    }
    // Each range leaves the pending list once, as vk_gpu_written removes one entry per call: equal ranges are
    // neighbours in w, and gone[] marks the ones that found their entry.
    int kept = 0, removed = 0;
    for (int i = 0; i < nspend; i++) {
        uint64_t key[2] = {spend[i].lo, spend[i].hi};
        int lo = 0, hi = m;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (wb_cmp(w[mid], key) < 0) lo = mid + 1;
            else hi = mid;
        }
        while (lo < m && gone[lo] && !wb_cmp(w[lo], key)) lo++;
        if (removed < m && lo < m && !wb_cmp(w[lo], key)) gone[lo] = 1, removed++;
        else spend[kept++] = spend[i];
    }
    if (kept != nspend) nspend = kept, spend_dirty = 1;
    plat_mutex_unlock(&shadow_lock);
    plat_mutex_unlock_shared(&pool_lock);
    plat_mutex_lock(&gen_lock);
    uint32_t g = ++gen_now;
    // Every range of the batch is stamped as written, less the pages this batch sealed: a sealed page keeps its old
    // guest bytes, so it is stamped by the copy that opens them instead. w and seal are both ascending.
    for (int i = 0, s = 0; i < m; i++) {
        if (nvn_gpu_only(w[i][0])) continue;
        for (uint64_t a = w[i][0] & ~0xfffull, hi = (w[i][1] + 0xfff) & ~0xfffull; a < hi;
             a += 1ull << GEN_PAGE_SHIFT) {
            while (s < nseal && seal[s] < a) s++;
            if (s < nseal && seal[s] == a) continue;   // sealed: its stamp waits for the copy that opens it
            stamp_page(a, g);
        }
    }
    plat_mutex_unlock(&gen_lock);
    free(seal);
    free(cp);
    free(gone);
    free(w);
}

// As vk_gpu_written, for a write the GPU made into buffer buf. When buf is the VRAM mirror of the pool holding the range,
// the mirror has the GPU's bytes, and vk_mem_mirror_end copies them into guest memory (or vk_mem_gpu_kept keeps them in
// the mirror). The stamp still tells every other reader of guest memory (the texture checks), but the pool's next
// mirror_sync need not copy those pages in again. In Hilltop those copies were about 57 MiB of the 58 MiB a frame the
// syncs uploaded, and the copy of a kept page put stale guest bytes over the GPU's. So such a stamp
// is noted per page (own_chunk) when the mirror already holds every older write of the page: its previous stamp is
// covered by the mirror's mgen or is itself such a stamp. A page with a newer write the mirror lacks fails that test, and
// a CPU write stamped later gives the page a newer generation, so mirror_sync copies both as before.
void vk_gpu_written_into(uint64_t addr, uint64_t size, VkBuffer buf) {
    if (!buf || !vk_mirror_on || shadow_on || !size || nvn_gpu_only(addr)) {
        vk_gpu_written(addr, size);
        return;
    }
    pool_lock_shared();
    const pidx_ent *e = pidx_find(&pool_index, addr, size);
    PoolRec *r = e ? e->buf : NULL;
    if (!r || ms_get(r) != MS_ON || r->mbuf != buf || addr < r->import_lo || addr + size > r->import_hi) {
        plat_mutex_unlock_shared(&pool_lock);
        vk_guest_written(addr, size);
        return;
    }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    uint32_t g = ++gen_now, mgen = r->mgen;
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 1);
        if (!s) continue;
        uint64_t ci = a >> GEN_CHUNK_SHIFT;
        if (!own_chunk[ci] && !(own_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t))))
            rt_fatal("vk: out of memory for page generations");
        uint32_t *o = &own_chunk[ci][(a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)], prior = *s;
        stamp_page(a, g);
        *o = prior <= mgen || prior == *o ? g : 0;
    }
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
}

static void shadow_log(uint64_t frame) {
    if (!shadow_on) return;
    plat_mutex_lock(&shadow_lock);
    rt_log("mem: shadow at frame %llu: %lld pools %.1f MiB; since the last: %llu pool syncs, %llu queries, %.1f MiB "
           "copied in (%llu untracked), %.1f MiB written back, %.2f ms, %d GPU writes pending",
           (unsigned long long)frame, (long long)s_live_n, s_live_bytes / 1048576.0, (unsigned long long)s_syncs,
           (unsigned long long)s_queries, s_copied / 1048576.0, (unsigned long long)s_untracked, s_wb_bytes / 1048576.0,
           qpc_ms(s_ticks), nspend);
    s_syncs = s_queries = s_copied = s_untracked = s_wb_bytes = s_ticks = 0;
    plat_mutex_unlock(&shadow_lock);
    if (wb_on > 0) {
        plat_mutex_lock(&wb_lock);
        rt_log("mem: lazy write-back at frame %llu: %d pages sealed; since the last: %.1f MiB left for a touch, %llu "
               "pages sealed, %llu touched (%.1f MiB copied then)", (unsigned long long)frame, (int)wb_nsealed,
               wb_skipped / 1048576.0, (unsigned long long)wb_seals, (unsigned long long)wb_touches,
               wb_touch_bytes / 1048576.0);
        wb_skipped = wb_seals = wb_touches = wb_touch_bytes = 0;
        plat_mutex_unlock(&wb_lock);
    }
}

void vk_mem_dirty_submit(void) {
    if (!(rt_trace & RT_TRACE_PERF) || vk_mirror_on) return;   // the mirror's own line replaces the census
    uint64_t t0 = kern_now_ns();
    uint32_t prev = dirty_sub;   // the submit whose replay noted the bindings
    dirty_sub = prev + 1 ? prev + 1 : 1;
    uint64_t eager[DC_N] = {0}, lazy[DC_N] = {0}, ln[DC_N] = {0}, lb[DC_N] = {0}, untracked = 0, queries = 0;
    pool_lock_shared();
    if (dcap < pool_index.n) {
        dcap = pool_index.n * 2;
        dlist = realloc(dlist, (size_t)dcap * sizeof *dlist);
        if (!dlist) rt_fatal("vk: out of memory for the dirty census");
    }
    int n = 0;
    for (int i = 0; i < pool_index.n; i++) {
        PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT && r->bufbound) dlist[n++] = r;
    }
    gen_lock_x();
    uint32_t g = ++gen_now, since = dirty_gen;
    dirty_gen = g;
    for (int i = 0; i < n;) {   // one query per run of pools (sorted by address)
        uint64_t lo = page_lo(dlist[i]), hi = page_hi(dlist[i]);
        int j = i + 1;
        for (; j < n && page_lo(dlist[j]) <= hi + DIRTY_GAP; j++)
            if (page_hi(dlist[j]) > hi) hi = page_hi(dlist[j]);
        queries++;
        int ok = ww_stamp(lo, hi, g, WW_DIRTY);
        for (int k = i; k < j; k++) {
            dlist[k]->untracked = 0;
            if (!ok) queries++, dlist[k]->untracked = (uint8_t)!ww_stamp(page_lo(dlist[k]), page_hi(dlist[k]), g, WW_DIRTY);
        }
        i = j;
    }
    uint64_t covered = 0;
    for (int i = 0; i < n; i++) {
        PoolRec *r = dlist[i];
        int k = dc_class(r);
        ln[k]++, lb[k] += r->size;
        int fresh = r->synced_gen == DIRTY_UNSYNCED, lz = !fresh && r->bound_sub == prev;
        uint32_t ls = r->synced_gen;
        if (fresh || r->bound_sub == prev) r->synced_gen = g;
        if (r->untracked) { untracked += r->size; continue; }
        if (fresh) continue;   // counted as first use
        uint64_t lo = page_lo(r), hi = page_hi(r), ne = 0, nl = 0;
        if (lo < covered) lo = covered;   // overlapping pools: each page once
        if (lo >= hi) continue;
        covered = hi;
        for (uint64_t a = lo; a < hi;) {
            uint64_t ci = a >> GEN_CHUNK_SHIFT, ce = (ci + 1) << GEN_CHUNK_SHIFT;
            if (ce > hi) ce = hi;
            const uint32_t *ch = ci < GEN_CHUNKS ? gen_chunk[ci] : NULL;
            if (ch)
                for (uint64_t x = a; x < ce; x += 1ull << GEN_PAGE_SHIFT) {
                    uint32_t v = ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
                    ne += v > since;
                    nl += lz && v > ls;
                }
            a = ce;
        }
        eager[k] += ne << GEN_PAGE_SHIFT, lazy[k] += nl << GEN_PAGE_SHIFT;
    }
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    uint64_t tot = 0;
    for (int k = 0; k < DC_N; k++) tot += eager[k];
    plat_mutex_lock(&dirty_lock);
    for (int k = 0; k < DC_N; k++) {
        dfr.eager[k] += eager[k], dfr.lazy[k] += lazy[k], dfr.first[k] += first_pending[k];
        first_pending[k] = 0;
        live_n[k] = ln[k], live_bytes[k] = lb[k], ever_snap_n[k] = ever_n[k], ever_snap_bytes[k] = ever_bytes[k];
    }
    live_untracked = untracked;
    dfr.submits++, dfr.queries += queries;
    if (tot > dfr.sub_max) dfr.sub_max = tot;
    dfr.scan_ns += kern_now_ns() - t0;
    plat_mutex_unlock(&dirty_lock);
}

static void dirty_frame(uint64_t frame) {
    plat_mutex_lock(&dirty_lock);
    if (dwin.frames < DIRTY_WIN)
        for (int k = 0; k < DC_N; k++) dwin_kib[dwin.frames][k] = (uint32_t)(dfr.eager[k] >> 10);
    for (int k = 0; k < DC_N; k++) dwin.eager[k] += dfr.eager[k], dwin.lazy[k] += dfr.lazy[k], dwin.first[k] += dfr.first[k];
    dwin.submits += dfr.submits, dwin.queries += dfr.queries, dwin.scan_ns += dfr.scan_ns, dwin.frames++;
    if (dfr.sub_max > dwin.sub_max) dwin.sub_max = dfr.sub_max;
    memset(&dfr, 0, sizeof dfr);
    if (frame % DIRTY_WIN) { plat_mutex_unlock(&dirty_lock); return; }
    DirtyAcc w = dwin;
    uint32_t kib[DIRTY_WIN][DC_N];
    memcpy(kib, dwin_kib, sizeof kib);
    uint64_t lv[DC_N], lvb[DC_N], ev[DC_N], evb[DC_N], unt = live_untracked;
    for (int k = 0; k < DC_N; k++) lv[k] = live_n[k], lvb[k] = live_bytes[k], ev[k] = ever_snap_n[k], evb[k] = ever_snap_bytes[k];
    memset(&dwin, 0, sizeof dwin);
    plat_mutex_unlock(&dirty_lock);
    if (!w.frames) return;
    uint64_t fl[2][3][2];
    nvn_flush_stats(fl);
    double f = (double)w.frames, M = 1048576.0;
    uint64_t sub_tot = 0;
    for (int k = 0; k < DC_N; k++) sub_tot += w.eager[k];
#define C3(a, d) (a)[0] / (d), (a)[1] / (d), (a)[2] / (d)
    rt_log("mem: dirty frame %llu: frames %llu submits_per_frame %.2f; per frame, cached/uncached/other: eager_mib "
           "%.3f/%.3f/%.3f lazy_mib %.3f/%.3f/%.3f first_mib %.3f/%.3f/%.3f flush_pool_calls %.1f/%.1f/%.1f "
           "flush_pool_mib %.3f/%.3f/%.3f flush_buf_calls %.1f/%.1f/%.1f flush_buf_mib %.3f/%.3f/%.3f; per submit: "
           "avg_mib %.3f max_mib %.3f; mirror: live_pools %llu/%llu/%llu live_mib %.1f/%.1f/%.1f ever_pools %llu/%llu/%llu "
           "ever_mib %.1f/%.1f/%.1f untracked_mib %.1f; cost: queries_per_frame %.1f scan_ms_per_frame %.3f",
           (unsigned long long)frame, (unsigned long long)w.frames, w.submits / f,
           C3(w.eager, f * M), C3(w.lazy, f * M), C3(w.first, f * M),
           fl[0][0][0] / f, fl[0][1][0] / f, fl[0][2][0] / f, fl[0][0][1] / f / M, fl[0][1][1] / f / M, fl[0][2][1] / f / M,
           fl[1][0][0] / f, fl[1][1][0] / f, fl[1][2][0] / f, fl[1][0][1] / f / M, fl[1][1][1] / f / M, fl[1][2][1] / f / M,
           w.submits ? sub_tot / (double)w.submits / M : 0.0, w.sub_max / M,
           (unsigned long long)lv[0], (unsigned long long)lv[1], (unsigned long long)lv[2], C3(lvb, M),
           (unsigned long long)ev[0], (unsigned long long)ev[1], (unsigned long long)ev[2], C3(evb, M), unt / M,
           w.queries / f, w.scan_ns / 1e6 / f);
#undef C3
    char line[DIRTY_WIN * 36 + 64];
    size_t k = (size_t)snprintf(line, sizeof line, "mem: dirty frames KiB cached/uncached/other:");
    for (uint64_t i = 0; i < w.frames && i < DIRTY_WIN; i++)
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, " %u/%u/%u", kib[i][0], kib[i][1], kib[i][2]);
    rt_log("%s", line);
}

// ---------------------------------------------------------------- VRAM mirror of buffer-bound pools
// An imported pool keeps vertices, indices, constants
// and storage buffers in system memory, so a discrete GPU reads them over PCIe on every draw (GTX 1050 Ti and RTX
// 3090 Ti: about 110 ms of GPU time per frame in the field). In mirror mode (device class discrete, TOTK_MEMORY) an
// imported pool gets a device-local buffer over its imported range the first time the replay uses it as a vertex,
// index, uniform, storage or indirect buffer (vk_mem_use). From then on those bindings, and the buffer copies, fills
// and uniform updates that touch the pool (vk_mem_pick), use the mirror at the same offsets. Guest memory stays what
// the CPU reads and writes; textures and every host path that reads or writes pool bytes keep using it.
// - CPU -> mirror: the first access of a submission to a mirror syncs it (mirror_sync): the 2 MiB write-watch
//   snapshots of its regions for this epoch (snap_regions, shared with the texture checks: the OS bits keep one
//   consumer, the shared page generations) and every page stamped after the mirror's mgen goes into a copy, unless
//   that stamp reported a GPU write the mirror already holds (vk_gpu_written_into); so do the
//   nvnMemoryPoolFlushMappedRange ranges. The first use copies the whole range. The copies run in a prologue command
//   buffer that vk_sync.c submits right before the part that accessed the mirror first, between full barriers.
//   vk_mem_mirror_begin snaps, in a few merged queries, the regions of the pools the next submission is likely to
//   access (presnap_wanted): those accessed every P submissions, for P up to TOTK_MIRROR_PRESNAP.
// - GPU -> guest: what the replay writes into a mirror (storage buffers its shaders write, buffer copies, fills,
//   uniform updates, texture-to-buffer copies: vk_mem_gpu_wrote) goes back into the imported buffer at the end of the
//   submission, exact byte ranges, before the fence signals (vk_mem_mirror_end).
// - TOTK_MEMORY=verify: every submission also copies up to VERIFY_BYTES of mirror pages, round robin, into host memory;
//   at its fence the pages the CPU has not written since their sync are compared with guest memory.
// A pool stays imported when its mirror would not fit the VRAM budget, its range is not write-watched, it overlaps
// another pool or the allocation fails.
#define VERIFY_BYTES (4u << 20)
#define VERIFY_SAMPLES 64
#define VERIFY_LOG 16
enum { MR_BUDGET, MR_UNTRACKED, MR_OVERLAP, MR_ALLOC, MR_N };

int vk_mirror_on;
static int mirror_verify;
static int mirror_sparse;
static int mirror_gap = 2;             // pre-snap: hot pools at most this many 2 MiB regions apart share one query
static int mirror_presnap = 8;         // pre-snap: longest submission period predicted (TOTK_MIRROR_PRESNAP, presnap_wanted)
static int mirror_own = 1;             // TOTK_MIRROR_OWN: mirror_sync skips the pages own_chunk notes (rt_knob mode)
static PoolRec **mlist;                // pool_lock: mirrored pools by import_lo
static int nm, mcap;
static uint32_t msub;                  // replay: number of the submission being recorded
// The write-back tag of this thread's GPU writes (vk_mem_wb_tag): the place in the walk of the part that records them, so
// a held part can write back the ranges its own commands and the parts before it wrote (vk_mem_mirror_wb_held).
static _Thread_local uint32_t wb_tag;

uint32_t vk_mem_wb_tag(uint32_t tag) {
    uint32_t old = wb_tag;
    wb_tag = tag;
    return old;
}
static uint32_t mepoch;                // replay: its write-watch snapshot epoch (vk_texture_epoch)
typedef struct { PoolRec *r; uint64_t off, size; uint32_t tag; } MRange;   // tag: a write-back's (vk_mem_wb_tag)
static MRange *pend, *wbl;             // replay: copies for the next prologue; GPU writes into mirrors this submission
static int npend, cpend, nwbl, cwbl;
static _Thread_local VkBufferCopy *mregions;   // vkCmdCopyBuffer regions: the replay thread's and the write-back thread's
static _Thread_local int cregions;
static PlatMutex flush_lock = PLAT_MUTEX_INIT;
// The replay thread and its helper (vk_spec.h) both reach mirrors for the first time in a submission: a pool's sync and
// the list of pending copies are theirs one at a time. A pool's acc_sub is stored last, so the thread that reads it
// without the lock (vk_mem_use) sees a pool as synced only once its copies are queued.
static PlatMutex mirror_lock = PLAT_MUTEX_INIT;
// Takes mirror_lock only while helper threads exist, so with TOTK_REPLAY_PAR=0 (the default) the replay thread works
// the mirrors alone and takes no lock, as before the helpers. Returns whether it took the lock, for mirror_unlock:
// vk_spec_count is set once, by the replay thread, so a caller reads it once and pairs the two on that reading.
static int mem_workers;   // vk_mem_threads: the workers of sliced passes (vk_exec.c) bind mirrors too
// TOTK_MEMORY=verify reads r->kept on the completion thread (kept_page), so it always takes the lock.
static int mirror_shared(void) {
    return mirror_verify || vk_spec_count() || __atomic_load_n(&mem_workers, __ATOMIC_ACQUIRE);
}
static int mirror_lock_if_shared(void) {
    if (!mirror_shared()) return 0;
    if (plat_mutex_trylock(&mirror_lock)) return 1;
    uint64_t t0 = kern_now_ns();   // contended: TOTK_TRACE=perf counts the long waits (perf_lock_wait)
    plat_mutex_lock(&mirror_lock);
    if (rt_trace & RT_TRACE_PERF) perf_lock_wait(PERF_LOCK_MIRROR, kern_now_ns() - t0);
    return 1;
}
static void mirror_unlock(int locked) {
    if (locked) plat_mutex_unlock(&mirror_lock);
}

typedef struct {
    uint64_t subs, syncs, upload_bytes, upload_copies, first_n, first_bytes, wb_bytes, wb_copies, wb_ranges;
    uint64_t presnap_queries, sync_queries, presnap_t, sync_t, prologues, presnap_bytes, own_bytes;
    uint64_t sync_bytes, presnap_pools, presnap_used, presnap_unused_bytes;
    uint64_t early_n, early_bytes;   // the held parts' write-backs (wb_job_run): command buffers submitted, their bytes
} MirrorAcc;
static MirrorAcc mrep;                 // replay: this submission
static MirrorAcc macc;                 // mirror_stat_lock: since the last summary
static PlatMutex mirror_stat_lock = PLAT_MUTEX_INIT;
static volatile int64_t m_live_n, m_live_bytes, m_refused_n[MR_N], m_refused_bytes[MR_N];
static volatile int64_t m_flush_calls, m_flush_bytes, v_pages, v_skipped, v_mismatch;

static void sparse_cache_open(void);

static void mirror_mode(int discrete, uint32_t vendor) {
    const char *e = cfg_value("TOTK_MEMORY");
    int mode = discrete ? 1 : 0;   // 0 import, 1 mirror, 2 verify
    const char *why = discrete ? "auto, discrete gpu" : "auto, uma";
    if (strcmp(e, "auto")) {
        if (!strcmp(e, "import")) mode = 0, why = "TOTK_MEMORY=import";
        else if (!strcmp(e, "mirror")) mode = 1, why = "TOTK_MEMORY=mirror";
        else if (!strcmp(e, "verify")) mode = 2, why = "TOTK_MEMORY=verify";
        else rt_log("mem: TOTK_MEMORY=%s is not import, mirror, verify or auto: auto", e);
    }
    vk_mirror_on = mode > 0;
    mirror_verify = mode == 2;
    // The sparse write-back dispatch (sparse_ranges): by default on NVIDIA, the one vendor it was measured on;
    // TOTK_SPARSE_MIRROR_COPY=1 turns it on elsewhere, 0 off.
    const char *sparse = cfg_get("TOTK_SPARSE_MIRROR_COPY");
    mirror_sparse = vk_mirror_on && (sparse && *sparse ? strcmp(sparse, "0") != 0 : vendor == 0x10de);
    if (vk_mirror_on)
        rt_log("mem: small write-back ranges are copied by %s (TOTK_SPARSE_MIRROR_COPY=%s)",
               mirror_sparse ? "one compute dispatch a pool" : "copy regions", sparse && *sparse ? sparse : "auto");
    if (mirror_sparse) sparse_cache_open();
    if (vk_mirror_on) {
        mirror_own = rt_knob("MIRROR_OWN", "1");
        if (mirror_own != 1)
            rt_log("mem: pages the mirrors' own GPU writes stamped are copied in again%s (TOTK_MIRROR_OWN=%s)",
                   mirror_own == 2 ? " in the A blocks" : "", mirror_own == 2 ? "ab" : "0");
    }
    const char *g = cfg_get("TOTK_MIRROR_GAP");
    mirror_gap = g && *g ? atoi(g) : 2;
    if (mirror_gap < 0) mirror_gap = 0;
    const char *ps = cfg_get("TOTK_MIRROR_PRESNAP");
    mirror_presnap = ps && *ps ? atoi(ps) : 8;
    if (mirror_presnap < 0) mirror_presnap = 0;
    // Submissions a region stays unwritten before it is frozen. 60 rather than 300: in a fight 100 of the 257 MiB
    // queried a frame lay in regions idle for 30-299 submissions, and a thaw by a write costs about 60 us.
    const char *fi = cfg_get("TOTK_FREEZE_IDLE");
    freeze_idle = fi && *fi ? atoi(fi) : 60;
    if (freeze_idle < 0) freeze_idle = 0;
    const char *sh = cfg_get("TOTK_SHADOW");
#if defined(_WIN32) || defined(__APPLE__)
    shadow_on = sh && *sh && strcmp(sh, "0");
#else
    shadow_on = !sh || !*sh || strcmp(sh, "0");
#endif
    if (shadow_on) rt_log("mem: shadow storage: CPU-visible pools are host-visible copies of guest memory, not imports%s",
                          sh && *sh ? " (TOTK_SHADOW)" : "");
    rt_log("mem: mode %s (%s): %s", mode == 2 ? "verify" : mode ? "mirror" : "import", why,
           mode ? "pools used as buffers get a device-local mirror" : "every CPU-visible pool is imported host memory");
}

static void range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size);
static void wb_range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size);

// The write-back ranges a thread recording slices of parallel passes adds (vk_mem_count_quiet on), in a list of its own
// that vk_mem_mirror_end moves into wbl: every slice of a submission is recorded before its end, so the lists are not
// written then. With eight threads adding to wbl under mirror_lock, a profile had 7% of the slices' time waiting for
// the lock in vk_mem_gpu_wrote.
// The kept GPU writes of the submission (vk_mem_gpu_kept), as whole pages, go into a second list of each thread's
// (k), which vk_mem_mirror_end merges into the pools' kept lists (kept_take).
// lock: the owner adds to a and the write-back thread takes a held part's ranges out of it (wb_job_run) while the owner
// records slices of a later pass; uncontended otherwise.
typedef struct WbLocal { MRange *a; int n, cap; MRange *k; int nk, ck; struct WbLocal *next; PlatMutex lock; } WbLocal;
static WbLocal *wb_locals;   // every thread's, added under mirror_lock and kept until exit
static _Thread_local WbLocal *wb_mine;

static WbLocal *wb_local(void) {
    if (!wb_mine) {
        WbLocal *w = calloc(1, sizeof *w);
        if (!w) rt_fatal("vk: out of memory");
        plat_mutex_lock(&mirror_lock);
        w->next = wb_locals;
        __atomic_store_n(&wb_locals, w, __ATOMIC_RELEASE);
        plat_mutex_unlock(&mirror_lock);
        wb_mine = w;
    }
    return wb_mine;
}

static void wb_local_add(PoolRec *r, uint64_t off, uint64_t size) {
    WbLocal *w = wb_local();
    plat_mutex_lock(&w->lock);
    wb_range_add(&w->a, &w->n, &w->cap, r, off, size);
    plat_mutex_unlock(&w->lock);
}

static void wb_locals_take(void) {   // replay thread, no slice being recorded
    plat_mutex_lock(&mirror_lock);
    for (WbLocal *w = wb_locals; w; w = w->next) {
        plat_mutex_lock(&w->lock);
        for (int i = 0; i < w->n; i++) {
            if (nwbl == cwbl) grow((void **)&wbl, &cwbl, sizeof *wbl);
            wbl[nwbl++] = w->a[i];
        }
        w->n = 0;
        plat_mutex_unlock(&w->lock);
    }
    plat_mutex_unlock(&mirror_lock);
}

// range_add for a write-back list: the range carries this thread's tag, and only a range of the same tag grows.
static void wb_range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size) {
    if (*n && (*a)[*n - 1].r == r && (*a)[*n - 1].off + (*a)[*n - 1].size == off && (*a)[*n - 1].tag == wb_tag) {
        (*a)[*n - 1].size += size;
        return;
    }
    if (*n == *cap) grow((void **)a, cap, sizeof **a);
    (*a)[(*n)++] = (MRange){r, off, size, wb_tag};
}

static void range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size) {
    if (*n && (*a)[*n - 1].r == r && (*a)[*n - 1].off + (*a)[*n - 1].size == off) {
        (*a)[*n - 1].size += size;
        return;
    }
    if (*n == *cap) grow((void **)a, cap, sizeof **a);
    (*a)[*n] = (MRange){r, off, size};
    __atomic_store_n(n, *n + 1, __ATOMIC_RELAXED);   // npend: vk_mem_mirror_pending reads it without mirror_lock
}

#define MRANGE_LESS(x, y) ((x)->r != (y)->r ? (uintptr_t)(x)->r < (uintptr_t)(y)->r : (x)->off < (y)->off)
MSORT_DEFINE(sort_ranges, MRange, MRANGE_LESS)   // by pool, then offset, without a call per comparison (msort.h)

// One dispatch for a pool's many small write-back ranges (TOTK_SPARSE_MIRROR_COPY, on by default on NVIDIA; 0 copies
// every range with vkCmdCopyBuffer). A copy region costs the GPU about 0.7 us whatever its size: in the Hilltop Layer of Surf Hills
// (3.2.0) 5,500 write-back ranges a frame of 870 bytes on average took 4.2 ms of an RTX 5090's GPU time as copy
// regions and 0.4 ms as one workgroup each. The transfer path keeps bulk copies, where it is faster (the mirror
// uploads, 80 KiB a range, took 4.5 ms as copies and 11 to 15 as a dispatch): the uploads, a pool's ranges over 4 KiB
// on average, and any single range over SPARSE_RANGE_MAX or not word-aligned, since one 32-lane workgroup copies a
// range. Also any unsupported device limit.
static struct {
    int tried;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout set;
    VkPipelineLayout layout;
    VkPipeline pipeline;
} sparse_copy;

typedef struct {
    uint32_t start, words;
} SparseRange;

typedef struct {
    VkBuffer buffer;
    VmaAllocation allocation;
    SparseRange *ranges;
    VkDeviceSize bytes;
    uint32_t used;
} SparseBatch;

enum { SPARSE_CACHE_COUNT = 8, SPARSE_CACHE_BYTES = 2 * 1024 * 1024 };
static struct {
    PlatMutex lock;
    SparseBatch *ready[SPARSE_CACHE_COUNT];
    VkDeviceSize bytes;
    int n, closed;
} sparse_cache;   // ready buffers have reached their submission's completion callback

static void sparse_cache_open(void) {
    plat_mutex_lock(&sparse_cache.lock);
    sparse_cache.closed = 0;
    plat_mutex_unlock(&sparse_cache.lock);
}

static SparseBatch *sparse_cache_take(VkDeviceSize bytes) {
    SparseBatch *batch = NULL;
    plat_mutex_lock(&sparse_cache.lock);
    int best = -1;
    for (int i = 0; i < sparse_cache.n; i++) {
        VkDeviceSize size = sparse_cache.ready[i]->bytes;
        if (size >= bytes && (best < 0 || size < sparse_cache.ready[best]->bytes)) best = i;
    }
    if (best >= 0) {
        batch = sparse_cache.ready[best];
        sparse_cache.ready[best] = sparse_cache.ready[--sparse_cache.n];
        sparse_cache.bytes -= batch->bytes;
    }
    plat_mutex_unlock(&sparse_cache.lock);
    if (batch) batch->used = 0;
    return batch;
}

static int sparse_copy_init(VkCtx *c) {
    if (sparse_copy.tried) return sparse_copy.pipeline != VK_NULL_HANDLE;
    sparse_copy.tried = 1;
    sparse_copy.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    if (!sparse_copy.push) return 0;
    VkDescriptorSetLayoutBinding bindings[3];
    for (uint32_t i = 0; i < 3; i++)
        bindings[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                     VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo set = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 3, .pBindings = bindings};
    if (vkCreateDescriptorSetLayout(c->device, &set, NULL, &sparse_copy.set) != VK_SUCCESS) return 0;
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                         .setLayoutCount = 1, .pSetLayouts = &sparse_copy.set,
                                         .pushConstantRangeCount = 1, .pPushConstantRanges = &push};
    if (vkCreatePipelineLayout(c->device, &layout, NULL, &sparse_copy.layout) != VK_SUCCESS) return 0;
    VkShaderModule shader;
    VkShaderModuleCreateInfo module = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                       .codeSize = sizeof mirror_sparse_spv, .pCode = mirror_sparse_spv};
    if (vkCreateShaderModule(c->device, &module, NULL, &shader) != VK_SUCCESS) return 0;
    VkComputePipelineCreateInfo pipeline = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = shader, .pName = "main"},
        .layout = sparse_copy.layout
    };
    VkResult result = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &pipeline, NULL, &sparse_copy.pipeline);
    vkDestroyShaderModule(c->device, shader, NULL);
    return result == VK_SUCCESS;
}

static SparseBatch *sparse_batch_create(VkCtx *c, int capacity) {
    VkDeviceSize bytes = (VkDeviceSize)capacity * sizeof(SparseRange);
    if (bytes > c->props.limits.maxStorageBufferRange || !sparse_copy_init(c)) return NULL;
    SparseBatch *cached = sparse_cache_take(bytes);
    if (cached) return cached;
    SparseBatch *batch = calloc(1, sizeof *batch);
    if (!batch) return NULL;
    VkBufferCreateInfo buffer = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo allocation = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };
    VmaAllocationInfo mapped;
    if (vmaCreateBuffer(c->vma, &buffer, &allocation, &batch->buffer, &batch->allocation, &mapped) != VK_SUCCESS) {
        free(batch);
        return NULL;
    }
    if (!mapped.pMappedData) {
        vmaDestroyBuffer(c->vma, batch->buffer, batch->allocation);
        free(batch);
        return NULL;
    }
    batch->bytes = bytes;
    batch->ranges = mapped.pMappedData;
    return batch;
}

static void sparse_batch_free(void *arg) {
    SparseBatch *batch = arg;
    vmaDestroyBuffer(vk_ctx()->vma, batch->buffer, batch->allocation);
    free(batch);
}

static void sparse_batch_complete(void *arg) {
    SparseBatch *batch = arg;
    plat_mutex_lock(&sparse_cache.lock);
    if (!sparse_cache.closed && sparse_cache.n < SPARSE_CACHE_COUNT &&
        batch->bytes <= SPARSE_CACHE_BYTES - sparse_cache.bytes) {
        sparse_cache.ready[sparse_cache.n++] = batch;
        sparse_cache.bytes += batch->bytes;
        plat_mutex_unlock(&sparse_cache.lock);
        return;
    }
    plat_mutex_unlock(&sparse_cache.lock);
    sparse_batch_free(batch);
}

static void sparse_cache_close(void) {
    SparseBatch *ready[SPARSE_CACHE_COUNT];
    plat_mutex_lock(&sparse_cache.lock);
    sparse_cache.closed = 1;
    int n = sparse_cache.n;
    memcpy(ready, sparse_cache.ready, (size_t)n * sizeof *ready);
    sparse_cache.n = 0;
    sparse_cache.bytes = 0;
    plat_mutex_unlock(&sparse_cache.lock);
    for (int i = 0; i < n; i++) sparse_batch_free(ready[i]);
}

#define SPARSE_RANGE_MAX 65536u

// The pool's merged ranges mregions[0, count): the ones the dispatch can take (word-aligned, at most SPARSE_RANGE_MAX)
// are moved to the front, and their count is returned when the dispatch should copy them; 0 leaves every range to
// vkCmdCopyBuffer.
static int sparse_ranges(VkCtx *c, uint64_t size, int count) {
    if (count < 64 || size > c->props.limits.maxStorageBufferRange) return 0;
    int small = 0;
    uint64_t total = 0;
    for (int i = 0; i < count; i++) {
        VkBufferCopy region = mregions[i];
        if ((region.srcOffset | region.size) & 3 || region.size > SPARSE_RANGE_MAX) continue;
        mregions[i] = mregions[small], mregions[small++] = region;
        total += region.size;
    }
    if (small < 64 || (uint32_t)small > c->props.limits.maxComputeWorkGroupCount[0]) return 0;
    return total <= 4096u * (uint64_t)small ? small : 0;
}

static void sparse_dispatch(VkCommandBuffer cb, SparseBatch *batch, PoolRec *pool, uint64_t size, int count) {
    uint32_t first = batch->used;
    for (int i = 0; i < count; i++) {
        const VkBufferCopy *region = &mregions[i];
        batch->ranges[batch->used++] = (SparseRange){(uint32_t)(region->srcOffset / 4), (uint32_t)(region->size / 4)};
    }
    VkDescriptorBufferInfo infos[3] = {{pool->mbuf, 0, size}, {pool->buf, 0, size}, {batch->buffer, 0, batch->bytes}};
    VkWriteDescriptorSet writes[3];
    for (uint32_t i = 0; i < 3; i++)
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                           .dstBinding = i, .descriptorCount = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[i]};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, sparse_copy.pipeline);
    rec_push(sparse_copy.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, sparse_copy.layout, 0, 3, writes);
    vkCmdPushConstants(cb, sparse_copy.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof first, &first);
    vkCmdDispatch(cb, (uint32_t)count, 1, 1);
}

// Merge ranges by pool before recording either a transfer or an exact sparse write-back dispatch.
// to_mirror: buf -> mbuf, else mbuf -> buf. Returns the command count; accumulates bytes and ranges moved.
static int copy_ranges(VkCommandBuffer cb, MRange *a, int n, int to_mirror, uint64_t *bytes, uint64_t *copies,
                       uint64_t fence) {
    static _Thread_local MRange *sbuf;   // the replaying thread's (vk_mem_mirror_begin and _end) and the write-back thread's
    static _Thread_local size_t scap;
    sort_ranges(a, (size_t)n, &sbuf, &scap);
    int rec = 0;
    VkCtx *c = mirror_sparse && !to_mirror ? vk_ctx() : NULL;
    SparseBatch *sparse = NULL;
    int sparse_tried = 0;
    for (int i = 0; i < n;) {
        PoolRec *r = a[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        int k = 0;
        for (; i < n && a[i].r == r; i++) {
            uint64_t lo = a[i].off, hi = a[i].off + a[i].size;
            if (hi > lim) hi = lim;
            if (lo >= hi) continue;
            if (k && lo <= mregions[k - 1].srcOffset + mregions[k - 1].size) {
                uint64_t end = mregions[k - 1].srcOffset + mregions[k - 1].size;
                if (hi > end) mregions[k - 1].size += hi - end;
                continue;
            }
            if (k == cregions) grow((void **)&mregions, &cregions, sizeof *mregions);
            mregions[k++] = (VkBufferCopy){lo, lo, hi - lo};
        }
        if (!k) continue;
        int dispatched = c ? sparse_ranges(c, lim, k) : 0;   // the first ones go to the dispatch, the rest are copied
        if (dispatched) {
            if (!sparse_tried) {
                sparse = sparse_batch_create(c, n);
                sparse_tried = 1;
            }
            if (sparse) {
                sparse_dispatch(cb, sparse, r, lim, dispatched);
                rec += 4;
            } else dispatched = 0;
        }
        if (dispatched < k) {
            if (to_mirror) vkCmdCopyBuffer(cb, r->buf, r->mbuf, (uint32_t)(k - dispatched), mregions + dispatched);
            else vkCmdCopyBuffer(cb, r->mbuf, r->buf, (uint32_t)(k - dispatched), mregions + dispatched);
            rec++;
        }
        for (int j = 0; j < k; j++) *bytes += mregions[j].size;
        *copies += (uint64_t)k;
    }
    if (sparse) vk_completion_add(fence, sparse_batch_complete, sparse);
    return rec;
}

static void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                     VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                           .dstStageMask = ds, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void refuse(PoolRec *r, int why, const char *what) {
    ms_set(r, MS_REFUSED);
    uint64_t n = r->import_hi - r->import_lo;
    if (plat_inc64(&m_refused_n[why]) <= 4) {
        char f[160];
        flag_names(r->flags, f, sizeof f);
        rt_log("mem: pool %#llx+%#llx (%s) stays imported: %s", (unsigned long long)r->lo, (unsigned long long)r->size,
               f, what);
    }
    plat_add64(&m_refused_bytes[why], (int64_t)n);
}

// A mirror of BIG_MIRROR bytes or more gets a log line with the times of its parts (mirror_create) and of recording
// the prologue that holds its first full copy (vk_mem_mirror_prologue): the first use of the 450 MiB terrain arena of
// a layer stalled one frame for 250 ms or more (#397). Replay thread (mirror_lock when shared), like the mirror's
// creation and the prologue.
#define BIG_MIRROR (64ull << 20)
static struct {
    int pending;                               // a line waits for the prologue
    uint64_t lo, size;                         // the pool's address and the mirror's size
    uint64_t hold;                             // ns: vk_mem_use's wait for the walk and mirror_lock before it
    uint64_t lock, budget, snap, alloc, total; // ns: pool_lock and the overlap check, vk_mem_vram, gen_lock and
                                               // snap_regions, vmaCreateBuffer, all of mirror_create
    uint64_t queries, pages;                   // the snapshot's write-watch queries and the written pages they found
} big_mirror;

// Replay thread: the pool's mirror on its first buffer use.
static void mirror_create(PoolRec *r) {
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo;
    uint64_t t0 = qpc();
    // The state is stored once, when it is final. A helper thread that reads it before that (vk_mem_use) sees MS_NONE
    // and gives its part up; it never sees a refusal that turns into a mirror.
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    if (r->kind != POOL_IMPORT || !r->buf || !n) return ms_set(r, MS_REFUSED);
    pool_lock_shared();
    int i = pidx_upper(&pool_index, r->lo) - 1;
    while (i >= 0 && pool_index.e[i].buf != r) i--;
    int alone = i >= 0 && (i == 0 || pool_index.maxhi[i - 1] <= pool_index.e[i].lo) &&
                (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= pool_index.e[i].hi);
    plat_mutex_unlock_shared(&pool_lock);
    if (i < 0) return ms_set(r, MS_REFUSED);   // finalized meanwhile
    if (!alone) return refuse(r, MR_OVERLAP, "it overlaps another pool");
    uint64_t r0 = r->import_lo >> SNAP_SHIFT, r1 = (r->import_hi - 1) >> SNAP_SHIFT;
    if (r1 >= SNAP_REGIONS) return refuse(r, MR_UNTRACKED, "outside the write-watched address space");
    uint64_t t1 = qpc();
    uint64_t used, budget;
    int exact;
    if (vk_mem_vram(&used, &budget, &exact) && used + n + budget / 8 > budget) {
        char why[128];
        snprintf(why, sizeof why, "its %.1f MiB mirror does not fit the VRAM budget (%.0f of %.0f MiB in use)",
                 n / 1048576.0, used / 1048576.0, budget / 1048576.0);
        return refuse(r, MR_BUDGET, why);
    }
    uint64_t t2 = qpc();
    gen_lock_x();
    uint64_t p0 = ww_pages, q = snap_regions(r0, r1, mepoch), wpages = ww_pages - p0;
    mrep.sync_queries += q;
    int untracked = 0;
    for (uint64_t k = r0; k <= r1; k++) untracked |= snap_untracked[k];
    r->mgen = gen_now;   // the whole-range copy below holds every write observed so far
    plat_mutex_unlock(&gen_lock);
    if (untracked) return refuse(r, MR_UNTRACKED, "its range is not write-watched");
    uint64_t t3 = qpc();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkResult res = vmaCreateBuffer(c->vma, &bi, &ai, &r->mbuf, &r->malloc_, NULL);
    uint64_t t4 = qpc();
    if (res != VK_SUCCESS) {
        r->mbuf = VK_NULL_HANDLE, r->malloc_ = NULL;
        char why[96];
        snprintf(why, sizeof why, "device-local allocation of %.1f MiB failed (%d)", n / 1048576.0, (int)res);
        return refuse(r, MR_ALLOC, why);
    }
    plat_mutex_lock(&pool_lock);
    if (nm == mcap) grow((void **)&mlist, &mcap, sizeof *mlist);
    int at = nm;
    while (at > 0 && mlist[at - 1]->import_lo > r->import_lo) at--;
    memmove(mlist + at + 1, mlist + at, (size_t)(nm - at) * sizeof *mlist);
    mlist[at] = r;
    nm++;
    plat_mutex_unlock(&pool_lock);
    r->acc_sub = msub, r->sub_hist = 1;
    ms_set(r, MS_ON);   // after mbuf and acc_sub: what a reader of MS_ON goes on to read
    range_add(&pend, &npend, &cpend, r, 0, n);
    if (n >= BIG_MIRROR) {
        big_mirror.pending = 1, big_mirror.lo = r->lo, big_mirror.size = n, big_mirror.hold = 0;
        big_mirror.lock = t1 - t0, big_mirror.budget = t2 - t1, big_mirror.snap = t3 - t2, big_mirror.alloc = t4 - t3;
        big_mirror.total = qpc() - t0, big_mirror.queries = q, big_mirror.pages = wpages;
    }
    mrep.first_n++, mrep.first_bytes += n;
    plat_inc64(&m_live_n);
    plat_add64(&m_live_bytes, (int64_t)n);
}

static void mirror_forget(PoolRec *r) {
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < nm; i++)
        if (mlist[i] == r) {
            memmove(mlist + i, mlist + i + 1, (size_t)(nm - i - 1) * sizeof *mlist);
            nm--;
            plat_dec64(&m_live_n);
            plat_add64(&m_live_bytes, -(int64_t)(r->import_hi - r->import_lo));
            break;
        }
}

// Finalized mirrored pools waiting for mirror_retire; vk_mem_shutdown destroys those whose callback never ran.
static PoolRec **retiring;
static int nretiring, cretiring;
static PlatMutex retire_lock = PLAT_MUTEX_INIT;

static void mirror_retiring(PoolRec *r) {
    plat_mutex_lock(&retire_lock);
    if (nretiring == cretiring) grow((void **)&retiring, &cretiring, sizeof *retiring);
    retiring[nretiring++] = r;
    plat_mutex_unlock(&retire_lock);
}

static void mirror_retire(void *arg) {
    plat_mutex_lock(&retire_lock);
    for (int i = 0; i < nretiring; i++)
        if (retiring[i] == arg) { retiring[i] = retiring[--nretiring]; break; }
    plat_mutex_unlock(&retire_lock);
    destroy_rec(vk_ctx(), arg);
}

// The CPU rewrote the pages of the n runs at run (offsets from import_lo, ascending and disjoint, as mirror_sync queues
// them): they are no longer kept GPU writes (mirror_lock held). One merge of the two sorted lists into a scratch array
// that then becomes the pool's list. Cutting page by page, each cut a binary search and, when it split or emptied a
// range, a memmove of the list's tail, held mirror_lock through a sync of the 450 MiB arena in the Hilltop Layer of
// Surf Hills (3.2.0), where the CPU rewrites the terrain records' pages every frame: the slice workers waited 3.9 ms a
// frame for the lock (up to 1.9 ms at a time) and the replay took 2.4 ms longer.
static void kept_cut(PoolRec *r, const MRange *run, int n) {
    static struct KeptRange *out;   // mirror_lock (or the replay thread alone): the next list, swapped with r->kept
    static int cout;
    while (cout < r->nkept + n) grow((void **)&out, &cout, sizeof *out);   // a run splits at most one range
    struct KeptRange *in = r->kept;
    int i = 0, o = 0;
    for (int j = 0; j < n && i < r->nkept; j++) {
        uint64_t a = run[j].off, b = a + run[j].size;
        while (i < r->nkept && in[i].hi <= a) out[o++] = in[i++];
        while (i < r->nkept && in[i].lo < b) {
            if (in[i].lo < a) out[o++] = (struct KeptRange){in[i].lo, a};
            if (in[i].hi > b) {   // what is left above the run may meet the next one
                in[i].lo = b;
                break;
            }
            i++;
        }
    }
    while (i < r->nkept) out[o++] = in[i++];
    r->kept = out, out = in;
    int c = r->ckept;
    r->ckept = cout, cout = c;
    r->nkept = o;
}

// The first access of this submission (the replay thread, or a slice worker through vk_mem_use and vk_mem_pick):
// queues the copies of the pages written since the last sync.
// The write-watch snapshot of the pool's regions runs before mirror_lock (#317: late in a long session one
// GetWriteWatch took 0.38 ms, and held under mirror_lock it made the replay thread and six slice workers wait 2.7 s per
// 60 frames). mepoch only changes on the replay thread between submissions, and the body below re-checks snap_at under
// gen_lock, so it queries again for any region the early snapshot did not cover. The page scan, the other long part (up
// to 1.9 ms for one pool on Hill, where the replay thread and the slice workers reach the same pools at the start of a
// submission), runs under the pool's own lock (sync_lock) into a list of the thread's; mirror_lock, which every pool's
// sync, the write-back ranges and the prologue share, is taken only to move that list into pend, cut the kept list and
// add the flush ranges. A thread that needs this pool still waits for its sync; threads syncing other pools do not.
// Lock order: sync_lock first; gen_lock is taken and dropped before mirror_lock, which is never held with gen_lock here.
static void mirror_sync(PoolRec *r) {
    {
        uint64_t e0 = r->import_lo >> SNAP_SHIFT, e1 = (r->import_hi - 1) >> SNAP_SHIFT;
        int need = 0;
        plat_mutex_lock_shared(&gen_lock);
        for (uint64_t k = e0; k <= e1; k++)
            if (snap_at[k] != mepoch) { need = 1; break; }
        plat_mutex_unlock_shared(&gen_lock);
        if (need) {
            gen_lock_x();
            mrep.sync_queries += snap_regions(e0, e1, mepoch);   // gen_lock held exclusively, as the other increments
            plat_mutex_unlock(&gen_lock);
        }
    }
    const int shared = mirror_shared();
    if (shared) plat_mutex_lock(&r->sync_lock);
    if (__atomic_load_n(&r->acc_sub, __ATOMIC_ACQUIRE) == msub) {   // the other thread did it meanwhile
        if (shared) plat_mutex_unlock(&r->sync_lock);
        return;
    }
    static _Thread_local MRange *loc;
    static _Thread_local int nloc, cloc;
    nloc = 0;
    uint64_t t0 = qpc();
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t k = r0; k <= r1; k++)
        if (snap_at[k] != mepoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        gen_lock_x();
        uint64_t b0 = ww_bytes;
        mrep.sync_queries += snap_regions(r0, r1, mepoch);
        mrep.sync_bytes += ww_bytes - b0;
    }
    uint32_t since = r->mgen;
    uint64_t own = 0;
    int skip_own = RT_KNOB_ON(mirror_own);
    for (uint64_t k = r0; k <= r1; k++) {
        if (reg_max[k] <= since) continue;
        uint64_t a = k << SNAP_SHIFT, b = a + (1ull << SNAP_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (!ch) continue;
        const uint32_t *oc = skip_own ? own_chunk[a >> GEN_CHUNK_SHIFT] : NULL;
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT) {
            uint32_t i = (x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1), v = ch[i];
            if (v <= since) continue;
            if (oc && oc[i] == v) {   // the mirror's own GPU write is the page's newest (vk_gpu_written_into)
                own++;
                continue;
            }
            range_add(&loc, &nloc, &cloc, r, x - lo, 1ull << GEN_PAGE_SHIFT);
        }
    }
    r->mgen = gen_now;
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);

    int locked = mirror_lock_if_shared();   // paired with mirror_unlock on its own reading, as everywhere
    mrep.own_bytes += own << GEN_PAGE_SHIFT;
    int p0 = npend > 0 && pend[npend - 1].r == r ? npend - 1 : npend;   // range_add may extend the last entry
    for (int i = 0; i < nloc; i++) range_add(&pend, &npend, &cpend, r, loc[i].off, loc[i].size);
    // The pages the CPU rewrote get the guest's bytes in the mirror: they are no longer GPU-only. A kept page whose
    // newest stamp is the GPU write that kept it is left out above, so it stays kept.
    if (r->nkept && npend > p0) kept_cut(r, pend + p0, npend - p0);
    if (__atomic_load_n(&r->fl_hi, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&flush_lock);
        uint64_t a = r->fl_lo, b = r->fl_hi;
        r->fl_hi = 0;
        plat_mutex_unlock(&flush_lock);
        a = a > lo ? a & ~0xfffull : lo;
        b = b < hi ? (b + 0xfff) & ~0xfffull : hi;
        // Pages wholly inside a kept GPU write hold the GPU's bytes, not the guest's: a flush range that merely spans them
        // (the pool's flushes are one bounding box) must not copy the stale guest bytes over them. A real CPU write to
        // such a page still comes in through its page generation above.
        uint64_t x = a - lo, e = b - lo;
        int k = 0, n = r->nkept;
        while (k < n) {   // first kept range ending above x
            int m = (k + n) / 2;
            if (r->kept[m].hi <= x) k = m + 1;
            else n = m;
        }
        for (; x < e && k < r->nkept && r->kept[k].lo < e; k++) {
            if (r->kept[k].lo > x) range_add(&pend, &npend, &cpend, r, x, r->kept[k].lo - x);
            if (r->kept[k].hi > x) x = r->kept[k].hi;
        }
        if (x < e) range_add(&pend, &npend, &cpend, r, x, e - x);
    }
    mrep.syncs++;
    mrep.sync_t += qpc() - t0;
    uint32_t d = msub - r->acc_sub;
    r->sub_hist = d >= 32 ? 1u : r->sub_hist << d | 1u;
    __atomic_store_n(&r->acc_sub, msub, __ATOMIC_RELEASE);
    mirror_unlock(locked);
    if (shared) plat_mutex_unlock(&r->sync_lock);
}


void vk_mem_use(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    int state = ms_get(r);
    if (state == MS_NONE) {
        // A new mirror changes what every later bind of the pool resolves to: the helper leaves it to the replay
        // thread, and a part that ran while one was made is dropped (vk_mem_mirror_gen). The workers of sliced passes
        // make it under mirror_lock, and the first of them to get the lock does.
        if (vk_spec_helper()) { vk_spec_fail(VK_SPEC_MIRROR); return; }
        uint64_t ph = perf_ph_begin_cpu();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
        uint64_t t0 = qpc();
        vk_texture_hold_barrier();   // the walk, with a pass end held back: its slices resolve the pool as it was
        int locked = mirror_lock_if_shared();
        uint64_t t1 = qpc();
        if (ms_get(r) == MS_NONE) {
            mirror_create(r);
            if (big_mirror.pending && big_mirror.lo == r->lo) big_mirror.hold = t1 - t0;
        }
        mirror_unlock(locked);
        perf_ph_end(PERF_PH_MIRROR, ph);
        state = ms_get(r);
    }
    if (state != MS_ON) return;
    if (__atomic_load_n(&r->acc_sub, __ATOMIC_ACQUIRE) != msub) {
        uint64_t ph = perf_ph_begin();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
        mirror_sync(r);
        perf_ph_end(PERF_PH_MSYNC, ph);
    }
    sp->buffer = r->mbuf;
}

void vk_mem_pick(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (!r || ms_get(r) != MS_ON) return;
    if (__atomic_load_n(&r->acc_sub, __ATOMIC_ACQUIRE) != msub) {
        uint64_t ph = perf_ph_begin();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
        mirror_sync(r);
        perf_ph_end(PERF_PH_MSYNC, ph);
    }
    sp->buffer = r->mbuf;
}

// vk_exec.c par_start_workers, before any worker records: from here on the mirrors are shared, so mirror_lock is taken.
void vk_mem_threads(void) { __atomic_store_n(&mem_workers, 1, __ATOMIC_RELEASE); }

uint64_t vk_mem_mirror_gen(void) { return __atomic_load_n(&mirror_gen, __ATOMIC_ACQUIRE); }
int vk_mem_shadow_on(void) { return shadow_on; }

int vk_mem_mirror_probe(uint64_t addr) {
    VkMemSpan sp;
    if (!vk_mirror_on || !vk_mem_resolve(addr, 1, &sp) || !sp.pool) return 0;
    mepoch = vk_texture_epoch();
    vk_mem_use(&sp);
    return ms_get(sp.pool) == MS_ON;
}

// A range already added in this submission adds nothing (copy_ranges merges overlaps): each thread remembers the
// ranges it added, so the same storage buffer bound by draw after draw takes mirror_lock once. A profile on eight
// threads had 6% of the parallel slices' recording waiting for the lock here, and the busy area adds about 7,600
// write-back ranges a frame that vk_mem_mirror_end sorts. The ranges are remembered with their write-back tag
// (vk_mem_wb_tag): a held part's write-back (vk_mem_mirror_wb_held) takes ranges out of the lists in the middle of the
// submission, and a range the GPU writes again after that carries a higher tag (vk_exec.c wb_tag_walk only grows), so
// it is added again and copied again later.
void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size) {
    vk_spec_event();   // a bind the GPU writes through: never the helper thread's
    PoolRec *r = sp->pool;
    if (!r || ms_get(r) != MS_ON || sp->buffer != r->mbuf || !size) return;
    static _Thread_local struct { const PoolRec *r; uint64_t off, size; uint32_t sub, tag; } seen[128];
    __typeof__(seen[0]) *k = &seen[((uintptr_t)r >> 6 ^ sp->offset >> 4) & 127];
    uint32_t sub = __atomic_load_n(&msub, __ATOMIC_RELAXED);
    if (k->r == r && k->sub == sub && k->tag == wb_tag && k->off == sp->offset && k->size >= size) return;
    if (count_quiet) wb_local_add(r, sp->offset, size);
    else {
        int locked = mirror_lock_if_shared();
        wb_range_add(&wbl, &nwbl, &cwbl, r, sp->offset, size);
        mirror_unlock(locked);
    }
    *k = (__typeof__(*k)){r, sp->offset, size, sub, wb_tag};
}

// A GPU write that stays in the mirror (the geometry stage's storage, vk_exec_draw.c gs_storage_writeback). Re-uploads
// (mirror_sync) work on whole 4 KiB pages, so the bytes of the range that share a page with other data go back to guest
// memory as usual, and a CPU write or flush there copies the page, GPU bytes included, back into the mirror. The pages
// wholly inside the range are noted in the pool's kept list, which mirror_sync leaves out of its flush ranges.
// The thread notes them in a list of its own, without mirror_lock, and vk_mem_mirror_end merges every thread's into
// the pools' lists (kept_take). Nothing reads a pool's list for this submission's writes before that: the first bind of
// a pool in a submission syncs it (vk_mem_use), so its sync and flush ranges come before any write the GPU keeps there,
// and the uniform snapshot asks at the start of the next submission (vk_mem_kept_overlaps). An insertion under
// mirror_lock per range, a memmove of the list's tail each, made the slice workers wait for the lock.
// Returns whether it kept any page; 0 means every byte went the vk_mem_gpu_wrote way (no mirror, or no whole page), so
// guest memory will hold the GPU's bytes and the caller treats the write as an ordinary one.
int vk_mem_gpu_kept(const VkMemSpan *sp, uint64_t size) {
    vk_spec_event();
    PoolRec *r = sp->pool;
    if (!r || ms_get(r) != MS_ON || sp->buffer != r->mbuf || !size) return 0;
    const uint64_t page = 1ull << GEN_PAGE_SHIFT, a = r->import_lo + sp->offset, b = a + size;
    uint64_t pa = (a + page - 1) & ~(page - 1), pb = b & ~(page - 1);
    if (pa >= pb) return vk_mem_gpu_wrote(sp, size), 0;   // no whole page
    // As in vk_mem_gpu_wrote, a range this thread already kept in this submission is in the list.
    static _Thread_local struct { const PoolRec *r; uint64_t off, size; uint32_t sub, tag; } seen[128];
    __typeof__(seen[0]) *k = &seen[((uintptr_t)r >> 6 ^ sp->offset >> 4) & 127];
    uint32_t sub = __atomic_load_n(&msub, __ATOMIC_RELAXED);
    if (k->r == r && k->sub == sub && k->tag == wb_tag && k->off == sp->offset && k->size >= size) return 1;
    if (a < pa) vk_mem_gpu_wrote(&(VkMemSpan){.pool = r, .buffer = sp->buffer, .offset = sp->offset}, pa - a);
    if (pb < b) vk_mem_gpu_wrote(&(VkMemSpan){.pool = r, .buffer = sp->buffer, .offset = pb - r->import_lo}, b - pb);
    WbLocal *w = wb_local();
    range_add(&w->k, &w->nk, &w->ck, r, pa - r->import_lo, pb - pa);
    *k = (__typeof__(*k)){r, sp->offset, size, sub, wb_tag};
    return 1;
}

// One pool's kept ranges of the submission (sorted by offset) merged into its list, overlapping or touching ranges
// joined (mirror_lock held).
static void kept_merge(PoolRec *r, const MRange *add, int n) {
    static struct KeptRange *out;   // the next list, swapped with r->kept
    static int cout;
    while (cout < r->nkept + n) grow((void **)&out, &cout, sizeof *out);
    int i = 0, j = 0, o = 0;
    while (i < r->nkept || j < n) {
        struct KeptRange x;
        if (j >= n || (i < r->nkept && r->kept[i].lo <= add[j].off)) x = r->kept[i++];
        else x = (struct KeptRange){add[j].off, add[j].off + add[j].size}, j++;
        if (o && x.lo <= out[o - 1].hi) {
            if (x.hi > out[o - 1].hi) out[o - 1].hi = x.hi;
        } else out[o++] = x;
    }
    struct KeptRange *old = r->kept;
    int c = r->ckept;
    r->kept = out, r->ckept = cout, r->nkept = o;
    out = old, cout = c;
}

// vk_mem_mirror_end, no slice being recorded: every thread's kept ranges of the submission into the pools' lists.
static void kept_take(void) {
    static MRange *all, *sbuf;
    static int call;
    static size_t scap;
    int n = 0;
    // Most submissions keep nothing: skip the lock and the sort. The slices have ended, so no thread adds to its list now.
    for (WbLocal *w = wb_locals; w; w = w->next) n += w->nk;
    if (!n) return;
    n = 0;
    plat_mutex_lock(&mirror_lock);
    for (WbLocal *w = wb_locals; w; w = w->next) {
        while (call < n + w->nk) grow((void **)&all, &call, sizeof *all);
        if (w->nk) memcpy(all + n, w->k, sizeof *all * (size_t)w->nk);
        n += w->nk, w->nk = 0;
    }
    sort_ranges(all, (size_t)n, &sbuf, &scap);
    for (int i = 0, j; i < n; i = j) {
        for (j = i + 1; j < n && all[j].r == all[i].r; j++) {}
        kept_merge(all[i].r, all + i, j - i);
    }
    plat_mutex_unlock(&mirror_lock);
}

// Whether a kept GPU write (vk_mem_gpu_kept) lies in [addr, addr+size): guest memory there is stale, so
// vk_usnap.c vk_usnap_begin does not copy it into a uniform snapshot.
int vk_mem_kept_overlaps(uint64_t addr, uint64_t size) {
    VkMemSpan sp;
    if (!size || !vk_mem_resolve(addr, size, &sp) || !sp.pool) return 0;
    PoolRec *r = sp.pool;
    if (ms_get(r) != MS_ON) return 0;
    uint64_t lo = addr - r->import_lo, hi = lo + size;
    int locked = mirror_lock_if_shared(), hit = 0;
    for (int a = 0, b = r->nkept; a < b && !hit;) {
        int k = (a + b) / 2;
        if (r->kept[k].hi <= lo) a = k + 1;
        else if (r->kept[k].lo >= hi) b = k;
        else hit = 1;
    }
    mirror_unlock(locked);
    return hit;
}

void nvn_pool_flushed(const pool_t *p, uint64_t off, uint64_t size) {
    if (!vk_mirror_on || !size || !p->mem) return;
    plat_inc64(&m_flush_calls);
    plat_add64(&m_flush_bytes, (int64_t)size);
    pool_lock_shared();
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)p->mem, 0);
    PoolRec *r = e && e->key == p ? e->buf : NULL;
    if (r && ms_get(r) == MS_ON) {
        uint64_t a = (uintptr_t)p->mem + off, b = a + size;
        plat_mutex_lock(&flush_lock);
        if (!r->fl_hi) r->fl_lo = a, r->fl_hi = b;
        else {
            if (a < r->fl_lo) r->fl_lo = a;
            if (b > r->fl_hi) r->fl_hi = b;
        }
        plat_mutex_unlock(&flush_lock);
    }
    plat_mutex_unlock_shared(&pool_lock);
}

// gen_lock exclusive: freezes up to FREEZE_PER_SUB candidate regions still idle. The query after the
// protection collects the writes made before it took effect.
static void freeze_pass(void) {
    for (int done = 0; nfcand && done < FREEZE_PER_SUB;) {
        uint64_t j = fcand[--nfcand], a = j << REG_SHIFT;
        if (ww_ep - reg_last[j] < (uint32_t)freeze_idle || mem_frozen(a)) continue;
        done++;
        if (!mem_freeze(a)) { n_frz_refused++; continue; }
        ww_stamp(a, a + (1ull << REG_SHIFT), ++gen_now, WW_FREEZE);
    }
}

// Whether vk_mem_mirror_begin snaps pool r's regions for the submission after prev: the submissions p and 2p before the
// next one synced it, for some p from 1 to TOTK_MIRROR_PRESNAP (8 by default), as with a pool one of a frame's
// submissions uses every frame. TOTK_MIRROR_PRESNAP=0 snaps the previous submission's pools instead, as before. A pool
// missed here is snapped by its first sync (mirror_sync), in the middle of the recording, under mirror_lock and
// gen_lock, and the slice workers wait for the lock while one runs. Bananza's submissions (4 a frame on Hilltop and
// Canyon Layer) use disjoint pools: none of the previous submission's pools was synced by the next one, so that
// pre-snapshot (about 200 MiB a frame) was all waste and the syncs still made about 55 queries a frame.
// With periods up to 8, 99% of the pre-snapped pools were synced by the submission, the syncs made 2.5 queries a
// frame, and on Canyon Layer the tracking took 0.8 ms a frame instead of 1.4 to 1.55, the waits over 100 us for
// mirror_lock fell from about 30 to 2 per 60 frames, and the replay took 11.3 ms a frame instead of 11.9 (two
// rounds); Hilltop's frame time did not change.
// No slice records here, so sub_hist is read unlocked.
static int presnap_wanted(const PoolRec *r, uint32_t prev) {
    uint32_t d = prev - r->acc_sub;
    if (!r->acc_sub || d >= 32) return 0;
    uint32_t h = r->sub_hist << d;   // bit i: submission prev - i synced it
    if (!mirror_presnap) return h & 1;
    for (int p = 1; p <= mirror_presnap && 2 * p <= 32; p++)
        if ((h >> (p - 1) & 1) && (h >> (2 * p - 1) & 1)) return 1;
    return 0;
}

static void mirror_begin(void) {
    if (!vk_mirror_on) {   // import mode: the texture checks' snapshots still skip frozen regions
        if (!freeze_idle) return;
        gen_lock_x();
        ww_ep++;
        freeze_pass();
        plat_mutex_unlock(&gen_lock);
        return;
    }
    uint64_t t0 = qpc();
    uint32_t prev = msub;
    msub = msub + 1 ? msub + 1 : 1;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    mepoch = vk_texture_epoch();
    uint64_t q = 0, run0 = 0, run1 = 0;
    int have = 0;
    pool_lock_shared();
    gen_lock_x();
    ww_ep++;
    uint64_t b0 = ww_bytes;
    for (int i = 0; i < nm; i++) {
        PoolRec *r = mlist[i];
        if (r->snap_sub == prev && prev) {   // pre-snapped for the previous submission: did it use the pool?
            if (r->acc_sub == prev) mrep.presnap_used++;
            else mrep.presnap_unused_bytes += r->import_hi - r->import_lo;
        }
        if (!presnap_wanted(r, prev)) continue;
        r->snap_sub = msub;
        mrep.presnap_pools++;
        uint64_t a = r->import_lo >> SNAP_SHIFT, b = (r->import_hi - 1) >> SNAP_SHIFT;
        if (have && a <= run1 + 1 + (uint64_t)mirror_gap) {
            if (b > run1) run1 = b;
            continue;
        }
        if (have) q += snap_regions(run0, run1, mepoch);
        run0 = a, run1 = b, have = 1;
    }
    if (have) q += snap_regions(run0, run1, mepoch);
    mrep.presnap_bytes += ww_bytes - b0;
    if (freeze_idle) freeze_pass();
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    mrep.subs++;
    mrep.presnap_queries += q;
    mrep.presnap_t += qpc() - t0;
}

// TOTK_TRACE=perf: a part of the replay (perf.c ph_frame).
void vk_mem_mirror_begin(void) {
    uint64_t ph = perf_ph_begin();
    mirror_begin();
    perf_ph_end(PERF_PH_MIRROR_BEGIN, ph);
}

// Whether copies are queued for the next prologue; vk_sync.c prologue skips its command buffer on 0. Read without
// mirror_lock: a helper queues a part's copies while it translates the part, before the replay thread joins it, and
// vk_mem_mirror_prologue reads the count again under the lock. The atomic accesses make the unlocked read defined.
int vk_mem_mirror_pending(void) { return __atomic_load_n(&npend, __ATOMIC_RELAXED); }

void vk_mem_mirror_prologue(VkCommandBuffer cb) {
    int locked = mirror_lock_if_shared();   // a helper may be queueing a sync
    if (npend) {
        uint64_t t0 = qpc(), b0 = mrep.upload_bytes;
        int n = npend;
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        copy_ranges(cb, pend, npend, 1, &mrep.upload_bytes, &mrep.upload_copies, 0);
        mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        __atomic_store_n(&npend, 0, __ATOMIC_RELAXED);
        mrep.prologues++;
        if (big_mirror.pending) {   // its first full copy was in pend (mirror_create queues it before any prologue)
            static int32_t lines;
            big_mirror.pending = 0;
            if (plat_inc32(&lines) <= 32)
                rt_log("mem: mirror of %.0f MiB at %#llx made in %.1f ms on first use (after %.1f ms waiting for the walk "
                       "and the mirror lock): pool lookup %.1f, budget check %.1f, snapshot %.1f (%llu queries, %llu "
                       "written pages), vmaCreateBuffer %.1f; its prologue (%d ranges, %.1f MiB) recorded in %.2f ms",
                       big_mirror.size / 1048576.0, (unsigned long long)big_mirror.lo, qpc_ms(big_mirror.total),
                       qpc_ms(big_mirror.hold), qpc_ms(big_mirror.lock), qpc_ms(big_mirror.budget), qpc_ms(big_mirror.snap),
                       (unsigned long long)big_mirror.queries, (unsigned long long)big_mirror.pages,
                       qpc_ms(big_mirror.alloc), n, (mrep.upload_bytes - b0) / 1048576.0, qpc_ms(qpc() - t0));
        }
    }
    mirror_unlock(locked);
}

// ---- verify mode
typedef struct { PoolRec *r; uint64_t off, len, dst; uint32_t mgen; } VSample;
typedef struct { uint32_t sub; int n; VkBuffer buf; VmaAllocation alloc; const uint8_t *map; VSample s[VERIFY_SAMPLES]; } VJob;
static int ver_i;
static uint64_t ver_off;

// The CPU wrote the page since the mirror's sync that mgen records: stamped later, or its write-watch bit is set. A
// later stamp that mirror_sync skips as the mirror's own GPU write does not count, so verify compares those pages: the
// skip is only right while the guest bytes equal the mirror's. Read under gen_lock: a write-watch query (snap_regions,
// vk_mem_changed) clears the OS bits before it stamps the pages, both under the lock held exclusively, and a page read
// between the two had neither, so a page the CPU had just rewritten counted as a mismatch.
static int page_dirty(uint64_t a, uint32_t mgen) {
    plat_mutex_lock_shared(&gen_lock);
    uint32_t *s = gen_slot(a, 0);
    int dirty = 0;
    if (s && *s > mgen) {   // s is set, so the chunk index is in range
        const uint32_t *oc = RT_KNOB_ON(mirror_own) ? own_chunk[a >> GEN_CHUNK_SHIFT] : NULL;
        dirty = !oc || oc[(a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] != *s;
    }
    uint64_t p[1];
    if (!dirty) dirty = mem_written_peek(a, 1ull << GEN_PAGE_SHIFT, p, 1) != 0;   // UINT64_MAX (not watched) counts
    plat_mutex_unlock_shared(&gen_lock);
    return dirty;
}

static int kept_page(PoolRec *r, uint64_t off) {   // completion thread: takes mirror_lock whatever the replay's mode
    plat_mutex_lock(&mirror_lock);
    int hit = 0;
    for (int a = 0, b = r->nkept; a < b && !hit;) {
        int k = (a + b) / 2;
        if (r->kept[k].hi <= off) a = k + 1;
        else if (r->kept[k].lo > off) b = k;
        else hit = 1;
    }
    plat_mutex_unlock(&mirror_lock);
    return hit;
}

static void verify_cb(void *arg) {   // completion thread
    VJob *j = arg;
    static volatile int32_t logged;
    uint64_t pages = 0, skipped = 0, bad = 0;
    for (int i = 0; i < j->n; i++) {
        const VSample *s = &j->s[i];
        PoolRec *r = s->r;
        uint32_t wb = __atomic_load_n(&r->wb_sub, __ATOMIC_RELAXED);
        if (wb != j->sub && wb - j->sub < 0x80000000u) {   // a later submission wrote back into it
            skipped += s->len >> GEN_PAGE_SHIFT;
            continue;
        }
        for (uint64_t o = 0; o < s->len; o += 1ull << GEN_PAGE_SHIFT) {
            uint64_t a = r->import_lo + s->off + o;
            const uint8_t *m = j->map + s->dst + o, *g = (const uint8_t *)(uintptr_t)a;
            pages++;
            if (!memcmp(m, g, 1ull << GEN_PAGE_SHIFT)) continue;
            if (kept_page(r, s->off + o)) { skipped++; continue; }   // a kept GPU write: guest memory is stale on purpose
            if (page_dirty(a, s->mgen)) { skipped++; continue; }
            bad++;
            if (plat_inc32(&logged) <= VERIFY_LOG) {
                int k = 0, e = (1 << GEN_PAGE_SHIFT) - 1;
                while (k < e && m[k] == g[k]) k++;
                while (e > k && m[e] == g[e]) e--;
                char f[160];
                flag_names(r->flags, f, sizeof f);
                rt_log("mem: verify mismatch: pool %#llx+%#llx (%s) offset %#llx: bytes %#x..%#x of the page differ, "
                       "mirror %02x guest %02x at the first (submission %u)", (unsigned long long)r->lo,
                       (unsigned long long)r->size, f, (unsigned long long)(a - r->lo), k, e, m[k], g[k], j->sub);
            }
        }
    }
    plat_add64(&v_pages, (int64_t)pages);
    plat_add64(&v_skipped, (int64_t)skipped);
    plat_add64(&v_mismatch, (int64_t)bad);
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}

static int verify_record(VkCommandBuffer cb, uint64_t fence) {
    VJob *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk: out of memory");
    uint64_t left = VERIFY_BYTES, dst = 0;
    pool_lock_shared();
    for (int tries = 0; nm && left && j->n < VERIFY_SAMPLES && tries <= nm; tries++) {
        if (ver_i >= nm) ver_i = 0, ver_off = 0;
        PoolRec *r = mlist[ver_i];
        uint64_t size = r->import_hi - r->import_lo;
        if (ver_off >= size) { ver_i++, ver_off = 0; continue; }
        uint64_t len = size - ver_off < left ? size - ver_off : left;
        j->s[j->n++] = (VSample){r, ver_off, len, dst, r->mgen};
        dst += len, left -= len, ver_off += len;
        if (ver_off >= size) ver_i++, ver_off = 0;
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (!j->n) { free(j); return 0; }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = dst,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS) { free(j); return 0; }
    j->map = info.pMappedData;
    j->sub = msub;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < j->n; i++) {
        VkBufferCopy c = {j->s[i].off, j->s[i].dst, j->s[i].len};
        vkCmdCopyBuffer(cb, j->s[i].r->mbuf, j->buf, 1, &c);
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
             VK_ACCESS_2_HOST_READ_BIT);
    vk_completion_add(fence, verify_cb, j);
    return j->n + 2;
}

// TOTK_WB_EARLY (on by default; "ab" for an in-run A/B; a number above 1 sets the least KiB, default 256): the least
// KiB of noted GPU writes worth a held part's write-back (vk_mem_mirror_wb_held), 0 when off. What no held part's
// write-back takes goes back at the end of the submission, after its last full barrier, as before.
int vk_mem_wb_early_min(void) {
    static int mode = -1, min_kib = 256;
    if (mode < 0) {
        const char *e = cfg_get("TOTK_WB_EARLY");
        mode = rt_knob("WB_EARLY", "1");
        if (e && atoi(e) > 1) min_kib = atoi(e), mode = 1;
    }
    return RT_KNOB_ON(mode) ? min_kib : 0;
}

// Moves the ranges tagged upto or lower from a[0, *n) to the end of out (keeping the others in order); their bytes.
static uint64_t wb_take_tagged(MRange *a, int *n, uint32_t upto, MRange **out, int *nout, int *cout) {
    uint64_t bytes = 0;
    int k = 0;
    for (int i = 0; i < *n; i++) {
        if (a[i].tag > upto) { a[k++] = a[i]; continue; }
        if (*nout == *cout) grow((void **)out, cout, sizeof **out);
        (*out)[(*nout)++] = a[i];
        bytes += a[i].size;
    }
    *n = k;
    return bytes;
}

// ---- write-backs of held parts (TOTK_WB_EARLY), on a thread of their own
// A held part has been submitted (vk_exec.c par_settle_front): the GPU writes tagged upto or lower, the walk's and every
// slice thread's, were all recorded into that part or the ones before it in queue order (slices of a later held pass,
// still being recorded, tag theirs higher). This thread takes them out of the lists, sorts them, records their
// write-back into a command buffer of its own pool and submits it, while the walk goes on: done at the held part's end
// by the walk, the 13,700 ranges of a held pass in the Hilltop Layer of Surf Hills kept the part off the queue 400 us,
// while the GPU waited for it. The submission's end (vk_mem_mirror_end) waits for the jobs, so its last part, which
// signals the fence, follows their copies in queue order. A copy that lands after a later part wrote the same range again
// copies newer bytes, and that write is noted again with its own tag and copied at its own time.
typedef struct { uint64_t fence; uint32_t upto, sub; uint64_t min_bytes; MRange *own; int nown; } WbJob;
#define WB_JOBS 16
static struct {
    PlatMutex lock;
    PlatCond cv, idle;
    PlatThread *t;
    WbJob q[WB_JOBS];
    int head, n, busy;
    VkCommandPool pool;
    VkCommandBuffer *free;   // command buffers whose submission completed (wb_cb_done)
    int nfree, cfree;
    uint64_t bytes, copies, ranges, jobs;   // what the thread wrote back, for mirror_end's counters
} WB = {.lock = PLAT_MUTEX_INIT, .cv = PLAT_COND_INIT, .idle = PLAT_COND_INIT};

static void wb_cb_done(void *arg) {   // completion thread
    plat_mutex_lock(&WB.lock);
    if (WB.nfree == WB.cfree) grow((void **)&WB.free, &WB.cfree, sizeof *WB.free);
    WB.free[WB.nfree++] = arg;
    plat_mutex_unlock(&WB.lock);
}

static void wb_job_run(const WbJob *j) {
    static MRange *h;
    static int ch;
    int nh = 0;
    uint64_t bytes = 0;
    for (int i = 0; i < j->nown; i++) {
        if (nh == ch) grow((void **)&h, &ch, sizeof *h);
        h[nh++] = j->own[i], bytes += j->own[i].size;
    }
    plat_mutex_lock(&mirror_lock);   // the list of lists
    for (WbLocal *w = wb_locals; w; w = w->next) {
        plat_mutex_lock(&w->lock);
        bytes += wb_take_tagged(w->a, &w->n, j->upto, &h, &nh, &ch);
        plat_mutex_unlock(&w->lock);
    }
    plat_mutex_unlock(&mirror_lock);
    if (!nh) return;
    VkCtx *c = vk_ctx();
    if (!WB.pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &WB.pool) != VK_SUCCESS) rt_fatal("vk: write-back command pool");
    }
    VkCommandBuffer cb = VK_NULL_HANDLE;
    plat_mutex_lock(&WB.lock);
    if (WB.nfree) cb = WB.free[--WB.nfree];
    plat_mutex_unlock(&WB.lock);
    if (cb) vkResetCommandBuffer(cb, 0);
    else {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = WB.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        if (vkAllocateCommandBuffers(c->device, &ai, &cb) != VK_SUCCESS) rt_fatal("vk: write-back command buffer");
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COPY_BIT;
    VkAccessFlags2 reads = VK_ACCESS_2_TRANSFER_READ_BIT, writes = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    if (mirror_sparse) {
        stage |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        reads |= VK_ACCESS_2_SHADER_READ_BIT;
        writes |= VK_ACCESS_2_SHADER_WRITE_BIT;
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, stage, reads);
    for (int i = 0; i < nh; i++) __atomic_store_n(&h[i].r->wb_sub, j->sub, __ATOMIC_RELAXED);
    uint64_t wbytes = 0, wcopies = 0;
    copy_ranges(cb, h, nh, 0, &wbytes, &wcopies, j->fence);
    mbarrier(cb, stage, writes, VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    vk_queue_lock();
    VkResult r = vkQueueSubmit(c->gfx_queue, 1, &si, VK_NULL_HANDLE);
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: write-back submit of value %llu failed (%d)", (unsigned long long)j->fence, (int)r);
    }
    vk_completion_add(j->fence, wb_cb_done, cb);
    plat_mutex_lock(&WB.lock);
    WB.bytes += wbytes, WB.copies += wcopies, WB.ranges += (uint64_t)nh, WB.jobs++;
    plat_mutex_unlock(&WB.lock);
}

static void wb_thread(void *arg) {
    (void)arg;
    plat_mutex_lock(&WB.lock);
    for (;;) {
        while (!WB.n) plat_cond_wait(&WB.cv, &WB.lock);
        WbJob j = WB.q[WB.head];
        WB.busy = 1;
        plat_mutex_unlock(&WB.lock);
        wb_job_run(&j);
        free(j.own);
        plat_mutex_lock(&WB.lock);
        WB.head = (WB.head + 1) % WB_JOBS, WB.n--, WB.busy = 0;
        plat_cond_broadcast(&WB.idle);
    }
}

// The replay thread: every job queued so far is submitted.
static void wb_jobs_wait(void) {
    if (!WB.t) return;
    plat_mutex_lock(&WB.lock);
    while (WB.n) plat_cond_wait(&WB.idle, &WB.lock);
    plat_mutex_unlock(&WB.lock);
}

static void wb_shutdown(VkCtx *c) {   // vk_mem_shutdown, the device idle: the pool goes with its command buffers
    wb_jobs_wait();
    plat_mutex_lock(&WB.lock);
    if (WB.pool) vkDestroyCommandPool(c->device, WB.pool, NULL);
    WB.pool = VK_NULL_HANDLE, WB.nfree = 0;
    plat_mutex_unlock(&WB.lock);
}

void vk_mem_mirror_wb_held(uint64_t fence, uint32_t upto, uint64_t min_bytes) {
    if (!vk_mirror_on) return;
    // Enough to be worth a command buffer? The walk's own list here, the slice threads' under their locks.
    uint64_t bytes = 0;
    for (int i = 0; i < nwbl; i++) bytes += wbl[i].tag <= upto ? wbl[i].size : 0;
    for (WbLocal *w = __atomic_load_n(&wb_locals, __ATOMIC_ACQUIRE); w && bytes < min_bytes; w = w->next) {
        plat_mutex_lock(&w->lock);
        for (int i = 0; i < w->n; i++) bytes += w->a[i].tag <= upto ? w->a[i].size : 0;
        plat_mutex_unlock(&w->lock);
    }
    if (bytes < min_bytes) return;
    WbJob j = {fence, upto, msub, min_bytes, NULL, 0};
    int cown = 0;
    wb_take_tagged(wbl, &nwbl, upto, &j.own, &j.nown, &cown);   // the walk's list is this thread's to change
    plat_mutex_lock(&WB.lock);
    if (!WB.t) {
        WB.t = plat_thread_new(wb_thread, NULL, "vk write-back");
        if (!WB.t) rt_fatal("vk: cannot start the write-back thread");
    }
    while (WB.n == WB_JOBS) plat_cond_wait(&WB.idle, &WB.lock);
    WB.q[(WB.head + WB.n) % WB_JOBS] = j;
    WB.n++;
    plat_cond_signal(&WB.cv);
    plat_mutex_unlock(&WB.lock);
}

int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence) {
    if (!vk_mirror_on) return 0;
    int rec = 0;
    wb_jobs_wait();   // their copies go to the queue ahead of the last part, and the lists are this thread's again
    if (WB.t) {
        plat_mutex_lock(&WB.lock);
        mrep.wb_bytes += WB.bytes, mrep.wb_copies += WB.copies, mrep.wb_ranges += WB.ranges;
        mrep.early_n += WB.jobs, mrep.early_bytes += WB.bytes;
        WB.bytes = WB.copies = WB.ranges = WB.jobs = 0;
        plat_mutex_unlock(&WB.lock);
    }
    if (__atomic_load_n(&wb_locals, __ATOMIC_ACQUIRE)) wb_locals_take(), kept_take();
    if (nwbl) {
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COPY_BIT;
        VkAccessFlags2 reads = VK_ACCESS_2_TRANSFER_READ_BIT, writes = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        if (mirror_sparse) {
            stage |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            reads |= VK_ACCESS_2_SHADER_READ_BIT;
            writes |= VK_ACCESS_2_SHADER_WRITE_BIT;
        }
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, stage, reads);
        for (int i = 0; i < nwbl; i++) wbl[i].r->wb_sub = msub;
        mrep.wb_ranges += (uint64_t)nwbl;
        rec += copy_ranges(cb, wbl, nwbl, 0, &mrep.wb_bytes, &mrep.wb_copies, fence) + 2;
        mbarrier(cb, stage, writes,
                 VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        nwbl = 0;
    }
    if (mirror_verify) rec += verify_record(cb, fence);
    plat_mutex_lock(&mirror_stat_lock);
    uint64_t *d = (uint64_t *)&macc, *s = (uint64_t *)&mrep;
    for (size_t i = 0; i < sizeof macc / sizeof(uint64_t); i++) d[i] += s[i];
    plat_mutex_unlock(&mirror_stat_lock);
    memset(&mrep, 0, sizeof mrep);
    return rec;
}

// One summary line: with TOTK_TRACE=perf every 60 frames, else at frames 600 and 1800.
static void mirror_log(uint64_t frame) {
    static uint64_t last, ww_last;
    plat_mutex_lock(&gen_lock);
    uint64_t wc = ww_calls, wb = ww_bytes, wt = ww_ticks, dc = ww_dirty_calls, dt = ww_dirty_ticks, wp = ww_pages;
    ww_pages = 0;
    uint64_t nr = n_frz_refused;
    ww_calls = ww_bytes = ww_ticks = ww_dirty_calls = ww_dirty_ticks = n_frz_refused = 0;
    uint64_t wcpu = ww_cpu_ns, who[WW_WHO_N], sl_ns = ww_slow_ns, sl_cpu = ww_slow_cpu, sl_a = ww_slow_addr,
             sl_n = ww_slow_size, sl_p = ww_slow_pages;
    int sl_who = ww_slow_who;
    memcpy(who, ww_who_ns, sizeof who);
    memset(ww_who_ns, 0, sizeof ww_who_ns);
    ww_cpu_ns = ww_slow_ns = ww_slow_cpu = ww_slow_addr = ww_slow_size = ww_slow_pages = 0;
    plat_mutex_unlock(&gen_lock);
    double wf = frame > ww_last ? (double)(frame - ww_last) : 1.0;
    ww_last = frame;
    uint64_t fz, ff, ft, fo;
    mem_freeze_stats(&fz, &ff, &ft, &fo);
    if (rt_trace & RT_TRACE_PERF)
        rt_log("mem: write-watch frame %llu: per frame, every consumer: %.1f queries over %.1f MiB, %.3f ms (%.1f found "
               "%.0f written pages, %.3f ms); frozen %.0f MiB; since the last line: freezes %llu, thaws by a write %llu, by a file read "
               "or mapping change %llu, refused %llu",
               (unsigned long long)frame, wc / wf, wb / wf / 1048576.0, qpc_ms(wt) / wf, dc / wf, wp / wf, qpc_ms(dt) / wf,
               fz * 2.0, (unsigned long long)ff, (unsigned long long)ft, (unsigned long long)fo, (unsigned long long)nr);
    if (rt_trace & RT_TRACE_PERF) {
        // #317: the queries' wall time against their CPU time, the slowest one, and what else asked the kernel about
        // guest memory meanwhile (VirtualQuery and VirtualProtect by kind, the pointer guards' cache misses).
        PlatVmStat vm[PLAT_VM_N];
        plat_vm_stats(vm);
        uint64_t gc, gm;
        rt_guard_stats(&gc, &gm);
        char vq[PLAT_VM_N][96];
        static const char *const vm_name[PLAT_VM_N] = {"readable", "state", "lazy heap", "protect"};
        for (int k = 0; k < PLAT_VM_N; k++)
            snprintf(vq[k], sizeof vq[k], "%s %.1f (%.3f ms, longest %.2f ms, run %.0f MiB)", vm_name[k],
                     vm[k].calls / wf, vm[k].ns / 1e6 / wf, vm[k].max_ns / 1e6, vm[k].max_run / 1048576.0);
        rt_log("mem: write-watch frame %llu: per frame, on the CPU %.3f ms of %.3f ms; by consumer ms: %s %.3f, %s %.3f, "
               "%s %.3f, %s %.3f, %s %.3f; slowest %.3f ms (%.3f on the CPU) over %.1f MiB at %#llx, %llu pages, by %s; "
               "host memory calls per frame: %s, %s, %s, %s; pointer guards %.0f, cache misses %.1f",
               (unsigned long long)frame, wcpu / 1e6 / wf, qpc_ms(wt) / wf, ww_who_name[0], who[0] / 1e6 / wf,
               ww_who_name[1], who[1] / 1e6 / wf, ww_who_name[2], who[2] / 1e6 / wf, ww_who_name[3], who[3] / 1e6 / wf,
               ww_who_name[4], who[4] / 1e6 / wf, sl_ns / 1e6, sl_cpu / 1e6, sl_n / 1048576.0, (unsigned long long)sl_a,
               (unsigned long long)sl_p, ww_who_name[sl_who], vq[0], vq[1], vq[2], vq[3], gc / wf, gm / wf);
    }
    if (!vk_mirror_on) return;
    plat_mutex_lock(&mirror_stat_lock);
    MirrorAcc a = macc;
    memset(&macc, 0, sizeof macc);
    plat_mutex_unlock(&mirror_stat_lock);
    double f = frame > last ? (double)(frame - last) : 1.0, M = 1048576.0;
    last = frame;
    uint64_t fc = (uint64_t)plat_xchg64(&m_flush_calls, 0), fb = (uint64_t)plat_xchg64(&m_flush_bytes, 0);
    char ver[160] = "";
    if (mirror_verify)
        snprintf(ver, sizeof ver, "; verify: pages compared %lld, skipped %lld (written since), mismatches %lld",
                 (long long)plat_xchg64(&v_pages, 0), (long long)plat_xchg64(&v_skipped, 0),
                 (long long)plat_xchg64(&v_mismatch, 0));
    rt_log("mem: mirror frame %llu: pools %lld, %.1f MiB device-local; stay imported: budget %lld (%.1f MiB), untracked "
           "%lld, overlapping %lld, failed %lld; per frame: submits %.2f, syncs %.1f, upload %.3f MiB in %.1f ranges, "
           "first use %.2f pools %.3f MiB, own GPU writes not copied in %.3f MiB, write-back %.3f MiB in %.1f ranges "
           "(before the end: %.2f command buffers, %.3f MiB), flush calls %.1f (%.3f MiB), queries %.1f pre-snap (%.1f MiB) + %.1f sync, "
           "prologues %.2f, tracking cpu %.3f ms (pre-snap %.3f, syncs %.3f), sync snaps %.1f MiB, pre-snapped pools %.1f "
           "(used %.1f, unused %.1f MiB)%s",
           (unsigned long long)frame, (long long)m_live_n, m_live_bytes / M, (long long)m_refused_n[MR_BUDGET],
           m_refused_bytes[MR_BUDGET] / M, (long long)m_refused_n[MR_UNTRACKED], (long long)m_refused_n[MR_OVERLAP],
           (long long)m_refused_n[MR_ALLOC], a.subs / f, a.syncs / f, a.upload_bytes / f / M, a.upload_copies / f,
           a.first_n / f, a.first_bytes / f / M, a.own_bytes / f / M, a.wb_bytes / f / M, a.wb_copies / f,
           a.early_n / f, a.early_bytes / f / M, fc / f, fb / f / M,
           a.presnap_queries / f, a.presnap_bytes / f / M, a.sync_queries / f, a.prologues / f, qpc_ms(a.presnap_t + a.sync_t) / f,
           qpc_ms(a.presnap_t) / f, qpc_ms(a.sync_t) / f, a.sync_bytes / f / M, a.presnap_pools / f, a.presnap_used / f,
           a.presnap_unused_bytes / f / M, ver);

}

// With TOTK_TRACE=perf every 600 frames: what holds device-local memory, to see what grows over a long session.
// VMA's blocks and allocations on the device-local heaps, texture entries with their own memory, the VRAM mirrors and
// the objects waiting for their last submission to retire.
static void vram_log(uint64_t frame) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    uint64_t blocks = 0, block_bytes = 0, allocs = 0, alloc_bytes = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            blocks += vb[h].statistics.blockCount, block_bytes += vb[h].statistics.blockBytes,
                allocs += vb[h].statistics.allocationCount, alloc_bytes += vb[h].statistics.allocationBytes;
    VkTexCensus t;
    vk_texture_census(&t);
    const double M = 1048576.0;
    rt_log("mem: vram frame %llu: vma %llu blocks %.0f MiB, %llu allocations %.0f MiB; textures live %llu (own memory %llu, "
           "%.0f MiB), released %llu (own memory %llu, %.0f MiB); mirrors %lld %.0f MiB; releases pending %d",
           (unsigned long long)frame, (unsigned long long)blocks, block_bytes / M, (unsigned long long)allocs,
           alloc_bytes / M, (unsigned long long)t.live, (unsigned long long)t.own_live, t.bytes_live / M,
           (unsigned long long)t.released, (unsigned long long)t.own_released, t.bytes_released / M, (long long)m_live_n,
           m_live_bytes / M, dead_pending());
}

// Device class: INTEGRATED -> uma; a discrete GPU with a DEVICE_LOCAL heap of 1 GiB or more beside a
// heap that is not device-local (host memory) -> discrete; otherwise uma (import everywhere). Sets the mirror mode.
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m) {
    int big_local = -1, host_heap = -1;
    for (uint32_t h = 0; h < m->memoryHeapCount; h++) {
        if (m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (m->memoryHeaps[h].size >= (1ull << 30) && big_local < 0) big_local = (int)h;
        } else if (host_heap < 0) host_heap = (int)h;
    }
    static const char *const types[] = {"other", "integrated", "discrete", "virtual", "cpu"};
    int t = (int)p->deviceType;
    const char *tn = t >= 0 && t <= 4 ? types[t] : "unknown";
    const char *cls, *why;
    if (p->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) cls = "uma", why = "integrated gpu";
    else if (p->deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) cls = "uma", why = "not a discrete gpu";
    else if (big_local < 0) cls = "uma", why = "no device-local heap of 1 GiB or more";
    else if (host_heap < 0) cls = "uma", why = "every heap is device-local";
    else cls = "discrete", why = "device-local heap separate from host memory";
    int discrete = cls[0] == 'd';
    char line[2048];
    size_t k = (size_t)snprintf(line, sizeof line, "mem: class %s (%s; device type %d %s, %s)", cls, why, t, tn, p->deviceName);
    for (uint32_t h = 0; h < m->memoryHeapCount; h++)
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; heap %u %.0f MiB flags %#x%s%s", h,
                              m->memoryHeaps[h].size / 1048576.0, m->memoryHeaps[h].flags,
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT ? " MULTI_INSTANCE" : "");
    rt_log("%s", line);
    k = (size_t)snprintf(line, sizeof line, "mem: types");
    for (uint32_t i = 0; i < m->memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m->memoryTypes[i].propertyFlags;
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; %u heap %u flags %#x%s%s%s%s%s", i,
                              m->memoryTypes[i].heapIndex, f, f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " HOST_VISIBLE" : "",
                              f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " HOST_COHERENT" : "",
                              f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " HOST_CACHED" : "",
                              f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT ? " LAZY" : "");
    }
    rt_log("%s", line);
    mirror_mode(discrete, p->vendorID);
}

// ---------------------------------------------------------------- upload batches and deferred destruction
typedef struct { VkBuffer buf; VmaAllocation alloc; } Staging;
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    uint64_t seq;
    Staging *stg;
    int nstg, cap;
    int ts, timed;   // its timestamp pair in up_pool (-1: none), and whether this use of it writes the pair
} Batch;

// The perf line's upload figures (TOTK_TRACE=perf, the full overlay). Upload batches go to the queue outside
// vk_sync.c's timed submissions, and the ASTC decoding of texture uploads records into them, so their GPU time, their
// count, the bytes staged for them and the time threads wait for them are counted here (perf_gpu_line). Each batch owns
// one timestamp pair, kept with its reused command buffer; batches past UP_PAIRS go untimed.
#define UP_PAIRS 256
static VkQueryPool up_pool;
static int up_state, up_pairs;   // up_state: 0 not set up, 1 ready, -1 no timestamps
static float up_period;
static volatile int64_t up_gpu_ns, up_timed, up_batches, up_wait_ns, up_bytes;

// A timestamp pool of n pairs for the perf line's side timings (upload batches, vk_present.c's blit), or NULL when
// the graphics queue has no 64-bit timestamps. *period: ns per tick.
VkQueryPool vk_ts_pool(uint32_t pairs, float *period) {
    VkCtx *c = vk_ctx();
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    if (!qp) return VK_NULL_HANDLE;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    free(qp);
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * pairs};
    VkQueryPool p = VK_NULL_HANDLE;
    // ponytail: a counter narrower than 64 bits goes untimed; mask it as vk_sync.c does if one shows up
    if (bits < 64 || c->props.limits.timestampPeriod <= 0 || vkCreateQueryPool(c->device, &qi, NULL, &p) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    *period = c->props.limits.timestampPeriod;
    return p;
}

static int up_ts_ready(void) {
    if (!up_state) up_state = (up_pool = vk_ts_pool(UP_PAIRS, &up_period)) ? 1 : -1;
    return up_state > 0;
}

void vk_upload_perf(uint64_t *gpu_ns, uint64_t *timed, uint64_t *batches, uint64_t *wait_ns, uint64_t *bytes) {
    *gpu_ns = (uint64_t)plat_xchg64(&up_gpu_ns, 0), *timed = (uint64_t)plat_xchg64(&up_timed, 0);
    *batches = (uint64_t)plat_xchg64(&up_batches, 0), *wait_ns = (uint64_t)plat_xchg64(&up_wait_ns, 0);
    *bytes = (uint64_t)plat_xchg64(&up_bytes, 0);
}

enum { DEAD_IMAGE = 1, DEAD_IMAGE_VMA, DEAD_VIEW, DEAD_BUFFER_VMA };
typedef struct { int kind; uint64_t handle; VmaAllocation alloc; uint64_t seq, tl; } Dead;

static PlatRMutex cmd_lock;
static PlatOnce cmd_once = PLAT_ONCE_INIT;
static VkCommandPool cmd_pool;
static Batch open_b;
static int open_active;
static Batch *inflight;
static int ninflight, cap_inflight;
static Batch *spare;                  // completed batches whose command buffer and fence are reused
static int nspare, cap_spare;
static uint64_t next_seq, done_seq;   // batches handed out / completed (all batches up to done_seq)
// vk_upload_flush(1) waits for a batch's fence with cmd_lock let go, so the replay's own flushes (every split part) do
// not wait behind another thread's GPU wait: pinned is the batch being waited on, whose fence retire keeps out of the
// spare list until the waiter is done with it (pinned_done); wait_lock lets one thread wait at a time.
static uint64_t pinned;               // cmd_lock: seq of the batch a waiter waits on, 0 none
static int pinned_done;               // cmd_lock: retire met it: the waiter puts it in the spare list
static Batch pinned_batch;
static PlatMutex wait_lock = PLAT_MUTEX_INIT;
static Dead *dead;
static int ndead, cap_dead;
static uint64_t tl_protected;         // newest submission being recorded or submitted; guards deferred destruction

static void cmd_init(void *p) {
    (void)p;
    return;
}

static void grow(void **arr, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 16;
    void *n = realloc(*arr, (size_t)*cap * elem);
    if (!n) rt_fatal("vk: out of memory (upload bookkeeping)");
    *arr = n;
}

static void free_staging(VkCtx *c, Batch *b) {
    for (int i = 0; i < b->nstg; i++) vmaDestroyBuffer(c->vma, b->stg[i].buf, b->stg[i].alloc);
    b->nstg = 0;
}

// Batches complete in submission order (one queue): retire from the front while their fences are signalled.
static void retire(VkCtx *c, int wait_all) {
    int k = 0;
    for (; k < ninflight; k++) {
        Batch *b = &inflight[k];
        if (wait_all) vkWaitForFences(c->device, 1, &b->fence, VK_TRUE, UINT64_MAX);
        else if (vkGetFenceStatus(c->device, b->fence) != VK_SUCCESS) break;
        uint64_t v[2];
        if (b->timed && vkGetQueryPoolResults(c->device, up_pool, 2 * (uint32_t)b->ts, 2, sizeof v, v, sizeof v[0],
                                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && v[1] >= v[0]) {
            plat_add64(&up_gpu_ns, (int64_t)((double)(v[1] - v[0]) * up_period));
            plat_inc64(&up_timed);
        }
        free_staging(c, b);
        done_seq = b->seq;
        if (b->seq == pinned) {   // a waiter still waits on its fence (vk_upload_flush)
            pinned_batch = *b, pinned_done = 1;
            continue;
        }
        if (nspare == cap_spare) grow((void **)&spare, &cap_spare, sizeof *spare);
        spare[nspare++] = *b;
    }
    memmove(inflight, inflight + k, (size_t)(ninflight - k) * sizeof *inflight);
    ninflight -= k;
}

static void collect_locked(VkCtx *c) {
    retire(c, 0);
    uint64_t tl = 0;
    if (c->timeline) vkGetSemaphoreCounterValue(c->device, c->timeline, &tl);
    int k = 0;
    for (int i = 0; i < ndead; i++) {
        Dead *d = &dead[i];
        if (d->seq > done_seq || d->tl > tl) { dead[k++] = *d; continue; }
        switch (d->kind) {
        case DEAD_IMAGE: vkDestroyImage(c->device, (VkImage)(uintptr_t)d->handle, NULL); break;
        case DEAD_IMAGE_VMA: vmaDestroyImage(c->vma, (VkImage)(uintptr_t)d->handle, d->alloc); break;
        case DEAD_VIEW: vkDestroyImageView(c->device, (VkImageView)(uintptr_t)d->handle, NULL); break;
        case DEAD_BUFFER_VMA: vmaDestroyBuffer(c->vma, (VkBuffer)(uintptr_t)d->handle, d->alloc); break;
        }
    }
    ndead = k;
}

VkCommandBuffer vk_upload_begin(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    if (!c) rt_fatal("vk: upload before vk_init");
    if (open_active) return open_b.cb;
    if (!cmd_pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &cmd_pool) != VK_SUCCESS) rt_fatal("vk: upload command pool");
    }
    retire(c, 0);
    memset(&open_b, 0, sizeof open_b);
    if (nspare) {
        Batch s = spare[--nspare];
        open_b.cb = s.cb, open_b.fence = s.fence, open_b.stg = s.stg, open_b.cap = s.cap, open_b.ts = s.ts;
        vkResetCommandBuffer(open_b.cb, 0);
        vkResetFences(c->device, 1, &open_b.fence);
    } else {
        open_b.ts = up_pairs < UP_PAIRS ? up_pairs++ : -1;
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(c->device, &ai, &open_b.cb) != VK_SUCCESS ||
            vkCreateFence(c->device, &fi, NULL, &open_b.fence) != VK_SUCCESS)
            rt_fatal("vk: upload command buffer");
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(open_b.cb, &bi);
    if (open_b.ts >= 0 && perf_gpu_timing() && up_ts_ready()) {
        vkCmdResetQueryPool(open_b.cb, up_pool, 2 * (uint32_t)open_b.ts, 2);
        vkCmdWriteTimestamp(open_b.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, up_pool, 2 * (uint32_t)open_b.ts);
        open_b.timed = 1;
    }
    open_b.seq = ++next_seq;
    vk_crumb(open_b.cb, VK_CRUMB_UPLOAD, open_b.seq);
    open_active = 1;
    return open_b.cb;
}

void vk_upload_end(void) { plat_rmutex_unlock(&cmd_lock); }

// The open batch was numbered when vk_upload_begin opened it, and whichever thread flushes it, vk_upload_flush returns
// that number: a caller can hold it for vk_upload_done without flushing the batch itself.
uint64_t vk_upload_seq(void) { return open_b.seq; }

// Staging memory comes from a pool of its own that keeps STG_MIN_BLOCKS blocks from its first use (at boot). In the
// default pools VMA creates a block under the lock of its memory type (VmaBlockVector::Allocate), so a block made by
// the headroom thread (vk_headroom.c) or a burst of uploads held every staging allocation of the render thread: one
// took 47-50 ms, for a few bytes, in the first frame after a save load. Larger buffers than a block go to the default
// pool.
#define STG_BLOCK (64ull << 20)
#define STG_MIN_BLOCKS 2
static VmaPool stg_pool;
static int stg_pool_tried;

void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc) {
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (!__atomic_load_n(&stg_pool_tried, __ATOMIC_ACQUIRE)) {
        plat_once(&cmd_once, cmd_init, NULL);
        plat_rmutex_lock(&cmd_lock);
        if (!stg_pool_tried) {
            uint32_t type;
            VmaPoolCreateInfo pi = {.blockSize = STG_BLOCK, .minBlockCount = STG_MIN_BLOCKS};
            if (vmaFindMemoryTypeIndexForBufferInfo(c->vma, &bi, &aci, &type) == VK_SUCCESS) {
                pi.memoryTypeIndex = type;
                if (vmaCreatePool(c->vma, &pi, &stg_pool) != VK_SUCCESS) stg_pool = NULL;
            }
            rt_log("vk: staging pool %s (%d blocks of %llu MiB kept)", stg_pool ? "created" : "unavailable: default pool",
                   STG_MIN_BLOCKS, (unsigned long long)(STG_BLOCK >> 20));
            __atomic_store_n(&stg_pool_tried, 1, __ATOMIC_RELEASE);
        }
        plat_rmutex_unlock(&cmd_lock);
    }
    if (stg_pool && bi.size <= STG_BLOCK / 2) aci.pool = stg_pool;
    VmaAllocationInfo info;
    VkBuffer b;
    VmaAllocation a;
    VkResult r = vmaCreateBuffer(c->vma, &bi, &aci, &b, &a, &info);
    if (r != VK_SUCCESS) rt_fatal("vk: staging buffer of %llu bytes failed (%d)", (unsigned long long)size, (int)r);
    plat_add64(&up_bytes, (int64_t)bi.size);
    *buf = b;
    if (alloc) *alloc = a;   // the caller destroys it (vmaDestroyBuffer) after vk_upload_flush(1)
    else {
        if (open_b.nstg == open_b.cap) grow((void **)&open_b.stg, &open_b.cap, sizeof *open_b.stg);
        open_b.stg[open_b.nstg++] = (Staging){b, a};
    }
    return info.pMappedData;
}

uint64_t vk_upload_flush(int wait) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    uint64_t seq = next_seq;
    if (c && open_active) {
        for (int i = 0; i < open_b.nstg; i++) vmaFlushAllocation(c->vma, open_b.stg[i].alloc, 0, VK_WHOLE_SIZE);
        vk_crumb(open_b.cb, VK_CRUMB_UPLOAD_END, open_b.seq);
        if (open_b.timed)
            vkCmdWriteTimestamp(open_b.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, up_pool, 2 * (uint32_t)open_b.ts + 1);
        vkEndCommandBuffer(open_b.cb);
        plat_inc64(&up_batches);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &open_b.cb};
        rec_submit_wait();   // a replay part the recorder thread is still submitting precedes this batch (vk_rec.c rec_submit)
        vk_queue_lock();
        VkResult r = vkQueueSubmit(c->gfx_queue, 1, &si, open_b.fence);
        vk_queue_unlock();
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: upload submit failed (%d)", (int)r);
        }
        if (ninflight == cap_inflight) grow((void **)&inflight, &cap_inflight, sizeof *inflight);
        inflight[ninflight++] = open_b;
        open_active = 0;
    }
    if (c && wait) {
        uint64_t w0 = ninflight && done_seq < seq ? kern_now_ns() : 0;
        // Without cmd_lock when this thread holds it once (a caller holding it around the flush keeps it for the wait).
        int unlocked = cmd_lock.depth == 1;
        if (unlocked && ninflight && done_seq < seq) {
            plat_rmutex_unlock(&cmd_lock);
            plat_mutex_lock(&wait_lock);
            plat_rmutex_lock(&cmd_lock);
        }
        while (ninflight && done_seq < seq) {
            VkFence f = inflight[0].fence;
            if (unlocked) pinned = inflight[0].seq, pinned_done = 0, plat_rmutex_unlock(&cmd_lock);
            // A lost device returns at once and its fence never signals: without this check the loop spins forever
            // (a guest thread at 100% of a core and an idle GPU, with no fault reported).
            VkResult w = vkWaitForFences(c->device, 1, &f, VK_TRUE, UINT64_MAX);
            if (unlocked) plat_rmutex_lock(&cmd_lock);
            if (w != VK_SUCCESS && w != VK_TIMEOUT) {
                vk_device_fault_report(w);
                rt_fatal("vk: upload wait failed (%d)", (int)w);
            }
            retire(c, 0);
            if (unlocked) {
                if (pinned_done) {
                    if (nspare == cap_spare) grow((void **)&spare, &cap_spare, sizeof *spare);
                    spare[nspare++] = pinned_batch;
                }
                pinned = 0, pinned_done = 0;
            }
        }
        if (unlocked && w0) plat_mutex_unlock(&wait_lock);
        if (w0) plat_add64(&up_wait_ns, (int64_t)(kern_now_ns() - w0));
    }
    if (c) collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
    return seq;
}

int vk_upload_done(uint64_t seq) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    if (c && done_seq < seq) retire(c, 0);
    int d = done_seq >= seq;
    plat_rmutex_unlock(&cmd_lock);
    return d;
}

static void defer(int kind, uint64_t handle, VmaAllocation alloc) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (ndead == cap_dead) grow((void **)&dead, &cap_dead, sizeof *dead);
    // The open batch may record into the object: it counts as the newest user.
    dead[ndead++] = (Dead){kind, handle, alloc, open_active ? open_b.seq : next_seq, tl_protected};
    plat_rmutex_unlock(&cmd_lock);
}

static int dead_pending(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    int n = ndead;
    plat_rmutex_unlock(&cmd_lock);
    return n;
}

void vk_defer_image(VkImage img, VmaAllocation alloc) {
    if (img) defer(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc);
}
void vk_defer_view(VkImageView view) { if (view) defer(DEAD_VIEW, (uint64_t)(uintptr_t)view, NULL); }
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc) {
    if (buf) defer(DEAD_BUFFER_VMA, (uint64_t)(uintptr_t)buf, alloc);
}

void vk_mem_protect_until(uint64_t timeline_value) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    // New retirements conservatively share the frontier; already queued objects keep their earlier protection value.
    if (timeline_value > tl_protected) tl_protected = timeline_value;
    plat_rmutex_unlock(&cmd_lock);
}

void vk_mem_collect(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
    vk_mem_pressure_check();
}

void vk_mem_stats(VkMemStats *s) {
    memset(s, 0, sizeof *s);
    pool_lock_shared();
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT) s->imported_pools++;
        else if (r->kind == POOL_PHYS) s->physical_pools++;
        else s->virtual_pools++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    s->import_failures = (uint64_t)n_import_failures;
    s->upload_batches = next_seq;
}

// Fills out with the GPU-side numbers that explain the "other private" bucket of mem_commit_report (memory.c). The
// VMA heap numbers are the device-local heaps, as vram_log counts them; on an integrated GPU they are system memory.
void vk_mem_gpu_summary(char *out, size_t n) {
    VkCtx *c = vk_ctx();
    if (!c || !c->vma) {
        if (n) *out = 0;
        return;
    }
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    uint64_t dev = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) dev += vb[h].statistics.blockBytes;
    uint64_t stg = 0;
    if (__atomic_load_n(&stg_pool_tried, __ATOMIC_ACQUIRE) && stg_pool) {
        VmaStatistics ps;
        vmaGetPoolStatistics(c->vma, stg_pool, &ps);
        stg = ps.blockBytes;
    }
    snprintf(out, n, "; gpu: device-local %.0f MiB, mirrors %.0f MiB, staging pool %.0f MiB, usnap arenas %.0f MiB",
             dev / 1048576.0, m_live_bytes / 1048576.0, stg / 1048576.0, vk_usnap_arena_bytes() / 1048576.0);
}

void vk_mem_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    vk_upload_flush(1);
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    if (mirror_sparse) sparse_cache_close();
    if (sparse_copy.pipeline) vkDestroyPipeline(c->device, sparse_copy.pipeline, NULL);
    if (sparse_copy.layout) vkDestroyPipelineLayout(c->device, sparse_copy.layout, NULL);
    if (sparse_copy.set) vkDestroyDescriptorSetLayout(c->device, sparse_copy.set, NULL);
    memset(&sparse_copy, 0, sizeof sparse_copy);
    vk_texture_shutdown();
    plat_rmutex_lock(&cmd_lock);
    retire(c, 1);
    uint64_t keep = tl_protected;
    tl_protected = 0;
    for (int i = 0; i < ndead; i++) dead[i].seq = 0, dead[i].tl = 0;   // the device is idle
    collect_locked(c);
    tl_protected = keep;
    for (int i = 0; i < nspare; i++) {
        vkDestroyFence(c->device, spare[i].fence, NULL);
        free(spare[i].stg);
    }
    nspare = 0;
    if (cmd_pool) vkDestroyCommandPool(c->device, cmd_pool, NULL);
    wb_shutdown(c);
    cmd_pool = VK_NULL_HANDLE;
    if (stg_pool) vmaDestroyPool(c->vma, stg_pool);   // after retire: no staging buffer is left in it
    stg_pool = NULL;
    stg_pool_tried = 0;
    plat_rmutex_unlock(&cmd_lock);
    plat_mutex_lock(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) destroy_rec(c, pool_index.e[i].buf);
    pool_index.n = 0;
    nm = npend = nwbl = 0;   // mirrors went with their pools
    for (WbLocal *w = wb_locals; w; w = w->next) w->n = w->nk = 0;   // pending kept ranges point at pools just freed
    plat_mutex_lock(&shadow_lock);
    nstouch = nspend = nsblocked = spend_dirty = 0;   // shadows too
    s_live_n = s_live_bytes = 0;
    plat_mutex_unlock(&shadow_lock);
    for (int i = 0; i < nretiring; i++) destroy_rec(c, retiring[i]);   // finalized ones whose callback never ran
    nretiring = 0;
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    nphys = 0;
    plat_mutex_unlock(&pool_lock);
    plat_mutex_lock(&gen_lock);
    for (uint64_t i = 0; i < GEN_CHUNKS; i++) free(gen_chunk[i]), gen_chunk[i] = NULL;
    plat_mutex_unlock(&gen_lock);
}
