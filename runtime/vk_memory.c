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
    // Staged mirror (TOTK_MIRROR_STAGED, "staged mirror" below). staged is set by vk_pool_init and never changes;
    // sfeed is set by mirror_create before it stores MS_ON and never changes.
    uint8_t staged;                    // lazy pool of TOTK_MIRROR_STAGED's size: its first GPU use makes a mirror, no import
    uint8_t sfeed;                     // its mirror is filled from staging buffers and written back through readbacks
    uint8_t szero;                     // mirror_lock: a small staged mirror the next prologue zeroes and fills (staged_upload)
    uint8_t mrefused;                  // 1 + the MR_ reason of a refused mirror (refuse), for staged_import's count
    volatile int32_t sfell;            // imported after all (staged_import): the line is logged once
    struct KeptRange *own;             // own_lock, TOTK_STAGED_WB=fault: bytes only the mirror holds (staged_fetch)
    int nown, cown;
    int own_ix;                        // own_lock: 1 + its place in st_own while own holds a range, else 0
    uint8_t *cpu_read;                 // per page: the CPU fetched it, so GPU writes to it go back at the fence
    // TOTK_STAGED_FETCH_TEST, per page: the submission of the last recorded write, that of the reference bytes, and
    // the reference (ft_check).
    uint32_t *t_rec, *t_ref;
    uint8_t **t_page;
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
static uint64_t staged_min;              // TOTK_MIRROR_STAGED in bytes (mirror_mode): 0 off, 1 every lazy pool
#define STG_SMALL (16ull << 20)          // a staged mirror up to this size is first filled in a prologue (staged_upload)
static int staged_fault = 1;             // TOTK_STAGED_WB (mirror_mode): 1 fault (GPU-owned pages), 0 fence (copy-back)
static int staged_fetch(uint64_t page, int write, uint64_t rip, int kernel);
static void staged_forget(PoolRec *r);   // vk_pool_fini: a staged pool's owned pages open without a fetch
static int fetch_test;                   // TOTK_STAGED_FETCH_TEST (fetch exerciser below)
static void ft_frame(uint64_t frame);
static void staged_frame_log(uint64_t frame);   // vk_mem_census: what the staged pools did in the frame
// A staged pool that has not been imported: lazy stays 1 until staged_import imports it after all, and lazy_import
// stores 0 with release order only once r->buf is complete, so a reader that sees 0 here may read r->buf.
static inline int staged_pending(const PoolRec *r) { return r->staged && __atomic_load_n(&r->lazy, __ATOMIC_ACQUIRE); }

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
// where: for import_pool's line and the import log.
static int lazy_import(PoolRec *r, int at, const char *where) {
    if (!__atomic_load_n(&r->lazy, __ATOMIC_ACQUIRE)) return r->buf != VK_NULL_HANDLE;
    uint64_t ph = perf_ph_begin_cpu();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
    uint64_t t0 = qpc();
    plat_mutex_lock(&lazy_lock);
    if (r->lazy) {
        ImportTimes t = {0};
        uint64_t t1 = qpc();
        int ok = import_pool(vk_ctx(), r, r->lo, r->size, where, &t);
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
        if (import_log_on()) import_log(where, r, at, &t, ok);
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
    free(r->own);
    free(r->cpu_read);
    if (r->t_page)
        for (uint64_t i = 0; i <= (r->import_hi - r->import_lo) >> 12; i++) free(r->t_page[i]);
    free(r->t_page);
    free(r->t_rec);
    free(r->t_ref);
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
            // A staged pool is never imported by a resolve (resolve_locked); its mirror is fed from staging buffers.
            // Imported pools in shadow mode never get here (lazy_enabled), and mirror_mode clears staged_min unless
            // the mirror is on.
            r->staged = staged_min && p->size >= staged_min;
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

static void staged_released(const PoolRec *r, int imported, uint64_t ns);

static void pool_fini(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    uint64_t t_fini = qpc();
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
    if (r->sfeed && staged_fault) staged_forget(r);   // its owned pages open without a fetch: the storage goes away
    if (r->kind == POOL_IMPORT) mem_thaw(r->lo, r->size);   // the storage goes back to the guest writable
    // A pool never imported (a staged pool, TOTK_MIRROR_STAGED, the only lazy pool a resolve marks used): the GPU never
    // read or wrote its storage, only the VRAM mirror and staging buffers that the CPU filled at record time, and its
    // write-backs into guest memory check at the fence that the pool is still live (staged_wb_done). So its release
    // waits for nothing: with hundreds of small pools staged, a layer's exit released about 215 pools in a row, and
    // the queue-idle wait below costs a driver round trip each even with the GPU idle. No resolve finds the pool once it
    // is out of the index, so no import can start after this read.
    int never_imported = r->kind == POOL_IMPORT && __atomic_load_n(&r->lazy, __ATOMIC_ACQUIRE);
    // Images cut from this range go first; a used buffer or memory waits until the GPU is done with it, since the
    // guest gets the storage back when this returns.
    vk_texture_range_gone(r->lo, r->size, r->alloc);
    if (r->used && r->kind != POOL_VIRT && !never_imported) {
        uint64_t t0 = qpc();
        vk_upload_flush(1);
        uint64_t t1 = qpc();
        pool_fini_wait(c, r);
        pool_fini_note(r->size, t1 - t0, qpc() - t1);
    }
    vk_mem_collect();
    if (r->staged) staged_released(r, !never_imported, qpc() - t_fini);   // before the record may go
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
    // A copy without a buffer is a staged pool's (resolve_locked); once the pool was imported after all
    // (staged_import), the locked lookup gives the import and puts it in the copy.
    if (h && !h->buf && (!h->pool || !staged_pending(h->pool))) h = NULL;
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
    // First GPU use. A staged pool is not imported here: it resolves with no buffer, and VK_MEM_USE or VK_MEM_PICK
    // gives the span its mirror (vk_mem_use), so callers that only read the guest bytes, or ask whether an address
    // is backed, cost no import. staged_import imports it when the mirror path cannot serve.
    if (r && r->kind == POOL_IMPORT && !r->staged && __atomic_load_n(&r->lazy, __ATOMIC_ACQUIRE))
        lazy_import(r, (int)(e - pool_index.e), "on first GPU use");
    int st = r && r->kind == POOL_IMPORT && staged_pending(r);
    VkBuffer ib = r && r->kind == POOL_IMPORT && !st ? r->buf : VK_NULL_HANDLE;   // staged_pending: no torn r->buf
    if (r && (r->kind == POOL_IMPORT ? ib != VK_NULL_HANDLE || st : r->kind == POOL_PHYS)) {
        int i = (int)(e - pool_index.e);
        if ((i == 0 || pool_index.maxhi[i - 1] <= e->lo) && (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= e->hi)) {
            // pool_gen is stable here: it changes only under the exclusive lock.
            ResolveHit *h = rhit_add(__atomic_load_n(&pool_gen, __ATOMIC_RELAXED), e->lo, e->hi);
            h->cls = rec_class(r);
            if (r->kind == POOL_IMPORT) h->buf = ib, h->base = r->import_lo, h->end = r->import_hi, h->alloc = NULL, h->pool = r;
            else h->buf = r->buf, h->base = r->lo, h->end = r->lo + r->size, h->alloc = r->alloc, h->pool = NULL;
        }
    }
    if (r && r->kind == POOL_IMPORT && (ib || st)) {
        out->cls = rec_class(r);
        out->pool = r;
        out->buffer = ib;
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
    __atomic_store_n(&mem_probe_frame, frame, __ATOMIC_RELAXED);
    if (fetch_test) ft_frame(frame);   // TOTK_STAGED_FETCH_TEST
    lazy_frame_log(frame);
    staged_frame_log(frame);
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
    // Staged mirrors (staged_upload, staged_wb_record): bytes and ranges copied in through staging, write-back bytes
    // in flight left out, CPU time of the prologue's part; write-back bytes and ranges, their recording time.
    uint64_t st_up_bytes, st_up_ranges, st_skip_bytes, st_up_t, st_wb_bytes, st_wb_ranges, st_wb_t;
} MirrorAcc;
static MirrorAcc mrep;                 // replay: this submission
static MirrorAcc macc;                 // mirror_stat_lock: since the last summary
static PlatMutex mirror_stat_lock = PLAT_MUTEX_INIT;
static volatile int64_t m_live_n, m_live_bytes, m_refused_n[MR_N], m_refused_bytes[MR_N];
static volatile int64_t m_flush_calls, m_flush_bytes, v_pages, v_skipped, v_mismatch, v_owned;

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
    // TOTK_MIRROR_STAGED=<MiB>, <KiB>k or all ("staged mirror" below): off (0) by default. It needs the mirror, the
    // deferred imports (a pool imported at nvnMemoryPoolInitialize has paid the import already) and no shadow storage.
    // all stages every lazily imported pool (staged_min 1 byte), the small ones included.
    const char *ms = cfg_get("TOTK_MIRROR_STAGED");
    uint64_t want = 0;
    if (ms && *ms) {
        char *end = NULL;
        uint64_t v = strtoull(ms, &end, 10);
        want = !strcmp(ms, "all") ? 1 : end && (*end == 'k' || *end == 'K') ? v << 10 : v << 20;
    }
    if (want) {
        const char *lz = cfg_get("TOTK_POOL_LAZY");
        const char *off = !vk_mirror_on ? "the VRAM mirror is off" : shadow_on ? "shadow storage is on"
                        : lz && !strcmp(lz, "0") ? "TOTK_POOL_LAZY=0 imports every pool at once" : NULL;
        if (off) rt_log("mem: TOTK_MIRROR_STAGED=%s ignored: %s", ms, off);
        else {
            staged_min = want;
            const char *wb = cfg_get("TOTK_STAGED_WB");
            staged_fault = !(wb && !strcmp(wb, "fence"));
            if (wb && *wb && strcmp(wb, "fence") && strcmp(wb, "fault"))
                rt_log("mem: TOTK_STAGED_WB=%s is not fence or fault: fault", wb);
            char which[64];
            if (want == 1) snprintf(which, sizeof which, "of every size");
            else if (want % (1ull << 20)) snprintf(which, sizeof which, "of %llu KiB or more", (unsigned long long)(want >> 10));
            else snprintf(which, sizeof which, "of %llu MiB or more", (unsigned long long)(want >> 20));
            rt_log("mem: imported pools %s are not imported: their VRAM mirror is filled from staging buffers (pools up "
                   "to %llu MiB in the prologue of the part that first uses them), and the bytes the GPU writes %s "
                   "(TOTK_MIRROR_STAGED=%s, TOTK_STAGED_WB=%s)", which, (unsigned long long)(STG_SMALL >> 20),
                   staged_fault ? "stay in it until a CPU access fetches them" : "go back to guest memory at each fence",
                   ms, staged_fault ? "fault" : "fence");
#ifdef _WIN32
            if (staged_fault) mem_guard_hook(staged_fetch);
            else mem_probe_on();   // TOTK_STAGED_READ_PROBE: staged_wb_done protects the pages it wrote (memory.c read probe)
#else
            if (staged_fault) rt_log("mem: TOTK_STAGED_WB=fault needs page faults this platform does not route: fence"), staged_fault = 0;
#endif
            const char *ftv = cfg_get("TOTK_STAGED_FETCH_TEST");
            fetch_test = staged_fault && ftv && *ftv ? atoi(ftv) : 0;
            if (fetch_test < 0) fetch_test = 0;
            if (fetch_test)
                rt_log("mem: staged fetch test: %d GPU-owned pages a frame are touched and their fetches checked against "
                       "a reference copy (TOTK_STAGED_FETCH_TEST)", fetch_test);
        }
    }
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
    r->mrefused = (uint8_t)(why + 1);   // before the state: vk_mem_use reads it after MS_REFUSED (staged_import)
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
typedef struct {
    int pending;                               // a line waits for the prologue
    uint64_t lo, size;                         // the pool's address and the mirror's size
    uint64_t hold;                             // ns: vk_mem_use's wait for the walk and mirror_lock before it
    uint64_t lock, budget, snap, alloc, total; // ns: pool_lock and the overlap check, vk_mem_vram, gen_lock and
                                               // snap_regions, vmaCreateBuffer, all of mirror_create
    uint64_t queries, pages;                   // the snapshot's write-watch queries and the written pages they found
} BigMirror;
static BigMirror big_mirror;

// ---- staged mirror (TOTK_MIRROR_STAGED=<MiB>, <KiB>k or all; off by default)
// An imported pool of at least that size is never imported for the mirror's sake: the terrain arena of each layer of
// Surf Hills is a 450 MiB CPU_CACHED pool, and its import at the first GPU use (65-85 ms in vkAllocateMemory on NVIDIA)
// plus the full copy import -> mirror stalled one frame (#397). Importing it earlier only moved the cost: the driver
// made the next vkQueueSubmit2 wait about as long. The same frame also imported about 276 small pools (73 MiB: 197 of
// 256 KiB, the rest 4 to 28 KiB) in 36-42 ms, almost all of it a per-call cost of vkAllocateMemory (about 0.1 ms
// whatever the size), and about 215 more on each reload of the layer; all stages them too. Here the pool keeps a VRAM
// mirror as any buffer-bound pool does, and the bytes travel through host-visible staging memory that the CPU fills:
// - Resolve: resolve_locked does not import the pool; its spans carry no buffer until VK_MEM_USE or VK_MEM_PICK gives
//   them the mirror (vk_mem_use makes it on the first use, as for other pools). A path that needs the pool's own buffer
//   imports it then (staged_import, logged once per pool with the reason and counted by reason in the staged line): a
//   mirror refused (budget, overlap, no write watch, allocation) or the first use on a TOTK_REPLAY_PAR helper thread,
//   which may not make a mirror. Every buffer user of a resolved span calls VK_MEM_USE or VK_MEM_PICK first
//   (vk_exec*.c, vk_dump.c): binds, draws, indirect-reset targets and aligned bind_range descriptors use, copies, fills,
//   updates and vk_exec_read_buffer pick, and a pick of a staged pool makes its mirror as a use does. The rest only read
//   guest memory on the CPU (the uniform snapshot, vk_usnap.c; the indirect-reset list read from guest memory,
//   vk_exec_draw.c vk_exec_reset_indirect; counter words, unaligned bind_range copies, dumps) or ask whether an address
//   is backed, so a pool that only those reach is never imported nor mirrored (the first-use census of the small pools
//   had 70 of 276 first reached by the uniform snapshot).
// - First fill of a large pool (staged_fill, over STG_SMALL): the thread that made the mirror zeroes it on the GPU and
//   copies the pages ever written in STG_CHUNK pieces, with mirror_lock let go, through STG_SLOTS staging buffers of the
//   upload batch's staging pool, each piece a copy in an upload batch of its own that goes to the queue at once
//   (vk_upload_flush(0)); a buffer is reused once its batch is done. The batches precede in queue order every part that
//   uses the mirror, since no part that records a use of it is submitted before the recording of the thread that fills
//   it is done.
// - First fill of a small pool (STG_SMALL or less): mirror_create queues the whole mirror in pend and marks it szero;
//   the prologue of the part that made it (staged_upload) zeroes it and copies the pages ever written along with that
//   prologue's other staged uploads, through the same staging buffers. The 276 small pools of a layer's first frame
//   thus cost no submission of their own, where staged_fill would submit one upload batch or more for each.
// - CPU -> mirror after that: mirror_sync queues the written pages as for any mirror, and the prologue
//   (vk_mem_mirror_prologue) copies them from guest memory into staging buffers it frees at the submission's fence
//   (staged_upload), outside mirror_lock, then records staging -> mirror. Bytes of a write-back of an earlier
//   submission still in flight (below) are left out: guest memory gets them only at that fence, and the mirror
//   already holds them, as shadow_copy does for the shadow storage.
// - Release: a pool never imported releases without waiting for the GPU (vk_pool_fini, staged_released).
// - GPU -> guest: vk_mem_mirror_end records mirror -> readback buffer for these pools' write-back ranges
//   (staged_wb_record), and at the fence, before the submission's other callbacks report the writes (the job is queued
//   at vk_mem_mirror_begin), the completion thread copies the bytes into guest memory (staged_wb_done). That copy is a
//   CPU write: the next write-watch query stamps those pages, and the mirror takes them in again (counted in the
//   staged line as upload). A held part's early write-back (TOTK_WB_EARLY, vk_mem_mirror_wb_held) leaves these
//   ranges to vk_mem_mirror_end (wb_early_takes): it copies into a pool's import, which a staged pool lacks.
// - TOTK_MEMORY=verify works unchanged: it compares the mirror with guest memory. Pages a staged write-back just
//   wrote count as written since their sync and are skipped. It also checks the zero fills: a page with no stamp
//   whose guest bytes are not zero would show as a mismatch.
// Eglantine (Eden) never imports guest memory for its buffers either: video_core/buffer_cache/buffer_cache.h
// BufferCache::CreateBuffer makes a device buffer over the used range and clears it on the GPU (ClearBuffer),
// SynchronizeBuffer collects the CPU-modified ranges minus the GPU-modified ones, and MappedUploadMemory copies them
// into one staging buffer and records staging -> buffer in the command stream, as staged_upload does. Differences: Eden
// makes buffers over the ranges the GPU uses (64 KiB pages, joining overlaps), here one mirror covers one pool; and its
// memory tracker starts every page CPU-modified (video_core/buffer_cache/word_manager.h WordManager), so its first
// sync uploads the whole range, where the write-watch stamps here leave the never-written pages to the zero fill.
// Changes outside this file: vk_exec.c vk_exec_read_buffer picks before it tests the buffer, vk_sync.c passes the
// fence to vk_mem_mirror_prologue.
#define STG_CHUNK (16ull << 20)   // a piece of the first fill: STG_SLOTS of them fit the staging pool's kept blocks
#define STG_SLOTS 4

// TOTK_STAGED_INJECT (a test of the failure paths, off by default; logged when on):
// - import: every staged pool's mirror is refused (as a failed device-local allocation) and its import after all
//   fails (staged_import): the uses of its spans must skip them, as those of a pool whose first-use import failed;
// - fence: from the batch of the first piece of the first staged_fill that will reuse a staging buffer, every upload
//   batch reads as on a lost device (batch_status, upload_wait_locked): the fill's wait for that buffer must end the
//   process with the fault report instead of spinning.
enum { INJECT_NONE, INJECT_IMPORT, INJECT_FENCE };
static int staged_inject(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_get("TOTK_STAGED_INJECT");
        int m = !e || !*e ? INJECT_NONE : !strcmp(e, "import") ? INJECT_IMPORT : !strcmp(e, "fence") ? INJECT_FENCE : -1;
        if (m < 0) rt_log("mem: TOTK_STAGED_INJECT=%s is not import or fence: ignored", e), m = INJECT_NONE;
        if (m) rt_log("mem: TOTK_STAGED_INJECT=%s: %s", e, m == INJECT_IMPORT ? "staged pools get neither a mirror nor an import"
                      : "upload batches from a staged first fill's on read as on a lost device");
        mode = m;
    }
    return mode;
}

// The write-backs of one submission: one job per submission while the mode is on, queued for its fence when its
// recording begins. Ranges: offsets from import_lo of their pool, sorted by pool and offset, merged; roff where each
// sits in the readback buffer; lo and serial identify the live pool at the fence.
typedef struct { PoolRec *r; uint64_t lo, serial, off, size, roff; } StRange;
typedef struct StJob {
    uint64_t fence;
    int ended;                          // st_lock: recorded (vk_mem_mirror_end): its ranges block the later uploads
    StRange *w;
    int n, cap;
    StRange *o;                         // TOTK_STAGED_WB=fault: the GPU writes that stay in the mirror (own_take)
    int no, co;
    uint32_t sub;                       // msub of the submission
    VkBuffer tbuf;                      // TOTK_STAGED_FETCH_TEST: the reference copy of the o ranges (staged_ref_record)
    VmaAllocation talloc;
    const uint8_t *tmap;
    uint64_t bytes;
    VkBuffer buf;
    VmaAllocation alloc;
    const uint8_t *map;
    struct StJob *next;                 // st_lock: on st_jobs
} StJob;
static PlatMutex st_lock = PLAT_MUTEX_INIT;   // st_jobs and the blocked union; never held with mirror_lock or pool_lock
static StJob *st_jobs;                  // every job not freed yet (vk_mem_shutdown frees those whose fence never came)
static StJob *st_cur;                   // replay: the job of the submission being recorded
static MRange *stq, *stw;               // replay: staged ranges the prologue took out of pend; write-backs at the end
static int nstq, cstq, nstw, cstw;
static MRange *st_blk;                  // st_lock: union of the write-back ranges in flight (st_blk_update)
static int nst_blk, cst_blk, st_blk_dirty = 1;
static uint64_t st_blk_fence;           // st_lock: the prologue fence it was built for
static volatile int32_t n_staged_imports, n_staged_fills, n_staged_import_failures;
static volatile int64_t st_fill_ns, st_fill_bytes, st_wbcpu_ns, st_wbcpu_bytes;
// What became of the staged pools, [0] since the start (the staged line), [1] since the last present (staged_frame_log):
// mirrors whose first fill went into a prologue (small: count, mirror bytes, written bytes copied, CPU copy time) or
// through staged_fill (big); imports after all by reason (the MR_ reasons of a refused mirror, then the first use on a
// helper thread and a refusal without a reason, as for a pool finalized meanwhile) with their bytes, and those whose
// import failed as well (left unbacked, staged_import) with their bytes; releases (all,
// never imported and so with no GPU wait, of those the ones only CPU paths had resolved, with a mirror, imported after
// all) and the time vk_pool_fini took for them.
enum { SI_HELPER = MR_N, SI_GONE, SI_N };
static const char *const si_name[SI_N] = {"budget", "untracked", "overlapping", "allocation", "helper thread", "other"};
typedef struct {
    int64_t small_n, small_bytes, small_copied, small_ns, big_n, imp_n[SI_N], imp_bytes[SI_N], imp_failed, imp_failed_bytes;
    int64_t rel_n, rel_never, rel_cpu, rel_mirror, rel_imported, rel_ns;
    int64_t drop_n, drop_bytes;   // released pools that dropped GPU-owned bytes unfetched (staged_forget), and the bytes
} StCount;
static volatile StCount stc[2];
#define ST_ADD(field, v) (plat_add64(&stc[0].field, (int64_t)(v)), plat_add64(&stc[1].field, (int64_t)(v)))

// Every present (vk_mem_census): a frame in which staged pools got a mirror, were imported after all or were released
// gets a line (the first 256), so the first frame of a layer and the frames of its release show what the staged
// path did there, beside lazy_frame_log's imports.
static void staged_frame_log(uint64_t frame) {
    static int lines;
    if (!staged_min || lines >= 256) return;
    volatile int64_t *src = (volatile int64_t *)&stc[1];   // StCount is all int64_t counters
    int any = 0;
    for (size_t i = 0; i < sizeof(StCount) / sizeof(int64_t) && !any; i++) any = src[i] != 0;
    if (!any) return;
    StCount c;
    int64_t *d = (int64_t *)&c;
    for (size_t i = 0; i < sizeof c / sizeof(int64_t); i++) d[i] = plat_xchg64(&src[i], 0);
    lines++;
    const double M = 1048576.0;
    int64_t imp = 0;
    for (int k = 0; k < SI_N; k++) imp += c.imp_n[k];
    rt_log("mem: frame %llu: staged pools: %lld mirrors filled in a prologue (%.1f MiB, %.1f MiB of written pages "
           "copied in %.1f ms), %lld by staged_fill; imported after all %lld (budget %lld, untracked %lld, overlapping %lld, "
           "allocation %lld, helper thread %lld, other %lld), import failed and left unbacked %lld (%.1f MiB); released "
           "%lld in %.1f ms (never imported %lld, with a mirror %lld, imported %lld), %lld of them dropping %.3f MiB of "
           "GPU-owned bytes unfetched",
           (unsigned long long)frame, (long long)c.small_n, c.small_bytes / M, c.small_copied / M, c.small_ns / 1e6,
           (long long)c.big_n, (long long)imp, (long long)c.imp_n[MR_BUDGET], (long long)c.imp_n[MR_UNTRACKED],
           (long long)c.imp_n[MR_OVERLAP], (long long)c.imp_n[MR_ALLOC], (long long)c.imp_n[SI_HELPER],
           (long long)c.imp_n[SI_GONE], (long long)c.imp_failed, c.imp_failed_bytes / M, (long long)c.rel_n,
           c.rel_ns / 1e6, (long long)c.rel_never,
           (long long)c.rel_mirror, (long long)c.rel_imported, (long long)c.drop_n, c.drop_bytes / M);
}

// vk_pool_fini, the pool out of the index and its record still alive: the release's counts (stc).
static void staged_released(const PoolRec *r, int imported, uint64_t ns) {
    ST_ADD(rel_n, 1), ST_ADD(rel_ns, ns);
    if (imported) ST_ADD(rel_imported, 1);
    else {
        ST_ADD(rel_never, 1);
        if (!r->mbuf && r->used) ST_ADD(rel_cpu, 1);   // resolved, but only by paths that read guest memory
    }
    if (r->mbuf) ST_ADD(rel_mirror, 1);
}

typedef struct { VkBuffer buf; VmaAllocation alloc; } StBuf;
static void staged_free_buffer(void *arg) {   // completion thread: a prologue's staging buffer, after its fence
    StBuf *s = arg;
    vmaDestroyBuffer(vk_ctx()->vma, s->buf, s->alloc);
    free(s);
}

// ---- GPU-owned pages (TOTK_STAGED_WB=fault, the default with TOTK_MIRROR_STAGED)
// A staged mirror's write-backs are not copied into guest memory: the mirror alone holds the bytes the GPU wrote, and
// the pool keeps them as a list of byte ranges (r->own, offsets from import_lo, sorted and disjoint). At the fence of
// the submission that wrote them (staged_wb_done, ahead of the submission's other callbacks, so before the game sees
// the fence), the ranges join the list and their pages become PAGE_NOACCESS (memory.c mem_guard_own), unless already
// owned: the GPU rewrites the same pages every frame, so in steady state no protection changes. The first CPU access
// to such a page, a game thread's or ours, faults into memory.c probe_fault, which calls staged_fetch: it copies the
// owned bytes of the page and of the other owned pages of its 64 KiB block from the mirror into guest memory, through
// a command buffer of its own it waits for, drops them from the list, and opens the pages. The access then runs on the merged
// page: the GPU's bytes where it wrote, the guest's elsewhere, as an import would hold them. A host path that hands the
// range to the kernel or a driver fetches the same way (memory.c mem_thaw, mem_touch, win32.c plat_mem_read); a pool
// that goes away drops its pages unfetched (staged_forget). plat_mem_readable counts owned pages as readable.
// Uploads into the mirror leave the owned bytes out (st_blk_update), and copy the other bytes of an owned page without
// opening it (memory.c mem_probe_copy). A page a fetch brought owned bytes in for is marked CPU-read (r->cpu_read): the
// GPU's later writes to it go back at the fence as with TOTK_STAGED_WB=fence (vk_mem_mirror_end), so a page the CPU
// keeps reading does not fault every frame.
// Why at the fence and not when the write is recorded: at the fence every owned range is complete in the mirror, so a
// fetch only waits for its own copy (queued after everything submitted) and never for a submission still being
// recorded, which the faulting thread could itself be recording, or need own_lock for (its prologue's st_blk_update)
// while the fetch holds it. A CPU access between the GPU's write and the end of its submission sees the guest's old
// bytes, as it does with a mirrored imported pool until the end-of-submission write-back runs. Past that the window is
// longer than an import's: the timeline value is reached with guest memory still stale and the pages still open until
// the completion thread runs staged_wb_done. What the game can learn completion from (NVN front end, nvn_api_*.c):
// - syncs: nvnSyncWait (nvn_api_core.c, a poll with timeout 0), nvnQueueFinish (vk_backend.c vk_be_finish), a command
//   buffer's WaitSync (nvn_api_cmd.c wait_submission) and TotK's LOD guard all ask vk_sync.c vk_sync_fence_done or
//   vk_sync_fence_wait, whose done value completion_main stores only after every callback of the values reached has
//   run, staged_wb_done among them;
// - event values (nvnCommandBufferSignalEvent, read with nvnEventGetValue or from the event's pool memory) and counter
//   and occlusion reports (nvnCommandBufferReportCounter): the completion thread writes them into guest memory in
//   callbacks the recording queues (vk_exec_xfer.c event_cb, counter_cb, oq_cb), after staged_wb_done, which
//   vk_mem_mirror_begin queues before the recording starts; the queue runs in the order callbacks were added;
// - window syncs (nvn_api_obj.c nvnWindowAcquireTexture) are signalled at once and say nothing about the GPU's work;
// - memory the GPU writes, polled by the game. A staged pool's GPU writes reach guest memory only through
//   staged_wb_done's copy-back or a fetch, never early, and staged_wb_done protects a submission's owned ranges before
//   its copy-back bytes go in, so a flag the copy-back carries is not seen ahead of data the same submission left
//   owned. GPU writes into an imported pool (below the TOTK_MIRROR_STAGED size, imported after all, or never given a
//   mirror) land in guest memory before the timeline value is signalled: at the GPU's own pace without a mirror, in
//   the end-of-submission write-back copy with one. A game that polls such a flag, written by the GPU after data in a
//   staged pool, could see the flag and then the stale data. Ordering that would take protecting the pages when the
//   write is recorded and a fetch that waits for an unfinished submission without own_lock (see above); with
//   TOTK_MIRROR_STAGED=all only the pools imported after all are such writers, and the staged line counts them. With
//   an import (TOTK_MEMORY=import) there is no window.
// Eglantine (Eden) does the same on a guest CPU read of GPU-written buffer memory: video_core/gpu.cpp OnCPURead
// downloads the page-aligned area at once and waits for it (RasterizerVulkan::GetFlushArea, FlushRegion,
// BufferCache::DownloadMemory), and BufferCache::NoteCpuRead remembers those pages so that later GPU writes to them
// are copied back at fences (cpu_read_ranges). Differences: Eden finds the read through its page table
// (core/memory.cpp HandleRasterizerDownload), here a page protection fault does; Eden flushes on CPU reads only, here a
// CPU write to an owned page fetches first too, since the guest bytes of the rest of the page must be merged with the
// GPU's; and the area here is the 64 KiB block of owned pages (NVN's pool page size), where Eden downloads the read's
// pages, so a CPU scan through a GPU-written buffer pays one GPU round trip per 64 KiB instead of per 4 KiB.
#define OWN_BLOCK 0x10000ull
#define OWN_LOG 32
static PlatMutex own_lock = PLAT_MUTEX_INIT;   // st_pools and every pool's own list; taken after pool_lock, st_lock
static PoolRec **st_pools;              // own_lock: the live staged mirrors by import_lo (staged_fetch finds the pool here)
static int nst_pools, cst_pools;
static PoolRec **st_own;                // own_lock: those whose own list holds a range (PoolRec.own_ix), in no order
static int nst_own, cst_own;
static volatile uint64_t own_gen;       // own_lock: bumped by every change of an own list (st_blk_update)
// Since the last staged line: protection changes, fetches by [write][game thread] and by kernel paths, their pages,
// bytes and time, the longest; owned bytes now.
static volatile int64_t own_protects, own_fetch[2][2], own_kfetch, own_fpages, own_fbytes, own_fns, own_fmax;
static volatile int64_t own_seen[3], own_pushed, own_take_ns, own_pfail;   // own_take: pages by state, queued, time,
                                                                           // protection failures
// Since the last staged line: fetches that found no owned byte, fetches whose copy waited over a second, pages switched
// to the fence copy-back (r->cpu_read), and the write-back pieces and bytes of CPU-read pages that the fence copied
// back (staged_split_own).
static volatile int64_t own_fempty, own_long, own_switched, own_fb_ranges, own_fb_bytes;
static volatile int64_t own_drop_bytes;   // since the start: owned bytes released pools dropped unfetched (staged_forget)
// Since the last staged line: copy-back pages found owned again that opened without a fetch, copy-back pieces kept
// owned instead (staged_copy_back), and fetches the completion thread made inside staged_wb_done (none expected).
static volatile int64_t own_cb_opened, own_cb_kept, own_cfetch;
// Since the last staged line, own_take's time by part (ns): thawing frozen regions, committing lazy-heap pages,
// protecting, the runs of pages it protected, and the merge and page scan under own_lock.
static volatile int64_t own_take_part[5];
static volatile int64_t st_wbdone_max;   // since the last staged line: the longest staged_wb_done (ns), a fence's delay
static _Thread_local int st_in_wb;   // staged_wb_done runs on this thread now: staged_fetch counts a fetch it causes
static volatile int64_t ft_touch[4], ft_cmp_bytes, ft_cmp_pieces, ft_bad, ft_skip;
static PlatMutex ft_lock = PLAT_MUTEX_INIT;   // TOTK_STAGED_FETCH_TEST: a touch from its pick to its access (staged_forget)
static void ft_ref_take(PoolRec *r, const StRange *add, int n, const uint8_t *tmap, uint32_t sub);
static void ft_check(PoolRec *r, const struct KeptRange *cut, const VkBufferCopy *reg, int n, const uint8_t *map,
                     const uint32_t *rec0, uint64_t w0off);
static struct { VkCommandPool pool; VkCommandBuffer cb; VkFence fence; } fc;   // own_lock: staged_fetch's copy
typedef struct { uint64_t frame, addr, rip, ns; int write, guest, kernel, pages; } OwnRec;
static OwnRec own_rec[OWN_LOG];
static volatile int32_t own_nrec, own_logged;

// With TOTK_MIRROR_STAGED=all there are hundreds to thousands of staged mirrors, and few of them own any byte: the
// lookups below are binary searches of st_pools, and the loops that want the owned bytes (st_blk_update, the fetch
// exerciser, own_log) walk st_own, not every staged pool.
// own_lock held: the index of the first staged pool whose import_lo is above a.
static int st_upper(uint64_t a) {
    int lo = 0, hi = nst_pools;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (st_pools[mid]->import_lo <= a) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// own_lock held: the staged pool whose mirror holds address a, or NULL.
static PoolRec *st_find(uint64_t a) {
    int i = st_upper(a) - 1;
    return i >= 0 && a < st_pools[i]->import_hi ? st_pools[i] : NULL;
}

// own_lock held, after r->own changed: r is on st_own while its list holds a range.
static void own_listed(PoolRec *r) {
    if (r->nown && !r->own_ix) {
        if (nst_own == cst_own) grow((void **)&st_own, &cst_own, sizeof *st_own);
        st_own[nst_own++] = r;
        r->own_ix = nst_own;
    } else if (!r->nown && r->own_ix) {
        PoolRec *last = st_own[--nst_own];   // the last one takes r's place (r itself when it is the last)
        st_own[r->own_ix - 1] = last;
        last->own_ix = r->own_ix;
        r->own_ix = 0;
    }
}

// own_lock held: the sorted ranges add[0..n) (one pool's, offsets) joined to r->own.
static void own_merge(PoolRec *r, const StRange *add, int n) {
    static struct KeptRange *out;
    static int cout;
    while (cout < r->nown + n) grow((void **)&out, &cout, sizeof *out);
    int i = 0, j = 0, o = 0;
    while (i < r->nown || j < n) {
        struct KeptRange x;
        if (j >= n || (i < r->nown && r->own[i].lo <= add[j].off)) x = r->own[i++];
        else x = (struct KeptRange){add[j].off, add[j].off + add[j].size}, j++;
        if (o && x.lo <= out[o - 1].hi) {
            if (x.hi > out[o - 1].hi) out[o - 1].hi = x.hi;
        } else out[o++] = x;
    }
    struct KeptRange *old = r->own;
    int c = r->cown;
    r->own = out, r->cown = cout, r->nown = o;
    out = old, cout = c;
    own_listed(r);
}

// own_lock held: the sorted, disjoint pieces cut[0..n) (offsets) leave r->own.
static void own_cut(PoolRec *r, const struct KeptRange *cut, int n) {
    static struct KeptRange *out;
    static int cout;
    while (cout < r->nown + n) grow((void **)&out, &cout, sizeof *out);
    struct KeptRange *in = r->own;
    int i = 0, o = 0;
    for (int j = 0; j < n && i < r->nown; j++) {
        uint64_t a = cut[j].lo, b = cut[j].hi;
        while (i < r->nown && in[i].hi <= a) out[o++] = in[i++];
        while (i < r->nown && in[i].lo < b) {
            if (in[i].lo < a) out[o++] = (struct KeptRange){in[i].lo, a};
            if (in[i].hi > b) {
                in[i].lo = b;
                break;
            }
            i++;
        }
    }
    while (i < r->nown) out[o++] = in[i++];
    r->own = out, out = in;
    int c = r->cown;
    r->cown = cout, cout = c;
    r->nown = o;
    own_listed(r);
}

// vk_mem_mirror_end, fault mode: the staged write-back ranges in stw split by page; those on pages the CPU has read
// (fetched) stay in stw for the fence copy-back, the rest become the job's owned ranges.
static void staged_split_own(void) {
    static MRange *sbuf, *fb;
    static size_t scap;
    static int nfb, cfb;
    StJob *j = st_cur;
    if (!j) rt_fatal("mem: a staged write-back outside a submission's recording");
    sort_ranges(stw, (size_t)nstw, &sbuf, &scap);
    nfb = 0;
    for (int i = 0; i < nstw; i++) {
        PoolRec *r = stw[i].r;
        uint64_t lim = r->import_hi - r->import_lo, a = stw[i].off, b = a + stw[i].size;
        if (b > lim) b = lim;
        for (uint64_t x = a; x < b;) {
            uint64_t e = (x & ~0xfffull) + 0x1000;
            if (e > b) e = b;
            if (r->cpu_read && __atomic_load_n(&r->cpu_read[x >> 12], __ATOMIC_RELAXED)) {
                range_add(&fb, &nfb, &cfb, r, x, e - x);
            } else if (j->no && j->o[j->no - 1].r == r && j->o[j->no - 1].off + j->o[j->no - 1].size >= x) {
                StRange *l = &j->o[j->no - 1];
                if (e > l->off + l->size) l->size = e - l->off;
            } else {
                if (j->no == j->co) grow((void **)&j->o, &j->co, sizeof *j->o);
                j->o[j->no++] = (StRange){r, r->lo, r->storage_serial, x, e - x, 0};
            }
            x = e;
        }
    }
    nstw = 0;
    uint64_t fbb = 0;
    for (int i = 0; i < nfb; i++) range_add(&stw, &nstw, &cstw, fb[i].r, fb[i].off, fb[i].size), fbb += fb[i].size;
    plat_add64(&own_fb_ranges, nfb), plat_add64(&own_fb_bytes, (int64_t)fbb);
}

static void own_settle_failed(PoolRec *r, const uint64_t *pg, int n);

// staged_wb_done, the pool found live: the job's owned ranges of pool r (sorted) join its list, and the pages that were
// not owned yet are protected, a run of them at a time (memory.c mem_guard_own_range); those that could not be are
// settled at once (own_settle_failed). Without pool_lock: the protection of the first GPU writes into a new layer's
// terrain arena took up to about 120 ms, and a game thread that initializes or finalizes a pool meanwhile waits for
// pool_lock exclusively, and every resolve then waits behind it (the reloads of the Hilltop route showed waits of
// 122-172 ms). The record stays alive without it: a staged mirror's record is freed by mirror_retire, on this thread;
// a pool released meanwhile has left st_pools (staged_forget, under own_lock), and then nothing is taken.
static void own_take(PoolRec *r, const StRange *add, int n, const uint8_t *tmap, uint32_t sub) {
    static uint64_t *pg, *fl;   // completion thread: pages to protect, pages whose protection failed
    static int npg, cpg, cfl;
    int nfl = 0;
    npg = 0;
    uint64_t t0 = qpc();
    plat_mutex_lock(&own_lock);
    if (st_find(r->import_lo) != r) {   // released since staged_wb_done found it live
        plat_mutex_unlock(&own_lock);
        return;
    }
    if (tmap && r->t_page) ft_ref_take(r, add, n, tmap, sub);   // TOTK_STAGED_FETCH_TEST
    own_merge(r, add, n);
    for (int i = 0; i < n; i++)
        for (uint64_t p = (r->import_lo + add[i].off) & ~0xfffull; p < r->import_lo + add[i].off + add[i].size; p += 0x1000) {
            // Owned: protected already. Busy: a fetch that waits for own_lock (it will take these ranges too), or a claim
            // of a moment: memory.c probe_fault claims the state of every page that faults in a chunk with guards, a
            // lazy-heap commit or a frozen region's write as well, and leaving such a page out would leave its owned
            // bytes on an open page. mem_guard_own_range waits the claim out and protects the page if it is open then; after
            // a fetch that protects a page with nothing owned on it, whose next access fetches nothing and opens it.
            int st = mem_guard_state(p);
            plat_inc64(&own_seen[st]);
            if (st == 1 || (npg && pg[npg - 1] == p)) continue;
            if (npg == cpg) grow((void **)&pg, &cpg, sizeof *pg);
            pg[npg++] = p;
        }
    __atomic_add_fetch(&own_gen, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&own_lock);
    plat_add64(&own_take_part[4], (int64_t)(qpc() - t0));   // the part under own_lock
    for (int i = 0; i < npg;) {
        int k = i + 1;
        while (k < npg && pg[k] == pg[k - 1] + 0x1000) k++;
        // A frozen region could not catch writes to pages opened later. A page of the lazy heap the CPU never touched is
        // still reserved, and VirtualProtect fails there (the arena's pages are mostly such at its first use): a run
        // whose protection fails is committed (mem_touch_keep) and protected again. Committing every run first, as
        // before, asked VirtualQuery about each run's 64 KiB, about 90% of own_take's time, and at the first fences
        // of a reloaded layer, whose new arena the GPU writes in thousands of scattered pages, one staged_wb_done took
        // 125-163 ms, a delay of every later fence the game waits for.
        uint64_t n_run = pg[k - 1] + 0x1000 - pg[i], ta = qpc();
        mem_thaw_keep(pg[i], n_run);
        uint64_t tb = qpc(), t_commit = 0;
        int failed, made = mem_guard_own_range(pg[i], n_run, &failed);
        if (failed) {
            uint64_t tc = qpc();
            mem_touch_keep(pg[i], n_run);
            t_commit = qpc() - tc;
            made += mem_guard_own_range(pg[i], n_run, &failed);
        }
        plat_add64(&own_protects, made);
        plat_add64(&own_take_part[0], (int64_t)(tb - ta)), plat_add64(&own_take_part[1], (int64_t)t_commit);
        plat_add64(&own_take_part[2], (int64_t)(qpc() - tb - t_commit)), plat_inc64(&own_take_part[3]);
        for (; failed && i < k; i++)
            if (!mem_guard_state(pg[i])) {   // still open: VirtualProtect refused it
                plat_inc64(&own_pfail);
                if (nfl == cfl) grow((void **)&fl, &cfl, sizeof *fl);
                fl[nfl++] = pg[i];
            }
        i = k;
    }
    if (nfl) own_settle_failed(r, fl, nfl);
    plat_add64(&own_pushed, npg);
    plat_add64(&own_take_ns, (int64_t)(qpc() - t0));
}

// own_lock held (it guards fc): the regions reg[0..n) of r's mirror (srcOffset) copied into a new staging buffer of
// total bytes (dstOffset) and waited for; returns its mapping, *sb and *sa for the caller to destroy. A command buffer
// and fence of their own, not the upload batch: the faulting thread may be recording into the open batch (a texture
// upload copying pool bytes), which a flush here would end under it. The copy runs after everything submitted so far
// (queue order); the owned ranges' writers finished at their fences already. page: for the line of a long wait.
static const uint8_t *own_gpu_read(PoolRec *r, const VkBufferCopy *reg, int n, uint64_t total, VkBuffer *sb,
                                   VmaAllocation *sa, uint64_t page) {
    VkCtx *c = vk_ctx();
    if (!fc.pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateCommandPool(c->device, &pi, NULL, &fc.pool) != VK_SUCCESS || (ai.commandPool = fc.pool,
            vkAllocateCommandBuffers(c->device, &ai, &fc.cb) != VK_SUCCESS) ||
            vkCreateFence(c->device, &fi, NULL, &fc.fence) != VK_SUCCESS)
            rt_fatal("vk: command buffer for staged fetches");
    }
    const uint8_t *map = vk_staging(total, sb, sa);
    vkResetCommandBuffer(fc.cb, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(fc.cb, &bi);
    mbarrier(fc.cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyBuffer(fc.cb, r->mbuf, *sb, (uint32_t)n, reg);
    mbarrier(fc.cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
             VK_ACCESS_2_HOST_READ_BIT);
    vkEndCommandBuffer(fc.cb);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &fc.cb};
    vkResetFences(c->device, 1, &fc.fence);
    vk_queue_lock();
    VkResult res = vkQueueSubmit(c->gfx_queue, 1, &si, fc.fence);
    vk_queue_unlock();
    for (int waited = 0; res == VK_SUCCESS;) {   // a wait over a second gets a line each second (a hang, not a slow GPU)
        res = vkWaitForFences(c->device, 1, &fc.fence, VK_TRUE, 1000000000ull);
        if (res != VK_TIMEOUT) break;
        res = VK_SUCCESS;
        if (!waited++) plat_inc64(&own_long);
        rt_log("mem: a staged fetch at %#llx (%s) has waited %d s for its copy", (unsigned long long)page,
               kern_cpu() ? "game thread" : "runtime thread", waited);
    }
    if (res != VK_SUCCESS) {
        vk_device_fault_report(res);
        rt_fatal("vk: staged fetch failed (%d)", (int)res);
    }
    vmaInvalidateAllocation(c->vma, *sa, 0, VK_WHOLE_SIZE);
    return map;
}

// own_take, completion thread: VirtualProtect refused to make these pages of r PAGE_NOACCESS
// (memory.c mem_guard_own), so owned bytes sit on open pages, where a CPU read gets the guest's stale bytes and a CPU
// write would be overwritten by the next fetch. Before, only own_pfail counted them. Now their owned bytes are fetched
// at once, a GPU round trip on the completion thread for a case that should not arise (own_take commits lazy-heap
// pages and thaws frozen regions first), and the pages are marked CPU-read, so the GPU's later writes to them go back
// at the fence instead of being owned. Nothing stops the CPU touching such a page before the bytes are in, and a CPU
// write to an owned byte in that time is lost; the first 16 pages get a line.
static void own_settle_failed(PoolRec *r, const uint64_t *pg, int npg) {
    static struct KeptRange *cut;
    static VkBufferCopy *reg;
    static int ccut, creg;
    static volatile int32_t logged;
    int n = 0;
    uint64_t total = 0;
    plat_mutex_lock(&own_lock);
    for (int i = 0; i < npg && st_find(r->import_lo) == r; i++) {   // (a pool released meanwhile: nothing to repair)
        if (mem_guard_state(pg[i])) continue;   // owned or claimed by now: not one of the open pages this repairs
        uint64_t lo = pg[i] - r->import_lo, hi = lo + 0x1000;
        int k = 0, m = r->nown;
        while (k < m) {   // the first owned range ending above lo
            int mid = (k + m) / 2;
            if (r->own[mid].hi <= lo) k = mid + 1;
            else m = mid;
        }
        uint64_t bytes = 0;
        for (; k < r->nown && r->own[k].lo < hi; k++) {
            uint64_t a = r->own[k].lo > lo ? r->own[k].lo : lo, b = r->own[k].hi < hi ? r->own[k].hi : hi;
            if (n == ccut) grow((void **)&cut, &ccut, sizeof *cut);
            if (n == creg) grow((void **)&reg, &creg, sizeof *reg);
            cut[n] = (struct KeptRange){a, b}, reg[n] = (VkBufferCopy){a, total, b - a}, n++;
            total += b - a, bytes += b - a;
        }
        if (!__atomic_exchange_n(&r->cpu_read[lo >> 12], 1, __ATOMIC_RELAXED)) plat_inc64(&own_switched);
        if (plat_inc32(&logged) <= 16)
            rt_log("mem: staged pool %#llx: page %#llx could not be protected for its GPU-owned bytes; %llu bytes fetched "
                   "at once and the page goes back to the fence copy-back", (unsigned long long)r->lo,
                   (unsigned long long)pg[i], (unsigned long long)bytes);
    }
    if (total) {
        VkBuffer sb;
        VmaAllocation sa;
        const uint8_t *map = own_gpu_read(r, reg, n, total, &sb, &sa, pg[0]);
        for (int i = 0; i < n; i++)   // open pages, read-write (own_take thawed and committed them)
            memcpy((void *)(uintptr_t)(r->import_lo + cut[i].lo), map + reg[i].dstOffset, cut[i].hi - cut[i].lo);
        vmaDestroyBuffer(vk_ctx()->vma, sb, sa);
        own_cut(r, cut, n);
        __atomic_add_fetch(&own_gen, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&own_lock);
}

// memory.c probe_fault and prb_release: page (claimed, read-write) is GPU-owned. kernel: a host path that hands it to
// the kernel or a driver (rip: its caller). Fetches the owned bytes of the page and of the other owned pages of its
// 64 KiB block that no other thread is settling, then clears them (mem_guard_clear) before own_lock goes, so a range
// staged_wb_done adds meanwhile either is in this fetch or finds the page open and protects it again.
static int staged_fetch(uint64_t page, int write, uint64_t rip, int kernel) {
    uint64_t t0 = qpc();
    plat_mutex_lock(&own_lock);
    PoolRec *r = st_find(page);
    if (!r) {   // the pool went away (staged_forget) while this thread waited for the lock
        mem_guard_clear(page);
        plat_mutex_unlock(&own_lock);
        return 1;
    }
    uint64_t w0 = page & ~(OWN_BLOCK - 1), w1 = w0 + OWN_BLOCK;
    if (w0 < r->import_lo) w0 = r->import_lo;
    if (w1 > r->import_hi) w1 = r->import_hi;
    uint32_t mine = 0;   // bit k: page w0 + k * 4 KiB is claimed here
    for (uint64_t p = w0; p < w1; p += 0x1000)
        if (p == page || mem_guard_tryclaim(p)) mine |= 1u << ((p - w0) >> 12);
    static struct KeptRange *cut;   // own_lock
    static VkBufferCopy *reg;
    static int ccut, creg;
    uint32_t got = 0;   // bit k: page w0 + k * 4 KiB had owned bytes this fetch brings in
    int n = 0;
    uint64_t total = 0, lo = w0 - r->import_lo, hi = w1 - r->import_lo;
    int k = 0, m = r->nown;
    while (k < m) {   // first owned range ending above lo
        int mid = (k + m) / 2;
        if (r->own[mid].hi <= lo) k = mid + 1;
        else m = mid;
    }
    for (; k < r->nown && r->own[k].lo < hi; k++) {
        uint64_t a = r->own[k].lo > lo ? r->own[k].lo : lo, b = r->own[k].hi < hi ? r->own[k].hi : hi;
        for (uint64_t x = a; x < b;) {
            uint64_t e = (x & ~0xfffull) + 0x1000;
            if (e > b) e = b;
            if (mine >> ((x - lo) >> 12) & 1) {
                got |= 1u << ((x - lo) >> 12);
                if (n == ccut) grow((void **)&cut, &ccut, sizeof *cut);
                if (n == creg) grow((void **)&reg, &creg, sizeof *reg);
                if (n && cut[n - 1].hi == x) cut[n - 1].hi = e, reg[n - 1].size += e - x;
                else cut[n] = (struct KeptRange){x, e}, reg[n] = (VkBufferCopy){x, total, e - x}, n++;
                total += e - x;
            }
            x = e;
        }
    }
    if (total) {
        uint32_t rec0[OWN_BLOCK >> 12];   // TOTK_STAGED_FETCH_TEST: the pages' last recorded writes before the copy
        if (r->t_rec)
            for (uint64_t p = w0; p < w1; p += 0x1000)
                rec0[(p - w0) >> 12] = __atomic_load_n(&r->t_rec[(p - r->import_lo) >> 12], __ATOMIC_RELAXED);
        VkBuffer sb;
        VmaAllocation sa;
        const uint8_t *map = own_gpu_read(r, reg, n, total, &sb, &sa, page);
        if (r->t_rec) ft_check(r, cut, reg, n, map, rec0, w0 - r->import_lo);
        for (int i = 0; i < n; i++)   // the claimed pages are read-write
            memcpy((void *)(uintptr_t)(r->import_lo + cut[i].lo), map + reg[i].dstOffset, cut[i].hi - cut[i].lo);
        vmaDestroyBuffer(vk_ctx()->vma, sb, sa);
        own_cut(r, cut, n);
        __atomic_add_fetch(&own_gen, 1, __ATOMIC_RELEASE);
    }
    // Only a page whose owned bytes came in is marked CPU-read (later GPU writes to it go back at the fence): a page
    // that was protected with nothing owned on it (own_take's claimed case) just opens, and its next GPU write is
    // owned again rather than copied back at every fence from then on.
    // The fetched bytes are uploaded into the mirror once more at the page's next sync: the copy into guest memory sets
    // the page's write-watch bit. Left so: the reset that would avoid it cannot tell that copy from a write by another
    // thread to the page, which is read-write while claimed, and a missed write would leave the mirror stale; the cost
    // is one page upload per fetched page, and fetches are rare (none on the Hilltop route without the exerciser).
    int pages = 0;
    for (uint64_t p = w0; p < w1; p += 0x1000)
        if (mine >> ((p - w0) >> 12) & 1) {
            if ((got >> ((p - w0) >> 12) & 1) && !__atomic_exchange_n(&r->cpu_read[(p - r->import_lo) >> 12], 1,
                                                                       __ATOMIC_RELAXED))
                plat_inc64(&own_switched);
            mem_guard_clear(p);
            pages++;
        }
    if (!total) plat_inc64(&own_fempty);
    plat_mutex_unlock(&own_lock);
    uint64_t ns = qpc() - t0;
    int guest = kern_cpu() != NULL;
    if (st_in_wb) plat_inc64(&own_cfetch);   // the completion thread inside staged_wb_done
    if (kernel) plat_inc64(&own_kfetch);
    else plat_inc64(&own_fetch[write ? 1 : 0][guest]);
    plat_add64(&own_fpages, pages), plat_add64(&own_fbytes, (int64_t)total), plat_add64(&own_fns, (int64_t)ns);
    for (int64_t mx = own_fmax; (int64_t)ns > mx && !__atomic_compare_exchange_n(&own_fmax, &mx, (int64_t)ns, 0,
                                                                               __ATOMIC_RELAXED, __ATOMIC_RELAXED);) {}
    int32_t i = __atomic_fetch_add(&own_nrec, 1, __ATOMIC_RELAXED);
    if (i < OWN_LOG)
        own_rec[i] = (OwnRec){__atomic_load_n(&mem_probe_frame, __ATOMIC_RELAXED), page, rip, ns, write, guest, kernel, pages};
    return 1;
}

// vk_pool_fini, the pool out of the index: its owned pages open again without a fetch (the storage goes back to the
// guest, and a pool made later at these addresses starts with plain pages), after any fetch of it has ended.
// The first 32 get a line with what they dropped.
static void staged_forget(PoolRec *r) {
    static volatile int32_t logged;
    if (fetch_test) plat_mutex_lock(&ft_lock);   // no touch of the exerciser is between its pick and its access
    uint64_t t0 = qpc(), bytes = 0;
    plat_mutex_lock(&own_lock);
    int at = st_upper(r->import_lo) - 1;   // st_pools stays sorted: staged_fetch's binary search
    if (at < 0 || st_pools[at] != r)
        for (at = 0; at < nst_pools && st_pools[at] != r; at++) {}
    if (at < nst_pools) memmove(st_pools + at, st_pools + at + 1, (size_t)(nst_pools - at - 1) * sizeof *st_pools), nst_pools--;
    int claimed, opened = mem_guard_drop(r->import_lo, r->import_hi - r->import_lo, &claimed), ranges = r->nown;
    for (int k = 0; k < r->nown; k++) bytes += r->own[k].hi - r->own[k].lo;
    free(r->own);
    r->own = NULL, r->nown = r->cown = 0;
    own_listed(r);
    __atomic_add_fetch(&own_gen, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&own_lock);
    if (fetch_test) plat_mutex_unlock(&ft_lock);
    uint64_t frame = __atomic_load_n(&mem_probe_frame, __ATOMIC_RELAXED), n = r->import_hi - r->import_lo;
    plat_add64(&own_drop_bytes, (int64_t)bytes);
    if (bytes) ST_ADD(drop_n, 1), ST_ADD(drop_bytes, bytes);
    // The first 32 that dropped or opened anything, or are large: with all pools staged, a layer's exit releases
    // hundreds of small pools that own nothing, and the staged frame line counts those.
    if ((ranges || opened || claimed || n >= BIG_MIRROR) && plat_inc32(&logged) <= 32)
        rt_log("mem: staged pool %#llx of %.2f MiB released at frame %llu: %d owned ranges (%.1f MiB) dropped unfetched, "
               "%d pages opened, %d left to the threads that claimed them, %.2f ms", (unsigned long long)r->lo,
               n / 1048576.0, (unsigned long long)frame, ranges, bytes / 1048576.0, opened, claimed, qpc_ms(qpc() - t0));
}

// vk_mem_shutdown, the device idle, before the pools and the staging pool go: the owned pages are fetched while the
// mirrors still hold their bytes (memory.c mem_thaw, as for a kernel path), then the pools leave st_pools, the pages
// left owned open (mem_guard_drop) and staged_fetch's command pool and fence are destroyed. A fault on a page of
// theirs after this finds no pool and opens the page. A pool's record must not outlive this in st_pools: vk_mem_shutdown
// frees the records without vk_pool_fini, and a probe may start a new device in the same process.
static void staged_own_shutdown(VkCtx *c) {
    static struct { uint64_t lo, hi; } *rg;
    static int crg;
    if (fetch_test) plat_mutex_lock(&ft_lock);
    plat_mutex_lock(&own_lock);
    int n = nst_pools;
    while (crg < n) grow((void **)&rg, &crg, sizeof *rg);
    for (int i = 0; i < n; i++) rg[i].lo = st_pools[i]->import_lo, rg[i].hi = st_pools[i]->import_hi;
    plat_mutex_unlock(&own_lock);
    for (int i = 0; i < n; i++) mem_thaw(rg[i].lo, rg[i].hi - rg[i].lo);   // staged_fetch takes own_lock
    plat_mutex_lock(&own_lock);
    for (int i = 0; i < nst_pools; i++) {
        PoolRec *r = st_pools[i];
        int claimed;
        mem_guard_drop(r->import_lo, r->import_hi - r->import_lo, &claimed);
        free(r->own);
        r->own = NULL, r->nown = r->cown = r->own_ix = 0;
    }
    nst_pools = nst_own = 0;
    __atomic_add_fetch(&own_gen, 1, __ATOMIC_RELEASE);
    if (fc.pool) {   // the command buffer goes with its pool
        vkDestroyFence(c->device, fc.fence, NULL);
        vkDestroyCommandPool(c->device, fc.pool, NULL);
        memset(&fc, 0, sizeof fc);
    }
    plat_mutex_unlock(&own_lock);
    if (fetch_test) plat_mutex_unlock(&ft_lock);
}

// With the staged line: the owned pages and fetches since the last one, and the first OWN_LOG fetches.
static void own_log(uint64_t frame, double f) {
    uint64_t bytes = 0, cpu_read = 0;
    int ranges = 0;
    plat_mutex_lock(&own_lock);
    for (int i = 0; i < nst_own; i++) {
        const PoolRec *r = st_own[i];
        ranges += r->nown;
        for (int k = 0; k < r->nown; k++) bytes += r->own[k].hi - r->own[k].lo;
    }
    // Every staged pool's marks: a byte a page, about 0.3 MB for the arena and all the small pools of a layer, once a
    // summary line (every 60 frames with TOTK_TRACE=perf).
    for (int i = 0; i < nst_pools; i++) {
        const PoolRec *r = st_pools[i];
        for (uint64_t p = 0; p <= (r->import_hi - r->import_lo) >> 12; p++) cpu_read += r->cpu_read[p];
    }
    plat_mutex_unlock(&own_lock);
    int64_t fe = plat_xchg64(&own_fempty, 0), lw = plat_xchg64(&own_long, 0), sw = plat_xchg64(&own_switched, 0);
    int64_t fbr = plat_xchg64(&own_fb_ranges, 0), fbb = plat_xchg64(&own_fb_bytes, 0);
    int64_t cbo = plat_xchg64(&own_cb_opened, 0), cbk = plat_xchg64(&own_cb_kept, 0), cf = plat_xchg64(&own_cfetch, 0);
    rt_log("mem: staged own frame %llu: since the last line: fetches without an owned byte %lld, fetches that waited over "
           "1 s %lld, fetches inside staged_wb_done %lld; pages switched to the fence copy-back %lld, CPU-read pages now "
           "%llu; copy-back onto pages owned again: opened %lld, pieces kept owned %lld; per frame: their GPU writes "
           "copied back at the fence %.1f pieces, %.3f MiB (time in the staged frame line's write-back)",
           (unsigned long long)frame, (long long)fe, (long long)lw, (long long)cf, (long long)sw,
           (unsigned long long)cpu_read, (long long)cbo, (long long)cbk, fbr / f, fbb / 1048576.0 / f);
    int64_t pr = plat_xchg64(&own_protects, 0), gr = plat_xchg64(&own_fetch[0][1], 0), gw = plat_xchg64(&own_fetch[1][1], 0);
    int64_t rr = plat_xchg64(&own_fetch[0][0], 0), rw = plat_xchg64(&own_fetch[1][0], 0), kf = plat_xchg64(&own_kfetch, 0);
    int64_t fp = plat_xchg64(&own_fpages, 0), fb = plat_xchg64(&own_fbytes, 0), fn = plat_xchg64(&own_fns, 0);
    int64_t fm = plat_xchg64(&own_fmax, 0);
    int64_t s0 = plat_xchg64(&own_seen[0], 0), s1 = plat_xchg64(&own_seen[1], 0), s2 = plat_xchg64(&own_seen[2], 0);
    int64_t pu = plat_xchg64(&own_pushed, 0), tn = plat_xchg64(&own_take_ns, 0), pf = plat_xchg64(&own_pfail, 0);
    int64_t tp[5];
    for (int k = 0; k < 5; k++) tp[k] = plat_xchg64(&own_take_part[k], 0);
    rt_log("mem: staged own frame %llu: per frame: fence-time pages seen %.1f open, %.1f owned, %.1f busy; %.1f queued, "
           "%.1f failed to protect; own_take %.3f ms (merge and scan %.3f, thaw %.3f, commit %.3f, protect %.3f in %.1f "
           "runs); since the last line the longest staged_wb_done %.2f ms", (unsigned long long)frame, s0 / f, s1 / f,
           s2 / f, pu / f, pf / f, tn / 1e6 / f, tp[4] / 1e6 / f, tp[0] / 1e6 / f, tp[1] / 1e6 / f, tp[2] / 1e6 / f,
           tp[3] / f, plat_xchg64(&st_wbdone_max, 0) / 1e6);
    rt_log("mem: staged own frame %llu: owned now %d ranges, %.1f MiB; per frame: pages made owned (VirtualProtect calls) "
           "%.1f; since the last line: fetches game read %lld, game write %lld, runtime read %lld, runtime write %lld, "
           "kernel paths %lld; %lld pages, %.2f MiB, %.2f ms in all, longest %.2f ms",
           (unsigned long long)frame, ranges, bytes / 1048576.0, pr / f, (long long)gr, (long long)gw, (long long)rr,
           (long long)rw, (long long)kf, (long long)fp, fb / 1048576.0, fn / 1e6, fm / 1e6);
    uintptr_t base = plat_exe_base();
    int32_t nr = __atomic_load_n(&own_nrec, __ATOMIC_RELAXED);
    if (nr > OWN_LOG) nr = OWN_LOG;
    for (int32_t i = own_logged; i < nr; i++) {
        const OwnRec *x = &own_rec[i];
        plat_mutex_lock(&own_lock);
        PoolRec *r = st_find(x->addr);
        uint64_t off = r ? x->addr - r->lo : 0;
        plat_mutex_unlock(&own_lock);
        rt_log("mem: staged fetch %d: frame %llu, %s %#llx (pool offset %#llx%s), %s, %s exe+%#llx, %d pages, %.3f ms", i,
               (unsigned long long)x->frame, x->kernel ? "kernel path at" : x->write ? "write" : "read",
               (unsigned long long)x->addr, (unsigned long long)off, r ? "" : ", pool gone", x->guest ? "game thread" : "runtime thread",
               x->kernel ? "caller" : "rip", (unsigned long long)(x->rip >= base ? x->rip - base : x->rip), x->pages,
               x->ns / 1e6);
    }
    own_logged = nr;
}

// ---- fetch exerciser (TOTK_STAGED_FETCH_TEST=<pages per frame>, off by default; a test of TOTK_STAGED_WB=fault)
// On Hilltop nothing reads a staged pool's GPU-owned pages, so staged_fetch never runs on its own. This mode touches
// that many randomly chosen owned pages a frame through the real fault path: half from a thread of its own (a runtime
// thread) and half from the present path (vk_mem_census, on the thread that presented, which is the game's own thread
// when the replay is synchronous, the default; a game thread inside the runtime, not lifted code), alternating a read
// and a write (a locked OR of 0 into one byte: a write access that leaves the byte as it was, whatever another thread
// writes there meanwhile). Every 8th frame its thread also hands the owned page's 64 KiB block to a kernel path,
// mem_thaw and mem_touch in turn, which fetch the owned pages there first. ft_lock holds a touch from its pick to its
// access, so that the pool cannot be released in between (staged_forget takes it).
// Every fetch it causes is checked: vk_mem_mirror_end also copies the submission's owned ranges into a readback buffer
// (staged_ref_record), own_take keeps those bytes per page as a reference, apart from guest memory, with the
// submission that wrote them, and staged_fetch compares the bytes it fetched with the reference on every page that no
// write recorded since that submission can have changed (the page's last recorded write, r->t_rec, stamped by
// vk_mem_gpu_wrote and vk_mem_gpu_kept, is the reference's submission, before and after the copy). Every 600 frames
// the pages' CPU-read marks are cleared, so fetched pages go back to being owned and the switch to the fence copy-back
// repeats.
static uint64_t ft_rng = 0x9e3779b97f4a7c15ull;   // own_lock

static uint64_t ft_next(void) {
    ft_rng ^= ft_rng << 13, ft_rng ^= ft_rng >> 7, ft_rng ^= ft_rng << 17;
    return ft_rng;
}

// ft_lock held: a byte of a randomly chosen owned page (0: none), and the pool's bounds.
static uint64_t ft_pick(uint64_t *lo, uint64_t *hi) {
    uint64_t a = 0;
    plat_mutex_lock(&own_lock);
    for (int tries = 0; tries < 8 && !a && nst_own; tries++) {   // st_own: the pools that own a byte
        PoolRec *r = st_own[ft_next() % (uint64_t)nst_own];
        if (!r->nown) continue;
        const struct KeptRange *k = &r->own[ft_next() % (uint64_t)r->nown];
        uint64_t x = r->import_lo + k->lo + ft_next() % (k->hi - k->lo);
        if (mem_guard_state(x) == 1) a = x, *lo = r->import_lo, *hi = r->import_hi;
    }
    plat_mutex_unlock(&own_lock);
    return a;
}

static void ft_access(uint64_t a, int write) {
    volatile uint8_t *p = (volatile uint8_t *)(uintptr_t)a;
    if (!write) (void)*p;
#if defined(__x86_64__)
    // Not an atomic fetch-or of 0: compilers turn that into a fenced load, which would fault as a read.
    else __asm__ volatile("lock orb $0, %0" : "+m"(*p) : : "memory");
#else
    else __atomic_fetch_add(p, 0, __ATOMIC_SEQ_CST);
#endif
}

// kind 0 runtime thread, 1 present path, each on one thread; reads and writes in turn, also across calls (at one
// touch a call, the parity of the loop index alone would only ever read).
static void ft_touch_pages(int n, int kind) {
    static unsigned turn[2];
    for (int i = 0; i < n; i++) {
        uint64_t lo, hi;
        plat_mutex_lock(&ft_lock);
        uint64_t a = ft_pick(&lo, &hi);
        if (a) ft_access(a, turn[kind]++ & 1), plat_inc64(&ft_touch[kind]);
        plat_mutex_unlock(&ft_lock);
        if (!a) return;
    }
}

static void ft_main(void *arg) {
    (void)arg;
    uint64_t seen = 0, kernel = 0;
    for (;;) {
        uint64_t f = __atomic_load_n(&mem_probe_frame, __ATOMIC_RELAXED);
        if (f == seen) {
            plat_sleep_ms(1);
            continue;
        }
        seen = f;
        ft_touch_pages((fetch_test + 1) / 2, 0);
        if (f % 8 == 0) {   // a kernel path: mem_thaw and mem_touch settle the owned pages of their range first
            uint64_t lo, hi;
            plat_mutex_lock(&ft_lock);
            uint64_t a = ft_pick(&lo, &hi);
            if (a) {
                uint64_t b0 = a & ~(OWN_BLOCK - 1), b1 = b0 + OWN_BLOCK;
                if (b0 < lo) b0 = lo;
                if (b1 > hi) b1 = hi;
                if (kernel++ & 1) mem_touch(b0, b1 - b0), plat_inc64(&ft_touch[3]);
                else mem_thaw(b0, b1 - b0), plat_inc64(&ft_touch[2]);
            }
            plat_mutex_unlock(&ft_lock);
        }
    }
}

// vk_mem_census, every present.
static void ft_frame(uint64_t frame) {
    static int started;
    if (!fetch_test || !__atomic_load_n(&nst_pools, __ATOMIC_RELAXED)) return;
    if (!started) started = plat_thread_spawn(ft_main, NULL, "staged fetch test", 0);
    ft_touch_pages(fetch_test / 2, 1);
    if (frame % 600 == 0) {   // the fetched pages become owned again at their next GPU write
        plat_mutex_lock(&own_lock);
        for (int i = 0; i < nst_pools; i++)
            memset(st_pools[i]->cpu_read, 0, (size_t)((st_pools[i]->import_hi - st_pools[i]->import_lo) >> 12) + 1);
        plat_mutex_unlock(&own_lock);
    }
}

// Record time (vk_mem_gpu_wrote, vk_mem_gpu_kept): the pages of a write into a staged mirror carry the submission.
static void ft_stamp(PoolRec *r, uint64_t off, uint64_t size) {
    if (!r->t_rec || !size) return;
    uint32_t s = __atomic_load_n(&msub, __ATOMIC_RELAXED);
    for (uint64_t p = off >> 12; p <= (off + size - 1) >> 12; p++) __atomic_store_n(&r->t_rec[p], s, __ATOMIC_RELAXED);
}

// vk_mem_mirror_end, after staged_split_own: the job's owned ranges also go mirror -> a readback buffer of the job (the
// reference); their roff is where. The caller's barriers order and expose the copies as for staged_wb_record.
static int staged_ref_record(VkCommandBuffer cb) {
    static VkBufferCopy *reg;
    static int creg;
    StJob *j = st_cur;
    uint64_t total = 0;
    for (int i = 0; i < j->no; i++) j->o[i].roff = total, total += j->o[i].size;
    if (!total) return 0;
    j->tmap = vk_staging(total, &j->tbuf, &j->talloc);
    int rec = 0;
    for (int i = 0; i < j->no;) {
        PoolRec *r = j->o[i].r;
        int k = 0;
        for (; i < j->no && j->o[i].r == r; i++) {
            if (k == creg) grow((void **)&reg, &creg, sizeof *reg);
            reg[k++] = (VkBufferCopy){j->o[i].off, j->o[i].roff, j->o[i].size};
        }
        vkCmdCopyBuffer(cb, r->mbuf, j->tbuf, (uint32_t)k, reg);
        rec++;
    }
    return rec;
}

// own_take, own_lock held: the reference bytes of the job's ranges add[0..n) of pool r, from the job's readback.
static void ft_ref_take(PoolRec *r, const StRange *add, int n, const uint8_t *tmap, uint32_t sub) {
    for (int i = 0; i < n; i++)
        for (uint64_t x = add[i].off; x < add[i].off + add[i].size;) {
            uint64_t pg = x >> 12, e = (pg + 1) << 12;
            if (e > add[i].off + add[i].size) e = add[i].off + add[i].size;
            if (!r->t_page[pg] && !(r->t_page[pg] = calloc(1, 0x1000))) rt_fatal("vk: out of memory");
            memcpy(r->t_page[pg] + (x & 0xfff), tmap + add[i].roff + (x - add[i].off), e - x);
            r->t_ref[pg] = sub;
            x = e;
        }
}

// staged_fetch, own_lock held, after the copy: the fetched pieces against the reference, on the pages whose last
// recorded write is the reference's and was so before the copy too (rec0).
static void ft_check(PoolRec *r, const struct KeptRange *cut, const VkBufferCopy *reg, int n, const uint8_t *map,
                     const uint32_t *rec0, uint64_t w0off) {
    static int32_t logged;
    for (int i = 0; i < n; i++)
        for (uint64_t x = cut[i].lo; x < cut[i].hi;) {
            uint64_t pg = x >> 12, e = (pg + 1) << 12;
            if (e > cut[i].hi) e = cut[i].hi;
            uint32_t now = __atomic_load_n(&r->t_rec[pg], __ATOMIC_RELAXED), was = rec0[(x - w0off) >> 12];
            if (now != was || now != r->t_ref[pg] || !r->t_page[pg]) {
                plat_add64(&ft_skip, (int64_t)(e - x));
            } else {
                const uint8_t *got = map + reg[i].dstOffset + (x - cut[i].lo), *want = r->t_page[pg] + (x & 0xfff);
                plat_add64(&ft_cmp_bytes, (int64_t)(e - x));
                plat_inc64(&ft_cmp_pieces);
                if (memcmp(got, want, e - x)) {
                    plat_inc64(&ft_bad);
                    if (plat_inc32(&logged) <= 16) {
                        uint64_t k = 0;
                        while (k < e - x && got[k] == want[k]) k++;
                        rt_log("mem: staged fetch test mismatch: pool %#llx offset %#llx (%llu bytes, first at +%llu: fetched "
                               "%02x, reference %02x of submission %u)", (unsigned long long)r->lo,
                               (unsigned long long)(r->import_lo + x - r->lo), (unsigned long long)(e - x),
                               (unsigned long long)k, got[k], want[k], r->t_ref[pg]);
                    }
                }
            }
            x = e;
        }
}

static void ft_log(uint64_t frame) {
    if (!fetch_test) return;
    rt_log("mem: staged fetch test frame %llu: since the last line: touches runtime thread %lld, present path %lld, kernel "
           "path mem_thaw %lld, mem_touch %lld; fetched bytes compared %lld in %lld pieces, mismatches %lld, not "
           "comparable %lld bytes", (unsigned long long)frame,
           (long long)plat_xchg64(&ft_touch[0], 0), (long long)plat_xchg64(&ft_touch[1], 0),
           (long long)plat_xchg64(&ft_touch[2], 0), (long long)plat_xchg64(&ft_touch[3], 0),
           (long long)plat_xchg64(&ft_cmp_bytes, 0), (long long)plat_xchg64(&ft_cmp_pieces, 0),
           (long long)plat_xchg64(&ft_bad, 0), (long long)plat_xchg64(&ft_skip, 0));
}

// TOTK_STAGED_INJECT=import: the import fails as lazy_import's does (no GPU buffer, counted in n_import_failures). The
// caller holds pool_lock shared.
static int inject_import_fail(PoolRec *r) {
    plat_mutex_lock(&lazy_lock);
    if (r->lazy) {
        long n = plat_inc32(&n_import_failures);
        if (n <= 8) rt_log("vk: pool %#llx (flags %#x, %#llx bytes) has no GPU buffer (%ld so far; TOTK_STAGED_INJECT=import)",
                           (unsigned long long)r->lo, r->flags, (unsigned long long)r->size, n);
        __atomic_store_n(&r->lazy, 0, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&lazy_lock);
    return r->buf != VK_NULL_HANDLE;
}

// The span's pool resolved without a buffer (staged_pending) and the mirror cannot serve this use: import it after
// all, as lazy_import does on a first GPU use. Logged once per pool (the first 32) and counted by reason (stc: an MR_
// reason of the refused mirror, SI_HELPER or SI_GONE). When the import fails too, the pool ends as one whose first-use
// import failed (resolve_locked, lazy_import): lazy_import has logged and counted it (n_import_failures), the pool is
// no longer staged_pending and has no buffer, so every later resolve reports its addresses as unbacked, and the users
// of those skip them. This span stays without a buffer, and its user skips it the same way (VK_MEM_USE and VK_MEM_PICK
// in vk.h); the pool gets a line (the first 32) and is counted in the staged lines (imp_failed).
static void staged_import(VkMemSpan *sp, int why) {
    static const char *const text[SI_N] = {"its VRAM mirror does not fit the budget", "its range is not write-watched",
                                           "it overlaps or shares a page with another pool",
                                           "the device-local allocation failed",
                                           "first used on a TOTK_REPLAY_PAR helper thread", "its VRAM mirror was refused"};
    PoolRec *r = sp->pool;
    uint64_t t0 = qpc();
    pool_lock_shared();   // as resolve_locked: the record stays alive while lazy_import runs
    int ok = staged_inject() == INJECT_IMPORT ? inject_import_fail(r) : lazy_import(r, -1, "for a staged pool after all");
    plat_mutex_unlock_shared(&pool_lock);
    if (!ok) {
        sp->buffer = VK_NULL_HANDLE;
        if (!plat_xchg32(&r->sfell, 1)) {   // once per pool: the spans resolved before the failure come here too
            long n = plat_inc32(&n_staged_import_failures);
            ST_ADD(imp_failed, 1), ST_ADD(imp_failed_bytes, r->size);
            if (n <= 32)
                rt_log("mem: staged pool %#llx+%#llx has neither a mirror nor an import (%s): its uses are skipped as "
                       "those of an unbacked pool (%ld so far)", (unsigned long long)r->lo, (unsigned long long)r->size,
                       text[why], n);
        }
        return;
    }
    if (!plat_xchg32(&r->sfell, 1)) {
        long n = plat_inc32(&n_staged_imports);
        ST_ADD(imp_n[why], 1), ST_ADD(imp_bytes[why], r->size);
        if (n <= 32)
            rt_log("mem: staged pool %#llx+%#llx imported after all: %s (%.1f ms; %ld so far)", (unsigned long long)r->lo,
                   (unsigned long long)r->size, text[why], qpc_ms(qpc() - t0), n);
    }
    sp->buffer = r->buf;
}

// The pages of staged pool r the write-watch ever saw written, as runs (offsets from import_lo) appended to *a, from
// the page generations; returns their bytes. A first fill copies only these and zeroes the rest of the mirror. Guest
// memory is write-watched from its reservation (memory.c watched, win32.c plat_mem_reserve), and mirror_create's
// snapshot has just stamped every page written since the last query, so a page with no stamp has never been written
// and still holds the zeros it was committed with. Reading such pages is what made a full copy slow: in the Hilltop
// Layer the 450 MiB arena had little written, and the copy of all of it ran at 2 GiB/s (219 ms), a demand-zero or
// lazy-heap commit fault per page. A page written after the scan is stamped after the mirror's mgen and copied in by
// the next sync, as with the full copy.
static uint64_t staged_runs(PoolRec *r, MRange **a, int *n, int *cap) {
    uint64_t copied = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t x = r->import_lo; x < r->import_hi;) {
        uint64_t ce = ((x >> GEN_CHUNK_SHIFT) + 1) << GEN_CHUNK_SHIFT;
        if (ce > r->import_hi) ce = r->import_hi;
        const uint32_t *ch = gen_chunk[x >> GEN_CHUNK_SHIFT];
        for (; ch && x < ce; x += 1ull << GEN_PAGE_SHIFT)
            if (ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)])
                range_add(a, n, cap, r, x - r->import_lo, 1ull << GEN_PAGE_SHIFT), copied += 1ull << GEN_PAGE_SHIFT;
        x = ce;
    }
    plat_mutex_unlock_shared(&gen_lock);
    return copied;
}

static void upload_wait(uint64_t seq);
static volatile uint64_t inject_lost;

// The first fill of a large staged mirror (over STG_SMALL), on the thread that made it, mirror_lock let go (see
// above): the pages staged_runs finds are copied in, the rest of the mirror is zeroed on the GPU. A staging buffer is
// reused once the batch of its previous piece is done, waited for on that batch's fence (upload_wait), which reports a
// lost device. bm: mirror_create's times, for the line.
static void staged_fill(PoolRec *r, const BigMirror *bm) {
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo, t0 = qpc(), t_alloc = 0, t_copy = 0, t_wait = 0, t_scan, pieces = 0;
    static _Thread_local MRange *run;
    static _Thread_local int nrun, crun;
    nrun = 0;
    uint64_t copied = staged_runs(r, &run, &nrun, &crun);
    t_scan = qpc() - t0;
    VkCommandBuffer cb = vk_upload_begin();
    vkCmdFillBuffer(cb, r->mbuf, 0, VK_WHOLE_SIZE, 0);   // the pages never written
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
             VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vk_upload_end();
    struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; uint64_t seq; } slot[STG_SLOTS];
    memset(slot, 0, sizeof slot);
    static _Thread_local VkBufferCopy *reg;
    static _Thread_local int creg;
    for (int i = 0; i < nrun; pieces++) {   // the runs in pieces of at most STG_CHUNK bytes of staging
        typeof(slot[0]) *s = &slot[pieces % STG_SLOTS];
        uint64_t t = qpc();
        if (!s->buf) {
            // Owned here (alloc non-NULL), so taken outside vk_upload_begin: vk_staging touches the open batch only
            // for the buffers the batch owns.
            s->map = vk_staging(STG_CHUNK, &s->buf, &s->alloc);
            t_alloc += qpc() - t;
        } else {
            upload_wait(s->seq);   // its previous piece's batch: a copy of 16 MiB, about 1 ms
            t_wait += qpc() - t;
        }
        t = qpc();
        uint64_t used = 0;
        int k = 0;
        while (i < nrun && used < STG_CHUNK) {
            uint64_t len = run[i].size < STG_CHUNK - used ? run[i].size : STG_CHUNK - used;
            memcpy(s->map + used, (const void *)(uintptr_t)(r->import_lo + run[i].off), len);
            if (k == creg) grow((void **)&reg, &creg, sizeof *reg);
            reg[k++] = (VkBufferCopy){used, run[i].off, len};
            used += len;
            if (len == run[i].size) i++;
            else run[i].off += len, run[i].size -= len;
        }
        vmaFlushAllocation(c->vma, s->alloc, 0, used);
        t_copy += qpc() - t;
        cb = vk_upload_begin();
        vkCmdCopyBuffer(cb, s->buf, r->mbuf, (uint32_t)k, reg);
        s->seq = vk_upload_seq();
        // TOTK_STAGED_INJECT=fence: this piece's batch and every later one read as lost (retire, upload_wait_locked),
        // when a later piece will wait for this buffer.
        if (!pieces && copied > STG_SLOTS * STG_CHUNK && staged_inject() == INJECT_FENCE && !inject_lost)
            __atomic_store_n(&inject_lost, s->seq, __ATOMIC_RELAXED);
        vk_upload_end();
        vk_upload_flush(0);
    }
    cb = vk_upload_begin();
    // Later submissions (every part that uses the mirror) see the fill and the copies of all the pieces.
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    // Freed once this batch, the newest of the fill, is done (and the submission being recorded).
    for (int i = 0; i < STG_SLOTS; i++) vk_defer_buffer(slot[i].buf, slot[i].alloc);
    vk_upload_end();
    vk_upload_flush(0);
    uint64_t total = qpc() - t0;
    plat_add64(&st_fill_ns, (int64_t)total);
    plat_add64(&st_fill_bytes, (int64_t)copied);
    long nf = plat_inc32(&n_staged_fills);
    ST_ADD(big_n, 1);
    if (nf <= 32)
        rt_log("mem: staged mirror of %.0f MiB at %#llx made in %.1f ms on first use (after %.1f ms waiting for the walk "
               "and the mirror lock): pool lookup %.1f, budget check %.1f, snapshot %.1f (%llu queries, %llu written "
               "pages), vmaCreateBuffer %.1f; first fill %.1f ms: %.1f MiB ever written in %d runs, copied in %llu pieces "
               "of up to %llu MiB, the rest zeroed: page scan %.1f, staging allocation %.1f, CPU copy %.1f (%.2f GiB/s), "
               "waits for a free staging buffer %.1f, recording and submitting %.1f",
               n / 1048576.0, (unsigned long long)r->lo, qpc_ms(bm->total), qpc_ms(bm->hold), qpc_ms(bm->lock),
               qpc_ms(bm->budget), qpc_ms(bm->snap), (unsigned long long)bm->queries, (unsigned long long)bm->pages,
               qpc_ms(bm->alloc), qpc_ms(total), copied / 1048576.0, nrun, (unsigned long long)pieces,
               (unsigned long long)(STG_CHUNK >> 20), qpc_ms(t_scan), qpc_ms(t_alloc), qpc_ms(t_copy),
               t_copy ? copied / 1073741824.0 / (t_copy / 1e9) : 0.0, qpc_ms(t_wait),
               qpc_ms(total - t_scan - t_alloc - t_copy - t_wait));
}

#define MRANGE_BEFORE(x, pool, o) ((uintptr_t)(x)->r < (uintptr_t)(pool) || ((x)->r == (pool) && (x)->off + (x)->size <= (o)))

// st_lock held: st_blk becomes the union of the write-back ranges of the jobs recorded before the submission of
// `fence` and not yet copied into guest memory (a submission's own write-backs come after its uploads in GPU order:
// the last part's prologue is recorded after vk_mem_mirror_end).
static void st_blk_update(uint64_t fence) {
    static MRange *sbuf;
    static size_t scap;
    static uint64_t owngen;
    uint64_t og = __atomic_load_n(&own_gen, __ATOMIC_ACQUIRE);
    if (!st_blk_dirty && st_blk_fence == fence && owngen == og) return;
    nst_blk = 0;
    for (StJob *j = st_jobs; j; j = j->next) {
        if (!j->ended || j->fence >= fence) continue;
        for (int i = 0; i < j->n; i++) range_add(&st_blk, &nst_blk, &cst_blk, j->w[i].r, j->w[i].off, j->w[i].size);
        for (int i = 0; i < j->no; i++) range_add(&st_blk, &nst_blk, &cst_blk, j->o[i].r, j->o[i].off, j->o[i].size);
    }
    if (staged_fault) {   // the bytes only the mirrors hold: guest memory there is stale
        // The pools whose own list holds a range, not every staged pool: with TOTK_MIRROR_STAGED=all hundreds to
        // thousands of them, a cache miss each, at every prologue after an own list changed (most fences).
        plat_mutex_lock(&own_lock);
        og = __atomic_load_n(&own_gen, __ATOMIC_ACQUIRE);
        for (int i = 0; i < nst_own; i++)
            for (int k = 0; k < st_own[i]->nown; k++)
                range_add(&st_blk, &nst_blk, &cst_blk, st_own[i], st_own[i]->own[k].lo,
                          st_own[i]->own[k].hi - st_own[i]->own[k].lo);
        plat_mutex_unlock(&own_lock);
    }
    owngen = og;
    sort_ranges(st_blk, (size_t)nst_blk, &sbuf, &scap);
    int o = 0;
    for (int i = 0; i < nst_blk; i++) {
        if (o && st_blk[o - 1].r == st_blk[i].r && st_blk[i].off <= st_blk[o - 1].off + st_blk[o - 1].size) {
            uint64_t end = st_blk[i].off + st_blk[i].size;
            if (end > st_blk[o - 1].off + st_blk[o - 1].size) st_blk[o - 1].size = end - st_blk[o - 1].off;
        } else st_blk[o++] = st_blk[i];
    }
    nst_blk = o;
    st_blk_dirty = 0, st_blk_fence = fence;
}

// staged_upload: a piece of staging is full (used bytes) and the copies out of it are recorded; freed at the fence.
static void staged_piece_done(VkBuffer sb, VmaAllocation sa, uint64_t used, uint64_t fence) {
    vmaFlushAllocation(vk_ctx()->vma, sa, 0, used);
    StBuf *s = malloc(sizeof *s);
    if (!s) rt_fatal("vk: out of memory");
    *s = (StBuf){sb, sa};
    vk_completion_add(fence, staged_free_buffer, s);
}

// Replay thread, in the prologue of the part (fence: its submission's), mirror_lock not held: the ranges in stq go
// from guest memory into staging buffers and from there into their mirrors, minus the write-back bytes in flight.
// A small staged mirror made during the part (szero, mirror_create) is zeroed first, and the pages ever written
// (staged_runs) take the place of its queued whole range: the first fills of a layer's 276 small pools go into this one
// prologue, with no submission of their own. Staging comes in pieces of at most STG_CHUNK from the staging pool's kept
// blocks (vk_staging), so the up to 73 MiB of such a frame need no vkAllocateMemory of a larger buffer.
// The barriers around it are the prologue's. Returns whether it recorded zero fills: the prologue's last barrier then
// covers clears as well as copies.
static int staged_upload(VkCommandBuffer cb, uint64_t fence) {
    static MRange *sbuf, *cut;
    static size_t scap;
    static int ncut, ccut;
    static VkBufferCopy *reg;
    static int creg;
    static PoolRec **zl;   // the pools first filled here, in stq's order (by address of the record)
    static int nzl, czl;
    uint64_t t0 = qpc();
    sort_ranges(stq, (size_t)nstq, &sbuf, &scap);
    nzl = 0;
    int m = 0;
    for (int i = 0; i < nstq; i++) {
        PoolRec *r = stq[i].r;
        if (r->szero) {   // replay thread only from mirror_create's store on (mirror_lock taken since, by the prologue)
            r->szero = 0;
            if (nzl == czl) grow((void **)&zl, &czl, sizeof *zl);
            zl[nzl++] = r;
        }
        if (nzl && zl[nzl - 1] == r) continue;   // its ranges are the whole mirror (and syncs of it, none yet)
        stq[m++] = stq[i];
    }
    nstq = m;
    uint64_t z_bytes = 0, z_copied = 0, z_ns = 0;
    for (int z = 0; z < nzl; z++) {
        vkCmdFillBuffer(cb, zl[z]->mbuf, 0, VK_WHOLE_SIZE, 0);   // the pages never written
        z_bytes += zl[z]->import_hi - zl[z]->import_lo;
        z_copied += staged_runs(zl[z], &stq, &nstq, &cstq);
    }
    if (nzl) {
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT);   // the written pages land over the zeros
        sort_ranges(stq, (size_t)nstq, &sbuf, &scap);
    }
    // A job leaves st_jobs only after its bytes are in guest memory or on the pool's own list (staged_wb_done), so a
    // range blocked here has its bytes in the mirror and one not blocked has them in guest memory; the copies need not
    // hold the lock. No job ends meanwhile: vk_mem_mirror_end runs on this thread. A fetch that empties an own range
    // after this only makes guest memory equal the mirror there.
    plat_mutex_lock(&st_lock);
    st_blk_update(fence);
    ncut = 0;
    uint64_t total = 0, skipped = 0;
    for (int i = 0; i < nstq;) {   // merged per pool, clamped to the mirror, then the blocked bytes cut out
        PoolRec *r = stq[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        for (; i < nstq && stq[i].r == r;) {
            uint64_t a = stq[i].off, b = a + stq[i].size;
            for (i++; i < nstq && stq[i].r == r && stq[i].off <= b; i++)
                if (stq[i].off + stq[i].size > b) b = stq[i].off + stq[i].size;
            if (b > lim) b = lim;
            if (a >= b) continue;
            int lo = 0, hi = nst_blk;   // the first blocked range of r that ends above a
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (MRANGE_BEFORE(&st_blk[mid], r, a)) lo = mid + 1;
                else hi = mid;
            }
            for (int k = lo; a < b && k < nst_blk && st_blk[k].r == r && st_blk[k].off < b; k++) {
                uint64_t ka = st_blk[k].off, kb = ka + st_blk[k].size;
                if (ka > a) range_add(&cut, &ncut, &ccut, r, a, ka - a), total += ka - a;
                uint64_t e = kb < b ? kb : b;
                if (e > a) skipped += e - (ka > a ? ka : a), a = e;
            }
            if (a < b) range_add(&cut, &ncut, &ccut, r, a, b - a), total += b - a;
        }
    }
    plat_mutex_unlock(&st_lock);
    if (total) {
        VkBuffer sb = VK_NULL_HANDLE;
        VmaAllocation sa = NULL;
        uint8_t *map = NULL;
        uint64_t used = 0, room = 0, left = total;
        int probe = staged_fault || mem_probe_on() > 0;   // owned or probed pages: copied without opening them
        for (int i = 0, zi = 0; i < ncut;) {
            PoolRec *r = cut[i].r;
            while (zi < nzl && (uintptr_t)zl[zi] < (uintptr_t)r) zi++;
            int first = zi < nzl && zl[zi] == r;   // its first fill: timed apart
            uint64_t tz = first ? qpc() : 0;
            int k = 0;
            for (uint64_t done = 0; i < ncut && cut[i].r == r;) {
                if (used == room) {   // the next piece of staging; the copies out of the full one first
                    if (k) vkCmdCopyBuffer(cb, sb, r->mbuf, (uint32_t)k, reg), k = 0;
                    if (sb) staged_piece_done(sb, sa, used, fence);
                    room = left < STG_CHUNK ? left : STG_CHUNK;
                    map = vk_staging(room, &sb, &sa);
                    used = 0;
                }
                uint64_t len = cut[i].size - done, src = r->import_lo + cut[i].off + done;
                if (len > room - used) len = room - used;
                if (probe) mem_probe_copy(map + used, src, len);   // the guard stays on
                else memcpy(map + used, (const void *)(uintptr_t)src, len);
                if (k == creg) grow((void **)&reg, &creg, sizeof *reg);
                reg[k++] = (VkBufferCopy){used, cut[i].off + done, len};
                used += len, left -= len, done += len;
                if (done == cut[i].size) i++, done = 0;
            }
            if (k) vkCmdCopyBuffer(cb, sb, r->mbuf, (uint32_t)k, reg);
            if (first) z_ns += qpc() - tz;
        }
        if (sb) staged_piece_done(sb, sa, used, fence);
    }
    if (nzl) ST_ADD(small_n, nzl), ST_ADD(small_bytes, z_bytes), ST_ADD(small_copied, z_copied), ST_ADD(small_ns, z_ns);
    mrep.upload_bytes += total, mrep.upload_copies += (uint64_t)ncut;   // the mirror line's totals include them
    mrep.st_up_bytes += total, mrep.st_up_ranges += (uint64_t)ncut, mrep.st_skip_bytes += skipped;
    mrep.st_up_t += qpc() - t0;
    nstq = 0;
    return nzl > 0;
}

// vk_mem_mirror_end: the write-back ranges of staged mirrors (stw) go mirror -> a readback buffer of the submission's
// job; staged_wb_done copies them into guest memory at the fence. Returns the commands recorded. The caller's barriers
// order the copies after the submission's writes and make them visible to the host.
static int staged_wb_record(VkCommandBuffer cb) {
    static MRange *sbuf;
    static size_t scap;
    static VkBufferCopy *reg;
    static int creg;
    StJob *j = st_cur;
    if (!j) rt_fatal("mem: a staged write-back outside a submission's recording");
    uint64_t t0 = qpc();
    sort_ranges(stw, (size_t)nstw, &sbuf, &scap);
    for (int i = 0; i < nstw;) {
        PoolRec *r = stw[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        for (; i < nstw && stw[i].r == r;) {
            uint64_t a = stw[i].off, b = a + stw[i].size;
            for (i++; i < nstw && stw[i].r == r && stw[i].off <= b; i++)
                if (stw[i].off + stw[i].size > b) b = stw[i].off + stw[i].size;
            if (b > lim) b = lim;
            if (a >= b) continue;
            if (j->n == j->cap) grow((void **)&j->w, &j->cap, sizeof *j->w);
            j->w[j->n++] = (StRange){r, r->lo, r->storage_serial, a, b - a, j->bytes};
            j->bytes += b - a;
        }
    }
    nstw = 0;
    if (!j->bytes) return 0;
    // From the staging pool, whose blocks are kept (vk_staging): a new block of VMA's default pools would be a
    // vkAllocateMemory in the middle of a frame. Its memory is host-cached where the device has it (random access),
    // for the CPU reads at the fence.
    j->map = vk_staging(j->bytes, &j->buf, &j->alloc);
    int rec = 0;
    for (int i = 0; i < j->n;) {
        PoolRec *r = j->w[i].r;
        int k = 0;
        for (; i < j->n && j->w[i].r == r; i++) {
            if (k == creg) grow((void **)&reg, &creg, sizeof *reg);
            reg[k++] = (VkBufferCopy){j->w[i].off, j->w[i].roff, j->w[i].size};
        }
        vkCmdCopyBuffer(cb, r->mbuf, j->buf, (uint32_t)k, reg);
        rec++;
    }
    mrep.wb_bytes += j->bytes, mrep.wb_copies += (uint64_t)j->n;   // the mirror line's totals include them
    mrep.st_wb_bytes += j->bytes, mrep.st_wb_ranges += (uint64_t)j->n, mrep.st_wb_t += qpc() - t0;
    return rec;
}

// staged_copy_back: the bytes of one page that a copy-back carries (offsets from import_lo, roff in the readback).
typedef struct { uint64_t off, size, roff; } CbPiece;

// own_lock held: whether every owned byte of the page at offset pg lies in the sorted, disjoint pieces pc[0..n) of
// that page (a page owned with no byte on the list counts as covered).
static int own_covered(const PoolRec *r, uint64_t pg, const CbPiece *pc, int n) {
    uint64_t pe = pg + 0x1000;
    int k = 0, m = r->nown;
    while (k < m) {   // the first owned range ending above pg
        int mid = (k + m) / 2;
        if (r->own[mid].hi <= pg) k = mid + 1;
        else m = mid;
    }
    for (int q = 0; k < r->nown && r->own[k].lo < pe; k++) {
        uint64_t a = r->own[k].lo > pg ? r->own[k].lo : pg, b = r->own[k].hi < pe ? r->own[k].hi : pe;
        while (a < b) {
            while (q < n && pc[q].off + pc[q].size <= a) q++;
            if (q == n || pc[q].off > a) return 0;
            a = pc[q].off + pc[q].size;
        }
    }
    return 1;
}

// staged_wb_done, TOTK_STAGED_WB=fault, pool_lock shared, r live: the job's copy-back ranges w[i0..i1) of r go into
// guest memory. They are the GPU's writes to pages the CPU had read (r->cpu_read when the job was recorded), but such
// a page can be owned again by now: a submission recorded before the fetch that made the page CPU-read put its writes
// there on the own list, and its fence, after the fetch, protected the page again (own_take). Thawing such a page
// with mem_thaw would fetch it from the mirror, a GPU round trip on the completion thread. Instead, page by page,
// under own_lock:
// - open: the bytes are copied in after the lock is let go (only own_take, on this thread, protects pages);
// - owned, and every owned byte of the page is among the copy-back's bytes (own_covered): the page is claimed, the
//   bytes go in, its owned bytes leave the list and the page opens, with no fetch;
// - owned with other owned bytes too, or being settled by another thread: the copy-back's bytes join the own list
//   instead. The mirror holds them, and as the newest version (this fence is the latest that wrote them; later writes
//   come with later fences), so the page's next fetch brings them in with its other owned bytes, and the bytes of the
//   page that neither carries stay the guest's. A page found being settled is protected again once the claim ends if
//   it is open then (mem_guard_own), as own_take does: the claim may have been a fault on an open page.
static void staged_copy_back(PoolRec *r, const StJob *j, int i0, int i1) {
    static CbPiece *pc, *open;   // completion thread: the pieces by page; those for open pages
    static int cpc, cop;
    static StRange *keep;
    static int ckeep;
    static struct KeptRange *gone;
    static int cgone;
    static uint64_t *again;
    static int cagain;
    const uint64_t base = r->import_lo;
    int npc = 0, nop = 0, nkeep = 0, ngone = 0, nagain = 0;
    for (int x = i0; x < i1; x++)
        for (uint64_t a = j->w[x].off, b = a + j->w[x].size; a < b;) {
            uint64_t e = (a & ~0xfffull) + 0x1000;
            if (e > b) e = b;
            if (npc == cpc) grow((void **)&pc, &cpc, sizeof *pc);
            pc[npc++] = (CbPiece){a, e - a, j->w[x].roff + (a - j->w[x].off)};
            a = e;
        }
    plat_mutex_lock(&own_lock);
    for (int p0 = 0; p0 < npc;) {
        uint64_t pg = pc[p0].off & ~0xfffull;
        int p1 = p0 + 1;
        while (p1 < npc && (pc[p1].off & ~0xfffull) == pg) p1++;
        int st = mem_guard_state(base + pg);
        if (!st) {
            for (int q = p0; q < p1; q++) {
                if (nop == cop) grow((void **)&open, &cop, sizeof *open);
                open[nop++] = pc[q];
            }
        } else if (st == 1 && own_covered(r, pg, pc + p0, p1 - p0) && mem_guard_tryclaim(base + pg)) {
            for (int q = p0; q < p1; q++)   // the claimed page is read-write
                memcpy((void *)(uintptr_t)(base + pc[q].off), j->map + pc[q].roff, pc[q].size);
            if (ngone == cgone) grow((void **)&gone, &cgone, sizeof *gone);
            gone[ngone++] = (struct KeptRange){pg, pg + 0x1000};
        } else {
            for (int q = p0; q < p1; q++) {
                if (nkeep == ckeep) grow((void **)&keep, &ckeep, sizeof *keep);
                keep[nkeep++] = (StRange){r, r->lo, r->storage_serial, pc[q].off, pc[q].size, 0};
            }
            if (st == 2) {
                if (nagain == cagain) grow((void **)&again, &cagain, sizeof *again);
                again[nagain++] = base + pg;
            }
        }
        p0 = p1;
    }
    if (ngone) {
        own_cut(r, gone, ngone);
        for (int i = 0; i < ngone; i++) mem_guard_clear(base + gone[i].lo);   // after the cut: a fault now finds it open
    }
    if (nkeep) own_merge(r, keep, nkeep);
    if (ngone || nkeep) __atomic_add_fetch(&own_gen, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&own_lock);
    for (int i = 0; i < nagain; i++) plat_add64(&own_protects, mem_guard_own(again[i]));
    plat_add64(&own_cb_opened, ngone), plat_add64(&own_cb_kept, nkeep);
    for (int x = 0; x < nop;) {   // runs of pieces whose pages touch: one thaw each (guarded pages are not among them)
        uint64_t a = (base + open[x].off) & ~0xfffull, b = (base + open[x].off + open[x].size + 0xfff) & ~0xfffull;
        int y = x + 1;
        for (; y < nop && ((base + open[y].off) & ~0xfffull) <= b; y++) {
            uint64_t e2 = (base + open[y].off + open[y].size + 0xfff) & ~0xfffull;
            if (e2 > b) b = e2;
        }
        mem_thaw_keep(a, b - a);   // a frozen region is read-only; nothing to settle on these pages
        for (; x < y; x++) memcpy((void *)(uintptr_t)(base + open[x].off), j->map + open[x].roff, open[x].size);
    }
}

// Completion thread, at the job's fence and ahead of the submission's other callbacks (vk_gpu_written_many among them):
// for the pools still live (a pool finalized meanwhile has its storage back in the guest's hands), the owned ranges
// join the own lists and their pages are protected, then the copy-back bytes go into guest memory, then the job stops
// blocking the uploads. Protection first: the game may poll a word the copy-back writes (a page it reads every frame)
// as the sign that the submission's other writes are done, and those must not be readable stale once it is set (the
// window above).
static void staged_wb_done(void *arg) {
    StJob *j = arg;
    uint64_t t_done = qpc();
    st_in_wb = 1;
    if (j->no) {   // TOTK_STAGED_WB=fault: the writes that stay in the mirror
        if (j->tbuf) vmaInvalidateAllocation(vk_ctx()->vma, j->talloc, 0, VK_WHOLE_SIZE);   // TOTK_STAGED_FETCH_TEST
        for (int i = 0; i < j->no;) {
            PoolRec *r = j->o[i].r;
            pool_lock_shared();   // only for the check: own_take runs without it
            const pidx_ent *e = pidx_find(&pool_index, j->o[i].lo, 0);
            int live = e && e->buf == (void *)r && r->storage_serial == j->o[i].serial && ms_get(r) == MS_ON;
            plat_mutex_unlock_shared(&pool_lock);
            int k = i;
            for (; k < j->no && j->o[k].r == r; k++) {}
            if (live) own_take(r, j->o + i, k - i, j->tmap, j->sub);
            i = k;
        }
    }
    if (j->n) {
        uint64_t t0 = qpc();
        vmaInvalidateAllocation(vk_ctx()->vma, j->alloc, 0, VK_WHOLE_SIZE);
        pool_lock_shared();   // the pools stay as they are while the bytes go in (vk_gpu_written does the same)
        for (int i = 0; i < j->n;) {
            PoolRec *r = j->w[i].r;
            const pidx_ent *e = pidx_find(&pool_index, j->w[i].lo, 0);
            int live = e && e->buf == (void *)r && r->storage_serial == j->w[i].serial;
            int k = i;
            for (; k < j->n && j->w[k].r == r; k++) {}
            if (live && staged_fault && ms_get(r) == MS_ON) staged_copy_back(r, j, i, k), live = 0;
            for (int x = i; live && x < k;) {   // TOTK_STAGED_WB=fence: runs of ranges whose pages touch, one thaw each
                uint64_t base = r->import_lo, a = (base + j->w[x].off) & ~0xfffull;
                uint64_t b = (base + j->w[x].off + j->w[x].size + 0xfff) & ~0xfffull;
                int y = x + 1;
                for (; y < k && ((base + j->w[y].off) & ~0xfffull) <= b; y++) {
                    uint64_t e2 = (base + j->w[y].off + j->w[y].size + 0xfff) & ~0xfffull;
                    if (e2 > b) b = e2;
                }
                mem_probe_release(a, b - a);   // the read probe's pages from an earlier write-back, uncounted
                mem_thaw(a, b - a);   // a frozen region is read-only
                for (int z = x; z < y; z++)
                    memcpy((void *)(uintptr_t)(base + j->w[z].off), j->map + j->w[z].roff, j->w[z].size);
                // TOTK_STAGED_READ_PROBE: from here on the first CPU access to these pages is counted (memory.c).
                for (; x < y; x++) mem_probe_protect(base + j->w[x].off, j->w[x].size);
            }
            i = k;
        }
        plat_mutex_unlock_shared(&pool_lock);
        plat_add64(&st_wbcpu_ns, (int64_t)(qpc() - t0));
        plat_add64(&st_wbcpu_bytes, (int64_t)j->bytes);
    }
    plat_mutex_lock(&st_lock);
    for (StJob **p = &st_jobs; *p; p = &(*p)->next)
        if (*p == j) { *p = j->next; break; }
    if (j->ended && (j->n || j->no)) st_blk_dirty = 1;
    plat_mutex_unlock(&st_lock);
    if (j->buf) vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    if (j->tbuf) vmaDestroyBuffer(vk_ctx()->vma, j->tbuf, j->talloc);
    free(j->w);
    free(j->o);
    free(j);
    st_in_wb = 0;
    // Every later callback, and with them the value the game waits for, waits for this one.
    int64_t ns = (int64_t)(qpc() - t_done);
    for (int64_t mx = st_wbdone_max; ns > mx && !__atomic_compare_exchange_n(&st_wbdone_max, &mx, ns, 0, __ATOMIC_RELAXED,
                                                                            __ATOMIC_RELAXED);) {}
}

// vk_mem_shutdown, pool_lock held, the pools just freed: the staged ranges and the write-back jobs whose fence never
// came (vk_sync_shutdown dropped their callbacks).
static void staged_shutdown(VkCtx *c) {
    nstq = nstw = 0;
    plat_mutex_lock(&st_lock);
    while (st_jobs) {
        StJob *j = st_jobs;
        st_jobs = j->next;
        if (j->buf) vmaDestroyBuffer(c->vma, j->buf, j->alloc);
        if (j->tbuf) vmaDestroyBuffer(c->vma, j->tbuf, j->talloc);
        free(j->w);
        free(j->o);
        free(j);
    }
    st_cur = NULL, nst_blk = 0, st_blk_dirty = 1;
    plat_mutex_unlock(&st_lock);
}

// Replay thread: the pool's mirror on its first buffer use. A staged pool's mirror is not filled here: for a large one
// fill->pending tells the caller to fill it (staged_fill) once mirror_lock is let go, with the times of this part in
// *fill; a small one is filled by the next prologue (staged_upload).
static void mirror_create(PoolRec *r, BigMirror *fill) {
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo;
    uint64_t t0 = qpc();
    fill->pending = 0;
    // The state is stored once, when it is final. A helper thread that reads it before that (vk_mem_use) sees MS_NONE
    // and gives its part up; it never sees a refusal that turns into a mirror.
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    int staged = r->kind == POOL_IMPORT && staged_pending(r);   // no import: staged_fill copies the pool in
    if (r->kind != POOL_IMPORT || (!staged && !r->buf) || !n) return ms_set(r, MS_REFUSED);
    pool_lock_shared();
    int i = pidx_upper(&pool_index, r->lo) - 1;
    while (i >= 0 && pool_index.e[i].buf != r) i--;
    int alone = i >= 0 && (i == 0 || pool_index.maxhi[i - 1] <= pool_index.e[i].lo) &&
                (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= pool_index.e[i].hi);
    // A staged mirror's GPU-owned bytes are protected, fetched and dropped by 4 KiB page, with the page given to the
    // one pool whose import range holds it (staged_fetch): no other pool may reach into its pages. NVN pools are 4 KiB
    // aligned and sized, so this holds unless the import alignment is coarser than a page.
    int apart = alone && (!staged || !staged_fault ||
                          ((i == 0 || pool_index.maxhi[i - 1] <= r->import_lo) &&
                           (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= r->import_hi)));
    plat_mutex_unlock_shared(&pool_lock);
    if (i < 0) return ms_set(r, MS_REFUSED);   // finalized meanwhile
    if (!alone) return refuse(r, MR_OVERLAP, "it overlaps another pool");
    if (!apart) return refuse(r, MR_OVERLAP, "its staged mirror would share a page with another pool");
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
    if (staged && staged_inject() == INJECT_IMPORT) return refuse(r, MR_ALLOC, "TOTK_STAGED_INJECT=import");
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
    r->sfeed = (uint8_t)staged;
    if (staged && staged_fault) {   // staged_fetch finds the pool by address
        r->cpu_read = calloc((size_t)(n >> 12) + 1, 1);
        if (!r->cpu_read) rt_fatal("vk: out of memory");
        if (fetch_test) {
            r->t_rec = calloc((size_t)(n >> 12) + 1, sizeof *r->t_rec);
            r->t_ref = calloc((size_t)(n >> 12) + 1, sizeof *r->t_ref);
            r->t_page = calloc((size_t)(n >> 12) + 1, sizeof *r->t_page);
            if (!r->t_rec || !r->t_ref || !r->t_page) rt_fatal("vk: out of memory");
        }
        plat_mutex_lock(&own_lock);
        if (nst_pools == cst_pools) grow((void **)&st_pools, &cst_pools, sizeof *st_pools);
        int at = st_upper(r->import_lo);   // sorted by address: staged_fetch's lookup is a binary search
        memmove(st_pools + at + 1, st_pools + at, (size_t)(nst_pools - at) * sizeof *st_pools);
        st_pools[at] = r;
        nst_pools++;
        plat_mutex_unlock(&own_lock);
    }
    // A small staged mirror is first filled by the prologue of this part (staged_upload): the whole range queued here
    // stands for the zero fill and the written pages, so that the prologue runs (vk_mem_mirror_pending) and the range
    // goes the staged way (sfeed). Before MS_ON, like the other fields; under mirror_lock when shared, like pend.
    if (staged && n <= STG_SMALL) r->szero = 1, range_add(&pend, &npend, &cpend, r, 0, n);
    ms_set(r, MS_ON);   // after mbuf, acc_sub and sfeed: what a reader of MS_ON goes on to read
    BigMirror bm = {1, r->lo, n, 0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, qpc() - t0, q, wpages};
    if (staged && n > STG_SMALL) *fill = bm;   // filled by the caller (staged_fill), which logs it
    else if (!staged) {
        range_add(&pend, &npend, &cpend, r, 0, n);
        if (n >= BIG_MIRROR) big_mirror = bm;
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
        // make it under mirror_lock, and the first of them to get the lock does. The helper's part, dropped or not, is
        // still recorded: a staged pool's span gets the import there.
        if (vk_spec_helper()) {
            vk_spec_fail(VK_SPEC_MIRROR);
            if (!sp->buffer) staged_import(sp, SI_HELPER);
            return;
        }
        uint64_t ph = perf_ph_begin_cpu();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
        uint64_t t0 = qpc();
        vk_texture_hold_barrier();   // the walk, with a pass end held back: its slices resolve the pool as it was
        int locked = mirror_lock_if_shared();
        uint64_t t1 = qpc();
        BigMirror fill = {0};
        if (ms_get(r) == MS_NONE) {
            mirror_create(r, &fill);
            if (big_mirror.pending && big_mirror.lo == r->lo) big_mirror.hold = t1 - t0;
            fill.hold = t1 - t0;
        }
        mirror_unlock(locked);
        perf_ph_end(PERF_PH_MIRROR, ph);
        // Outside mirror_lock: other threads that reach the pool meanwhile bind the mirror at once, and the fill's
        // upload batches still precede every part that uses it (staged mirror, above).
        if (fill.pending) staged_fill(r, &fill);
        state = ms_get(r);
    }
    if (state != MS_ON) {
        if (!sp->buffer) staged_import(sp, r->mrefused ? r->mrefused - 1 : SI_GONE);   // a staged pool that stays without one
        return;
    }
    if (__atomic_load_n(&r->acc_sub, __ATOMIC_ACQUIRE) != msub) {
        uint64_t ph = perf_ph_begin();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
        mirror_sync(r);
        perf_ph_end(PERF_PH_MSYNC, ph);
    }
    sp->buffer = r->mbuf;
}

void vk_mem_pick(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (!r) return;
    // A staged pool resolved without a buffer: a copy, fill or update makes its mirror as a first buffer use does,
    // rather than import the pool (staged mirror, above).
    if (!sp->buffer) return vk_mem_use(sp);
    if (ms_get(r) != MS_ON) return;
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
    if (fetch_test) ft_stamp(r, sp->offset, size);   // TOTK_STAGED_FETCH_TEST
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
    if (fetch_test) ft_stamp(r, sp->offset, size);   // TOTK_STAGED_FETCH_TEST: a kept write changes the mirror too
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
    if (staged_min) {
        // The staged write-backs of this submission go into guest memory at its fence ahead of every callback the
        // recording queues for it, so the reports of the GPU writes (vk_exec_draw.c cb_done, vk_exec_xfer.c
        // note_gpu_write) find the bytes there. Recording runs under vk_sync.c's submission lock, where the value
        // being recorded is the newest issued one plus one.
        StJob *j = calloc(1, sizeof *j);
        if (!j) rt_fatal("vk: out of memory");
        j->fence = vk_sync_issued() + 1;
        plat_mutex_lock(&st_lock);
        j->next = st_jobs, st_jobs = j;
        plat_mutex_unlock(&st_lock);
        vk_completion_add(j->fence, staged_wb_done, j);
        st_cur = j;
    }
    uint32_t prev = msub;
    msub = msub + 1 ? msub + 1 : 1;
    if (st_cur) st_cur->sub = msub;
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

void vk_mem_mirror_prologue(VkCommandBuffer cb, uint64_t fence) {
    int locked = mirror_lock_if_shared();   // a helper may be queueing a sync
    int any = npend > 0;
    if (npend) {
        uint64_t t0 = qpc(), b0 = mrep.upload_bytes;
        int n = npend, k = n;
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        if (staged_min) {   // staged mirrors' ranges: staged_upload below, without mirror_lock
            k = 0;
            for (int i = 0; i < n; i++)
                if (pend[i].r->sfeed) range_add(&stq, &nstq, &cstq, pend[i].r, pend[i].off, pend[i].size);
                else pend[k++] = pend[i];
        }
        if (k) copy_ranges(cb, pend, k, 1, &mrep.upload_bytes, &mrep.upload_copies, 0);
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
    int fills = nstq ? staged_upload(cb, fence) : 0;   // the replay thread's list: no other thread adds to it
    if (any)
        mbarrier(cb, fills ? VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT : VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
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
    uint64_t pages = 0, skipped = 0, bad = 0, owned = 0;
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
            // A GPU-owned page (TOTK_STAGED_WB=fault): guest memory is stale there on purpose, and reading it would
            // fetch it. Verify cannot cover those pages; TOTK_STAGED_WB=fence, whose fence copies put the GPU's bytes
            // into guest memory, compares them.
            if (r->sfeed && staged_fault && mem_guard_state(a)) {
                owned++;
                continue;
            }
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
    plat_add64(&v_owned, (int64_t)owned);
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

// Whether a held part's write-back (vk_mem_mirror_wb_held) takes the range: tagged upto or lower, and not a staged
// mirror's (TOTK_MIRROR_STAGED, sfeed). A staged pool has no import for the write-back thread's copy to land in, so its
// ranges stay in the lists until vk_mem_mirror_end, which splits them off for the readback buffer of the submission's
// job (staged_wb_record) and, with TOTK_STAGED_WB=fault, for the pool's own list (staged_split_own), as when no held
// part wrote anything back. sfeed is set before MS_ON and never changes, and a range is only noted for a pool in
// MS_ON (vk_mem_gpu_wrote), so the read needs no lock.
static int wb_early_takes(const MRange *m, uint32_t upto) { return m->tag <= upto && !m->r->sfeed; }

// Moves the ranges wb_early_takes picks from a[0, *n) to the end of out (keeping the others in order); their bytes.
static uint64_t wb_take_tagged(MRange *a, int *n, uint32_t upto, MRange **out, int *nout, int *cout) {
    uint64_t bytes = 0;
    int k = 0;
    for (int i = 0; i < *n; i++) {
        if (!wb_early_takes(&a[i], upto)) { a[k++] = a[i]; continue; }
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
    for (int i = 0; i < nwbl; i++) bytes += wb_early_takes(&wbl[i], upto) ? wbl[i].size : 0;
    for (WbLocal *w = __atomic_load_n(&wb_locals, __ATOMIC_ACQUIRE); w && bytes < min_bytes; w = w->next) {
        plat_mutex_lock(&w->lock);
        for (int i = 0; i < w->n; i++) bytes += wb_early_takes(&w->a[i], upto) ? w->a[i].size : 0;
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
        int k = nwbl;
        if (staged_min) {   // staged mirrors write back through the job's readback buffer (staged_wb_record)
            k = 0;
            for (int i = 0; i < nwbl; i++)
                if (wbl[i].r->sfeed) range_add(&stw, &nstw, &cstw, wbl[i].r, wbl[i].off, wbl[i].size);
                else wbl[k++] = wbl[i];
        }
        if (k) rec += copy_ranges(cb, wbl, k, 0, &mrep.wb_bytes, &mrep.wb_copies, fence);
        if (nstw && staged_fault) staged_split_own();   // most stay in the mirror; pages the CPU read go back
        if (fetch_test && st_cur && st_cur->no) rec += staged_ref_record(cb);   // TOTK_STAGED_FETCH_TEST
        if (nstw) rec += staged_wb_record(cb);
        rec += 2;
        mbarrier(cb, stage, writes,
                 VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        nwbl = 0;
    }
    if (st_cur) {   // from here on the job's ranges keep the next submissions' uploads off those bytes
        plat_mutex_lock(&st_lock);
        st_cur->ended = 1;
        if (st_cur->n || st_cur->no) st_blk_dirty = 1;
        plat_mutex_unlock(&st_lock);
        st_cur = NULL;
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
    char ver[224] = "";
    if (mirror_verify)
        snprintf(ver, sizeof ver, "; verify: pages compared %lld, skipped %lld (written since), mismatches %lld, "
                 "GPU-owned not compared %lld", (long long)plat_xchg64(&v_pages, 0), (long long)plat_xchg64(&v_skipped, 0),
                 (long long)plat_xchg64(&v_mismatch, 0), (long long)plat_xchg64(&v_owned, 0));
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
    if (staged_min) {
        int64_t wn = plat_xchg64(&st_wbcpu_ns, 0), wb = plat_xchg64(&st_wbcpu_bytes, 0);
        rt_log("mem: staged frame %llu: per frame: upload %.3f MiB in %.1f ranges (%.3f MiB of write-backs in flight "
               "left out), CPU copy %.3f ms; write-back %.3f MiB in %.1f ranges, recorded in %.3f ms, %.3f MiB copied into "
               "guest memory in %.3f ms (completion thread); so far: first fills %d (%.0f MiB, %.1f ms in all), pools "
               "imported after all %d",
               (unsigned long long)frame, a.st_up_bytes / f / M, a.st_up_ranges / f, a.st_skip_bytes / f / M,
               qpc_ms(a.st_up_t) / f, a.st_wb_bytes / f / M, a.st_wb_ranges / f, qpc_ms(a.st_wb_t) / f, wb / f / M,
               qpc_ms((uint64_t)wn) / f, n_staged_fills, st_fill_bytes / M, qpc_ms((uint64_t)st_fill_ns),
               n_staged_imports);
        const volatile StCount *s = &stc[0];
        char imp[192];
        size_t o = 0;
        imp[0] = 0;
        for (int k = 0; k < SI_N && o < sizeof imp; k++)
            if (s->imp_n[k])
                o += (size_t)snprintf(imp + o, sizeof imp - o, ", %s %lld (%.1f MiB)", si_name[k], (long long)s->imp_n[k],
                                      s->imp_bytes[k] / M);
        rt_log("mem: staged pools frame %llu: so far: mirrors with the first fill in a prologue %lld (%.1f MiB, %.1f MiB of "
               "written pages copied in %.1f ms), by staged_fill %lld; imported after all %d%s; import failed and left "
               "unbacked %d (%.1f MiB); released %lld in %.1f ms "
               "(never imported, no GPU wait: %lld, %lld of them resolved only by CPU paths; with a mirror %lld; imported "
               "%lld), owned bytes dropped unfetched %.2f MiB",
               (unsigned long long)frame, (long long)s->small_n, s->small_bytes / M, s->small_copied / M,
               s->small_ns / 1e6, (long long)s->big_n, n_staged_imports, imp, n_staged_import_failures,
               s->imp_failed_bytes / M, (long long)s->rel_n, s->rel_ns / 1e6,
               (long long)s->rel_never, (long long)s->rel_cpu, (long long)s->rel_mirror, (long long)s->rel_imported,
               own_drop_bytes / M);
        mem_probe_log(frame, 0);
        if (staged_fault) own_log(frame, f), ft_log(frame);
    }

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

// TOTK_STAGED_INJECT=fence (staged_inject): the first upload batch that reads as on a lost device, and every later one;
// 0 none. Their fence status and fence wait give VK_ERROR_DEVICE_LOST, so retire stops in front of them as on a real
// loss.
static volatile uint64_t inject_lost;

static VkResult batch_status(VkCtx *c, const Batch *b) {
    uint64_t l = __atomic_load_n(&inject_lost, __ATOMIC_RELAXED);
    return l && b->seq >= l ? VK_ERROR_DEVICE_LOST : vkGetFenceStatus(c->device, b->fence);
}

// Batches complete in submission order (one queue): retire from the front while their fences are signalled.
static void retire(VkCtx *c, int wait_all) {
    int k = 0;
    for (; k < ninflight; k++) {
        Batch *b = &inflight[k];
        if (wait_all) vkWaitForFences(c->device, 1, &b->fence, VK_TRUE, UINT64_MAX);
        else if (batch_status(c, b) != VK_SUCCESS) break;
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

// cmd_lock held: the open batch goes to the queue.
static void submit_open_locked(VkCtx *c) {
    if (!open_active) return;
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

// cmd_lock held: waits until every upload batch up to seq has completed; they are all submitted. Without cmd_lock
// while it waits when this thread holds it once (a caller holding it around the flush keeps it for the wait). Each wait
// is on the oldest batch's fence. A lost device ends the process with the fault report: retire stops at the first
// fence that is not signalled, so done_seq would never reach seq and a loop on it would spin forever (a thread at 100%
// of a core and an idle GPU, with no fault reported). Eglantine (Eden) ends the same way: vulkan_wrapper.h
// Semaphore::Wait, under MasterSemaphore::Wait, throws vk::Exception on any result but VK_SUCCESS and VK_TIMEOUT.
static void upload_wait_locked(VkCtx *c, uint64_t seq) {
    uint64_t w0 = ninflight && done_seq < seq ? kern_now_ns() : 0;
    int unlocked = cmd_lock.depth == 1;
    if (unlocked && ninflight && done_seq < seq) {
        plat_rmutex_unlock(&cmd_lock);
        plat_mutex_lock(&wait_lock);
        plat_rmutex_lock(&cmd_lock);
    }
    while (ninflight && done_seq < seq) {
        VkFence f = inflight[0].fence;
        uint64_t fseq = inflight[0].seq;
        if (unlocked) pinned = fseq, pinned_done = 0, plat_rmutex_unlock(&cmd_lock);
        uint64_t lost = __atomic_load_n(&inject_lost, __ATOMIC_RELAXED);
        VkResult w = lost && fseq >= lost ? VK_ERROR_DEVICE_LOST : vkWaitForFences(c->device, 1, &f, VK_TRUE, UINT64_MAX);
        if (unlocked) plat_rmutex_lock(&cmd_lock);
        if (w != VK_SUCCESS && w != VK_TIMEOUT) {
            vk_device_fault_report(w);
            rt_fatal("vk: upload wait for batch %llu failed (%d)", (unsigned long long)fseq, (int)w);
        }
        retire(c, 0);
        // The wait said signalled and retire's vkGetFenceStatus did not: a lost device may answer a wait with
        // VK_SUCCESS and the status query with VK_ERROR_DEVICE_LOST, and the loop would spin on the same fence.
        if (w == VK_SUCCESS && ninflight && inflight[0].seq == fseq) {
            VkResult s = batch_status(c, &inflight[0]);
            if (s != VK_SUCCESS) {
                vk_device_fault_report(s);
                rt_fatal("vk: upload batch %llu: its fence wait returned, its status is %d", (unsigned long long)fseq, (int)s);
            }
        }
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

uint64_t vk_upload_flush(int wait) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    uint64_t seq = next_seq;
    if (c) submit_open_locked(c);
    if (c && wait) upload_wait_locked(c, seq);
    if (c) collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
    return seq;
}

// Waits until the upload batch seq (vk_upload_seq, vk_upload_flush) has completed, submitting it first if it is still
// open; the batches after it are left to run.
static void upload_wait(uint64_t seq) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    if (c && done_seq < seq) {
        if (open_active && open_b.seq <= seq) submit_open_locked(c);
        upload_wait_locked(c, seq);
        collect_locked(c);
    }
    plat_rmutex_unlock(&cmd_lock);
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
    if (staged_min && staged_fault) staged_own_shutdown(c);   // the GPU-owned bytes of staged mirrors, while they exist
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
    staged_shutdown(c);
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    nphys = 0;
    plat_mutex_unlock(&pool_lock);
    plat_mutex_lock(&gen_lock);
    for (uint64_t i = 0; i < GEN_CHUNKS; i++) free(gen_chunk[i]), gen_chunk[i] = NULL;
    plat_mutex_unlock(&gen_lock);
}
