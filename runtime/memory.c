// Adapted from MacNativeGames games/totk/runtime/services.c and loader.c @ b75be8c (MIT, Copyright (c) 2026
// MacNativeGames contributors), and LAS Recomp runtime/main.c and platform/windows.c @ da35b83 (MIT, Copyright (c)
// 2026 Peiikai).
// Guest address space and the memory SVCs: heap, module data permissions, stack moves,
// QueryMemory, physical memory in the alias region, GetInfo memory ids, and the kernel-region allocator.
//
// Every region is a plain reservation; pages are committed and decommitted at 4 KiB granularity. Decommitted pages
// come back zeroed, so no stale-page bookkeeping is needed. The stack and heap block and the alias region are
// reserved write-watched (platform.h plat_mem_reserve): mem_written_pages reports the pages the CPU wrote, which
// the Vulkan back end uploads again.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "layout.h"
#include "platform/platform.h"

#define PAGE 0x1000ull
#define GiB (1ull << 30)
#define LOW_SEARCH_FROM 0x100000000ull
// Switch 2 sizes (Donkey Kong Bananza): the game sizes its heap from TotalMemorySize (GetInfo 6), so the pool and the
// heap block are the Switch 2's, not the Switch 1's 0xCD500000 and 4 GiB. The heap is committed as the game asks
// (SetHeapSize), and Windows commit charge only grows with it.
#ifdef __APPLE__
// macOS maps nothing between 63 GiB and 448 GiB (a kernel guard region), so the stack and heap block, the alias
// region and the module window all share the 64 GiB from 448 GiB to 2^39, in whatever holes libmalloc has left. The
// block still holds the 768 MiB of stack and heap offset plus the 12 GiB heap cap, so it is 13 GiB here; the alias
// region is ALIAS_SIZE at most, since the 64 GiB of the other hosts fit in no hole of this window.
#define LOW_SIZE (13 * GiB)
// The alias region is where the game maps its pools, so it has to hold a pool-sized alias: ALIAS_MIN leaves room for
// the offsets those aliases are mapped at, ALIAS_SIZE is what a window with room gives, and mem_init steps down to
// the largest that fits, logging when it does.
#define ALIAS_SIZE (16 * GiB)
#define ALIAS_MIN (10 * GiB)
#else
#define LOW_SIZE (16 * GiB)
#define ALIAS_SIZE (64 * GiB)
#endif
#define STACK_OFF 0x10000000ull
#define STACK_SIZE 0x10000000ull
#define HEAP_OFF (STACK_OFF + STACK_SIZE + 0x10000000ull)
#define HEAP_CAP (12 * GiB)
#ifdef __APPLE__
// The window above the modules is fragmented by libmalloc's own reservations, and a 4 GiB alignment would give back
// up to 4 GiB of every hole: the alias region needs only the 2 MiB of the module bases.
#define ALIAS_ALIGN (2u << 20)
#else
#define ALIAS_ALIGN (4 * GiB)
#endif
#define ALIAS_PAGES (ALIAS_SIZE / PAGE)
static uint64_t alias_page_count = ALIAS_PAGES;   // pages of the alias region actually reserved (macOS may take fewer)
// Application pool: Switch 2 games are said to get about 9 GiB of the 12 (not measured on hardware; TOTK_APP_POOL
// overrides). TotK's retail Switch 1 pool was 0xCD500000.
#define DEFAULT_POOL (9 * GiB)
#define SYSTEM_RESOURCE_BUDGET (32ull << 20)
#define MAX_STACK_MAPS 4096
#define REEXEC_LIMIT 20

// Horizon memory states, permissions, attributes and results used here.
enum { ST_FREE = 0, ST_CODE = 3, ST_CODEDATA = 4, ST_NORMAL = 5, ST_STACK = 0xB, ST_THREADLOCAL = 0xC,
       ST_INACCESSIBLE = 0x10 };
enum { PERM_NONE = 0, PERM_R = 1, PERM_RW = 3, PERM_RX = 5 };
enum { ATTR_PERMISSION_LOCKED = 16 };
#define R_OK 0u
#define R_INVALID_ADDRESS 0xcc01u   // misaligned address
#define R_INVALID_SIZE 0xca01u
#define R_INVALID_NEW_PERM 0xd801u
#define R_INVALID_STATE 0xd401u     // InvalidCurrentMemory
#define R_INVALID_REGION 0xdc01u
#define R_LIMIT_REACHED 0x10801u

static RtGuestLayout L;
static uint64_t mods_lo, mods_hi;   // fixed module window
static PlatMutex lock = PLAT_MUTEX_INIT;
static uint64_t heap_size, alias_bytes;
static uint8_t *alias_pages;        // 1 per mapped 4 KiB page of the alias region
typedef struct { uint64_t dst, src, size; } StackMap;
static StackMap stack_maps[MAX_STACK_MAPS];
static unsigned nstack_maps;
static uint8_t *module_perm[MOD_COUNT];   // per data page of each module: perm | ATTR_PERMISSION_LOCKED

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

static int reserve_at(uint64_t at, uint64_t size, int watch) { return plat_mem_reserve(at, size, watch); }

// ---- frozen regions
// A 2 MiB region of the watched reservations that nothing wrote for a while can be made read-only (mem_freeze), so the
// back end's write-watch queries skip it: no page of a frozen region changes before a thaw. The first write faults and
// thaw_handler makes the region writable again before the write retries. A kernel write into guest memory (a file
// read) fails on a read-only page instead of faulting, so file reads thaw their buffer first (mem_thaw), and so does
// every commit, decommit and protection change here. A region stays thawed for FRZ_HOLD_MS after a mem_thaw: a file
// read still in flight writes its buffer later. frz_lock is a leaf lock: nothing writes guest memory under it.
// frz[r] is FRZ_WHOLE for a region that was committed throughout at its freeze, FRZ_HOLES for one with reserved holes.
#define FRZ_SHIFT 21
#define FRZ_SIZE (1ull << FRZ_SHIFT)
#define FRZ_HOLD_MS 5000u
enum { FRZ_HOLES = 1, FRZ_WHOLE = 2 };
static volatile uint8_t frz[RT_ADDRESS_LIMIT >> FRZ_SHIFT];
static uint32_t thaw_ms[RT_ADDRESS_LIMIT >> FRZ_SHIFT];   // frz_lock: GetTickCount of the last thaw request, 0 never
static volatile int32_t nfrz;
static PlatMutex frz_lock = PLAT_MUTEX_INIT;
static volatile int64_t frz_count[4];   // freezes; thaws by a write fault, by mem_thaw, by a mapping change
// TOTK_FRZ_WHOLE=0 (an A/B of the thaw, TOTK_TRACE=perf shows the difference): every region is thawed the way one with
// holes is, a query per read-only run, from the bottom up. Read at the first freeze (frz_lock), before which nothing
// is thawed.
static int frz_whole = -1;
// Regions that hold sealed pages (vk_memory.c, the lazy write-back) are never frozen: a freeze would make a sealed page
// readable. Set under frz_lock by mem_freeze_exclude and never cleared. thaw_hook is the back end's: it unseals the
// pages of a range before a kernel write into it (a file read), which fails on a sealed page as on a read-only one.
static volatile uint8_t frz_never[RT_ADDRESS_LIMIT >> FRZ_SHIFT];
static void (*thaw_hook)(uint64_t a, uint64_t n);

static int heap_lazy(void);

// frz_lock, with the time a thread waits for it recorded as an operation in flight (TOTK_TRACE=perf): a thaw of a
// whole pool holds the lock while it runs, and every write fault into a frozen region and every file read waits for it.
static void frz_lock_take(void) {
    if (plat_mutex_trylock(&frz_lock)) return;
    int op = plat_op_begin(PLAT_OP_FRZ_WAIT, 0, 0);
    plat_mutex_lock(&frz_lock);
    plat_op_end(op, 0);
}

static int watched(uint64_t addr, uint64_t size) {
    const RtGuestLayout *g = &L;
    return (addr >= g->aslr.base && size <= LOW_SIZE && addr - g->aslr.base <= LOW_SIZE - size) ||
           (addr >= g->alias.base && size <= g->alias.size && addr - g->alias.base <= g->alias.size - size);
}

// frz_lock held exclusively. why: 1 a write fault, 2 mem_thaw (holds the region thawed), 3 a mapping change here.
// writable: the caller already made the region read-write (thaw_range_locked, a run of FRZ_WHOLE regions).
static void thaw_locked(uint64_t r, int why, int writable) {
    if (why == 2) thaw_ms[r] = (uint32_t)plat_now_ms() | 1;
    int f = frz[r];
    if (!f) return;
    uint64_t a = r << FRZ_SHIFT, end = a + FRZ_SIZE, b;
    // A region committed throughout at its freeze is still committed throughout and read-only: nothing decommits a page
    // or changes its protection without thawing first (commit, decommit and protect below), the lazy heap's fault handler
    // and mem_touch commit reserved pages only, and the one race left (win32.c lazy_commit) can only make a page
    // read-write again. So one protection change thaws it, with no query. A query would report the whole run of
    // read-only pages after the address, all the frozen regions above it, and on Windows its cost grows with that run:
    // with a query per region, a layer exit's thaw of the 450 MiB terrain pool (vk_pool_fini, 226 regions) took 50 to
    // 100 ms under frz_lock, and the replay thread's next write-watch query waited for all of it.
    if (writable || (f == FRZ_WHOLE && plat_mem_protect(a, FRZ_SIZE, PLAT_PROT_RW))) a = end;
    // Otherwise the read-only runs only: freeze_locked left reserved holes (heap pages given back) as they were. Every
    // committed page of a frozen region was read-write at the freeze, and mapping changes thaw before they run, so these
    // are all ours.
    for (; a < end; a = b)
        if (plat_mem_state(a, end, &b) == PLAT_PROT_READ && !plat_mem_protect(a, b - a, PLAT_PROT_RW))
            rt_fatal("memory: thawing %#llx+%#llx failed", (unsigned long long)a, (unsigned long long)(b - a));
    __atomic_store_n(&frz[r], 0, __ATOMIC_RELEASE);
    plat_dec32(&nfrz);
    plat_inc64(&frz_count[why]);
}

// frz_lock held exclusively. From the top down: a region with holes still queries its runs (thaw_locked), and once
// the regions above it are writable, the read-only run its query reports ends inside it. Adjacent FRZ_WHOLE regions
// (up to FRZ_RUN of them) are made writable with one protection change: with one per 2 MiB region (30 to 40 us
// each), the terrain pool's 220 still made the replay's next write-watch query wait 5 to 9 ms, while with four calls of
// 1 to 2 ms it mostly got in between them. A run that the call refuses (one that crosses into another reservation) is
// thawed region by region.
#define FRZ_RUN 64
static void thaw_range_locked(uint64_t a, uint64_t n, int why) {
    if (!n || a >= RT_ADDRESS_LIMIT) return;
    uint64_t r0 = a >> FRZ_SHIFT, r1 = (a + n - 1 < RT_ADDRESS_LIMIT ? a + n - 1 : RT_ADDRESS_LIMIT - 1) >> FRZ_SHIFT;
    int op = -1;
    if (!frz_whole) {
        for (uint64_t r = r0; r <= r1; r++) {
            if (op < 0 && frz[r]) op = plat_op_begin(PLAT_OP_THAW, a, n);
            thaw_locked(r, why, 0);
        }
    } else {
        for (uint64_t r = r1 + 1; r-- > r0;) {
            if (op < 0 && frz[r]) op = plat_op_begin(PLAT_OP_THAW, a, n);
            uint64_t lo = r;
            while (frz[r] == FRZ_WHOLE && lo > r0 && frz[lo - 1] == FRZ_WHOLE && r - lo + 1 < FRZ_RUN) lo--;
            if (lo < r && plat_mem_protect(lo << FRZ_SHIFT, (r - lo + 1) << FRZ_SHIFT, PLAT_PROT_RW)) {
                for (uint64_t q = lo; q <= r; q++) thaw_locked(q, why, 1);
                r = lo;
                continue;
            }
            thaw_locked(r, why, 0);
        }
    }
    plat_op_end(op, 0);
}

static int never_frozen(uint64_t a, uint64_t n) {
    if (!n || a >= RT_ADDRESS_LIMIT) return 0;
    uint64_t r1 = (a + n - 1 < RT_ADDRESS_LIMIT ? a + n - 1 : RT_ADDRESS_LIMIT - 1) >> FRZ_SHIFT;
    for (uint64_t r = a >> FRZ_SHIFT; r <= r1; r++)
        if (__atomic_load_n(&frz_never[r], __ATOMIC_ACQUIRE)) return 1;
    return 0;
}

void mem_freeze_exclude(uint64_t a, uint64_t n) {
    if (!n || a >= RT_ADDRESS_LIMIT) return;
    uint64_t last = a + n - 1 < RT_ADDRESS_LIMIT ? a + n - 1 : RT_ADDRESS_LIMIT - 1;
    uint64_t r0 = a >> FRZ_SHIFT, r1 = last >> FRZ_SHIFT, r = r0;
    while (r <= r1 && __atomic_load_n(&frz_never[r], __ATOMIC_ACQUIRE)) r++;
    if (r > r1) return;   // the usual case: excluded since an earlier frame
    frz_lock_take();
    for (r = r0; r <= r1; r++) {
        thaw_locked(r, 3, 0);
        __atomic_store_n(&frz_never[r], 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&frz_lock);
}

void mem_thaw_hook(void (*fn)(uint64_t a, uint64_t n)) { thaw_hook = fn; }

void mem_thaw(uint64_t a, uint64_t n) {
    if (thaw_hook && never_frozen(a, n)) thaw_hook(a, n);   // before frz_lock: the hook takes its own lock only
    frz_lock_take();   // also when nothing is frozen: the hold keeps a read's buffer writable
    thaw_range_locked(a, n, 2);
    plat_mutex_unlock(&frz_lock);
}

// frz_lock held exclusively. Makes the committed pages of region r read-only when every one of them is read-write;
// reserved holes (heap pages given back) stay as they are, and thaw_locked finds the read-only runs again. Returns
// FRZ_WHOLE when the region is one committed run, FRZ_HOLES when it has holes, 0 when it was refused.
static int freeze_locked(uint64_t r) {
    uint64_t a = r << FRZ_SHIFT, end = a + FRZ_SIZE, run[16][2];
    int n = 0;
    for (uint64_t x = a, y; x < end; x = y) {
        int st = plat_mem_state(x, end, &y);
        if (st >= PLAT_PROT_NONE) {
            if (st != PLAT_PROT_RW || n == 16) return 0;
            run[n][0] = x, run[n][1] = y, n++;
        } else if (st != PLAT_MEM_RESERVED || (heap_lazy() && x >= L.heap.base && x < L.heap.base + heap_size)) return 0;
        // (a lazy-heap hole may be committed by fault_veh at any moment, so such a region is never frozen)
    }
    for (int i = 0; i < n; i++)
        if (!plat_mem_protect(run[i][0], run[i][1] - run[i][0], PLAT_PROT_READ)) {
            while (i--) plat_mem_protect(run[i][0], run[i][1] - run[i][0], PLAT_PROT_RW);
            return 0;
        }
    return !n ? 0 : n == 1 && run[0][0] == a && run[0][1] == end && frz_whole ? FRZ_WHOLE : FRZ_HOLES;
}

int mem_freeze(uint64_t a) {
    if ((a & (FRZ_SIZE - 1)) || !watched(a, FRZ_SIZE)) return 0;
    uint64_t r = a >> FRZ_SHIFT;
    int ok = 0, how;
    int op = plat_op_begin(PLAT_OP_FREEZE, a, FRZ_SIZE);
    frz_lock_take();
    if (frz_whole < 0) {
        const char *e = cfg_get("TOTK_FRZ_WHOLE");
        frz_whole = !(e && !strcmp(e, "0"));
        if (!frz_whole) rt_log("mem: frozen regions are thawed with a query per read-only run (TOTK_FRZ_WHOLE=0)");
    }
    if (frz[r]) ok = 1;
    else if (frz_never[r]) ok = 0;
    else if (thaw_ms[r] && (int32_t)((uint32_t)plat_now_ms() - thaw_ms[r]) < (int32_t)FRZ_HOLD_MS) ok = 0;   // |1 may be ahead
    else if ((how = freeze_locked(r))) {
        __atomic_store_n(&frz[r], (uint8_t)how, __ATOMIC_RELEASE);
        plat_inc32(&nfrz);
        plat_inc64(&frz_count[0]);
        ok = 1;
    }
    plat_mutex_unlock(&frz_lock);
    plat_op_end(op, 0);
    return ok;
}

int mem_frozen(uint64_t a) { return a < RT_ADDRESS_LIMIT && __atomic_load_n(&frz[a >> FRZ_SHIFT], __ATOMIC_ACQUIRE); }

void mem_freeze_stats(uint64_t *frozen, uint64_t *freezes, uint64_t *fault_thaws, uint64_t *other_thaws) {
    *frozen = (uint64_t)__atomic_load_n(&nfrz, __ATOMIC_RELAXED);
    *freezes = (uint64_t)plat_xchg64(&frz_count[0], 0);
    *fault_thaws = (uint64_t)plat_xchg64(&frz_count[1], 0);
    *other_thaws = (uint64_t)plat_xchg64(&frz_count[2], 0) + (uint64_t)plat_xchg64(&frz_count[3], 0);
}

// The first write into a frozen region thaws it and retries. A fault in a region another thread thawed meanwhile
// retries too; any other fault goes on to the crash handler.
// The platform's write-fault hook (plat_mem_fault_hook): 1 retries the write.
static int thaw_fault(uint64_t a) {
    if (a >= RT_ADDRESS_LIMIT || !watched(a, 1)) return 0;
    int op = plat_op_begin(PLAT_OP_FAULT_THAW, a & ~(FRZ_SIZE - 1), FRZ_SIZE);
    frz_lock_take();
    int was = frz[a >> FRZ_SHIFT];
    thaw_locked(a >> FRZ_SHIFT, 1, 0);
    plat_mutex_unlock(&frz_lock);
    plat_op_end(op, 0);
    if (was) return 1;
    uint64_t end;
    return plat_mem_state(a & ~(PAGE - 1), (a & ~(PAGE - 1)) + PAGE, &end) == PLAT_PROT_RW;
}

// Mapping changes thaw the regions they touch and run under frz_lock, so no freeze sees them half done.
static void commit(uint64_t a, uint64_t n, const char *what) {
    frz_lock_take();
    thaw_range_locked(a, n, 3);
    int ok = !n || plat_mem_commit(a, n);
    plat_mutex_unlock(&frz_lock);
    if (!ok) rt_fatal("memory: commit %s %#llx+%#llx failed", what, (unsigned long long)a, (unsigned long long)n);
}

static void decommit(uint64_t a, uint64_t n, const char *what) {
    if (thaw_hook && never_frozen(a, n)) thaw_hook(a, n);   // sealed pages become ordinary ones before they go
    frz_lock_take();
    thaw_range_locked(a, n, 3);
    int ok = !n || plat_mem_decommit(a, n);
    plat_mutex_unlock(&frz_lock);
    if (!ok) rt_fatal("memory: decommit %s %#llx+%#llx failed", what, (unsigned long long)a, (unsigned long long)n);
}

static void protect(uint64_t a, uint64_t n, int prot, const char *what) {
    frz_lock_take();
    thaw_range_locked(a, n, 3);
    int ok = !n || plat_mem_protect(a, n, prot);
    plat_mutex_unlock(&frz_lock);
    if (!ok) rt_fatal("memory: protect %s %#llx+%#llx failed", what, (unsigned long long)a, (unsigned long long)n);
}

// The process is re-run when a region is taken: Windows places DLLs, heaps and thread stacks at random addresses,
// and a fresh process gets a fresh layout.
static void reexec_or_fail(const char *what) {
    const char *e = cfg_get("TOTK_REEXEC");
    int n = e ? atoi(e) : 0;
    if (n >= REEXEC_LIMIT) rt_fatal("memory: %s is not free after %d attempts", what, n);
    char buf[16];
    snprintf(buf, sizeof buf, "%d", n + 1);
    plat_setenv("TOTK_REEXEC", buf);
    plat_reexec();
}

void mem_init(void) {
    if (L.pool) return;
    const char *ran = cfg_get("TOTK_REEXEC");
    if (ran && *ran && atoi(ran)) rt_log("mem: re-ran the process %d time%s for a free address space", atoi(ran),
                                          atoi(ran) == 1 ? "" : "s");
    // Fixed window: modules, kernel region, host-function addresses. The window's ends come from rt_layout, never from
    // layout.h's MOD_<NAME>_BASE/_SIZE: modules.c is generated for the machine the game is installed on, while this
    // library is compiled ahead of it (tools/ci/stub does it without a dump), so a compile-time base would reserve the
    // space of a layout the process does not have. rt_layout is in layout order, base ascending (tools/build_game.py
    // module_glue), so its first entry is the window's bottom and its last the window's top.
    mods_lo = rt_layout[0].base;
    mods_hi = align_up(rt_layout[rt_layout_count - 1].base + rt_layout[rt_layout_count - 1].size, 0x10000);
    if (mods_hi > RT_KERNEL_BASE) rt_fatal("memory: modules end at %#llx, inside the kernel region",
                                           (unsigned long long)mods_hi);
    if (!reserve_at(mods_lo, mods_hi - mods_lo, 0)) reexec_or_fail("the module window");
    if (!reserve_at(RT_KERNEL_BASE, RT_KERNEL_SIZE, 0)) reexec_or_fail("the kernel region");
    if (!reserve_at(RT_HOSTFN_BASE, RT_HOSTFN_SIZE, 0)) reexec_or_fail("the host-function window");

    // Stack and heap block, then the alias region above it, both searched below the module window.
    uint64_t low = 0, alias = 0, asize = ALIAS_SIZE;
#ifdef __APPLE__
    // Nothing below the module window is mappable on macOS, so both regions go into the holes above the host-function
    // window: plat_mem_reserve_in walks the region list and skips whatever the system and libmalloc already hold
    // there, and a fresh process (reexec_or_fail) gets a fresh layout when this one's holes are too small. The block
    // goes first: it is the fixed size the heap cap needs, and the alias region fits above it more often than below.
    if (!plat_mem_reserve_in(RT_HOSTFN_BASE + RT_HOSTFN_SIZE, RT_ADDRESS_LIMIT, LOW_SIZE, GiB, 1, &low))
        reexec_or_fail("a 13 GiB stack and heap block above the modules");
    asize = 0;
    for (uint64_t s = ALIAS_SIZE; s >= ALIAS_MIN && !asize; s -= 2 * GiB)
        if (plat_mem_reserve_in(RT_HOSTFN_BASE + RT_HOSTFN_SIZE, RT_ADDRESS_LIMIT, s, ALIAS_ALIGN, 1, &alias)) asize = s;
    if (!asize) reexec_or_fail("a 10 GiB alias region above the modules");
    if (asize != ALIAS_SIZE)
        rt_log("mem: alias region %llu MiB: the holes above the modules hold no %llu MiB one",
               (unsigned long long)(asize >> 20), (unsigned long long)(ALIAS_SIZE >> 20));
    alias_page_count = asize / PAGE;
#else
    for (uint64_t at = LOW_SEARCH_FROM; at + LOW_SIZE <= mods_lo && !low; at += GiB)
        if (reserve_at(at, LOW_SIZE, 1)) low = at;
    if (!low) reexec_or_fail("a 6 GiB stack and heap block below 2^39");
    for (uint64_t at = align_up(low + LOW_SIZE, ALIAS_ALIGN); at + ALIAS_SIZE <= mods_lo && !alias; at += ALIAS_ALIGN)
        if (reserve_at(at, ALIAS_SIZE, 1)) alias = at;
    if (!alias) reexec_or_fail("a 64 GiB alias region below 2^39");
#endif

    plat_mem_fault_hook(thaw_fault);   // before the crash handler
    alias_pages = calloc(alias_page_count, 1);
    if (!alias_pages) rt_fatal("memory: cannot allocate the alias page map");
    // totk.ini is read here, once the address space is claimed and before the first setting: the pool size below
    // depends on TOTK_APP_POOL and on the resolution settings (resolution.c decides them once, on this first call).
    cfg_init();
    const char *pool = cfg_get("TOTK_APP_POOL");
    L = (RtGuestLayout){
        .aslr = {low, RT_ADDRESS_LIMIT - low},
        .stack = {low + STACK_OFF, STACK_SIZE},
        .heap = {low + HEAP_OFF, HEAP_CAP},
        .alias = {alias, asize},
        .kernel = {RT_KERNEL_BASE, RT_KERNEL_SIZE},
        .hostfn = {RT_HOSTFN_BASE, RT_HOSTFN_SIZE},
        .pool = pool ? strtoull(pool, NULL, 0) : DEFAULT_POOL + res_heap_extra_bytes(),   // resolution.c
    };
    if (!L.pool) rt_fatal("memory: TOTK_APP_POOL is 0");
}

const RtGuestLayout *mem_layout(void) { return &L; }

// ---- commit report (TOTK_MEM_REPORT=N, and once at exit from perf_summary_flush in perf.c)
// Says where the process's commit goes: one plat_mem_walk, each committed run added to the bucket of the address range
// it starts in. The bounds are the ones mem_init set (L, mods_lo, mods_hi never change afterwards), so no lock is
// taken; the walk itself takes none either, so a game thread holding frz_lock or lock cannot block it.
enum { B_LOW, B_ALIAS, B_MODS, B_KERNEL, B_PRIVATE, B_MAPPED, B_IMAGE, B_N };
static void report_run(uint64_t base, uint64_t size, int kind, void *arg) {
    uint64_t *sum = arg;
    int b = kind == 2 ? B_IMAGE : kind == 1 ? B_MAPPED
          : base >= L.aslr.base && base - L.aslr.base < LOW_SIZE ? B_LOW
          : base >= L.alias.base && base - L.alias.base < L.alias.size ? B_ALIAS
          : base >= mods_lo && base < mods_hi ? B_MODS
          : base >= L.kernel.base && base - L.kernel.base < L.kernel.size ? B_KERNEL : B_PRIVATE;
    sum[b] += size;
}

// The GPU side (vk_memory.c) overrides this; builds without the Vulkan back end add nothing.
__attribute__((weak)) void vk_mem_gpu_summary(char *out, size_t n) { if (n) *out = 0; }

void mem_commit_report(const char *when) {
    if (!L.pool) return;
    uint64_t s[B_N] = {0};
    plat_mem_walk(report_run, s);
    // How much of the committed heap+stack the game has touched: the same range the B_LOW bucket counts.
    uint64_t resident = plat_mem_resident(L.aslr.base, LOW_SIZE);
    double proc, used, limit, M = 1048576.0;
    plat_commit_info(&proc, &used, &limit);
    char gpu[256];
    vk_mem_gpu_summary(gpu, sizeof gpu);
#ifdef _WIN32
    if (heap_lazy()) rt_log("mem: lazy heap: %llu commit faults so far", (unsigned long long)plat_mem_lazy_faults());
#endif
    rt_log("mem: commit %s: guest heap+stack %.0f MiB, alias %.0f, modules %.0f, kernel %.0f, other private %.0f, mapped "
           "%.0f, image %.0f; process commit %.0f MiB (limit %.0f), guest heap+stack resident %.0f MiB%s", when, s[B_LOW] / M,
           s[B_ALIAS] / M, s[B_MODS] / M, s[B_KERNEL] / M, s[B_PRIVATE] / M, s[B_MAPPED] / M, s[B_IMAGE] / M, proc, limit,
           resident / M, gpu);
}

// A range inside one watched reservation (stack and heap block, or alias region) only.
static uint64_t written_pages(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset) {
    if (!max || !size) return 0;
    if (!watched(addr, size)) return UINT64_MAX;
    uint64_t ph = perf_ph_begin();   // TOTK_TRACE=perf: a part of the replay (perf.c ph_frame)
    uint64_t n = plat_mem_written(addr, size, pages, max, reset);
    perf_ph_end(PERF_PH_WW, ph);
    return n;
}

uint64_t mem_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max) {
    return written_pages(addr, size, pages, max, 1);
}

uint64_t mem_written_peek(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max) {
    return written_pages(addr, size, pages, max, 0);
}

static uint64_t image_bytes(void) {
    uint64_t n = 0;
    for (int i = 0; i < ldr_count(); i++) n += ldr_module(i)->size;
    return align_up(n, PAGE);
}

static int overlaps(uint64_t a, uint64_t n, uint64_t b, uint64_t m) { return a < b + m && b < a + n; }
static int inside(uint64_t a, uint64_t n, RtRange r) { return a >= r.base && n <= r.size && a - r.base <= r.size - n; }

// ---- 0x01 SetHeapSize
// TOTK_HEAP_LAZY=1 (Windows only, ignored elsewhere): the heap stays reserved and its pages are committed when touched,
// so the commit charge follows what the game uses instead of what it asked for.
static int heap_lazy(void) {
#ifdef _WIN32
    static int on = -1;
    if (on < 0) on = cfg_bool("TOTK_HEAP_LAZY");
    return on;
#else
    return 0;
#endif
}

// Commits the reserved pages of [s, e) (64 KiB aligned, inside the heap) after asking the system for their state.
static void touch_query(uint64_t s, uint64_t e) {
    int op = plat_op_begin(PLAT_OP_TOUCH, s, e - s);
    frz_lock_take();   // freeze_locked refuses regions with holes, so a commit here never lands in a frozen one
    for (uint64_t x = s, y; x < e; x = y) {
        int st = plat_mem_state(x, e, &y);
        if (st == PLAT_MEM_RESERVED && !plat_mem_commit(x, y - x)) {
            plat_mutex_unlock(&frz_lock);
            rt_fatal("memory: lazy heap commit %#llx+%#llx failed", (unsigned long long)x, (unsigned long long)(y - x));
        }
    }
    plat_mutex_unlock(&frz_lock);
    plat_op_end(op, 0);
}

// TOTK_TOUCH_MAP: 1 (default) lets mem_touch skip the chunks the platform's commit map knows to be committed
// (plat_mem_lazy_run), 0 queries every chunk as before, verify queries the known chunks as well and logs each one found
// reserved (then commits it). Windows only, like the lazy heap.
#ifdef _WIN32
static int touch_map(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_get("TOTK_TOUCH_MAP");
        int m = !e || !*e ? 1 : !strcmp(e, "verify") ? 2 : strcmp(e, "0") != 0;
        if (m != 1 && heap_lazy())
            rt_log("mem: mem_touch %s (TOTK_TOUCH_MAP=%s)", m ? "checks the commit map against VirtualQuery" :
                   "queries every chunk", e);
        mode = m;
    }
    return mode;
}
#endif

// mem_touch since the start: calls, chunks the commit map answered, chunks queried, and known chunks verify found
// reserved (mem_touch_stats).
static volatile int64_t touch_n[4];

void mem_touch_stats(uint64_t out[4]) {
    for (int i = 0; i < 4; i++) out[i] = (uint64_t)__atomic_load_n(&touch_n[i], __ATOMIC_RELAXED);
}

// Commits the reserved pages of [a, a+n) that lie in the guest heap, in 64 KiB chunks, before something other than the
// CPU (a kernel file read, a driver importing host memory) accesses them: those accesses fail on a reserved page
// instead of faulting into fault_veh. Committed pages are left alone. No-op without TOTK_HEAP_LAZY.
// Chunks the platform's commit map knows (every chunk the fault handler or an earlier mem_touch committed) are skipped
// without a system call; only the others are queried with plat_mem_state, whose VirtualQuery reports the whole run of
// like pages after the address and so costs more the longer the committed stretch around the pool (win32.c lazy_map).
// On the first frame of the Hilltop Layer of Surf Hills (3.2.0, RTX 5090, one run each) the touch of its ~280 first-use
// pool imports fell from 15.3 ms to 0.5 ms, but their vkAllocateMemory took about 9 ms longer, as if the driver's page
// pinning waited for what the queries had waited for: the imports took 36.4 ms instead of 40.0 and 40.3.
void mem_touch(uint64_t a, uint64_t n) {
    if (!n || !heap_lazy()) return;
    uint64_t lo = L.heap.base, hi = lo + __atomic_load_n(&heap_size, __ATOMIC_RELAXED);
    uint64_t s = a & ~0xffffull, e = (a + n + 0xffff) & ~0xffffull;
    if (s < lo) s = lo;
    if (e > hi) e = hi;
    if (s >= e) return;
    plat_inc64(&touch_n[0]);
#ifdef _WIN32
    int map = touch_map();
#else
    int map = 0;
#endif
    for (uint64_t x = s, y; x < e; x = y) {
        int known = 0;
#ifdef _WIN32
        y = map ? plat_mem_lazy_run(x, e, &known) : e;
#else
        y = e;
#endif
        int64_t chunks = (int64_t)((y - x) >> 16);
        if (known) {
            plat_add64(&touch_n[1], chunks);
            if (map != 2) continue;
            // verify: the map said committed; a reserved page here would have failed the caller's kernel or driver access
            for (uint64_t u = x, v; u < y; u = v)
                if (plat_mem_state(u, y, &v) == PLAT_MEM_RESERVED) {
                    if (plat_inc64(&touch_n[3]) <= 16)
                        rt_log("mem: mem_touch: the commit map holds %#llx+%#llx, which is reserved", (unsigned long long)u,
                               (unsigned long long)(v - u));
                    touch_query(u, v);
                }
            continue;
        }
        plat_add64(&touch_n[2], chunks);
        touch_query(x, y);
#ifdef _WIN32
        if (map) plat_mem_lazy_mark(x, y);
#endif
    }
}

uint32_t mem_set_heap_size(uint64_t size, uint64_t *addr) {
    int op = plat_op_begin(PLAT_OP_HEAP_SIZE, L.heap.base, size);
    plat_mutex_lock(&lock);
    if (size > HEAP_CAP || size % 0x200000 || size + image_bytes() > L.pool) {
        plat_mutex_unlock(&lock);
        plat_op_end(op, 0);
        return R_INVALID_STATE;
    }
    // TOTK_TRACE=perf: each change after the first, the first 64 (a shrink decommits, and thaws the frozen regions it
    // covers)
    static int changes;
    if ((rt_trace & RT_TRACE_PERF) && heap_size && size != heap_size && ++changes <= 64)
        rt_log("mem: SetHeapSize %#llx -> %#llx (%s %llu MiB)", (unsigned long long)heap_size, (unsigned long long)size,
               size < heap_size ? "shrinks by" : "grows by",
               (unsigned long long)((size < heap_size ? heap_size - size : size - heap_size) >> 20));
    // TOTK_HEAP_LAZY: growing leaves the new range reserved; platform/win32.c fault_veh commits it on first touch and
    // mem_touch before a kernel or driver access. Shrinking decommits (a no-op on pages never committed).
    if (size > heap_size && !heap_lazy()) commit(L.heap.base + heap_size, size - heap_size, "heap");
#ifdef _WIN32
    // On a shrink the handler's range drops first, so no fault recommits what decommit gives back.
    if (heap_lazy() && size < heap_size) plat_mem_lazy(L.heap.base, size);
#endif
    if (size < heap_size) decommit(L.heap.base + size, heap_size - size, "heap");
    heap_size = size;
#ifdef _WIN32
    if (heap_lazy()) plat_mem_lazy(L.heap.base, heap_size);
#endif
    plat_mutex_unlock(&lock);
    plat_op_end(op, 0);
    if (addr) *addr = L.heap.base;
    return R_OK;
}

// ---- module data permissions (0x02, 0x03) and module pages for QueryMemory
static int module_index(uint64_t a) {
    for (int i = 0; i < ldr_count(); i++) {
        const RtLoaded *m = ldr_module(i);
        if (a - m->base < m->size) return i;
    }
    return -1;
}

static uint8_t *perm_table(int i) {   // under lock
    if (!module_perm[i]) {
        const RtLoaded *m = ldr_module(i);
        size_t n = (m->size - m->data_off) / PAGE;
        module_perm[i] = malloc(n);
        if (!module_perm[i]) rt_fatal("memory: module permission table");
        memset(module_perm[i], PERM_RW, n);
    }
    return module_perm[i];
}

static unsigned module_page_perm(int i, uint64_t off) {   // under lock
    const RtLoaded *m = ldr_module(i);
    if (off < m->data_off) return off < align_up(m->text_end, PAGE) ? PERM_RX : PERM_R;
    return module_perm[i] ? module_perm[i][(off - m->data_off) / PAGE] : PERM_RW;
}

// The module whose data segment holds [a, a+n), or -1.
static int module_data(uint64_t a, uint64_t n) {
    int i = module_index(a);
    if (i < 0) return -1;
    const RtLoaded *m = ldr_module(i);
    uint64_t off = a - m->base;
    return off >= m->data_off && n <= m->size - off ? i : -1;
}

uint32_t mem_set_permission(uint64_t a, uint64_t n, uint32_t perm) {
    if (a & (PAGE - 1)) return R_INVALID_ADDRESS;
    if (!n || (n & (PAGE - 1))) return R_INVALID_SIZE;
    if (perm != PERM_NONE && perm != PERM_R && perm != PERM_RW) return R_INVALID_NEW_PERM;
    int i = module_data(a, n);
    if (i < 0) return R_INVALID_STATE;
    const RtLoaded *m = ldr_module(i);
    plat_mutex_lock(&lock);
    uint8_t *t = perm_table(i) + (a - m->base - m->data_off) / PAGE;
    for (uint64_t p = 0; p < n / PAGE; p++)
        if (t[p] & ATTR_PERMISSION_LOCKED) {
            plat_mutex_unlock(&lock);
            return R_INVALID_STATE;
        }
    memset(t, (int)perm, n / PAGE);
    protect(a, n, perm == PERM_RW ? PLAT_PROT_RW : perm == PERM_R ? PLAT_PROT_READ : PLAT_PROT_NONE, "module data");
    plat_mutex_unlock(&lock);
    rt_log("%s data permission +%#llx size %#llx perm %u", m->name, (unsigned long long)(a - m->base),
           (unsigned long long)n, perm);
    return R_OK;
}

// The only module attribute the SDK sets is the irreversible RELRO lock.
uint32_t mem_set_attribute(uint64_t a, uint64_t n, uint32_t mask, uint32_t at) {
    if (a & (PAGE - 1)) return R_INVALID_ADDRESS;
    if (!n || (n & (PAGE - 1))) return R_INVALID_SIZE;
    if (mask != ATTR_PERMISSION_LOCKED || at != ATTR_PERMISSION_LOCKED) return R_INVALID_NEW_PERM;
    int i = module_data(a, n);
    if (i < 0) return R_INVALID_STATE;
    const RtLoaded *m = ldr_module(i);
    plat_mutex_lock(&lock);
    uint8_t *t = perm_table(i) + (a - m->base - m->data_off) / PAGE;
    for (uint64_t p = 0; p < n / PAGE; p++) t[p] |= ATTR_PERMISSION_LOCKED;
    plat_mutex_unlock(&lock);
    return R_OK;
}

// ---- 0x04/0x05 MapMemory, UnmapMemory: thread stacks move from heap or alias into the stack region by copy; the
// source is inaccessible while moved and gets the contents back on unmap.
static int alias_all_mapped(uint64_t a, uint64_t n) {   // under lock
    uint64_t first = (a - L.alias.base) / PAGE;
    for (uint64_t i = 0; i < n / PAGE; i++)
        if (!alias_pages[first + i]) return 0;
    return 1;
}

static uint32_t stack_move(int unmap, uint64_t dst, uint64_t src, uint64_t n) {
    if (!n || ((dst | src | n) & (PAGE - 1))) return R_INVALID_ADDRESS;
    if (!inside(dst, n, L.stack)) return R_INVALID_REGION;
    int in_heap = inside(src, n, (RtRange){L.heap.base, heap_size});
    int in_alias = inside(src, n, L.alias);
    if (!in_heap && !in_alias) return R_INVALID_REGION;
    plat_mutex_lock(&lock);
    uint32_t r = R_OK;
    if (in_alias && !alias_all_mapped(src, n)) {
        r = R_INVALID_REGION;
        goto out;
    }
    unsigned found = nstack_maps;
    for (unsigned i = 0; i < nstack_maps; i++)
        if (stack_maps[i].dst == dst && stack_maps[i].src == src && stack_maps[i].size == n) found = i;
    if (unmap) {
        if (found == nstack_maps) {
            r = R_INVALID_STATE;
            goto out;
        }
        protect(src, n, PLAT_PROT_RW, "moved stack source");
        memcpy((void *)(uintptr_t)src, (const void *)(uintptr_t)dst, n);
        decommit(dst, n, "moved stack");
        stack_maps[found] = stack_maps[--nstack_maps];
    } else {
        if (nstack_maps == MAX_STACK_MAPS) {
            r = R_INVALID_STATE;
            goto out;
        }
        for (unsigned i = 0; i < nstack_maps; i++)
            if (overlaps(dst, n, stack_maps[i].dst, stack_maps[i].size) ||
                overlaps(src, n, stack_maps[i].src, stack_maps[i].size)) {
                r = R_INVALID_REGION;
                goto out;
            }
        commit(dst, n, "moved stack");
        if (in_heap) mem_touch(src, n);   // protect() below fails on a page the lazy heap never committed
        memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, n);
        protect(src, n, PLAT_PROT_NONE, "moved stack source");
        stack_maps[nstack_maps++] = (StackMap){dst, src, n};
    }
out:
    plat_mutex_unlock(&lock);
    return r;
}

uint32_t mem_map(uint64_t dst, uint64_t src, uint64_t n) { return stack_move(0, dst, src, n); }
uint32_t mem_unmap(uint64_t dst, uint64_t src, uint64_t n) { return stack_move(1, dst, src, n); }

// ---- 0x2C/0x2D MapPhysicalMemory, UnmapPhysicalMemory. Horizon maps only the still-unmapped pages of the range and
// leaves mapped pages and their contents alone; TotK's heaps grow by re-mapping ranges they partly own.
static void alias_runs(uint64_t first, uint64_t count, uint8_t want, int do_commit) {   // under lock
    uint64_t i = 0;
    while (i < count) {
        if (alias_pages[first + i] != want) { i++; continue; }
        uint64_t j = i;
        while (j < count && alias_pages[first + j] == want) j++;
        uint64_t a = L.alias.base + (first + i) * PAGE, n = (j - i) * PAGE;
        if (do_commit) commit(a, n, "physical memory");
        else decommit(a, n, "physical memory");
        memset(alias_pages + first + i, !want, j - i);
        i = j;
    }
}

static uint32_t alias_map(int unmap, uint64_t a, uint64_t n) {
    if (!n || ((a | n) & (PAGE - 1))) return R_INVALID_ADDRESS;
    if (!inside(a, n, L.alias)) return R_INVALID_REGION;
    uint64_t first = (a - L.alias.base) / PAGE, count = n / PAGE;
    plat_mutex_lock(&lock);
    uint32_t r = R_OK;
    if (unmap) {
        for (unsigned i = 0; i < nstack_maps; i++)
            if (overlaps(a, n, stack_maps[i].src, stack_maps[i].size)) {
                r = R_INVALID_REGION;
                goto out;
            }
        uint64_t released = 0;
        for (uint64_t i = 0; i < count; i++) released += alias_pages[first + i];
        alias_runs(first, count, 1, 0);
        alias_bytes -= released * PAGE;
    } else {
        uint64_t fresh = 0;
        for (uint64_t i = 0; i < count; i++) fresh += !alias_pages[first + i];
        if (fresh * PAGE + alias_bytes + heap_size + image_bytes() > L.pool) {
            r = R_LIMIT_REACHED;   // the SDK maps LimitReached to its recoverable out-of-memory result
            goto out;
        }
        alias_runs(first, count, 0, 1);
        alias_bytes += fresh * PAGE;
    }
out:
    plat_mutex_unlock(&lock);
    if (r) rt_log("%sMapPhysicalMemory %#llx+%#llx failed %#x (alias %#llx used, heap %#llx)", unmap ? "Un" : "",
                  (unsigned long long)a, (unsigned long long)n, r, (unsigned long long)alias_bytes,
                  (unsigned long long)heap_size);
    return r;
}

uint32_t mem_map_physical(uint64_t a, uint64_t n) { return alias_map(0, a, n); }
uint32_t mem_unmap_physical(uint64_t a, uint64_t n) { return alias_map(1, a, n); }

// ---- 0x06 QueryMemory
// The region around a that no known region covers: everything else below 2^39 belongs to the host and is reported
// as taken (Code, RX), so the SDK never picks it.
static void taken_gap(uint64_t a, uint64_t *base, uint64_t *size) {
    uint64_t lo = 0, hi = RT_ADDRESS_LIMIT;
    RtRange known[3 + MOD_COUNT + 2];
    int n = 0;
    known[n++] = (RtRange){L.aslr.base, LOW_SIZE};
    known[n++] = L.alias;
    known[n++] = L.kernel;
    known[n++] = L.hostfn;
    for (int i = 0; i < ldr_count(); i++) known[n++] = (RtRange){ldr_module(i)->base, ldr_module(i)->size};
    for (int i = 0; i < n; i++) {
        uint64_t e = known[i].base + known[i].size;
        if (e <= a && e > lo) lo = e;
        if (known[i].base > a && known[i].base < hi) hi = known[i].base;
    }
    *base = lo;
    *size = hi - lo;
}

static uint64_t run_end(const uint8_t *p, uint64_t i, uint64_t n, uint8_t v) {
    while (i < n && (i & 7)) { if (p[i] != v) return i; i++; }
    uint64_t w = v ? 0x0101010101010101ull : 0;
    while (i + 8 <= n && *(const uint64_t *)(p + i) == w) i += 8;
    while (i < n && p[i] == v) i++;
    return i;
}

static uint64_t run_start(const uint8_t *p, uint64_t i, uint8_t v) {   // first index of the run that ends at i
    while (i > 0 && (i & 7)) { if (p[i - 1] != v) return i; i--; }
    uint64_t w = v ? 0x0101010101010101ull : 0;
    while (i >= 8 && *(const uint64_t *)(p + i - 8) == w) i -= 8;
    while (i > 0 && p[i - 1] == v) i--;
    return i;
}

uint32_t mem_query(uint64_t a, RtMemoryInfo *out) {
    RtMemoryInfo r = {.base = a & ~(PAGE - 1), .size = PAGE};
    uint64_t low = L.aslr.base;
    plat_mutex_lock_shared(&lock);
    int mi;
    if (a >= RT_ADDRESS_LIMIT) {
        r.base = RT_ADDRESS_LIMIT;
        r.size = 0 - RT_ADDRESS_LIMIT;
        r.state = ST_INACCESSIBLE;
    } else if ((mi = module_index(a)) >= 0) {
        const RtLoaded *m = ldr_module(mi);
        uint64_t off = (a - m->base) & ~(PAGE - 1), lo = off, hi = off + PAGE;
        unsigned perm = module_page_perm(mi, off);
        int data = off >= m->data_off;
        uint64_t limit = data ? m->data_off : 0, end = data ? m->size : m->data_off;
        while (lo > limit && module_page_perm(mi, lo - PAGE) == perm) lo -= PAGE;
        while (hi < end && module_page_perm(mi, hi) == perm) hi += PAGE;
        r = (RtMemoryInfo){.base = m->base + lo, .size = hi - lo, .state = data ? ST_CODEDATA : ST_CODE,
                           .perm = perm & 7, .attr = perm & ATTR_PERMISSION_LOCKED};
    } else if (a - L.kernel.base < L.kernel.size) {
        r = (RtMemoryInfo){.base = L.kernel.base, .size = L.kernel.size, .state = ST_THREADLOCAL, .perm = PERM_RW};
    } else if (a - L.hostfn.base < L.hostfn.size) {
        r = (RtMemoryInfo){.base = L.hostfn.base, .size = L.hostfn.size, .state = ST_INACCESSIBLE};
    } else if (a - L.alias.base < L.alias.size) {
        // Before the block: macOS has no room below the modules and can put the alias region under it.
        uint64_t p = (a - L.alias.base) / PAGE;
        uint8_t v = alias_pages[p];
        uint64_t s = run_start(alias_pages, p, v), e = run_end(alias_pages, p, alias_page_count, v);
        r = (RtMemoryInfo){.base = L.alias.base + s * PAGE, .size = (e - s) * PAGE, .state = v ? ST_NORMAL : ST_FREE,
                           .perm = v ? PERM_RW : PERM_NONE};
    } else if (a < low) {
        // Below the block. With none of our regions under a (always so on Windows, where the block is the lowest) this
        // is the free space at the bottom, up to the first region above a. On macOS the block lies above the modules,
        // the kernel and host-function areas and maybe the alias region; a gap between those is the host's, reported
        // taken like any other gap below 2^39 (taken_gap).
        taken_gap(a, &r.base, &r.size);
        if (r.base == 0) {
            r.state = ST_FREE;
        } else {
            r.state = ST_CODE;
            r.perm = PERM_RX;
        }
    } else if (a < low + LOW_SIZE) {
        uint64_t heap = L.heap.base;
        if (a >= heap && a < heap + heap_size) {
            r = (RtMemoryInfo){.base = heap, .size = heap_size, .state = ST_NORMAL, .perm = PERM_RW};
        } else if (a < heap) {
            r = (RtMemoryInfo){.base = low, .size = heap - low, .state = ST_FREE};
        } else {
            r = (RtMemoryInfo){.base = heap + heap_size, .size = low + LOW_SIZE - heap - heap_size, .state = ST_FREE};
        }
    } else {
        taken_gap(a, &r.base, &r.size);
        r.state = ST_CODE;
        r.perm = PERM_RX;
    }
    // Moved stacks: the destination is Stack, the source Normal without access; free and normal regions around them
    // are clipped (MNG services.c).
    for (unsigned i = 0; i < nstack_maps; i++) {
        const StackMap *m = &stack_maps[i];
        uint64_t end = m->dst + m->size, send = m->src + m->size;
        if (a >= m->dst && a < end) {
            r = (RtMemoryInfo){.base = m->dst, .size = m->size, .state = ST_STACK, .perm = PERM_RW};
            break;
        }
        if (a >= m->src && a < send) {
            r = (RtMemoryInfo){.base = m->src, .size = m->size, .state = ST_NORMAL, .perm = PERM_NONE, .attr = 4};
            break;
        }
        uint64_t lo = r.base, hi = r.base + r.size;
        uint64_t b = r.state == ST_FREE ? m->dst : r.state == ST_NORMAL ? m->src : 0;
        uint64_t e = b + m->size;
        if (!b) continue;
        if (e <= a && e > lo) lo = e;
        if (b > a && b < hi) hi = b;
        r.base = lo;
        r.size = hi - lo;
    }
    plat_mutex_unlock_shared(&lock);
    if (out) *out = r;
    return R_OK;
}

// Whether a read anywhere in [a, a + n) is sure not to fault, from this file's own map of the guest: 1 inside the
// heap's current size (committed, or in the lazy heap reserved until the first access commits it; frozen pages are
// read-only, still readable) or on mapped alias pages, in both cases outside a moved stack's source; 0 inside the heap
// block or the alias region otherwise; -1 anywhere else (modules, the stack region, the kernel region, host memory),
// for the caller to ask the platform. No system call: on Windows plat_mem_readable's VirtualQuery reports the whole
// run of like pages around a, so a pointer in a long read-write stretch of a committed heap (TOTK_HEAP_LAZY=0: all
// 9.8 GiB of it) cost milliseconds, and a GetWriteWatch over the same reservation waited for it (#317).
int mem_guest_readable(uint64_t a, uint64_t n) {
    if (!n) n = 1;
    if (a + n < a) return -1;
    int r = -1;
    plat_mutex_lock_shared(&lock);
    if (inside(a, n, (RtRange){L.aslr.base, LOW_SIZE}) && !inside(a, n, L.stack))
        r = inside(a, n, (RtRange){L.heap.base, heap_size});
    else if (inside(a, n, L.alias)) {
        uint64_t first = (a - L.alias.base) / PAGE, last = (a + n - 1 - L.alias.base) / PAGE;
        r = 1;
        for (uint64_t p = first; p <= last && r; p++) r = alias_pages[p] != 0;
    }
    for (unsigned i = 0; i < nstack_maps && r == 1; i++)
        if (overlaps(a, n, stack_maps[i].src, stack_maps[i].size)) r = 0;
    plat_mutex_unlock_shared(&lock);
    return r;
}

// ---- GetInfo memory ids
int mem_info(uint32_t id, uint64_t *out) {
    uint64_t v;
    switch (id) {
    case 2: v = L.alias.base; break;
    case 3: v = L.alias.size; break;
    case 4: v = L.heap.base; break;
    case 5: v = L.heap.size; break;
    case 6: case 21: v = L.pool; break;
    case 7: case 22:
        plat_mutex_lock_shared(&lock);
        v = image_bytes() + heap_size + alias_bytes;
        plat_mutex_unlock_shared(&lock);
        break;
    case 12: v = L.aslr.base; break;
    case 13: v = L.aslr.size; break;
    case 14: v = L.stack.base; break;
    case 15: v = L.stack.size; break;
    case 16: v = SYSTEM_RESOURCE_BUDGET; break;
    case 17: v = sizeof stack_maps + alias_page_count; break;   // system resource in use: our bookkeeping
    case 28: v = 0; break;   // AliasRegionExtraSize: no kernel-owned extension
    default: return 0;
    }
    *out = v;
    return 1;
}

// ---- kernel region: power-of-two blocks, a free list per size, bump allocation with lazy commit.
#define KCLASSES 29          // 64 B .. 256 MiB
#define KMIN_SHIFT 6
#define KCOMMIT 0x10000ull
typedef struct KFree { struct KFree *next; } KFree;
static KFree *kfree_list[KCLASSES];
static uint64_t kbump, kcommitted;
static PlatMutex klock = PLAT_MUTEX_INIT;

static int kclass(uint64_t size) {
    int s = KMIN_SHIFT;
    while ((1ull << s) < size) s++;
    return s - KMIN_SHIFT;
}

void *mem_kernel_alloc(uint64_t size, uint64_t align) {
    if (!L.pool) rt_fatal("mem_kernel_alloc before mem_init");
    if (align > KCOMMIT || (align & (align - 1))) rt_fatal("mem_kernel_alloc: alignment %#llx", (unsigned long long)align);
    uint64_t want = size > align ? size : align;
    int k = kclass(want ? want : 1);
    if (k >= KCLASSES) rt_fatal("mem_kernel_alloc: %#llx bytes", (unsigned long long)size);
    uint64_t bytes = 1ull << (k + KMIN_SHIFT), al = bytes < KCOMMIT ? bytes : KCOMMIT;
    plat_mutex_lock(&klock);
    KFree *f = kfree_list[k];
    void *p;
    if (f) {
        kfree_list[k] = f->next;
        p = f;
    } else {
        uint64_t a = align_up(kbump ? kbump : RT_KERNEL_BASE, al);
        if (a + bytes > RT_KERNEL_BASE + RT_KERNEL_SIZE) rt_fatal("kernel region exhausted (%#llx bytes)",
                                                                  (unsigned long long)bytes);
        uint64_t need = align_up(a + bytes, KCOMMIT);
        if (!kcommitted) kcommitted = RT_KERNEL_BASE;
        if (need > kcommitted) {
            commit(kcommitted, need - kcommitted, "kernel region");
            kcommitted = need;
        }
        kbump = a + bytes;
        p = (void *)(uintptr_t)a;
    }
    plat_mutex_unlock(&klock);
    memset(p, 0, size ? size : 1);
    return p;
}

void mem_kernel_free(void *p, uint64_t size) {
    if (!p) return;
    uint64_t a = (uintptr_t)p;
    if (a - RT_KERNEL_BASE >= RT_KERNEL_SIZE) rt_fatal("mem_kernel_free(%p): not in the kernel region", p);
    int k = kclass(size ? size : 1);
    // A block allocated with a larger alignment sits in a larger class; freeing it into the smaller class only
    // wastes the rest.
    plat_mutex_lock(&klock);
    KFree *f = p;
    f->next = kfree_list[k];
    kfree_list[k] = f;
    plat_mutex_unlock(&klock);
}
