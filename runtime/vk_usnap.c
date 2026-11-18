// Uniform blocks as of the submit call. NVN starts the GPU when nvnQueueSubmitCommands returns and the GPU reads the
// constants soon after; here the replay records for several milliseconds, and the GPU reads a pool's bytes when it
// runs the part that uses them (the VRAM mirror's sync copies run then, vk_memory.c "VRAM mirror"). Meanwhile the game,
// having handed the frame over, rewrites per-object uniform blocks on other threads with no fence: late draws read
// another object's matrix (phantom rocks, a flat far field; TotK 1.2.1's producer f_70008c6460, 14-24 ms after the
// submit). So at the start of a submission's recording every uniform range it binds is copied into an arena of its own,
// and the uniform bindings of the replay use the copy.
// - Only ranges of imported pools (memory the CPU writes) are copied, aligned out to 256 bytes (the largest
//   minUniformBufferOffsetAlignment) and at most 64 KiB from the bound address, overlapping ones merged.
// - A range the GPU wrote in a submission that has not completed is left out: its bytes reach guest memory only with
//   that submission's write-back, the mirror holds them until then.
// - A GPU write within the submission (UpdateUniformBuffer, a storage buffer a shader writes, a buffer copy or fill,
//   an image-to-buffer copy: vk_usnap_drop) drops the copies it touches from then on, so later draws read the result.
// - The arena goes back to a free list once the submission's fence is done (completion thread).
// The design follows MacNativeGames' TOTK_UBO_SNAPSHOT (games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m usnap_take,
// usnap_drop; MIT, Copyright (c) 2026 jhaizhou); the code is ours.
// The copy is taken where vk_exec_record starts: at the submit call when the replay runs on the submitting thread, later
// with TOTK_REPLAY_ASYNC. TOTK_UBO_SNAPSHOT=0 binds live memory as before; =ab alternates (TOTK_AB).
// TOTK_USNAP_FAST=0 runs collect, sort, drop and copy as they were; by default each step takes a shortcut (repeated
// ranges skipped, fewer radix passes, repeated GPU writes merged, and with a CPU copy one copy into the arena), and its
// comment says why the result is the same.
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "msort.h"
#include "nvn_api.h"
#include "vk.h"
#include "vk_spec.h"
#include "platform/platform.h"

#define ALIGN 256u
#define MAX_RANGE (64u << 10)
#define ARENA_MIN (4u << 20)
#define CALL_DEPTH 16

typedef struct Arena {
    struct Arena *next;
    VkBuffer buf;
    VmaAllocation alloc;
    uint8_t *map;
    uint64_t size;
} Arena;

typedef struct { uint64_t lo, hi; } Rng;
typedef struct { uint64_t lo, hi, off, dropped; } Reg;   // dropped: the drop's number (U.dseq), 0 while kept
typedef struct { uint64_t lo, hi, fence; } Wrote;

static struct {
    int mode;                   // rt_knob: 0 off, 1 on, 2 ab
    int on;                     // this submission has a snapshot
    int fast;                   // this submission takes the USNAP_FAST shortcuts (set in vk_usnap_begin)
    uint64_t fence, gen;
    uint64_t dseq;              // drops made so far (vk_usnap_drops)
    Arena *arena;
    uint8_t *cpu;               // this submission's copy for the CPU (vk_usnap_cpu_reads), as laid out in the arena
    Rng *col;                   // collected ranges
    int ncol, ccol;
    Reg *reg;                   // merged, sorted by lo, disjoint
    int nreg, creg;
    Wrote *wr;                  // GPU writes of submissions not known to be complete
    int nwr, cwr;
    // since the last summary
    uint64_t subs, regions, bytes, copy_ns, hits, drops, inflight, refused, off_subs;
    uint64_t cols, t_collect, t_sort, t_merge, t_copy;   // ranges collected, and the time of each step (summary)
    uint64_t wr_added;   // GPU write ranges added to wr (vk_usnap_drop), for the summary
    uint64_t wr_coalesced;   // GPU writes that raised the newest tail entry's fence instead (drop, USNAP_FAST)
    uint64_t repeats;    // collected ranges that repeated one collected earlier in the submission (add_range, USNAP_FAST)
} U;

// The helper threads (vk_spec.h) look regions up in vk_usnap_find while the replay thread drops some of them
// (vk_usnap_drop), so a region's dropped mark and U.gen are loaded and stored atomically. A helper part that read a
// region before it was dropped is itself dropped, because the generation moved (vk_exec.c par_join). The rest of U
// changes only in vk_usnap_begin and vk_usnap_end, where no helper part is out.
static void gen_bump(void) { __atomic_add_fetch(&U.gen, 1, __ATOMIC_RELEASE); }

// The arena is written once, in order, and lives in device-local memory through the BAR where the device has it: for
// the CPU it is uncached, and reading a 256-byte block back takes microseconds. A caller that reads many copied blocks
// (vk_exec_draw.c merge_pick, TOTK_DRAW_MERGE) asks for a second copy in cached host memory, made from the first
// submission that begins after the call; vk_usnap_bytes reads that one. One submission records at a time
// (vk_exec_record), so the one buffer, grown as needed, serves every submission.
static int cpu_reads;
static uint8_t *cpu_buf;
static uint64_t cpu_cap;

void vk_usnap_cpu_reads(void) { __atomic_store_n(&cpu_reads, 1, __ATOMIC_RELAXED); }

static Arena *free_arenas;
static PlatMutex free_lock = PLAT_MUTEX_INIT;

static void *grow(void *p, int *cap, size_t elem) {
    int n = *cap ? *cap * 2 : 1024;
    void *q = realloc(p, (size_t)n * elem);
    if (!q) rt_fatal("vk_usnap: out of memory");
    *cap = n;
    return q;
}

static volatile uint64_t arena_total;   // bytes of every arena created; arenas are only freed at shutdown

static Arena *arena_get(uint64_t need) {
    plat_mutex_lock(&free_lock);
    Arena **pp = &free_arenas, *a = NULL;
    for (; *pp; pp = &(*pp)->next)
        if ((*pp)->size >= need) {
            a = *pp, *pp = a->next;
            break;
        }
    plat_mutex_unlock(&free_lock);
    if (a) return a;
    a = calloc(1, sizeof *a);
    if (!a) rt_fatal("vk_usnap: out of memory");
    a->size = need < ARENA_MIN ? ARENA_MIN : (need + (1u << 20) - 1) & ~(uint64_t)((1u << 20) - 1);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = a->size,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    // Device-local through the BAR when the device has it (the GPU reads constants on every draw), else host memory.
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &a->buf, &a->alloc, &info);
    if (res != VK_SUCCESS) {
        rt_log("vk_usnap: arena of %llu bytes failed (%d): this submission binds live memory",
               (unsigned long long)a->size, (int)res);
        free(a);
        return NULL;
    }
    a->map = info.pMappedData;
    __atomic_fetch_add(&arena_total, a->size, __ATOMIC_RELAXED);
    rt_log("vk_usnap: new arena of %.1f MiB", a->size / 1048576.0);
    return a;
}

static void arena_done(void *arg) {   // completion thread: the GPU has read it
    Arena *a = arg;
    plat_mutex_lock(&free_lock);
    a->next = free_arenas, free_arenas = a;
    plat_mutex_unlock(&free_lock);
}

// Ranges collected this submission, through a direct-mapped cache on lo (hashed, so ranges a power of two apart do not
// share a slot). Under USNAP_FAST a range the cache still holds was collected earlier in this submission, and is not
// appended again. The merge in vk_usnap_begin takes the union of the ranges and a repeat adds nothing to it, so the
// merged regions are the same. An empty slot has hi 0, which no collected range has: a range ends above its address,
// and the address is not 0.
#define SEEN_BITS 8
static Rng seen[1 << SEEN_BITS];

static void add_range(Rng r) {
    if (U.fast) {
        Rng *s = &seen[((r.lo / ALIGN) * 0x9E3779B97F4A7C15ull) >> (64 - SEEN_BITS)];
        if (s->lo == r.lo && s->hi == r.hi) {
            U.repeats++;
            return;
        }
        *s = r;
    }
    if (U.ncol == U.ccol) U.col = grow(U.col, &U.ccol, sizeof *U.col);
    U.col[U.ncol++] = r;
}

static void collect(const op_hdr *h, int depth) {
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk_usnap: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        if (h->op == OP_END) return;
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (h->op == OP_CALL && depth < CALL_DEPTH) collect(nvn_op_first(((const p_jump *)a)->target), depth + 1);
        if (h->op == OP_BIND_UBO) {
            const p_bind_buf *b = a;
            if (b->addr && b->size) {
                uint64_t n = b->size < MAX_RANGE ? b->size : MAX_RANGE;
                U.cols++;
                add_range((Rng){b->addr & ~(uint64_t)(ALIGN - 1), b->addr + n});
            }
        }
        h = nvn_op_next(h);
    }
}

static int by_lo(const void *a, const void *b) {
    uint64_t x = ((const Rng *)a)->lo, y = ((const Rng *)b)->lo;
    return x < y ? -1 : x > y;
}
#define WROTE_LESS(x, y) ((x)->lo < (y)->lo)
MSORT_DEFINE(sort_wrote, Wrote, WROTE_LESS)   // by lo, without a call per comparison (msort.h)

#define RADIX_BITS 11
// The number of passes of the LSD radix sorts below (sort_col, sort_new_writes), 11 bits of the key a pass, for keys
// in [0, span]: as many passes as the span has 11-bit digits (at least one) under USNAP_FAST, three always without
// it. The skipped passes would each see one digit, 0, for every key, and a stable pass on one digit keeps the order,
// so the result is the same either way.
static int radix_passes(uint64_t span) {
    if (!U.fast) return 3;
    int p = 1;
    while (p < 3 && span >> (p * RADIX_BITS)) p++;
    return p;
}

// Sorts the collected ranges by lo. A town frame binds about 10,000 uniform ranges and qsort took half a millisecond
// of the copy's 0.96 ms on the game thread, so they go through an LSD radix sort on (lo - lowest) / ALIGN, 11 bits a
// pass; qsort is left for a span that does not fit in 33 bits (2 TiB).
static void sort_col(void) {
    static Rng *tmp;
    static int ctmp;
    int n = U.ncol;
    if (n < 2) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (U.col[i].lo < lo) lo = U.col[i].lo;
        if (U.col[i].lo > hi) hi = U.col[i].lo;
    }
    if ((hi - lo) / ALIGN >> (3 * RADIX_BITS)) {
        qsort(U.col, (size_t)n, sizeof *U.col, by_lo);
        return;
    }
    if (ctmp < n) {
        free(tmp);
        ctmp = U.ccol;
        if (!(tmp = malloc((size_t)ctmp * sizeof *tmp))) rt_fatal("vk_usnap: out of memory");
    }
    int passes = radix_passes((hi - lo) / ALIGN);
    Rng *src = U.col, *dst = tmp;
    for (int pass = 0; pass < passes; pass++) {
        int shift = pass * RADIX_BITS, count[1 << RADIX_BITS] = {0};
        for (int i = 0; i < n; i++) count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++;
        for (int k = 0, sum = 0; k < 1 << RADIX_BITS; k++) {
            int c = count[k];
            count[k] = sum, sum += c;
        }
        for (int i = 0; i < n; i++) dst[count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++] = src[i];
        Rng *t = src;
        src = dst, dst = t;
    }
    if (src != U.col) memcpy(U.col, src, (size_t)n * sizeof *src);   // an odd number of passes ends in tmp
}

// Keeps the GPU writes of submissions that are not done, merged into disjoint ranges sorted by lo (the terrain rewrites
// one block before every draw: hundreds of equal entries), each with the newest fence among its parts.
// The ranges the last prune left are sorted already: only those added since (about 7,400 a submission on six threads in
// the hills of Bananza 3.2.0) are sorted, by an LSD radix sort on (lo - lowest) / 16 like sort_col, and merged with
// them. Sorting the whole list again took about 0.38 ms a submission.
static int nwr_sorted;   // U.wr[0, nwr_sorted) is the last prune's: sorted by lo and disjoint

static void sort_new_writes(Wrote *a, int n, Wrote **tmp, size_t *cap) {
    if (n < 2) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (a[i].lo < lo) lo = a[i].lo;
        if (a[i].lo > hi) hi = a[i].lo;
    }
    if (*cap < (size_t)n) {
        Wrote *t = realloc(*tmp, (size_t)n * sizeof *t);
        if (!t) rt_fatal("vk_usnap: out of memory");
        *tmp = t, *cap = (size_t)n;
    }
    if ((hi - lo) / 16 >> (3 * RADIX_BITS)) {   // a span over 128 GiB: the merge sort
        sort_wrote(a, (size_t)n, tmp, cap);
        return;
    }
    int passes = radix_passes((hi - lo) / 16);
    Wrote *src = a, *dst = *tmp;
    for (int pass = 0; pass < passes; pass++) {
        int shift = pass * RADIX_BITS, count[1 << RADIX_BITS] = {0};
        for (int i = 0; i < n; i++) count[(src[i].lo - lo) / 16 >> shift & ((1 << RADIX_BITS) - 1)]++;
        for (int k = 0, sum = 0; k < 1 << RADIX_BITS; k++) {
            int c = count[k];
            count[k] = sum, sum += c;
        }
        for (int i = 0; i < n; i++) dst[count[(src[i].lo - lo) / 16 >> shift & ((1 << RADIX_BITS) - 1)]++] = src[i];
        Wrote *t = src;
        src = dst, dst = t;
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof *a);   // an odd number of passes ends in tmp
    // The buckets are 16 bytes wide, and GPU write ranges need not be 16-byte aligned (inline uniform updates, buffer
    // copies, query results): two ranges in one bucket keep the order they were added in, and the merge in wr_prune,
    // which wants them by lo, then dropped the lower one. An insertion pass puts them in order; the array is sorted but
    // inside buckets, so it moves only those few.
    for (int i = 1; i < n; i++) {
        Wrote w = a[i];
        int j = i;
        while (j > 0 && a[j - 1].lo > w.lo) a[j] = a[j - 1], j--;
        a[j] = w;
    }
}

static void wr_prune(void) {
    uint64_t done = vk_sync_fence_done();
    int old = nwr_sorted < U.nwr ? nwr_sorted : U.nwr, nold = 0, nnew = 0;
    static Wrote *fresh, *out, *tmp;   // the replaying thread's (vk_usnap_begin)
    static size_t cfresh, cout, ctmp;
    if (cfresh < (size_t)(U.nwr - old) || cout < (size_t)U.nwr) {
        size_t c = (size_t)U.nwr + 1024;
        Wrote *f = realloc(fresh, c * sizeof *f), *o = f ? realloc(out, c * sizeof *o) : NULL;
        if (!f || !o) rt_fatal("vk_usnap: out of memory");
        fresh = f, out = o, cfresh = cout = c;
    }
    for (int i = old; i < U.nwr; i++)
        if (U.wr[i].fence > done) fresh[nnew++] = U.wr[i];
    for (int i = 0; i < old; i++)
        if (U.wr[i].fence > done) U.wr[nold++] = U.wr[i];   // stays sorted
    sort_new_writes(fresh, nnew, &tmp, &ctmp);
    int m = 0;
    for (int a = 0, b = 0; a < nold || b < nnew;) {
        const Wrote *w = b >= nnew || (a < nold && U.wr[a].lo <= fresh[b].lo) ? &U.wr[a++] : &fresh[b++];
        if (m && w->lo <= out[m - 1].hi) {
            if (w->hi > out[m - 1].hi) out[m - 1].hi = w->hi;
            if (w->fence > out[m - 1].fence) out[m - 1].fence = w->fence;
        } else {
            out[m++] = *w;
        }
    }
    memcpy(U.wr, out, (size_t)m * sizeof *out);
    U.nwr = nwr_sorted = m;
}
static int wr_hits(uint64_t lo, uint64_t hi) {   // wr_prune ran: disjoint, sorted
    int a = 0, b = U.nwr;
    while (a < b) {
        int k = (a + b) / 2;
        if (U.wr[k].hi <= lo) a = k + 1;
        else b = k;
    }
    return a < U.nwr && U.wr[a].lo < hi;
}

static void summary(void) {
    if (U.subs + U.off_subs < 3600) return;
    double s = U.subs ? (double)U.subs : 1.0;
    rt_log("vk_usnap: %llu submissions with a snapshot, %llu without; per submission %.0f ranges, %.1f KiB copied in "
           "%.3f ms, %.0f uniform bindings from the copy, %.1f dropped by GPU writes, %.1f left out (GPU writes in "
           "flight), %.1f refused (not an imported pool); %.0f bindings collected (%.0f repeats skipped), %.0f GPU write "
           "ranges added (%.0f coalesced); ms: collect %.3f, sort and prune %.3f, "
           "merge %.3f, copy %.3f",
           (unsigned long long)U.subs, (unsigned long long)U.off_subs, U.regions / s, U.bytes / s / 1024.0,
           U.copy_ns / s / 1e6, U.hits / s, U.drops / s, U.inflight / s, U.refused / s, U.cols / s, U.repeats / s,
           U.wr_added / s, U.wr_coalesced / s, U.t_collect / s / 1e6,
           U.t_sort / s / 1e6, U.t_merge / s / 1e6, U.t_copy / s / 1e6);
    U.subs = U.off_subs = U.regions = U.bytes = U.copy_ns = U.hits = U.drops = U.inflight = U.refused = 0;
    U.cols = U.t_collect = U.t_sort = U.t_merge = U.t_copy = U.wr_added = 0;
    U.repeats = U.wr_coalesced = 0;
}

static int drop_local;    // TOTK_USNAP_DROP_LOCAL, for this submission (wr_local)
static void wr_take(void);

static void usnap_begin(const uint64_t *handles, int n, uint64_t fence) {
    if (!U.mode) U.mode = rt_knob("UBO_SNAPSHOT", "1") + 1;   // stored +1: 0 is "not read yet"
    U.on = 0, U.fence = fence, U.cpu = NULL, gen_bump();
    wr_take();   // whatever the mode: an earlier submission may have noted writes in the threads' lists
    summary();
    // USNAP_FAST is read once; with "ab" it is decided per submission, as the A/B switch is. drop() follows U.fast for
    // the writes this submission records.
    static int fast_mode;   // stored +1: 0 is "not read yet"
    if (!fast_mode) fast_mode = rt_knob("USNAP_FAST", "1") + 1;
    U.fast = RT_KNOB_ON(fast_mode - 1);
    static int local_mode;   // the same for TOTK_USNAP_DROP_LOCAL
    if (!local_mode) local_mode = rt_knob("USNAP_DROP_LOCAL", "1") + 1;
    drop_local = RT_KNOB_ON(local_mode - 1);
    if (!RT_KNOB_ON(U.mode - 1)) {
        U.off_subs++;
        return;
    }
    uint64_t t0 = kern_now_ns();
    U.ncol = 0;
    if (U.fast) memset(seen, 0, sizeof seen);   // no range is collected yet in this submission
    for (int i = 0; i < n; i++)
        if (handles[i]) collect(nvn_op_first(handles[i]), 0);
    uint64_t t1 = kern_now_ns();
    sort_col();
    wr_prune();
    uint64_t t2 = kern_now_ns();
    U.t_collect += t1 - t0, U.t_sort += t2 - t1;
    // merge, keeping only what lies wholly in one imported pool and no GPU write in flight touches
    U.nreg = 0;
    uint64_t total = 0;
    for (int i = 0; i < U.ncol;) {
        uint64_t lo = U.col[i].lo, hi = U.col[i].hi;
        for (i++; i < U.ncol && U.col[i].lo <= hi; i++)
            if (U.col[i].hi > hi) hi = U.col[i].hi;
        VkMemSpan sp;
        if (!vk_mem_resolve(lo, hi - lo, &sp) || sp.alloc || !sp.pool || sp.avail < hi - lo ||
            (sp.cls != VK_MCLS_HOST_CACHED && sp.cls != VK_MCLS_HOST_UNC)) {
            U.refused++;
            continue;
        }
        // A GPU write kept in the mirror (vk_mem_gpu_kept) never reaches guest memory, which stays stale there.
        if (wr_hits(lo, hi) || vk_mem_kept_overlaps(lo, hi - lo)) {
            U.inflight++;
            continue;
        }
        if (U.nreg == U.creg) U.reg = grow(U.reg, &U.creg, sizeof *U.reg);
        U.reg[U.nreg++] = (Reg){lo, hi, total, 0};
        total += (hi - lo + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    }
    uint64_t t3 = kern_now_ns();
    U.t_merge += t3 - t2;
    if (!U.nreg) return;
    if (!(U.arena = arena_get(total))) return;
    if (__atomic_load_n(&cpu_reads, __ATOMIC_RELAXED) && cpu_cap < total) {
        free(cpu_buf);
        cpu_cap = (cpu_buf = malloc(total)) ? total : 0;
    }
    U.cpu = cpu_cap >= total && __atomic_load_n(&cpu_reads, __ATOMIC_RELAXED) ? cpu_buf : NULL;
    if (U.cpu && U.fast) {
        // Each region is read from guest memory once, into the CPU copy at its offset; the arena then takes the whole
        // span in one copy. Every byte a binding can read ends up at the offset the per-region copy gave it. The padding
        // between regions is never bound (a binding lies inside one region, vk_usnap_find), so what it holds is unused.
        for (int i = 0; i < U.nreg; i++)
            memcpy(U.cpu + U.reg[i].off, (const void *)(uintptr_t)U.reg[i].lo, U.reg[i].hi - U.reg[i].lo);
        memcpy(U.arena->map, U.cpu, total);
    } else {
        for (int i = 0; i < U.nreg; i++) {
            const void *src = (const void *)(uintptr_t)U.reg[i].lo;
            if (U.cpu) src = memcpy(U.cpu + U.reg[i].off, src, U.reg[i].hi - U.reg[i].lo);   // the guest's bytes once
            memcpy(U.arena->map + U.reg[i].off, src, U.reg[i].hi - U.reg[i].lo);
        }
    }
    vmaFlushAllocation(vk_ctx()->vma, U.arena->alloc, 0, total);   // a no-op on coherent memory
    U.on = 1;
    uint64_t t4 = kern_now_ns();
    U.t_copy += t4 - t3;
    U.subs++, U.regions += (uint64_t)U.nreg, U.bytes += total, U.copy_ns += t4 - t0;
}

// TOTK_TRACE=perf: a part of the replay (perf.c ph_frame).
void vk_usnap_begin(const uint64_t *handles, int n, uint64_t fence) {
    uint64_t ph = perf_ph_begin();
    usnap_begin(handles, n, fence);
    perf_ph_end(PERF_PH_USNAP, ph);
}

void vk_usnap_end(uint64_t fence) {
    if (U.arena) vk_completion_add(fence, arena_done, U.arena), U.arena = NULL;
    U.on = 0, U.cpu = NULL, gen_bump();
}

// First region whose end is above a.
static int first_above(uint64_t a) {
    int lo = 0, hi = U.nreg;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (U.reg[m].hi <= a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

static _Thread_local int find_quiet;
static _Thread_local uint64_t find_upto;   // vk_usnap_view: only drops numbered below it count; 0 every drop

void vk_usnap_quiet(int on) { find_quiet = on; }

// A slice of a sliced pass whose end is held back (vk_exec.c par_defer) may be recorded after the walk has gone on
// past its pass and dropped copies with the GPU writes of later ops (a dispatch's storage buffer, an inline update):
// the slice then bound live memory, which the game rewrites for its next frame, and models flickered black. Its
// bindings follow the drops made before its pass started, as the walk's did.
uint64_t vk_usnap_drops(void) { return __atomic_load_n(&U.dseq, __ATOMIC_ACQUIRE); }
void vk_usnap_view(uint64_t upto) { find_upto = upto; }

int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out) {
    if (!U.on || !n) return 0;
    int i = first_above(addr);
    if (i >= U.nreg) return 0;
    const Reg *r = &U.reg[i];
    uint64_t d = __atomic_load_n(&r->dropped, __ATOMIC_ACQUIRE);
    if ((d && (!find_upto || d < find_upto)) || addr < r->lo || addr + n > r->hi) return 0;
    uint64_t off = r->off + (addr - r->lo);
    if (off % align) return 0;
    *out = (VkDescriptorBufferInfo){U.arena->buf, off, n};
    if (!vk_spec_helper() && !find_quiet) U.hits++;   // the replay thread's count: helpers and slices only read the table
    return 1;
}

// Parallel passes (vk_exec.c par_flush) record slices on several threads: drops are made under drop_lock. A drop
// that takes a copy away belongs to a place in the draw order, which slices recorded at once do not keep (a slice ahead
// in the pass may be recorded after it and bind memory where the walk would have bound the copy), so a slice whose
// stored range overlaps a copy (vk_usnap_covers) marks its pass's targets and they are not sliced again
// (vk_exec_par_order_hazard). A stored range no copy overlaps changes no binding: only the list of writes grows.
static PlatMutex drop_lock = PLAT_MUTEX_INIT;
const void *vk_usnap_bytes(uint64_t addr, uint64_t n) {
    VkDescriptorBufferInfo b;
    return vk_usnap_find(addr, n, 1, &b) ? (U.cpu ? U.cpu : U.arena->map) + b.offset : NULL;
}

// 1 when a copy made for this submission overlaps [addr, addr+n): only then does a drop of that range change what a
// later binding gets, and with it depend on the draw order. Reads what vk_usnap_begin set up; any recording thread.
int vk_usnap_covers(uint64_t addr, uint64_t n) {
    if (!U.on || !n) return 0;
    int i = first_above(addr);
    return i < U.nreg && U.reg[i].lo < addr + n;
}

// TOTK_USNAP_DROP_LOCAL (on by default; 0 as before; "ab" decides per submission, in vk_usnap_begin): a GPU write
// goes into a list of the noting thread's own instead of U.wr, and drop_lock is taken only when the write drops a copy
// (vk_usnap_covers). With every GPU-written binding of the slices (about 5,700 ranges a submission in the Hilltop Layer
// of Surf Hills, 3.2.0, one per terrain record) appended to U.wr under the lock, the slice workers queued for it.
// What each reader needs, and why the lists keep it:
// - The copies' dropped marks (vk_usnap_find, read by the walk's later draws, by the slices of later passes through the
//   drop count vk_usnap_view compares with, and by the draw merge and DLSS through vk_usnap_bytes) are made as before:
//   under drop_lock, in the thread that notes the write, before drop returns. vk_usnap_covers is the condition under
//   which the marking loop below visits a region at all, and what it reads (U.on, U.reg, U.nreg) changes only in
//   vk_usnap_begin and vk_usnap_end, which no drop overlaps.
// - U.wr, the writes in flight, is read by wr_prune and wr_hits in vk_usnap_begin, which wr_take precedes. Every
//   write of a submission is noted before its recording ends: vk_exec.c record_impl waits for every slice (par_settle_all,
//   par_flush), helper parts drop nothing (a write fails them, vk_exec_draw.c), and a slice's notes precede its
//   par_slice_done under pw_lock, which the walk takes before the pass ends. The next submission's vk_usnap_begin comes
//   after (one submission records at a time), so wr_take moves every write noted since the last one. The order in which
//   the lists join U.wr does not matter: wr_prune sorts the entries added since the last prune and merges overlaps
//   into one range with the newest fence (tests/usnap_prune_test.py, drop_local_cases).
// - vk_usnap_gpu_writing reads U.wr in the middle of a submission too (the indirect reset's guest read, #354): it also
//   looks through every thread's list, each under the list's own lock, which the owner takes to append (uncontended
//   but for that look).
typedef struct WrLocal { Wrote *a; int n, c; uint64_t added, coalesced; struct WrLocal *next; PlatMutex lock; } WrLocal;
static WrLocal *wr_locals;   // every thread's, linked under drop_lock and kept until exit
static _Thread_local WrLocal *wr_mine;

static void wr_local(uint64_t addr, uint64_t n, uint64_t fence) {
    WrLocal *w = wr_mine;
    if (!w) {
        if (!(w = calloc(1, sizeof *w))) rt_fatal("vk_usnap: out of memory");
        plat_mutex_lock(&drop_lock);
        w->next = wr_locals, wr_locals = w;
        plat_mutex_unlock(&drop_lock);
        wr_mine = w;
    }
    plat_mutex_lock(&w->lock);
    w->added++;
    Wrote *last = w->n ? &w->a[w->n - 1] : NULL;   // as drop coalesces the tail of U.wr
    if (U.fast && last && last->lo == addr && last->hi == addr + n) {
        if (fence > last->fence) last->fence = fence;
        w->coalesced++;
    } else {
        if (w->n == w->c) w->a = grow(w->a, &w->c, sizeof *w->a);
        w->a[w->n++] = (Wrote){addr, addr + n, fence};
    }
    plat_mutex_unlock(&w->lock);
}

static void wr_take(void) {   // vk_usnap_begin: no thread notes a write now
    plat_mutex_lock(&drop_lock);
    for (WrLocal *w = wr_locals; w; w = w->next) {
        plat_mutex_lock(&w->lock);
        while (U.cwr < U.nwr + w->n) U.wr = grow(U.wr, &U.cwr, sizeof *U.wr);
        if (w->n) memcpy(U.wr + U.nwr, w->a, (size_t)w->n * sizeof *w->a);
        U.nwr += w->n, U.wr_added += w->added, U.wr_coalesced += w->coalesced;
        w->n = 0, w->added = w->coalesced = 0;
        plat_mutex_unlock(&w->lock);
    }
    plat_mutex_unlock(&drop_lock);
}

// The copies [addr, addr+n) overlaps are dropped from here on (drop_lock held).
static void mark(uint64_t addr, uint64_t n) {
    if (U.on)
        for (int i = first_above(addr); i < U.nreg && U.reg[i].lo < addr + n; i++)
            if (!U.reg[i].dropped)
                __atomic_store_n(&U.reg[i].dropped, __atomic_add_fetch(&U.dseq, 1, __ATOMIC_ACQ_REL), __ATOMIC_RELEASE), U.drops++,
                    gen_bump();
}

static void drop_locked(uint64_t addr, uint64_t n, uint64_t fence);

static void drop(uint64_t addr, uint64_t n, uint64_t fence) {
    if (!U.mode || !n) return;
    if (drop_local) {
        wr_local(addr, n, fence);
        if (!vk_usnap_covers(addr, n)) return;   // no copy to drop
        plat_mutex_lock(&drop_lock);
        mark(addr, n);
        plat_mutex_unlock(&drop_lock);
        return;
    }
    plat_mutex_lock(&drop_lock);
    drop_locked(addr, n, fence);
    plat_mutex_unlock(&drop_lock);
}

static void drop_locked(uint64_t addr, uint64_t n, uint64_t fence) {   // drop_lock held, n > 0
    U.wr_added++;
    // The same range as the newest entry of the unsorted tail (U.wr[nwr_sorted, U.nwr); the terrain rewrites one block
    // before every draw) raises that entry's fence instead of adding a second entry. The newer of the two fences is
    // above the done fence exactly when either is, so wr_prune builds the same list from the one entry as from two.
    Wrote *last = U.nwr > nwr_sorted ? &U.wr[U.nwr - 1] : NULL;
    if (U.fast && last && last->lo == addr && last->hi == addr + n) {
        if (fence > last->fence) last->fence = fence;
        U.wr_coalesced++;
    } else {
        if (U.nwr == U.cwr) U.wr = grow(U.wr, &U.cwr, sizeof *U.wr);
        U.wr[U.nwr++] = (Wrote){addr, addr + n, fence};
    }
    mark(addr, n);
}

void vk_usnap_drop(uint64_t addr, uint64_t n) { drop(addr, n, U.fence); }

// vk_usnap_drop of each range [r[i][0], r[i][1]) (vk_exec_draw.c reset_batch, TOTK_RESET_BATCH): one take of drop_lock
// for them all instead of one each, while the slice threads take it for their own drops. The ranges go into U.wr and
// mark their copies under that one hold whatever TOTK_USNAP_DROP_LOCAL says: wr_local is not called here, because a
// thread's first wr_local takes drop_lock to link its list and would deadlock under it. U.wr and the threads' lists
// may hold a submission's writes side by side (wr_take appends the lists in any order; wr_prune sorts what was added).
void vk_usnap_drop_many(const uint64_t (*r)[2], uint32_t n) {
    if (!U.mode || !n) return;
    plat_mutex_lock(&drop_lock);
    for (uint32_t i = 0; i < n; i++)
        if (r[i][1] > r[i][0]) drop_locked(r[i][0], r[i][1] - r[i][0], U.fence);
    plat_mutex_unlock(&drop_lock);
}

// Whether a GPU write noted here (vk_usnap_drop: shader stores, copies, fills, reports, uniform updates, indirect
// resets) whose fence is not done overlaps [addr, addr+n): guest memory there may not hold the GPU's bytes yet. The
// sorted prefix of the last prune and the unsorted tail are both looked at, under drop_lock as the slices add to them,
// and with TOTK_USNAP_DROP_LOCAL the threads' own lists, each under its lock (wr_local).
int vk_usnap_gpu_writing(uint64_t addr, uint64_t n) {
    if (!n) return 0;
    const uint64_t done = vk_sync_fence_done(), hi = addr + n;
    int hit = 0;
    plat_mutex_lock(&drop_lock);
    int a = 0, b = nwr_sorted < U.nwr ? nwr_sorted : U.nwr;
    while (a < b) {   // the first sorted entry ending above addr
        int k = (a + b) / 2;
        if (U.wr[k].hi <= addr) a = k + 1;
        else b = k;
    }
    for (int end = nwr_sorted < U.nwr ? nwr_sorted : U.nwr; !hit && a < end && U.wr[a].lo < hi; a++) hit = U.wr[a].fence > done;
    for (int i = nwr_sorted; !hit && i < U.nwr; i++) hit = U.wr[i].lo < hi && addr < U.wr[i].hi && U.wr[i].fence > done;
    for (WrLocal *w = wr_locals; !hit && w; w = w->next) {   // linked under drop_lock
        plat_mutex_lock(&w->lock);
        for (int i = 0; !hit && i < w->n; i++) hit = w->a[i].lo < hi && addr < w->a[i].hi && w->a[i].fence > done;
        plat_mutex_unlock(&w->lock);
    }
    plat_mutex_unlock(&drop_lock);
    return hit;
}

uint64_t vk_usnap_gen(void) { return __atomic_load_n(&U.gen, __ATOMIC_ACQUIRE); }
uint64_t vk_usnap_arena_bytes(void) { return __atomic_load_n(&arena_total, __ATOMIC_RELAXED); }

// The arenas back on the free list (vk_exec_draw_shutdown: the device is idle, every completion ran).
void vk_usnap_shutdown(void) {
    plat_mutex_lock(&free_lock);
    for (Arena *a = free_arenas, *next; a; a = next) next = a->next, vmaDestroyBuffer(vk_ctx()->vma, a->buf, a->alloc), free(a);
    free_arenas = NULL;
    __atomic_store_n(&arena_total, 0, __ATOMIC_RELAXED);
    plat_mutex_unlock(&free_lock);
}
