"""runtime/vk_usnap.c wr_prune against a brute-force search: the GPU write ranges of the submissions in flight are kept
sorted and merged (a write whose fence completed leaves, merged or not), and wr_prune sorts only the ranges added since the last prune before merging them in. Over random
submissions (ranges of 4 bytes to 4 KiB, half of them not 16-byte aligned, a completed fence that moves forward),
every range a write still in flight overlaps must be found by wr_hits, and the kept list must stay sorted and disjoint.
A first case puts two unaligned ranges in one 16-byte bucket of the radix sort, the higher one added first.
Every case runs with U.fast 0 and 1 (TOTK_USNAP_FAST): the fast path sorts in as few 11-bit radix passes as the span
needs, so sort_new_writes must give the same array either way, checked at spans of 2047, 2048, 2^22 - 1 and 2^22
buckets, where the pass count changes; sort_col, which has no insertion pass to repair a short sort, must sort its
ranges at the same spans. add_range must keep exactly the distinct ranges it is given. The coalescing in drop() (a write equal to the newest unsorted entry raises its
fence instead of adding one) must leave wr_prune with the same list as the separate entries do.
TOTK_USNAP_DROP_LOCAL (drop_local_cases): writes noted by six threads into lists of their own (wr_local) and moved into
U.wr at the next submission's start (wr_take, the lists in a shuffled order) must leave wr_prune with the same list,
and the summary with the same counts, as the same writes made in their global order into U.wr.
vk_usnap_drop_many (drop_many_cases, TOTK_RESET_BATCH): an indirect reset's ranges added to U.wr under one take of
drop_lock while the threads note theirs in their own lists must give the same list and counts as the global order, and
mark every range. The fixture's lock fails a second take while held, as a thread's first wr_local under drop_lock
would deadlock.
add_range, by_lo, sort_col, radix_passes, sort_new_writes, wr_prune, wr_hits, wr_local, wr_take, drop_locked, drop and
vk_usnap_drop_many are cut out of the runtime source (the marking of copies, mark and vk_usnap_covers, is stubbed: no
copy is taken here; mark counts its calls). No GPU or game runs.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r'''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "msort.h"
static void rt_fatal(const char *m) { puts(m); fflush(stdout); abort(); }
#define RADIX_BITS 11
typedef struct { uint64_t lo, hi, fence; } Wrote;
typedef struct { uint64_t lo, hi; } Rng;
#define ALIGN 256u
static struct { Wrote *wr; int nwr, cwr; int fast; Rng *col; int ncol, ccol; uint64_t repeats; int mode, on;
                uint64_t wr_added, wr_coalesced, fence; } U;
static void *grow(void *p, int *cap, size_t elem) {
    int n = *cap ? *cap * 2 : 1024;
    void *q = realloc(p, (size_t)n * elem);
    if (!q) rt_fatal("out of memory");
    *cap = n;
    return q;
}
#define SEEN_BITS 8
static Rng seen[1 << SEEN_BITS];
static uint64_t done_fence;
static uint64_t vk_sync_fence_done(void) { return done_fence; }
#define WROTE_LESS(x, y) ((x)->lo < (y)->lo)
MSORT_DEFINE(sort_wrote, Wrote, WROTE_LESS)
static int nwr_sorted;
typedef int PlatMutex;
static PlatMutex drop_lock;
static void plat_mutex_lock(PlatMutex *m) { if ((*m)++) rt_fatal("FAIL: a lock taken while held (a deadlock)"); }
static void plat_mutex_unlock(PlatMutex *m) { (*m)--; }
static int drop_local;
static int vk_usnap_covers(uint64_t addr, uint64_t n) { (void)addr, (void)n; return 0; }
static uint64_t marks;
static void mark(uint64_t addr, uint64_t n) { (void)addr, (void)n, marks++; }
'''

CASES = r'''
static uint64_t rnd(void) { static uint64_t x = 88172645463325252ull; x ^= x << 13, x ^= x >> 7, x ^= x << 17; return x; }
static Wrote all[1000000];
static int checks;

static int prune_cases(void) {
    int nall = 0, fails = 0;
    U.nwr = 0, nwr_sorted = 0, done_fence = 0;
    // Two unaligned ranges in one 16-byte bucket, the higher added first: both must survive the prune.
    U.wr[U.nwr++] = (Wrote){0x1008, 0x1010, 1};
    U.wr[U.nwr++] = (Wrote){0x1000, 0x1004, 1};
    wr_prune();
    checks += 2;
    if (!wr_hits(0x1000, 0x1004)) fails++, puts("FAIL: [0x1000,0x1004) lost in its bucket");
    if (!wr_hits(0x1008, 0x1010)) fails++, puts("FAIL: [0x1008,0x1010) lost");
    // No write outlives its fence: a completed one is gone, and none is absorbed into a range that stays.
    U.nwr = 0, nwr_sorted = 0, done_fence = 1000;
    U.wr[U.nwr++] = (Wrote){0x5000, 0x6000, 1200};
    U.wr[U.nwr++] = (Wrote){0x5800, 0x7000, 5};
    wr_prune();
    wr_prune();
    checks += 2;
    if (!wr_hits(0x5400, 0x5500)) fails++, puts("FAIL: a write in flight left the list");
    if (U.nwr != 1 || U.wr[0].hi != 0x6000) fails++, puts("FAIL: a finished write was merged into a range in flight");
    done_fence = 0;
    U.nwr = 0, nwr_sorted = 0;
    for (uint64_t fence = 1; fence <= 300; fence++) {
        int add = (int)(rnd() % 3000);
        uint64_t base = rnd() % 4 ? 0x100000000ull : 0x7000000000ull * (1 + rnd() % 3);   /* spans past 128 GiB too */
        for (int i = 0; i < add; i++) {
            // Half the ranges start anywhere, as inline updates, copies and query results do; half on 16 bytes.
            uint64_t lo = base + (rnd() % 2 ? rnd() % (1 << 24) : rnd() % (1 << 20) * 16), n = 4 + rnd() % 4096;
            Wrote w = {lo, lo + n, fence};
            U.wr[U.nwr++] = w, all[nall++] = w;
        }
        if (rnd() % 2) {
            uint64_t d = fence - rnd() % 4;
            if (d > done_fence && d <= fence) done_fence = d;
        }
        wr_prune();
        for (int i = 1; i < U.nwr; i++, checks++)
            if (U.wr[i - 1].hi > U.wr[i].lo) fails++;   /* sorted and disjoint */
        for (int q = 0; q < 300; q++, checks++) {
            uint64_t lo = base + rnd() % (1 << 24), hi = lo + 1 + rnd() % 8192;
            int ref = 0;
            for (int i = 0; i < nall && !ref; i++) ref = all[i].fence > done_fence && all[i].lo < hi && lo < all[i].hi;
            if (ref && !wr_hits(lo, hi)) fails++;   /* a merged range may keep a completed part: only misses fail */
        }
    }
    return fails;
}

// sort_new_writes in each mode over spans where the fast pass count changes (1 to 2 passes at 2048 buckets, 2 to 3
// at 2^22): both modes must give the same array, sorted by lo. The lowest and highest bucket are always present, so
// the span is exactly the one named.
static Wrote a0[20000], a1[20000];
static int sort_cases(void) {
    static const uint64_t spans[] = {2047, 2048, (1u << 22) - 1, 1u << 22};
    static Wrote *tmp;
    static size_t cap;
    int fails = 0;
    for (int s = 0; s < 4; s++)
        for (int round = 0; round < 8; round++) {
            int n = 2 + (int)(rnd() % 19998);
            uint64_t base = 0x200000000ull + rnd() % 4096 * 16;
            for (int i = 0; i < n; i++) {
                uint64_t k = i == 0 ? 0 : i == 1 ? spans[s] : rnd() % (spans[s] + 1);
                uint64_t lo = base + k * 16 + rnd() % 16;   // unaligned inside the bucket, as in prune_cases
                a0[i] = (Wrote){lo, lo + 4 + rnd() % 64, (uint64_t)i};
            }
            a0[0].lo = base, a0[0].hi = base + 4;   // the lowest key is 0 in each mode
            a0[1].lo = base + spans[s] * 16, a0[1].hi = a0[1].lo + 4;
            memcpy(a1, a0, (size_t)n * sizeof *a0);
            U.fast = 0, sort_new_writes(a0, n, &tmp, &cap);
            U.fast = 1, sort_new_writes(a1, n, &tmp, &cap);
            checks += 2;
            if (memcmp(a0, a1, (size_t)n * sizeof *a0)) fails++, printf("FAIL: span %llu: the fast sort differs\n",
                                                                         (unsigned long long)spans[s]);
            for (int i = 1; i < n; i++)
                if (a1[i - 1].lo > a1[i].lo) { fails++, printf("FAIL: span %llu: not sorted\n",
                                                             (unsigned long long)spans[s]); break; }
        }
    return fails;
}

// The coalescing in drop(): a write with the same range as the newest entry of the unsorted tail raises that entry's
// fence. Over random writes that often repeat the previous range, the list wr_prune builds from the coalesced entries
// must equal the list it builds from every write, prune after prune.
static Wrote sep[200000], coal[200000];
static int coalesce_cases(void) {
    int fails = 0, nsep = 0, ncoal = 0, sorted_sep = 0, sorted_coal = 0;
    Wrote *wr = U.wr;
    uint64_t lo = 0x300000000ull, n = 64;
    done_fence = 0;
    for (uint64_t fence = 1; fence <= 200; fence++) {
        for (int i = 0, add = (int)(rnd() % 600); i < add; i++) {
            if (rnd() % 3) lo = 0x300000000ull + rnd() % (1 << 16) * 16, n = 16 + rnd() % 512;
            uint64_t f = fence - (rnd() % 4 == 0 && fence > 1);   // a later write may carry an older fence
            Wrote w = {lo, lo + n, f};
            sep[nsep++] = w;
            Wrote *last = ncoal > sorted_coal ? &coal[ncoal - 1] : NULL;
            if (last && last->lo == w.lo && last->hi == w.hi) {
                if (w.fence > last->fence) last->fence = w.fence;
            } else {
                coal[ncoal++] = w;
            }
        }
        if (rnd() % 2 && fence > 3) done_fence = fence - 1 - rnd() % 3;
        U.wr = sep, U.nwr = nsep, nwr_sorted = sorted_sep, wr_prune(), nsep = sorted_sep = U.nwr;
        U.wr = coal, U.nwr = ncoal, nwr_sorted = sorted_coal, wr_prune(), ncoal = sorted_coal = U.nwr;
        checks++;
        if (nsep != ncoal || memcmp(sep, coal, (size_t)nsep * sizeof *sep)) {
            fails++, printf("FAIL: fence %llu: the coalesced list differs\n", (unsigned long long)fence);
            break;
        }
    }
    U.wr = wr, U.nwr = 0, nwr_sorted = 0;
    return fails;
}

// sort_col at the spans where the fast pass count changes, in ALIGN units: its LSD passes alone must sort, since
// nothing repairs a short sort afterwards. Both modes must agree, and the result must be sorted by lo.
static Rng c0[20000];
static int col_cases(void) {
    static const uint64_t spans[] = {2047, 2048, (1u << 22) - 1, 1u << 22};
    int fails = 0;
    if (U.ccol < 20000) U.col = realloc(U.col, 20000 * sizeof *U.col), U.ccol = 20000;
    if (!U.col) return 1;
    for (int s = 0; s < 4; s++)
        for (int round = 0; round < 8; round++) {
            int n = 2 + (int)(rnd() % 19998);
            uint64_t base = 0x200000000ull + rnd() % 4096 * ALIGN;
            for (int i = 0; i < n; i++) {
                uint64_t k = i == 0 ? 0 : i == 1 ? spans[s] : rnd() % (spans[s] + 1);
                c0[i] = (Rng){base + k * ALIGN, base + k * ALIGN + 16 + rnd() % 4096};
            }
            Rng *out[2];
            static Rng r[2][20000];
            for (int fast = 0; fast < 2; fast++) {
                U.fast = fast, U.ncol = n;
                memcpy(U.col, c0, (size_t)n * sizeof *c0);
                sort_col();
                memcpy(r[fast], U.col, (size_t)n * sizeof *c0), out[fast] = r[fast];
            }
            checks += 2;
            if (memcmp(out[0], out[1], (size_t)n * sizeof *c0)) fails++, printf("FAIL: sort_col span %llu: the modes differ\n",
                                                                           (unsigned long long)spans[s]);
            for (int i = 1; i < n; i++)
                if (out[1][i - 1].lo > out[1][i].lo) { fails++, printf("FAIL: sort_col span %llu: not sorted\n",
                                                                      (unsigned long long)spans[s]); break; }
        }
    U.ncol = 0;
    return fails;
}

// add_range under USNAP_FAST skips a range its cache saw earlier in the submission. Over ranges that often repeat
// (and often share lo with a different hi), the distinct ranges it kept must be exactly the distinct ranges given.
static Rng given[50000];
static int by_lohi(const void *a, const void *b) {
    const Rng *x = a, *y = b;
    return x->lo != y->lo ? (x->lo < y->lo ? -1 : 1) : x->hi != y->hi ? (x->hi < y->hi ? -1 : 1) : 0;
}
static int uniq(Rng *a, int n) {
    qsort(a, (size_t)n, sizeof *a, by_lohi);
    int m = 0;
    for (int i = 0; i < n; i++)
        if (!m || by_lohi(&a[m - 1], &a[i])) a[m++] = a[i];
    return m;
}
static int seen_cases(void) {
    int fails = 0;
    for (int round = 0; round < 20; round++) {
        int n = 1 + (int)(rnd() % 50000), pool = 1 + (int)(rnd() % 3000);
        U.fast = 1, U.ncol = 0, U.repeats = 0;
        memset(seen, 0, sizeof seen);
        for (int i = 0; i < n; i++) {
            uint64_t k = rnd() % (uint64_t)pool;
            uint64_t lo = 0x400000000ull + k * ALIGN * (1 + k % 7), hi = lo + 16 + (rnd() % 3) * 256;
            given[i] = (Rng){lo, hi};
            add_range(given[i]);
        }
        int kept = U.ncol;
        int a = uniq(given, n), b = uniq(U.col, U.ncol);
        checks++;
        if (a != b || memcmp(given, U.col, (size_t)a * sizeof *given) || (uint64_t)(kept + (int)U.repeats) != (uint64_t)n)
            fails++, printf("FAIL: add_range round %d kept %d distinct of %d\n", round, b, a);
    }
    U.fast = 0, U.ncol = 0;
    return fails;
}

// TOTK_USNAP_DROP_LOCAL: per submission, random writes (often repeating a thread's previous range, sometimes with an
// older fence) noted by six threads. The reference makes them in their global order with drop_local 0 (into U.wr, under
// the lock); the other with drop_local 1, each into its thread's list, then wr_take with the lists linked in a shuffled
// order. After wr_prune both lists must be equal, and so must the counts the summary reports.
#define THREADS 6
static Wrote ref_wr[400000], loc_wr[400000];
static int drop_local_cases(void) {
    int fails = 0, nref = 0, sref = 0, nloc = 0, sloc = 0;
    Wrote *wr = U.wr;
    int cwr = U.cwr;
    WrLocal *thr[THREADS] = {0};
    uint64_t prev_lo[THREADS] = {0}, prev_n[THREADS] = {0};
    U.mode = 2, done_fence = 0;
    for (WrLocal *l = wr_locals, *next; l; l = next) next = l->next, free(l->a), free(l);   // the last round's threads
    wr_locals = NULL;
    for (uint64_t fence = 1; fence <= 150; fence++) {
        static struct { int t; uint64_t lo, n, f; } w[3000];
        int nw = (int)(rnd() % 3000);
        for (int i = 0; i < nw; i++) {
            int t = (int)(rnd() % THREADS);
            if (!prev_n[t] || rnd() % 3 == 0) prev_lo[t] = 0x500000000ull + rnd() % (1 << 16) * 16 + rnd() % 2 * 4, prev_n[t] = 4 + rnd() % 512;
            w[i].t = t, w[i].lo = prev_lo[t], w[i].n = prev_n[t], w[i].f = fence - (rnd() % 5 == 0 && fence > 1);
        }
        uint64_t added0 = U.wr_added, coal0 = U.wr_coalesced;
        // the reference: U.wr in the global order
        U.wr = ref_wr, U.cwr = 400000, U.nwr = nref, nwr_sorted = sref, drop_local = 0;
        for (int i = 0; i < nw; i++) drop(w[i].lo, w[i].n, w[i].f);
        uint64_t added_ref = U.wr_added - added0;
        if (rnd() % 2 && fence > 3) done_fence = fence - 1 - rnd() % 3;
        wr_prune(), nref = sref = U.nwr;
        // the threads' lists
        U.wr = loc_wr, U.cwr = 400000, U.nwr = nloc, nwr_sorted = sloc, drop_local = 1;
        U.wr_added = added0, U.wr_coalesced = coal0;
        for (int i = 0; i < nw; i++) wr_mine = thr[w[i].t], drop(w[i].lo, w[i].n, w[i].f), thr[w[i].t] = wr_mine;
        WrLocal *order[THREADS];
        int nl = 0;
        for (WrLocal *l = wr_locals; l; l = l->next) order[nl++] = l;
        for (int i = nl - 1; i > 0; i--) {
            int j = (int)(rnd() % (uint64_t)(i + 1));
            WrLocal *x = order[i];
            order[i] = order[j], order[j] = x;
        }
        wr_locals = NULL;
        for (int i = 0; i < nl; i++) order[i]->next = wr_locals, wr_locals = order[i];
        wr_take();
        checks++;
        if (U.wr_added - added0 != added_ref) fails++, printf("FAIL: fence %llu: %llu writes counted, %llu in the reference\n",
                                                              (unsigned long long)fence, (unsigned long long)(U.wr_added - added0),
                                                              (unsigned long long)added_ref);
        wr_prune(), nloc = sloc = U.nwr;
        checks++;
        if (nref != nloc || memcmp(ref_wr, loc_wr, (size_t)nref * sizeof *ref_wr)) {
            fails++, printf("FAIL: fence %llu: the threads' lists give %d ranges, the global order %d\n",
                            (unsigned long long)fence, nloc, nref);
            break;
        }
    }
    U.wr = wr, U.cwr = cwr, U.nwr = 0, nwr_sorted = 0, drop_local = 0, wr_mine = NULL;
    return fails;
}

// vk_usnap_drop_many with TOTK_USNAP_DROP_LOCAL on: per submission, random writes of 20-byte records and runs of them,
// some noted by six threads into their lists, the rest by the indirect reset (thread THREADS) in one vk_usnap_drop_many
// call that adds them to U.wr under drop_lock. The reference makes all of them in their global order into U.wr. After
// wr_take and wr_prune both lists must be equal, so must the write counts, and every range of the reset is marked.
static int drop_many_cases(void) {
    int fails = 0, nref = 0, sref = 0, nloc = 0, sloc = 0;
    Wrote *wr = U.wr;
    int cwr = U.cwr;
    WrLocal *thr[THREADS] = {0};
    U.mode = 2, done_fence = 0;
    for (WrLocal *l = wr_locals, *next; l; l = next) next = l->next, free(l->a), free(l);   // the last round's threads
    wr_locals = NULL;
    for (uint64_t fence = 1; fence <= 150; fence++) {
        static struct { int t; uint64_t lo, n; } w[3000];
        static uint64_t runs[3000][2];
        int nw = (int)(rnd() % 3000), nr = 0;
        for (int i = 0; i < nw; i++)
            w[i].t = (int)(rnd() % (THREADS + 1)), w[i].lo = 0x600000000ull + rnd() % (1 << 14) * 20,
            w[i].n = 20 * (1 + rnd() % 4);
        U.fence = fence;
        uint64_t added0 = U.wr_added, coal0 = U.wr_coalesced;
        U.wr = ref_wr, U.cwr = 400000, U.nwr = nref, nwr_sorted = sref, drop_local = 0;
        for (int i = 0; i < nw; i++) drop(w[i].lo, w[i].n, fence);
        uint64_t added_ref = U.wr_added - added0;
        if (rnd() % 2 && fence > 3) done_fence = fence - 1 - rnd() % 3;
        wr_prune(), nref = sref = U.nwr;
        U.wr = loc_wr, U.cwr = 400000, U.nwr = nloc, nwr_sorted = sloc, drop_local = 1;
        U.wr_added = added0, U.wr_coalesced = coal0;
        for (int i = 0; i < nw; i++)
            if (w[i].t == THREADS) runs[nr][0] = w[i].lo, runs[nr][1] = w[i].lo + w[i].n, nr++;
            else wr_mine = thr[w[i].t], drop(w[i].lo, w[i].n, fence), thr[w[i].t] = wr_mine;
        uint64_t marks0 = marks;
        wr_mine = NULL;   // the replay thread, which may have no list of its own yet
        vk_usnap_drop_many((const uint64_t (*)[2])runs, (uint32_t)nr);
        checks++;
        if (marks - marks0 != (uint64_t)nr) fails++, printf("FAIL: fence %llu: %llu of %d reset ranges marked\n",
                                                            (unsigned long long)fence, (unsigned long long)(marks - marks0), nr);
        wr_take();
        checks++;
        if (U.wr_added - added0 != added_ref) fails++, printf("FAIL: fence %llu: %llu writes counted, %llu in the reference\n",
                                                              (unsigned long long)fence, (unsigned long long)(U.wr_added - added0),
                                                              (unsigned long long)added_ref);
        wr_prune(), nloc = sloc = U.nwr;
        checks++;
        if (nref != nloc || memcmp(ref_wr, loc_wr, (size_t)nref * sizeof *ref_wr)) {
            fails++, printf("FAIL: fence %llu: the reset and the threads' lists give %d ranges, the global order %d\n",
                            (unsigned long long)fence, nloc, nref);
            break;
        }
    }
    U.wr = wr, U.cwr = cwr, U.nwr = 0, nwr_sorted = 0, drop_local = 0, wr_mine = NULL;
    return fails;
}

int main(void) {
    int fails = 0;
    U.cwr = 2000000, U.wr = malloc(sizeof *U.wr * (size_t)U.cwr);
    if (!U.wr) return 2;
    for (int fast = 0; fast < 2; fast++) {
        U.fast = fast;
        fails += prune_cases();
    }
    fails += sort_cases();
    fails += col_cases();
    fails += seen_cases();
    for (int fast = 0; fast < 2; fast++) {
        U.fast = fast;
        fails += coalesce_cases();
        fails += drop_local_cases();
        fails += drop_many_cases();
    }
    printf("usnap prune: %d checks, %d failures\n", checks, fails);
    return fails != 0;
}
'''


def function(text, start):
    i = text.index(start)
    depth, k = 0, text.index('{', i)
    while True:
        depth += text[k] == '{'
        depth -= text[k] == '}'
        if text[k] == '}' and not depth:
            return text[i:k + 1]
        k += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'build' / 'tests' / 'usnap-prune')
    args = parser.parse_args()
    source = (ROOT / 'runtime' / 'vk_usnap.c').read_text(encoding='utf-8')
    parts = [PRELUDE] + [function(source, start) for start in
                         ('static void add_range(', 'static int by_lo(', 'static int radix_passes(', 'static void sort_col(',
                          'static void sort_new_writes(', 'static void wr_prune(', 'static int wr_hits(')]
    start = source.index('typedef struct WrLocal')
    end = source.index('static _Thread_local WrLocal *wr_mine;', start) + len('static _Thread_local WrLocal *wr_mine;')
    parts += [source[start:end]] + [function(source, s) for s in ('static void wr_local(', 'static void wr_take(void) {')]
    # drop_locked holds the body drop takes the lock around (TOTK_RESET_BATCH); the source declares it before drop
    # and defines it after, so its definition is cut out by its full head and put first.
    parts += [function(source, s) for s in ('static void drop_locked(uint64_t addr, uint64_t n, uint64_t fence) {',
                                            'static void drop(', 'void vk_usnap_drop_many(')] + [CASES]
    args.out.mkdir(parents=True, exist_ok=True)
    src, exe = args.out / 'prune.c', args.out / 'prune.exe'
    src.write_text('\n'.join(parts), encoding='utf-8')
    for opt in ('-O0', '-O2'):
        subprocess.run([str(args.cc), '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'), str(src),
                        '-o', str(exe)], check=True, timeout=120)
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
        print(opt, run.stdout.strip())
        if run.returncode:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
