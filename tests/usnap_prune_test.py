"""runtime/vk_usnap.c wr_prune against a brute-force search: the GPU write ranges of the submissions in flight are kept
sorted and merged, and wr_prune sorts only the ranges added since the last prune before merging them in. Over random
submissions (ranges of 16 bytes to 4 KiB, a completed fence that moves forward), every range a write still in flight
overlaps must be found by wr_hits, and the kept list must stay sorted and disjoint. wr_prune, sort_new_writes and
wr_hits are cut out of the runtime source. No GPU or game runs.
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
static void rt_fatal(const char *m) { puts(m); abort(); }
#define RADIX_BITS 11
typedef struct { uint64_t lo, hi, fence; } Wrote;
static struct { Wrote *wr; int nwr, cwr; } U;
static uint64_t done_fence;
static uint64_t vk_sync_fence_done(void) { return done_fence; }
#define WROTE_LESS(x, y) ((x)->lo < (y)->lo)
MSORT_DEFINE(sort_wrote, Wrote, WROTE_LESS)
static int nwr_sorted;
'''

CASES = r'''
static uint64_t rnd(void) { static uint64_t x = 88172645463325252ull; x ^= x << 13, x ^= x >> 7, x ^= x << 17; return x; }
static Wrote all[1000000];
int main(void) {
    int nall = 0, fails = 0, checks = 0;
    U.cwr = 2000000, U.wr = malloc(sizeof *U.wr * (size_t)U.cwr);
    if (!U.wr) return 2;
    for (uint64_t fence = 1; fence <= 300; fence++) {
        int add = (int)(rnd() % 3000);
        uint64_t base = rnd() % 4 ? 0x100000000ull : 0x7000000000ull * (1 + rnd() % 3);   /* spans past 128 GiB too */
        for (int i = 0; i < add; i++) {
            uint64_t lo = base + rnd() % (1 << 20) * 16, n = 16 + rnd() % 4096;
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
            uint64_t lo = base + rnd() % (1 << 20) * 16, hi = lo + 1 + rnd() % 8192;
            int ref = 0;
            for (int i = 0; i < nall && !ref; i++) ref = all[i].fence > done_fence && all[i].lo < hi && lo < all[i].hi;
            if (ref && !wr_hits(lo, hi)) fails++;   /* a merged range may keep a completed part: only misses fail */
        }
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
                         ('static void sort_new_writes(', 'static void wr_prune(', 'static int wr_hits(')] + [CASES]
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
