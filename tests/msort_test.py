"""runtime/msort.h against the CRT's qsort: random, sorted, reversed, nearly sorted and all-equal lists of 0 to 5000
ranges keyed like copy_ranges (pool, then offset) must come out in the same order of keys, with every element kept
and equal keys in their input order (the sort is stable). No GPU or game runs.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "msort.h"
typedef struct { uintptr_t r; uint64_t off; uint32_t seq; } R;
#define LESS(x, y) ((x)->r != (y)->r ? (x)->r < (y)->r : (x)->off < (y)->off)
MSORT_DEFINE(sort_r, R, LESS)
static int cmp(const void *x, const void *y) {
    const R *a = x, *b = y;
    if (a->r != b->r) return a->r < b->r ? -1 : 1;
    if (a->off != b->off) return a->off < b->off ? -1 : 1;
    return a->seq < b->seq ? -1 : a->seq > b->seq;   /* stable reference */
}
static uint64_t rnd = 88172645463325252ull;
static uint64_t next(void) { rnd ^= rnd << 13, rnd ^= rnd >> 7, rnd ^= rnd << 17; return rnd; }
int main(void) {
    static R a[5000], b[5000];
    R *buf = NULL;
    size_t cap = 0;
    int fails = 0, checks = 0;
    const size_t sizes[] = {0, 1, 2, 31, 32, 33, 64, 100, 1000, 4097, 5000};
    for (int kind = 0; kind < 5; kind++)
        for (size_t s = 0; s < sizeof sizes / sizeof *sizes; s++) {
            size_t n = sizes[s];
            for (size_t i = 0; i < n; i++) {
                uint64_t v = kind == 0 ? next() : kind == 1 ? i : kind == 2 ? n - i : kind == 3 ? (i ^ (next() % 3 == 0)) : 7;
                a[i] = (R){(uintptr_t)(v % 5), v / 5 % 50, (uint32_t)i};
                b[i] = a[i];
            }
            sort_r(a, n, &buf, &cap);
            qsort(b, n, sizeof *b, cmp);
            for (size_t i = 0; i < n; i++, checks++)
                if (a[i].r != b[i].r || a[i].off != b[i].off || a[i].seq != b[i].seq) { fails++; break; }
        }
    free(buf);
    printf("msort: %d checks, %d failures\n", checks, fails);
    return fails != 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'build' / 'tests' / 'msort')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src, exe = args.out / 'msort.c', args.out / 'msort.exe'
    src.write_text(HARNESS, encoding='utf-8')
    for opt in ('-O0', '-O2'):
        subprocess.run([str(args.cc), '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'), str(src),
                        '-o', str(exe)], check=True, timeout=120)
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        print(opt, run.stdout.strip())
        if run.returncode:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
