"""runtime/vk_usnap.c drop numbering for deferred sliced passes (vk_exec.c TOTK_SLICE_DEFER). The slices of a deferred
pass are recorded while the replay thread replays the ops after it, and a slice must not see the uniform snapshot drops
those ops make. A thread that set vk_usnap_limit(seq) sees only the drops numbered up to seq, until it drops a range
itself. The cases: drops before and after the limit, a slice's own drop, a limit on one thread and drops on another,
and numbers that wrap past 2^32. first_above, dropped_by, vk_usnap_seq, vk_usnap_limit, vk_usnap_find and vk_usnap_drop
are cut out of the runtime source. No GPU or game runs.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r'''#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { void *buffer; uint64_t offset, range; } VkDescriptorBufferInfo;
typedef pthread_mutex_t PlatMutex;
#define PLAT_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
static void plat_mutex_lock(PlatMutex *m) { pthread_mutex_lock(m); }
static void plat_mutex_unlock(PlatMutex *m) { pthread_mutex_unlock(m); }
static void rt_fatal(const char *m) { puts(m); abort(); }
static int vk_spec_helper(void) { return 0; }
typedef struct { void *buf; } Arena;
typedef struct { uint64_t lo, hi, off; uint32_t dropped; } Reg;
typedef struct { uint64_t lo, hi, fence; } Wrote;
static struct {
    int mode, on;
    uint64_t fence, gen;
    Arena *arena;
    Reg *reg;
    int nreg;
    Wrote *wr;
    int nwr, cwr;
    uint64_t hits, drops, wr_added;
} U;
static void gen_bump(void) { __atomic_add_fetch(&U.gen, 1, __ATOMIC_RELEASE); }
static void *grow(void *p, int *cap, size_t elem) {
    int n = *cap ? *cap * 2 : 1024;
    void *q = realloc(p, (size_t)n * elem);
    if (!q) rt_fatal("out of memory");
    *cap = n;
    return q;
}
static _Thread_local int find_quiet;
static PlatMutex drop_lock = PLAT_MUTEX_INIT;
'''

CASES = r'''
static int fails, checks;
static void expect(int got, int want, const char *what) {
    checks++;
    if (got != want) fails++, printf("FAIL %s: %d, want %d\n", what, got, want);
}
static int found(uint64_t addr) { VkDescriptorBufferInfo b; return vk_usnap_find(addr, 16, 1, &b); }

static Arena arena;
static void begin(uint32_t seq) {   // a submission with four copied regions 0x1000 apart; drop_seq starts at seq
    static Reg reg[4];
    for (int i = 0; i < 4; i++) reg[i] = (Reg){0x10000 + 0x1000u * i, 0x10100 + 0x1000u * i, 0x100u * i, 0};
    U.reg = reg, U.nreg = 4, U.on = 1, U.mode = 1, U.arena = &arena;
    __atomic_store_n(&drop_seq, seq, __ATOMIC_RELEASE);
}

static uint32_t worker_limit;
static int worker_sees[4];
static pthread_barrier_t gate;
static void *worker(void *arg) {   // a slice thread: sets its limit, waits while the walk drops, then looks up
    (void)arg;
    vk_usnap_limit(worker_limit);
    pthread_barrier_wait(&gate);
    pthread_barrier_wait(&gate);
    for (int i = 0; i < 4; i++) worker_sees[i] = found(0x10000 + 0x1000u * i);
    vk_usnap_limit(0);
    return NULL;
}

int main(void) {
    static const uint32_t starts[] = {0, 1000, 0x7ffffffeu, 0xfffffffdu, 0xffffffffu};
    for (unsigned s = 0; s < sizeof starts / sizeof *starts; s++) {
        char what[96];
        // The walk drops region 0, a deferred pass starts (its limit), and the walk drops region 1 after the pass.
        begin(starts[s]);
        vk_usnap_drop(0x10000, 16);
        uint32_t limit = vk_usnap_seq();
        vk_usnap_drop(0x11000, 16);
        expect(found(0x10000), 0, "walk: a drop before the pass");
        expect(found(0x11000), 0, "walk: a drop after the pass");
        vk_usnap_limit(limit);
        snprintf(what, sizeof what, "slice from %#x: a drop before its pass", starts[s]);
        expect(found(0x10000), 0, what);
        snprintf(what, sizeof what, "slice from %#x: a drop after its pass", starts[s]);
        expect(found(0x11000), 1, what);
        expect(found(0x12000), 1, "slice: a region nobody dropped");
        vk_usnap_drop(0x13000, 16);   // the slice's own drop: from now on it sees every drop
        expect(found(0x13000), 0, "slice: its own drop");
        expect(found(0x11000), 0, "slice: after its own drop, the walk's later drop");
        vk_usnap_limit(0);

        // The same with the slice on another thread while the walk drops: the limit is the thread's own.
        begin(starts[s]);
        vk_usnap_drop(0x10000, 16);
        worker_limit = vk_usnap_seq();
        pthread_barrier_init(&gate, NULL, 2);
        pthread_t t;
        if (pthread_create(&t, NULL, worker, NULL)) return 2;
        pthread_barrier_wait(&gate);
        vk_usnap_drop(0x11000, 16);
        vk_usnap_drop(0x12000, 16);
        pthread_barrier_wait(&gate);
        pthread_join(t, NULL);
        pthread_barrier_destroy(&gate);
        snprintf(what, sizeof what, "thread from %#x: a drop before its pass", starts[s]);
        expect(worker_sees[0], 0, what);
        snprintf(what, sizeof what, "thread from %#x: drops after its pass", starts[s]);
        expect(worker_sees[1] && worker_sees[2], 1, what);
        expect(worker_sees[3], 1, "thread: a region nobody dropped");
        expect(found(0x11000) || found(0x12000), 0, "walk: its own drops, while a slice thread has a limit");
    }
    expect(dropped_by(0, 5), 0, "a region not dropped");
    expect(dropped_by(0xfffffff0u, 3), 1, "a drop numbered before the wrap, a limit after it");
    expect(dropped_by(3, 0xfffffff0u), 0, "a drop numbered after the wrap, a limit before it");
    printf("usnap seq: %d checks, %d failures\n", checks, fails);
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


def line(text, start):
    i = text.index(start)
    return text[i:text.index('\n', i)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'build' / 'tests' / 'usnap-seq')
    args = parser.parse_args()
    source = (ROOT / 'runtime' / 'vk_usnap.c').read_text(encoding='utf-8')
    parts = [PRELUDE,
             line(source, 'static uint32_t drop_seq;'),
             line(source, 'static _Thread_local uint32_t find_limit;'),
             line(source, 'static int dropped_by('),
             line(source, 'uint32_t vk_usnap_seq(void)'),
             line(source, 'void vk_usnap_limit(uint32_t seq)')]
    parts += [function(source, start) for start in
              ('static int first_above(', 'int vk_usnap_find(', 'void vk_usnap_drop(')] + [CASES]
    args.out.mkdir(parents=True, exist_ok=True)
    src, exe = args.out / 'seq.c', args.out / 'seq.exe'
    src.write_text('\n'.join(parts), encoding='utf-8')
    for opt in ('-O0', '-O2'):
        subprocess.run([str(args.cc), '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror', str(src), '-pthread',
                        '-o', str(exe)], check=True, timeout=120)
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
        print(opt, run.stdout.strip())
        if run.returncode:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
