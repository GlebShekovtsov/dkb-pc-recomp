"""CPU regression for the replay's written-range list.

The actual note_written, its hash and grow run against a plain list that searches every entry, the behaviour the table
replaced. Each call must give the same answer as that list: every distinct address and byte size once, in first-write
order, with one vk_mem_gpu_pending for each. The lazy flag stays only while every write is lazy, and the buffer the GPU
wrote stays only while every write of the range named the same one. No Vulkan, guest memory or driver is used.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from nvn2_compute_c0_test import function

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The draw context of the replay thread (vk_exec_draw.c DrawCtx), whose copy of the translator is no helper's: the
// helper check never fails.
typedef struct DrawCtx { char none; } DrawCtx;
#define DC_OF(ex) ((ex)->draw)
#define IS_HELPER(dc) ((void)(dc), 0)
typedef struct VkExec { DrawCtx *draw; } VkExec;
static DrawCtx ctx;
static VkExec exec = {&ctx};
typedef struct VkBuffer_T *VkBuffer;
#define VK_NULL_HANDLE NULL
#define VK_WB_LAZY (1ull << 63)   // vk.h: or-ed into a range's size; it travels in the written list only
typedef struct { uint64_t addr, size; VkBuffer buf; } Range;
typedef struct {
    Range *written;
    int nwritten, cwritten;
    uint32_t *written_slot;
    uint32_t written_slots;
} CbRes;

static CbRes res;
static Range *pending;
static int npending, cpending;
static unsigned failures;

static void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
enum { VK_SPEC_WRITE = 1 };
static int vk_spec_fail(int why) { (void)why; abort(); }
static CbRes *cb_res(VkExec *ex) { (void)ex; return &res; }
@GROW@
static void vk_mem_gpu_pending(uint64_t addr, uint64_t size) {
    if (npending == cpending) pending = grow(pending, &cpending, sizeof *pending);
    pending[npending++] = (Range){addr, size, VK_NULL_HANDLE};
}
@NOTE@

static Range *model;
static int nmodel, cmodel;

static void note_model(uint64_t addr, uint64_t size, VkBuffer buf) {
    for (int i = 0; i < nmodel; i++)
        if (model[i].addr == addr && (model[i].size & ~VK_WB_LAZY) == (size & ~VK_WB_LAZY)) {
            if (!(size & VK_WB_LAZY)) model[i].size &= ~VK_WB_LAZY;
            if (model[i].buf != buf) model[i].buf = VK_NULL_HANDLE;
            return;
        }
    if (nmodel == cmodel) model = grow(model, &cmodel, sizeof *model);
    model[nmodel++] = (Range){addr, size, buf};
}

// Two buffers the GPU may write a range through; note uses the first.
static char buffer_a, buffer_b;
#define BUF_A ((VkBuffer)(void *)&buffer_a)
#define BUF_B ((VkBuffer)(void *)&buffer_b)

static void note_buf(uint64_t addr, uint64_t size, VkBuffer buf) {
    note_written(&exec, addr, size, buf);
    note_model(addr, size, buf);
}
static void note(uint64_t addr, uint64_t size) { note_buf(addr, size, BUF_A); }

static void check(const char *name) {
    int same = res.nwritten == nmodel && npending == nmodel;
    for (int i = 0; same && i < nmodel; i++)
        same = res.written[i].addr == model[i].addr && res.written[i].size == model[i].size &&
               res.written[i].buf == model[i].buf && pending[i].addr == model[i].addr &&
               pending[i].size == (model[i].size & ~VK_WB_LAZY);   // the flag stays in the written list
    int table = res.written_slots && !(res.written_slots & (res.written_slots - 1)) &&
                (uint32_t)res.nwritten < res.written_slots;
    if (!same || !table) failures++, fprintf(stderr, "FAIL %s: %d ranges, %d in the model\n", name, res.nwritten, nmodel);
}

static void reset(void) {
    free(res.written), free(res.written_slot), free(pending), free(model);
    memset(&res, 0, sizeof res);
    pending = NULL, npending = 0, cpending = 0;
    model = NULL, nmodel = 0, cmodel = 0;
}

static uint64_t rng = 0x243f6a8885a308d3ull;
static uint64_t next(void) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return rng;
}

int main(void) {
    if (written_hash(0x7000, 0x80) != written_hash(0x7000, 0x80 | VK_WB_LAZY))
        failures++, fprintf(stderr, "FAIL the lazy flag changes the range's hash\n");

    note(0x1000, 0x40), note(0x1000, 0x40);
    check("a repeated range counts once");
    note(0x1000, 0x80), note(0x2000, 0x40), note(0x1000, 0x40);
    check("address and size each tell ranges apart");
    note(0, 0), note(0, 0), note(UINT64_MAX, UINT64_MAX), note(UINT64_MAX, UINT64_MAX);
    check("the extreme values are ordinary keys");
    reset();

    // The buffer stays while every write of a range names it; one write through another buffer, or through none, drops
    // it for good, since the range then holds bytes the first buffer did not write.
    note_buf(0x3000, 0x40, BUF_A), note_buf(0x3000, 0x40, BUF_A);
    note_buf(0x4000, 0x40, BUF_A), note_buf(0x4000, 0x40, BUF_B), note_buf(0x4000, 0x40, BUF_A);
    note_buf(0x5000, 0x40, BUF_B), note_buf(0x5000, 0x40, VK_NULL_HANDLE);
    note_buf(0x6000, 0x40, VK_NULL_HANDLE), note_buf(0x6000, 0x40, BUF_A);
    check("the written buffer");
    if (res.nwritten != 4 || res.written[0].buf != BUF_A || res.written[1].buf || res.written[2].buf || res.written[3].buf)
        failures++, fprintf(stderr, "FAIL the written buffer: a range kept a buffer another write did not name\n");
    reset();

    // Around each table growth, keep some entries lazy and clear others before rebuilding. Revisit every entry with
    // both flags afterwards: changing a range's lazy eligibility must not lose its slot or add another pending range.
    for (uint64_t i = 0; i < 5000; i++) {
        note(0x100000 + i * 0x100, 0x100 | (i & 1 ? VK_WB_LAZY : 0));
        if (i % 4 == 1) note_buf(0x100000 + i * 0x100, 0x100, BUF_B);
        check("sequential ranges");
    }
    for (uint64_t i = 0; i < 5000; i++) note(0x100000 + i * 0x100, 0x100 | VK_WB_LAZY);
    check("all sequential ranges again as lazy writes");
    for (uint64_t i = 0; i < 5000; i++) note(0x100000 + i * 0x100, 0x100);
    check("all sequential ranges again");
    for (uint64_t i = 0; i < 5000; i++) note(0x100000 + i * 0x100, 0x100 | VK_WB_LAZY);
    check("later lazy writes do not restore the flag");
    reset();

    // Ranges that repeat as a replay's do, with the same address under a few sizes.
    for (int i = 0; i < 40000; i++) {
        uint64_t r = next();
        note(0x80000000ull + (r % 3000) * 0x40, 0x40ull << (r >> 60 & 3));
    }
    check("mixed repeating ranges");
    reset();

    // Keys equal in the table's low bits: every one probes past the others.
    for (uint64_t i = 0; i < 2000; i++) note(i << 40, 0x10);
    for (uint64_t i = 0; i < 2000; i++) note(i << 40, 0x10);
    check("high address bits only");
    reset();

    // One field shared by every key, so a probe that lands on a neighbour needs the other field to tell them apart.
    for (uint64_t i = 1; i <= 2000; i++) note(0x1000, i);
    for (uint64_t i = 1; i <= 2000; i++) note(0x1000, i);
    check("one address under many sizes");
    reset();
    for (uint64_t i = 1; i <= 2000; i++) note(i * 0x100, 0x40);
    for (uint64_t i = 1; i <= 2000; i++) note(i * 0x100, 0x40);
    check("many addresses under one size");
    reset();

    // Only storage writes: the flag stays, even if two different buffers wrote the range. It never reaches pending.
    note(0x7000, 0x80 | VK_WB_LAZY), note(0x7000, 0x80 | VK_WB_LAZY);
    check("only lazy writes");
    if (res.nwritten != 1 || res.written[0].size != (0x80 | VK_WB_LAZY) || res.written[0].buf != BUF_A)
        failures++, fprintf(stderr, "FAIL repeated lazy writes lost the flag or buffer\n");
    note_buf(0x7000, 0x80 | VK_WB_LAZY, BUF_B);
    check("lazy writes through different buffers");
    if (res.nwritten != 1 || res.written[0].size != (0x80 | VK_WB_LAZY) || res.written[0].buf != VK_NULL_HANDLE)
        failures++, fprintf(stderr, "FAIL lazy writes through different buffers kept an owner or lost the flag\n");
    reset();

    // A storage write and a copy of the same range must merge in either order. With different buffers, keeping an
    // owner would let vk_memory.c vk_gpu_written_into claim the mirror has bytes it did not write, even with lazy
    // write-back disabled. An intervening range checks that merging preserves first-write order.
    const VkBuffer buffers[] = {BUF_A, BUF_B, VK_NULL_HANDLE};
    for (int first_lazy = 0; first_lazy < 2; first_lazy++) {
        for (unsigned b = 0; b < sizeof buffers / sizeof *buffers; b++) {
            note_buf(0x7000, 0x80 | (first_lazy ? VK_WB_LAZY : 0), BUF_A);
            note(0x8000, 0x40);
            note_buf(0x7000, 0x80 | (first_lazy ? 0 : VK_WB_LAZY), buffers[b]);
            note(0x7000, 0x80 | VK_WB_LAZY);
            check("mixed writes in either order");
            VkBuffer owner = b == 0 ? BUF_A : VK_NULL_HANDLE;
            if (res.nwritten != 2 || res.written[0].addr != 0x7000 || res.written[0].size != 0x80 ||
                res.written[0].buf != owner || res.written[1].addr != 0x8000 || npending != 2)
                failures++, fprintf(stderr, "FAIL mixed writes: first_lazy=%d, buffer=%u\n", first_lazy, b);
            reset();
        }
    }

    printf("note_written: %u failures\n", failures);
    return failures != 0;
}
'''


def source(draw):
    return FIXTURE.replace('@GROW@', function(draw, 'static void *grow(')).replace(
        '@NOTE@', function(draw, 'static uint32_t written_hash(') + '\n' + function(draw, 'static void note_written('))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc'))
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text(encoding='utf-8')
    if 'free(r->written_slot)' not in function(draw, 'static void cb_done('):
        print('cb_done does not free the table')
        return 1
    with tempfile.TemporaryDirectory() as temp:
        src = Path(temp) / 'fixture.c'
        src.write_text(source(draw), encoding='utf-8')
        for opt in ('-O0', '-O2'):
            exe = Path(temp) / ('fixture' + opt + ('.exe' if os.name == 'nt' else ''))
            compiled = subprocess.run([args.cc, '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror', str(src), '-o', str(exe)],
                                      capture_output=True, text=True, timeout=60)
            if compiled.returncode:
                print(compiled.stdout + compiled.stderr)
                return compiled.returncode
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
            print(opt, run.stdout + run.stderr, end='')
            if run.returncode:
                return run.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
