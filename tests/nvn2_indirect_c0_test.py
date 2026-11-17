"""Exercise the real indirect-dispatch branch with synthetic NVN 1/2 parameter banks and CPU Vulkan stubs.

Usage: python tests/nvn2_indirect_c0_test.py [--cc clang]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from nvn2_compute_c0_test import function

ROOT = Path(__file__).resolve().parents[1]


def source():
    production = (ROOT / 'runtime' / 'vk_exec_draw.c').read_text()
    before = r'''
#include <stdio.h>
#include <stdlib.h>
#include "nvn2_c0.h"
typedef struct { uint64_t ind, caller; } p_dispatch_ind;
typedef uint8_t *VkBuffer;
typedef unsigned VkPipeline;
typedef struct { uint32_t x, y, z; } VkDispatchIndirectCommand;
typedef struct { uint64_t srcOffset, dstOffset, size; } VkBufferCopy;
typedef struct { VkBuffer buffer; uint64_t offset; int cls; } VkMemSpan;
// The draw context fields op_dispatch_ind reaches (vk_exec_draw.c DrawCtx).
typedef struct DrawCtx {
    const Nvn2C0Layout *c0_lay;
    uint32_t c0_dim[3];
    int twin_dispatch;
    struct { int c0_valid[3]; uint64_t c0_off[3]; VkBuffer c0_buf[3]; } B;
} DrawCtx;
typedef struct { void *prog; int cb; unsigned recorded; DrawCtx *draw; } VkExec;
#define DC_OF(ex) ((ex)->draw)   // a translator that takes its context from VkExec.draw
typedef int GfxShader;
typedef int VkPipeInfo;
enum { W_PROGRAM, W_INDIRECT, W_PIPELINE, NVN_STAGE_CS=5, VK_PIPELINE_BIND_POINT_COMPUTE=1, VK_MUSE_IND=1 };
static DrawCtx ctx;
static int vk_dump_armed, bind_c0 = 1;
static uint8_t bank[NVN2_C0_BYTES];
static uint32_t arguments[4] = {0xdeadbeef,3,5,7}, groups[3];
static unsigned copies, dispatches, barriers;
static struct {
    const VkExec *ex;
    int kind, twin_dispatch;
    uint64_t work;
    unsigned calls, recorded, dispatches;
} noted;
// Keep the profiling callback bounded to one snapshot, including when the command was recorded.
static void note_work(const VkExec *ex, int kind, uint64_t work) {
    noted.ex = ex;
    noted.kind = kind;
    noted.work = work;
    noted.recorded = ex->recorded;
    noted.dispatches = dispatches;
    noted.twin_dispatch = ex->draw->twin_dispatch;
    noted.calls++;
}
static void warn_once(int code, const char *format, ...) { (void)code; (void)format; abort(); }
static int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *sp) {
    (void)addr; (void)size; *sp = (VkMemSpan){(uint8_t *)arguments,4,0}; return 1;
}
static void cb_res(VkExec *ex) { (void)ex; }
static void dummies_init(void) {}
static const GfxShader *shader_for(VkExec *ex, int stage, int *present, int *pending, int await) {
    (void)ex; (void)stage; (void)pending; (void)await; static GfxShader shader; *present = 1; return &shader;
}
static VkPipeline vk_pipeline_compute(const GfxShader *shader, const VkPipeInfo **info) {
    (void)shader; static VkPipeInfo pi; *info = &pi; return 1;
}
static void vk_exec_barrier(VkExec *ex) { (void)ex; barriers++; }
static void vk_exec_barrier_cs(VkExec *ex) { vk_exec_barrier(ex); }   // the compute barrier counts as one
static void vk_exec_note_dispatch(VkExec *ex) { (void)ex; }
#define VK_MEM_USE(sp) ((void)(sp))
#define VK_MEM_BOUND(sp) ((void)(sp))
static void bind_pipeline(VkExec *ex, unsigned point, VkPipeline pipe) { (void)ex; (void)point; (void)pipe; }
static void c0_lay_set(VkExec *ex, const GfxShader *shader) { (void)ex; (void)shader; }
static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    (void)info; if (!ex->draw->twin_dispatch) abort();
    ex->draw->B.c0_valid[2] = bind_c0; ex->draw->B.c0_off[2] = 0; ex->draw->B.c0_buf[2] = bank;
}
static void vkCmdCopyBuffer(int cb, VkBuffer src, VkBuffer dst, unsigned count, const VkBufferCopy *copy) {
    (void)cb;
    if (count != 1 || copy->dstOffset + copy->size > sizeof bank ||
        copy->srcOffset + copy->size > sizeof arguments) abort();
    memcpy(dst + copy->dstOffset, src + copy->srcOffset, (size_t)copy->size);
    copies++;
}
static void vkCmdDispatchIndirect(int cb, VkBuffer src, uint64_t offset) {
    (void)cb; memcpy(groups, src + offset, sizeof groups); dispatches++;
}
static void vk_mem_count(int cls, int use, uint64_t bytes) { (void)cls; (void)use; (void)bytes; }
static void vk_dump_dispatch(VkExec *ex, const GfxShader *shader, VkPipeline pipe, const VkPipeInfo *info,
                             const p_dispatch *dispatch) { (void)ex; (void)shader; (void)pipe; (void)info; (void)dispatch; }
'''
    after = r'''
static unsigned checks, failures;
#define CHECK(c, message) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL: %s\n", message); } } while (0)
static void run(const Nvn2C0Layout *layout, unsigned expected_copies) {
    VkExec ex = {.prog=&ex, .draw=&ctx};
    p_dispatch_ind d = {0,0};
    ctx.c0_lay = layout;
    copies = dispatches = barriers = 0;
    memset(&noted, 0, sizeof noted);
    op_dispatch_ind(&ex, &d);
    CHECK(dispatches == 1 && !memcmp(groups, arguments + 1, sizeof groups), "indirect dispatch keeps actual GPU arguments");
    CHECK(copies == expected_copies, "only the NVN 1 bank receives the legacy dimension copy");
    CHECK(ex.recorded == 1 + expected_copies && barriers == 1 + expected_copies, "copy bookkeeping follows the selected ABI");
    CHECK(ctx.B.c0_valid[2] == (bind_c0 && !expected_copies), "the NVN 2 packed bank stays reusable");
    CHECK(noted.calls == 1 && noted.ex == &ex, "indirect dispatch profiles its executor exactly once");
    CHECK(noted.kind == 3 && noted.work == 0, "indirect profiling keeps the GPU-only work count unknown");
    CHECK(noted.recorded == 1 + expected_copies && noted.dispatches == 1 && !noted.twin_dispatch,
          "profiling follows the recorded indirect dispatch and resource binding");
}

int main(void) {
    uint8_t expected[NVN2_C0_BYTES];
    // NVN 1 retains its original dimensions at +0x200 and preserves every other byte.
    memset(bank, 0xa5, sizeof bank);
    memcpy(expected, bank, sizeof bank);
    memcpy(expected + 0x200, arguments + 1, sizeof groups);
    run(NULL, 1);
    CHECK(!memcmp(bank, expected, sizeof bank), "NVN 1 dimension slot and surrounding bytes are preserved");

    uint32_t roots[8][64];
    for (unsigned t = 0; t < 8; t++)
        for (unsigned i = 0; i < 64; i++) roots[t][i] = 0xa0000000u + t * 0x100u + i;
    p_bind_buf ubo[16] = {{0}}, ssbo[16] = {{0}};
    ubo[0].addr = UINT64_C(0x1122334455667788), ubo[0].size = 0x1234;
    Nvn2C0Layout layout = {.ubos=1, .roots={64,64}};
    CHECK(nvn2_c0_pack(bank, sizeof bank, &layout, roots, ssbo, ubo), "synthetic roots cross the legacy +0x200 slot");
    memcpy(expected, bank, sizeof bank);
    run(&layout, 0);
    CHECK(!memcmp(bank, expected, sizeof bank), "NVN 2 declared root words survive indirect dispatch unchanged");

    layout.roots[1] = 60;
    CHECK(nvn2_c0_part(&layout, 1) == 0x200 && nvn2_c0_pack(bank, sizeof bank, &layout, roots, ssbo, ubo),
          "synthetic NVN 2 uniform table starts at +0x200");
    memcpy(expected, bank, sizeof bank);
    run(&layout, 0);
    CHECK(!memcmp(bank, expected, sizeof bank), "NVN 2 buffer address/size survive indirect dispatch unchanged");

    memcpy(bank, expected, sizeof bank);
    bind_c0 = 0;
    run(NULL, 0);
    CHECK(!memcmp(bank, expected, sizeof bank), "a shader without c0 does not trigger a bank copy");
    printf("indirect c0: %u checks, %u failures\n", checks, failures);
    return failures != 0;
}
'''
    return before + function(production, 'static void op_dispatch_ind(') + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc'))
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    out = ROOT / 'build' / 'test' / 'nvn2-indirect-c0'
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'indirect.c'
    src.write_text(source())
    exe = out / ('indirect.exe' if os.name == 'nt' else 'indirect')
    subprocess.run([args.cc, '-O2', '-Wall', '-Wextra', '-I', str(ROOT / 'runtime'), str(src), '-o', str(exe)], check=True)
    return subprocess.run([str(exe)]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
