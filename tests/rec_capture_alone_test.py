"""runtime/vk_rec.c captures a held pass end without the recorder thread (rec_capture_begin_for, used by vk_exec.c
par_defer with TOTK_VK_RECORDER=0, the default on AMD's proprietary driver).

vk_rec.c is compiled with Vulkan stubs that log each command made, and no recorder is started. The checks:

- during a capture of command buffer A, the wrappers make nothing into A, and a call into another command buffer B is
  made at once;
- rec_capture_play makes A's calls later, in their order, with their arguments (an end of rendering, a query end, an
  image copy with its regions);
- after rec_capture_end, calls into A are made at once again;
- a call that has no packet (a barrier with a pNext chain) spills the capture first (the spill function ends it), then
  is made after it;
- a capture that fills up spills, and the call that did not fit is made right after what the spill made (with no
  recorder nothing reads the ring, so it would be lost there): every query end of the run is made once, in order;
- rec_capture_alone answers 1 without the recorder, 0 for no command buffer.

Usage: python tests/rec_capture_alone_test.py --cc clang --deps /path/to/runtime/deps
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Stubs in a translation unit of their own, without the Vulkan headers: each logged command records its name, its
# command buffer and its first integer argument; the others do nothing.
STUBS = r'''
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { const char *name; void *cb; uint64_t a; } Call;
Call calls[64];
int ncalls;
static void note(const char *name, void *cb, uint64_t a) {
    if (ncalls < 64) calls[ncalls] = (Call){name, cb, a};
    ncalls++;
}
// The query ends made, in order: query_next is the index the next one must have, query_bad counts the others.
uint32_t query_next, query_bad, query_made;
void vkCmdEndRendering(void *cb) { note("EndRendering", cb, 0); }
void vkCmdEndQuery(void *cb, void *pool, uint32_t q) {
    (void)pool;
    note("EndQuery", cb, q);
    if (q >= 1000) query_bad += q - 1000 != query_next, query_next = q - 1000 + 1, query_made++;
}
typedef struct { uint32_t a[4]; int32_t o[3]; uint32_t b[4]; int32_t p[3]; uint32_t e[3]; } Region;   // VkImageCopy
void vkCmdCopyImage(void *cb, void *s, int sl, void *d, int dl, uint32_t n, const Region *r) {
    (void)s, (void)sl, (void)d, (void)dl;
    note("CopyImage", cb, (uint64_t)n << 32 | r[n - 1].e[0]);
}
void vkCmdPipelineBarrier2(void *cb, const void *di) { (void)di; note("PipelineBarrier2", cb, 0); }
#define NOP(n) void n(void) {}
NOP(vkBeginCommandBuffer) NOP(vkCmdBeginQuery) NOP(vkCmdBeginRendering) NOP(vkCmdBindDescriptorSets)
NOP(vkCmdBindIndexBuffer) NOP(vkCmdBindPipeline) NOP(vkCmdBindVertexBuffers2) NOP(vkCmdBlitImage)
NOP(vkCmdClearAttachments) NOP(vkCmdClearColorImage) NOP(vkCmdClearDepthStencilImage) NOP(vkCmdCopyBuffer)
NOP(vkCmdCopyBufferToImage) NOP(vkCmdCopyImageToBuffer) NOP(vkCmdDispatch) NOP(vkCmdDispatchIndirect)
NOP(vkCmdDraw) NOP(vkCmdDrawIndexed) NOP(vkCmdDrawIndexedIndirect) NOP(vkCmdFillBuffer) NOP(vkCmdPushConstants)
NOP(vkCmdResetQueryPool) NOP(vkCmdResolveImage) NOP(vkCmdSetBlendConstants) NOP(vkCmdSetCullMode)
NOP(vkCmdSetDepthBias) NOP(vkCmdSetDepthBiasEnable) NOP(vkCmdSetDepthCompareOp) NOP(vkCmdSetDepthTestEnable)
NOP(vkCmdSetDepthWriteEnable) NOP(vkCmdSetFrontFace) NOP(vkCmdSetPrimitiveRestartEnable)
NOP(vkCmdSetPrimitiveTopology) NOP(vkCmdSetRasterizerDiscardEnable) NOP(vkCmdSetScissorWithCount)
NOP(vkCmdSetStencilCompareMask) NOP(vkCmdSetStencilOp) NOP(vkCmdSetStencilReference) NOP(vkCmdSetStencilTestEnable)
NOP(vkCmdSetStencilWriteMask) NOP(vkCmdSetViewportWithCount) NOP(vkCmdWriteTimestamp) NOP(vkCmdWriteTimestamp2)
NOP(vkEndCommandBuffer) NOP(vkQueueSubmit2) NOP(plat_event_init) NOP(plat_event_set) NOP(plat_thread_priority)
NOP(plat_yield) NOP(vk_checkpoint) NOP(vk_checkpoint_compute) NOP(vk_crumb) NOP(vk_queue_lock) NOP(vk_queue_unlock)
NOP(vk_device_fault_report)
int plat_event_wait_ns(void) { return 0; }
int vk_sync_queue_submit(void) { return 0; }   // vk_sync.c: the part submit, VK_SUCCESS
uint64_t plat_now_ns(void) { return 0; }
void *plat_thread_new(void) { return NULL; }
int plat_thread_priority_get(void) { return 0; }
int cfg_int(const char *k) { (void)k; return 0; }
int rt_ab(void) { return 0; }
int rt_knob(const char *n, const char *d) { (void)n, (void)d; return 0; }
unsigned rt_trace;
void rt_log(const char *f, ...) { (void)f; }
void rt_fatal(const char *f, ...) { va_list ap; va_start(ap, f); vprintf(f, ap); va_end(ap); puts(""); exit(3); }
char vk_caps[256];
int vk_checkpoints_on(void) { return 0; }
const void *vk_checkpoint_tag(const char *k) { (void)k; return NULL; }
void *vk_ctx(void) { return NULL; }
'''

MAIN = r'''
#include <stdio.h>
#include <string.h>
#include "vk.h"
typedef struct { const char *name; void *cb; uint64_t a; } Call;
extern Call calls[64];
extern int ncalls;
extern uint32_t query_next, query_bad, query_made;
static int fails, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); puts(""); } } while (0)
static VkCommandBuffer A = (VkCommandBuffer)(uintptr_t)0x1000, B = (VkCommandBuffer)(uintptr_t)0x2000;
static uint32_t spill_lo;
static int spills;
static void spill(void) {   // as vk_exec.c par_spill: end the capture, then make what it held
    uint32_t hi = rec_capture_end();
    spills++;
    rec_capture_play(A, spill_lo, hi);
}
int main(void) {
    CHECK(rec_capture_alone(A), "no recorder: a capture of A can be taken alone");
    CHECK(!rec_capture_alone(VK_NULL_HANDLE), "no command buffer: none");
    uint32_t lo = rec_capture_begin_for(A);
    vkCmdEndQuery(A, VK_NULL_HANDLE, 7);
    vkCmdEndRendering(A);
    vkCmdEndRendering(B);   // another command buffer: made at once
    VkImageCopy r[2];
    memset(r, 0, sizeof r);
    r[1].extent.width = 33;
    vkCmdCopyImage(A, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, 2, r);
    r[1].extent.width = 99;   // the capture copied the regions
    uint32_t hi = rec_capture_end();
    CHECK(ncalls == 1 && calls[0].cb == B && !strcmp(calls[0].name, "EndRendering"), "only B's call made during the capture (%d)", ncalls);
    CHECK(hi > lo, "the capture holds packets");
    ncalls = 0;
    rec_capture_play(A, lo, hi);
    CHECK(ncalls == 3, "three calls played (%d)", ncalls);
    CHECK(ncalls >= 3 && !strcmp(calls[0].name, "EndQuery") && calls[0].a == 7 && calls[0].cb == A, "the query end first");
    CHECK(ncalls >= 3 && !strcmp(calls[1].name, "EndRendering") && calls[1].cb == A, "then the end of rendering");
    CHECK(ncalls >= 3 && !strcmp(calls[2].name, "CopyImage") && calls[2].a == ((uint64_t)2 << 32 | 33), "then the copy, as captured");
    ncalls = 0;
    vkCmdEndRendering(A);
    CHECK(ncalls == 1 && calls[0].cb == A, "after the capture A's calls are made at once");
    // A call without a packet: a barrier with a pNext chain spills the capture, then is made.
    ncalls = 0;
    rec_capture_reset();
    rec_capture_spill(spill);
    spill_lo = rec_capture_begin_for(A);
    vkCmdEndRendering(A);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .pNext = &mb};
    vkCmdPipelineBarrier2(A, &di);
    CHECK(spills == 1, "the capture spilled once (%d)", spills);
    CHECK(ncalls == 2 && !strcmp(calls[0].name, "EndRendering") && !strcmp(calls[1].name, "PipelineBarrier2"),
          "the held end of rendering first, then the barrier (%d calls)", ncalls);
    ncalls = 0;
    vkCmdEndRendering(A);
    CHECK(ncalls == 1, "after the spill A's calls are made at once");
    // A capture that fills up: the spill makes what it held, and the call that did not fit is made right after it,
    // not lost in a ring nobody reads. Query ends numbered from 1000 must all be made, once each, in order.
    rec_capture_reset();
    spills = 0, ncalls = 0;
    spill_lo = rec_capture_begin_for(A);
    uint32_t n = 0;
    while (spills == 0 && n < 100000) vkCmdEndQuery(A, VK_NULL_HANDLE, 1000 + n++);
    CHECK(spills == 1, "the full capture spilled once (%d)", spills);
    CHECK(query_made == n && query_bad == 0, "all %u query ends made in order (%u made, %u out of order)", n, query_made, query_bad);
    vkCmdEndQuery(A, VK_NULL_HANDLE, 1000 + n++);
    CHECK(query_made == n && query_bad == 0, "after the spill the next one is made at once (%u of %u)", query_made, n);
    printf("rec capture alone: %d checks, %d failures\n", checks, fails);
    return fails != 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--deps', type=Path, required=True, help='existing dependency root or include directory')
    parser.add_argument('--out', type=Path, default=ROOT / 'build' / 'tests' / 'rec-capture-alone')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    include = args.deps / 'include' if (args.deps / 'include').is_dir() else args.deps
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'stubs.c').write_text(STUBS, encoding='utf-8')
    (args.out / 'main.c').write_text(MAIN, encoding='utf-8')
    exe = args.out / 'rec_capture_alone.exe'
    for opt in ('-O0', '-O2'):
        subprocess.run([args.cc, '-std=gnu11', opt, '-I', str(ROOT / 'runtime'), '-I', str(include),
                        '-I', str(include / 'vma'), str(ROOT / 'runtime' / 'vk_rec.c'), str(args.out / 'main.c'),
                        str(args.out / 'stubs.c'), '-o', str(exe)], check=True, timeout=300)
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        print(opt, run.stdout.strip())
        if run.returncode:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
