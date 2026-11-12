"""Exercise the descriptor memo switches through the production functions.

The storage-bind serial of c0 (c0_info in vk_exec_draw.c, on a draw context of the fixture's) and the sampler generation of vk_texture.c
(descriptor_registered) are cut out of the runtime sources and linked against a scratch allocator in host memory. No
Vulkan loader or GPU is linked. Every switch is set to "ab", so each case runs with the switch off (bananza's
behaviour) and on, and flips it between calls the way TOTK_AB does inside one run.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from nvn2_compute_c0_test import draw_context, function

ROOT = Path(__file__).resolve().parents[1]


def source():
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    texture = (ROOT / 'runtime/vk_texture.c').read_text()
    prelude = r'''
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_hmap.h"
#include "vk_exec.h"
#pragma GCC diagnostic ignored "-Wunused-function"   // the switches of code this fixture does not cut
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_GS 2
#define NVN_STAGE_CS 5
#define SUPPORT_BYTES 1408u
static unsigned checks, failures;
#define CHECK(c,message) do { checks++; if (!(c)) { failures++; fprintf(stderr,"FAIL: %s\n",message); } } while (0)
void rt_fatal(const char *fmt,...) { (void)fmt; abort(); }
static int ab;   // every switch is "ab": on while this is 1
int rt_knob(const char *name,const char *def) { (void)name; (void)def; return 2; }
int rt_ab(void) { return ab; }
// Scratch in host memory: one 4 KiB block per copy, its index as the offset, so equal ranges mean the same copy.
static uint8_t scratch_mem[1024][4096];
static unsigned scratch_n;
static uint8_t *scratch(VkExec *ex,uint64_t size,VkBuffer *buf,VkDeviceSize *off) {
    (void)ex;
    if (size>4096 || scratch_n==1024) abort();
    *buf=(VkBuffer)(uintptr_t)0x5c, *off=scratch_n*4096ull;
    return scratch_mem[scratch_n++];
}
static const uint8_t *copy_at(VkDescriptorBufferInfo bi) { return scratch_mem[bi.offset/4096]; }
static unsigned fills;
@CTX@
static void c0_fill(DrawCtx *dc,uint8_t bank[VK_C0_BYTES],int stage,const p_bind_buf ssbo[16],const p_bind_buf ubo[16],
                    const uint32_t dim[3],uint32_t base_instance) {
    (void)dc; (void)ubo; (void)dim;
    memset(bank,0,VK_C0_BYTES);
    memcpy(bank,&ssbo[0].addr,8), memcpy(bank+8,&base_instance,4), bank[12]=(uint8_t)stage;
    fills++;
}
static DrawCtx ctx;   // the replay thread's: c0_lay NULL (NVN 1), the bank memo B, c0_binst, ssbo_serial
'''
    knobs = draw[draw.index('static int knob_on('):draw.index('static const char *const BSF_NAME[')]
    c0 = function(draw, 'static VkDescriptorBufferInfo c0_info(')
    # vk_texture.c: a slot's kind and serial are all descriptor_registered reads or writes.
    kinds = next(line for line in texture.splitlines() if line.startswith('enum { DESC_TEXTURE = 1'))
    gens = '\n'.join(line for line in texture.splitlines()
                     if line.startswith(('static uint32_t slots_gen', 'static uint32_t samp_gen',
                                         'static uint32_t reg_serial', '#define RGEN_N', 'static uint32_t rgen[')))
    gens += '\n' + function(texture, 'static uint32_t rgen_idx(') + '\n' + function(texture, 'static void rgen_bump(')
    registered = function(texture, 'static void descriptor_registered(')
    tests = r'''
static void c0_cases(VkExec *ex) {
    p_bind_buf x={.addr=0x10000,.size=0x100}, y={.addr=0x20000,.size=0x100};
    ex->ssbo[NVN_STAGE_VS][0]=x, ctx.ssbo_serial[NVN_STAGE_VS]++, ctx.c0_binst=0;
    memset(ctx.B.c0_valid,0,sizeof ctx.B.c0_valid);
    for (int on=0; on<2; on++) {
        ab=on, fills=0;
        VkDescriptorBufferInfo a=c0_info(ex,NVN_STAGE_VS), b=c0_info(ex,NVN_STAGE_VS);
        CHECK(fills==1 && a.offset==b.offset,"c0: unchanged inputs keep the bank");
        uint64_t got;
        memcpy(&got,copy_at(b),8);
        CHECK(got==ex->ssbo[NVN_STAGE_VS][0].addr,"c0: the bank was filled from the bound storage table");
        ex->ssbo[NVN_STAGE_VS][0]=on ? x : y, ctx.ssbo_serial[NVN_STAGE_VS]++;
        VkDescriptorBufferInfo c=c0_info(ex,NVN_STAGE_VS);
        memcpy(&got,copy_at(c),8);
        CHECK(got==ex->ssbo[NVN_STAGE_VS][0].addr,"c0: a storage bind gives a bank of the new table");
        ctx.c0_binst=7;
        VkDescriptorBufferInfo d=c0_info(ex,NVN_STAGE_VS);
        uint32_t bi;
        memcpy(&bi,copy_at(d)+8,4);
        CHECK(bi==7 && d.offset!=c.offset,"c0: a new base instance gives a new bank");
        ctx.c0_binst=0;
    }
    // On, then a bind and a call with the switch off, then a bind back and on: the serial moved, so the compare runs.
    ab=1;
    ex->ssbo[NVN_STAGE_VS][0]=x, ctx.ssbo_serial[NVN_STAGE_VS]++;
    c0_info(ex,NVN_STAGE_VS);
    ab=0;
    ex->ssbo[NVN_STAGE_VS][0]=y, ctx.ssbo_serial[NVN_STAGE_VS]++;
    c0_info(ex,NVN_STAGE_VS);
    ab=1;
    ex->ssbo[NVN_STAGE_VS][0]=x, ctx.ssbo_serial[NVN_STAGE_VS]++;
    uint64_t got;
    memcpy(&got,copy_at(c0_info(ex,NVN_STAGE_VS)),8);
    CHECK(got==x.addr,"c0: a flip between binds never answers with the other table's bank");
    // On: an unchanged serial skips both the fill and the compare.
    fills=0;
    c0_info(ex,NVN_STAGE_VS), c0_info(ex,NVN_STAGE_VS);
    CHECK(!fills,"c0 on: no storage bind since, no fill");
    // Off: the compare alone decides, also for a table written without a bind op (none exists, but the old path held).
    ab=0, fills=0;
    ex->ssbo[NVN_STAGE_VS][0]=y;
    memcpy(&got,copy_at(c0_info(ex,NVN_STAGE_VS)),8);
    CHECK(fills==1 && got==y.addr,"c0 off: the inputs are compared on every ask");
}

static void samp_gen_cases(void) {
    DescSlot d={0};
    for (int on=0; on<2; on++) {
        ab=on;
        d.kind=DESC_TEXTURE;
        uint32_t s0=samp_gen,t0=slots_gen;
        descriptor_registered(&d,DESC_TEXTURE);
        CHECK(slots_gen==t0+1 && samp_gen==s0+(on ? 0u : 1u),
              on ? "samp_gen on: a texture over a texture changes no sampler answer"
                 : "samp_gen off: every registration bumps it");
        s0=samp_gen;
        descriptor_registered(&d,DESC_SAMPLER);
        CHECK(samp_gen==s0+1 && d.kind==DESC_SAMPLER,"samp_gen: a sampler registration bumps it");
        s0=samp_gen;
        descriptor_registered(&d,DESC_TEXTURE);
        CHECK(samp_gen==s0+1 && d.kind==DESC_TEXTURE,"samp_gen: a texture over a sampler slot bumps it");
    }
}

int main(void) {
    static VkExec ex;
    ex.draw=&ctx;
    c0_cases(&ex);
    samp_gen_cases();
    printf("descriptor memo switches: %u checks, %u failures\n",checks,failures);
    return failures!=0;
}
'''
    adapter = 'typedef struct { uint32_t kind, serial; } DescSlot;\n'
    return '\n'.join((prelude.replace('@CTX@', draw_context(draw)), knobs, c0, kinds, adapter, gens, registered,
                      tests))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc'))
    parser.add_argument('--deps', type=Path, required=True,
                        help='runtime dependency root with include/vulkan and include/vma')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    out = ROOT / 'build/test/descriptor-memo-knobs'
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'knobs.c'
    src.write_text(source())
    failed = 0
    for opt in ('-O0', '-O2'):
        exe = out / ('knobs' + opt + ('.exe' if os.name == 'nt' else ''))
        command = [args.cc, opt, '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                   '-I', str(args.deps / 'include'), '-I', str(args.deps / 'include/vma'), str(src), '-o', str(exe)]
        run = subprocess.run(command, capture_output=True, text=True)
        (out / (opt + '-compile.log')).write_text(run.stdout + run.stderr)
        if run.returncode:
            print(run.stdout + run.stderr)
            return run.returncode
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        (out / (opt + '-run.log')).write_text(run.stdout + run.stderr)
        print(opt, run.stdout + run.stderr)
        failed |= run.returncode
    return failed


if __name__ == '__main__':
    raise SystemExit(main())
