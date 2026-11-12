"""CPU check of the GPU-write marking a helper part defers to its join, and of the pending mirror copy count.

Usage: python tests/texture_written_late_test.py --cc /path/to/clang --deps /path/to/runtime/deps
Only Vulkan/VMA headers are needed; no Vulkan library is linked. The fixture runs the production functions of
runtime/vk_texture.c (vk_texture_gpu_written_ready, vk_texture_gpu_written_late, vk_texture_release) and
runtime/vk_memory.c (range_add, vk_mem_mirror_pending) on one thread, in the orders the replay thread and a game
thread can produce. --source-root with an unmodified checkout reproduces the original failure: a release that lands
between a helper's check and the join leaves the render unrecorded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(text, name):
    match = re.search(r'^(?:static )?(?:inline )?[^\n;{}]+\b' + re.escape(name) + r'\([^\n]*\)\s*\{', text, re.M)
    if not match:
        raise ValueError(f'function not found: {name}')
    depth = 0
    for at in range(text.index('{', match.start()), len(text)):
        depth += (text[at] == '{') - (text[at] == '}')
        if not depth:
            return text[match.start():at + 1]
    raise ValueError(f'unterminated function: {name}')


def typedef(text, name):
    end = text.index('} ' + name + ';') + len('} ' + name + ';')
    starts = list(re.finditer(r'typedef struct(?:\s+\w+)?\s*\{', text[:end]))
    return text[starts[-1].start():end]


def statement(text, start):
    begin = text.index(start)
    end = text.index('\n', begin) if start.startswith('#') else text.index(';', begin) + 1
    return text[begin:end]


PRELUDE = r'''
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_spec.h"
#include "platform/platform.h"
unsigned rt_trace;
static unsigned checks;
void rt_log(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); }
void rt_fatal(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); exit(2); }
#define CHECK(c) do { checks++; if (!(c)) { fprintf(stderr, "check line %d: %s\n", __LINE__, #c); return 1; } } while (0)
// One thread: the orders a game thread and the replay thread can produce are played one after the other.
uint32_t plat_thread_id(void) { return 1; }
void plat_mutex_lock(PlatMutex *m) { if (m->p) rt_fatal("nonrecursive lock reentered\n"); m->p = (void *)1; }
void plat_mutex_unlock(PlatMutex *m) { if (!m->p) rt_fatal("unlock without lock\n"); m->p = NULL; }
int plat_mutex_trylock(PlatMutex *m) { if (m->p) return 0; m->p = (void *)1; return 1; }
uint64_t kern_now_ns(void) { return 0; }
void perf_lock_wait(int lock, uint64_t ns) { (void)lock; (void)ns; }
void plat_cond_broadcast(PlatCond *c) { (void)c; }
void plat_cond_wait(PlatCond *c, PlatMutex *m) { (void)c; (void)m; rt_fatal("unexpected fixture wait\n"); }
// The replay side the marking reports to.
static uint64_t fence;
static unsigned world_changes, frames_written, rewritten;
uint64_t vk_exec_recording_fence(void) { return fence; }
void vk_exec_frame_written(const tex_t *t) { (void)t; frames_written++; }
void vk_spec_world_changed(void) { world_changes++; }
int cfg_bool(const char *name) { (void)name; return 1; }   // TOTK_WB_EXACT: on, its default
'''

STUBS = r'''
int cfg_int(const char *name) { (void)name; return WB_SMALL_DEFAULT; }   // TOTK_WB_FENCE_LIMIT: its default
static void destroy_tex(Tex *e) { (void)e; rt_fatal("unexpected destroy: fewer than RELEASED_MAX are released\n"); }
// The page-generation snapshot needs the write watch; the fixture only counts it.
static int storage_rewritten(const Tex *e, uint32_t since, uint32_t *now) {
    (void)e; (void)since; rewritten++; *now = 77; return 0;
}
typedef struct PoolRec PoolRec;
'''

FIXTURE = r'''
static _Alignas(4096) uint8_t pixels[4][4096];

static Tex *resident(tex_t *t, unsigned i, int gpu_owned) {
    memset(pixels[i], 0x40 + i, 64);
    *t = (tex_t){.magic = NVN_TEX_MAGIC, .d = {.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA8,
        .width = 4, .height = 4, .depth = 1, .levels = 1}, .mem = pixels[i], .L = {.size = 64}, .serial = i + 1};
    Tex *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("fixture allocation\n");
    e->t = *t, e->pub.addr = (uintptr_t)pixels[i], e->pub.mirrored = 1, e->pub.gpu_owned = (uint8_t)gpu_owned;
    e->uploaded = 1;
    make_key(t, &e->key);
    nvn_hm_put(&by_key, &e->key, e);
    return e;
}

int main(void) {
    tex_t a, b, gone, cpu;
    Tex *ea = resident(&a, 0, 1), *eb = resident(&b, 1, 1), *ec = resident(&cpu, 3, 0);
    gone = a, gone.mem = pixels[2];

    // A helper part rendered into a; the helper found it ready, so the replay thread marks it at the join. In sync
    // replay a game thread's TextureFinalize may land in between. The render is still the part's: it is recorded.
    fence = 9;
    CHECK(vk_texture_gpu_written_ready(&a));
    vk_texture_release(&a);
    CHECK(ea->released && nreleased == 1 && rel_head == ea && !vk_texture_gpu_written_ready(&a));
    vk_texture_gpu_written_late(&a);
    CHECK(ea->wr_fence == 9 && ea->gpu_fence == 9 && ea->gpu_dirty && ea->gpu_gen == 77 && ea->gpu_seq == 1);
    CHECK(ea->listed && nowned == 1 && owned[0] == ea && small_pending);
    CHECK(ea->snap && !memcmp(ea->snap, pixels[0], 64));   // 64 bytes fill no page: the exact write-back copy
    CHECK(ea->released && nreleased == 1);                  // the release stands

    // The other order, release after the join, ends in the same state.
    CHECK(vk_texture_gpu_written_ready(&b));
    vk_texture_gpu_written_late(&b);
    vk_texture_release(&b);
    CHECK(eb->wr_fence == ea->wr_fence && eb->gpu_fence == ea->gpu_fence && eb->gpu_dirty && eb->gpu_gen == 77);
    CHECK(eb->gpu_seq == 2 && eb->listed && nowned == 2 && eb->snap && eb->released && nreleased == 2);
    CHECK(world_changes == 0 && rewritten == 2);   // both were GPU-owned already: no part in flight read otherwise

    // No entry: the helper does not defer it, and a late marking creates none.
    uint32_t keys = by_key.n;
    CHECK(!vk_texture_gpu_written_ready(&gone));
    vk_texture_gpu_written_late(&gone);
    CHECK(by_key.n == keys && nowned == 2 && rewritten == 2);

    // CPU-owned: not ready (the marking changes what a part reads), and a late marking leaves it alone.
    CHECK(!vk_texture_gpu_written_ready(&cpu));
    vk_texture_gpu_written_late(&cpu);
    CHECK(!ec->pub.gpu_owned && !ec->wr_fence && !ec->listed && world_changes == 0);
    CHECK(frames_written == 4 && !tex_lock.depth && !owned_lock.p);

    // Pending mirror copies: an adjacent range extends the last one, any other is appended; the count is what the
    // prologue reads.
    PoolRec *p = (PoolRec *)(uintptr_t)0x1000, *q = (PoolRec *)(uintptr_t)0x2000;
    CHECK(vk_mem_mirror_pending() == 0);
    range_add(&pend, &npend, &cpend, p, 0, 16);
    range_add(&pend, &npend, &cpend, p, 16, 16);
    CHECK(vk_mem_mirror_pending() == 1 && pend[0].size == 32);
    range_add(&pend, &npend, &cpend, q, 0, 16);
    for (unsigned i = 0; i < 40; i++) range_add(&pend, &npend, &cpend, p, 4096 * (i + 1), 16);
    CHECK(vk_mem_mirror_pending() == 42 && cpend >= 42 && pend[1].r == q && pend[41].off == 4096 * 40);

    printf("texture written late: %u checks passed (CPU only)\n", checks);
    free(ea->snap); free(eb->snap); free(ea); free(eb); free(ec);
    free(owned); free(pend); free(by_key.keys); free(by_key.vals);
    return 0;
}
'''


def source(root):
    files = {name: (root / 'runtime' / name).read_text(encoding='utf-8')
             for name in ('vk_texture.c', 'vk_memory.c', 'platform/common.c')}
    tex, memory, common = files.values()
    parts = [PRELUDE, typedef(tex, 'TexKey'), typedef(tex, 'Tex')]
    parts += [statement(tex, start) for start in (
        'static PlatRMutex tex_lock', 'static PlatOnce tex_once', 'static nvn_hmap by_key', 'static int nreleased',
        'static Tex *rel_head', 'static PlatMutex owned_lock', 'static Tex **owned', 'static int nowned',
        'static uint64_t max_owned', 'static uint64_t gpu_writes', 'static uint32_t tex_gen', 'static uint32_t slots_gen',
        '#define AGEN_N', 'static uint32_t agen[AGEN_N]', 'static volatile int32_t tex_ready', '#define RELEASED_MAX',
        'enum { SNAP_MAX', 'static int wb_exact', 'enum { WB_SMALL_DEFAULT', 'static int small_pending',
        'static uint64_t wb_limit', 'static int wb_trace', 'static int wb_cfg_read')]
    parts += [statement(common, 'static PlatMutex once_lock'), statement(common, 'static PlatCond once_done'), STUBS]
    parts += [typedef(memory, 'MRange'), statement(memory, 'static MRange *pend'), statement(memory, 'static int npend')]
    parts += [function(common, name) for name in ('plat_rmutex_lock', 'plat_rmutex_trylock', 'plat_rmutex_unlock', 'plat_once')]
    parts += [function(tex, name) for name in (
        'tex_init', 'lock', 'unlock', 'agen_idx', 'tex_gen_bump', 'make_key', 'owned_lower', 'own_add', 'rel_append',
        'sweep_released', 'fills_no_page', 'wb_cfg_init', 'vk_texture_gpu_written_ready')]
    parts += [statement(tex, 'static void gpu_written_entry(Tex *e);')]
    parts += [function(tex, name) for name in ('vk_texture_gpu_written_late', 'gpu_written_entry', 'vk_texture_release')]
    parts += [function(memory, name) for name in ('grow', 'range_add', 'vk_mem_mirror_pending')]
    parts += [FIXTURE]
    return '\n\n'.join(parts), files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT, help='checkout whose runtime/ is tested')
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--deps', type=Path, default=ROOT / 'runtime/deps')
    parser.add_argument('--out', type=Path, help='fixture and results directory')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with the project Clang executable')
    include = args.deps / 'include' if (args.deps / 'include').is_dir() else args.deps
    if not (include / 'vulkan/vulkan.h').is_file():
        parser.error('--deps must contain the Vulkan headers (directly or under include/)')
    root = args.source_root.resolve()
    out = (args.out or ROOT / 'build/tests/texture-written-late').resolve()
    out.mkdir(parents=True, exist_ok=True)
    text, files = source(root)
    fixture = out / 'fixture.c'
    fixture.write_text(text, encoding='utf-8')
    report = {'source_root': str(root),
              'sources': {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()},
              'fixture_sha256': hashlib.sha256(text.encode()).hexdigest(), 'runs': []}
    for opt in ('O0', 'O2'):
        exe = out / ('written-late-' + opt + ('.exe' if os.name == 'nt' else ''))
        command = [args.cc, '-' + opt, '-Wall', '-Werror=format', '-fno-strict-aliasing', '-I', str(root / 'runtime'),
                   '-I', str(include), '-I', str(include / 'vma'), str(fixture), '-o', str(exe)]
        built = subprocess.run(command, capture_output=True, text=True)
        (out / (opt + '-compile.log')).write_text(built.stdout + built.stderr + f'exit={built.returncode}\n')
        if built.returncode:
            print(built.stdout + built.stderr)
            return built.returncode
        compiler = Path(shutil.which(args.cc) or args.cc)
        objdump = compiler.with_name('llvm-objdump.exe' if os.name == 'nt' else 'llvm-objdump')
        if objdump.is_file():
            imports = subprocess.check_output([str(objdump), '-p', str(exe)], text=True)
            if re.search(r'DLL Name:.*vulkan', imports, re.IGNORECASE):
                raise RuntimeError('unexpected Vulkan loader import; refusing execution')
        ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
        (out / (opt + '-run.log')).write_text(ran.stdout + ran.stderr + f'exit={ran.returncode}\n')
        print(opt + ': ' + (ran.stdout.strip() or ran.stderr.strip()))
        report['runs'].append({'optimization': opt, 'exit': ran.returncode})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        if ran.returncode:
            return ran.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
