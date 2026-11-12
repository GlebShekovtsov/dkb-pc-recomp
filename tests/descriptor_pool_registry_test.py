"""CPU checks of real descriptor registration, backing resolution and lifetime policy; no Vulkan loader.

Usage: python tests/descriptor_pool_registry_test.py --cc /path/to/clang --deps /path/to/runtime/deps
Only Vulkan/VMA headers are needed. The fixture supplies original pool-index/resident-texture metadata, executes
actual production functions, and aborts if an image creation/upload dependency is reached. Single-thread only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from nvn2_compute_c0_test import function

ROOT = Path(__file__).resolve().parents[1]


def typedef(source, name):
    end = source.index('} ' + name + ';') + len('} ' + name + ';')
    starts = list(re.finditer(r'typedef struct(?:\s+\w+)?\s*\{', source[:end]))
    return source[starts[-1].start():end]


def statement(source, start):
    begin = source.index(start)
    return source[begin:source.index(';', begin) + 1]


PRELUDE = r'''
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_pool_index.h"
#include "tex.h"
#include "vk_exec.h"
#include "platform/platform.h"
#define DEV_MAGIC MAGIC('D', 'E', 'V', 0)
#define AGEN_N 4096
#define RGEN_N 4096
unsigned rt_trace;
const char rt_game_target[] = "dkb458752";
// Every TOTK_* switch reads as off, its default.
int rt_knob(const char *name, const char *def) { (void)name; (void)def; return 0; }
int rt_ab(void) { return 0; }
GfxStats gfx_counters;
static unsigned checks, dependencies;
void rt_log(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); }
void rt_fatal(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); exit(2); }
#define CHECK(c) do { checks++; if (!(c)) { fprintf(stderr, "descriptor check line %d: %s\n", __LINE__, #c); return 1; } } while (0)
// Deterministic single-thread primitives. Actual recursive-mutex/once implementations below enforce their nesting.
uint32_t plat_thread_id(void) { return 1; }
void plat_mutex_lock(PlatMutex *m) { if (m->p) rt_fatal("nonrecursive lock reentered\n"); m->p = (void *)1; }
void plat_mutex_unlock(PlatMutex *m) { if (!m->p) rt_fatal("unlock without lock\n"); m->p = NULL; }
void plat_mutex_lock_shared(PlatMutex *m) { plat_mutex_lock(m); }
void plat_mutex_unlock_shared(PlatMutex *m) { plat_mutex_unlock(m); }
void plat_cond_broadcast(PlatCond *c) { (void)c; }
void plat_cond_wait(PlatCond *c, PlatMutex *m) { (void)c; (void)m; rt_fatal("unexpected fixture wait\n"); }
'''

FIXTURE = r'''
static int query(dev_t_ *device, NVNdeviceInfo property) {
    int value = -1;
    Cpu c = {0}; c.x[0] = (uintptr_t)device, c.x[1] = property, c.x[2] = (uintptr_t)&value;
    nvnp_nvnDeviceGetInteger(&c);
    return value;
}
static void register_texture(dpool_t *pool, int id, const tex_t *t) {
    Cpu c = {0}; c.x[0] = (uintptr_t)pool, c.x[1] = id, c.x[2] = (uintptr_t)t;
    nvnp_nvnTexturePoolRegisterTexture(&c);
}
static void provide(PoolRec *record) {
    pidx_ent e = {.lo = record->lo, .hi = record->lo + record->size, .buf = record, .key = record};
    if (!pidx_insert(&pool_index, &e)) rt_fatal("pool-index allocation\n");
}
static const VkTexImage *lookup(const dpool_ref *pool, int id) {
    view_t view; int has_view;
    lock(); const VkTexImage *r = vk_texture_by_id_locked(pool, id, &view, &has_view); unlock();
    return r;
}
int main(void) {
    dev_t_ device; dev_builder_t builder = {0}; Cpu c = {0};
    c.x[0] = (uintptr_t)&device, c.x[1] = (uintptr_t)&builder; nvnp_nvnDeviceInitialize(&c); CHECK(c.x[0] == 1);
    int reserved = query(&device, NVN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS);
    int id = reserved + 1, count = id + 2;
    CHECK(reserved >= 0 && count <= query(&device, NVN_DEVICE_INFO_MAX_TEXTURE_POOL_SIZE));
    CHECK(query(&device, NVN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE) == NVN_DESCRIPTOR_BYTES);
    static _Alignas(4096) uint8_t bytes[2][32768], pixels[2][4096];
    const uint8_t colors[2][4] = {{255, 0, 0, 255}, {0, 255, 0, 255}};
    pool_t backing[2]; dpool_t tables[2]; dpool_ref refs[2]; PoolRec imported[2] = {{0}};
    tex_t textures[2] = {{0}}; Tex resident[2] = {{0}};
    for (unsigned i = 0; i < 2; i++) {
        backing[i] = (pool_t){.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = bytes[i], .size = sizeof bytes[i]};
        imported[i].kind = POOL_IMPORT, imported[i].lo = (uintptr_t)bytes[i], imported[i].size = sizeof bytes[i];
        provide(&imported[i]);
        memset(&c, 0, sizeof c); c.x[0] = (uintptr_t)&tables[i], c.x[1] = (uintptr_t)&backing[i];
        c.x[2] = 64, c.x[3] = count; nvnp_nvnTexturePoolInitialize(&c); CHECK(c.x[0] == 1);
        CHECK(nvn_dpool_snapshot(&tables[i], 1u << 20, &refs[i]));
        memcpy(pixels[i], colors[i], 4);
        textures[i] = (tex_t){.magic = NVN_TEX_MAGIC, .d = {.target = NVN_TEXTURE_TARGET_2D,
            .format = NVN_FORMAT_RGBA8, .width = 1, .height = 1, .depth = 1, .levels = 1,
            .swizzle_r = 2, .swizzle_g = 3, .swizzle_b = 4, .swizzle_a = 5},
            .mem = pixels[i], .L = {.size = sizeof pixels[i]}, .serial = i + 1};
        resident[i].t = textures[i], resident[i].pub.addr = (uintptr_t)pixels[i];
        resident[i].uploaded = 1, resident[i].checked = submit_epoch;
        memcpy(resident[i].pub.swz, (uint8_t[]){2, 3, 4, 5}, 4);
        make_key(&textures[i], &resident[i].key); nvn_hm_put(&by_key, &resident[i].key, &resident[i]);
        register_texture(&tables[i], id, &textures[i]);
    }
    CHECK(refs[0].addr != refs[1].addr);
    memset(&c, 0, sizeof c); c.x[0] = (uintptr_t)&device, c.x[1] = id, c.x[2] = reserved + 1;
    nvnp_nvnDeviceGetTextureHandle(&c); uint64_t handle = c.x[0]; CHECK(NVN_HANDLE_TEX(handle) == id);
    const unsigned order[3] = {0, 1, 0}; VkExec ex = {0};
    for (unsigned i = 0; i < 3; i++) {
        Cpu record = {0}; record.x[1] = (uintptr_t)&tables[order[i]];
        nvnp_nvnCommandBufferSetTexturePool(&record);
        apply_pool_case(&ex, recorded_op, &recorded_pool);
        const VkTexImage *image = lookup(&ex.tex_pool, NVN_HANDLE_TEX(handle));
        CHECK(image && !memcmp((void *)(uintptr_t)image->addr, colors[order[i]], 4));
    }
    CHECK(ex.tex_serial == 3);
    uint64_t token = vk_texture_slot_token(&refs[0], id); CHECK(token);
    lock(); CHECK(vk_texture_slot_token(&refs[0], id) == token); unlock(); // real recursive mutex path

    // Same bytes through another object and a shifted table origin must share the same registration/version.
    dpool_t wrapper = tables[0]; dpool_ref wrapped, shifted = refs[0];
    CHECK(nvn_dpool_snapshot(&wrapper, 1u << 20, &wrapped));
    shifted.addr += NVN_DESCRIPTOR_BYTES, shifted.count--;
    CHECK(vk_texture_slot_token(&wrapped, id) == token && vk_texture_slot_token(&shifted, id - 1) == token);
    CHECK(lookup(&shifted, id - 1)->addr == resident[0].pub.addr);
    // A stamped token is the same, and checked again without the lock it stays while the slot is not registered again.
    VkSlotStamp stamp, b_stamp;
    CHECK(vk_texture_slot_token_stamped(&refs[0], id, &stamp) == token && stamp.valid);
    CHECK(vk_texture_slot_token_again(&stamp) == token);
    uint64_t b_token = vk_texture_slot_token_stamped(&refs[1], id, &b_stamp);
    CHECK(vk_texture_slot_token_again(&b_stamp) == b_token);
    register_texture(&wrapper, id, &textures[1]);
    CHECK(!vk_texture_slot_token_again(&stamp));   // registered again: read anew under the lock
    CHECK(vk_texture_slot_token(&refs[0], id) != token && lookup(&refs[0], id)->addr == resident[1].pub.addr);
    CHECK(lookup(&refs[1], id)->addr == resident[1].pub.addr && vk_texture_slot_token(&refs[1], id) == b_token);
    uint64_t alias_token = vk_texture_slot_token(&refs[0], id);
    dpool_ref short_table = refs[0]; short_table.count = id;
    CHECK(!lookup(&short_table, id) && !vk_texture_slot_token(&refs[0], -1));
    VkSlotStamp none;
    CHECK(!vk_texture_slot_token_stamped(&refs[0], -1, &none) && !none.valid && !vk_texture_slot_token_again(&none));
    dpool_ref unmapped = {(uintptr_t)pixels[0], (uint32_t)count, 0}; CHECK(!lookup(&unmapped, id));

    // The actual SetPool API captures a value: finalizing/reinitializing the wrapper cannot redirect the recording.
    Cpu record = {0}; record.x[1] = (uintptr_t)&tables[0]; nvnp_nvnCommandBufferSetTexturePool(&record);
    p_set_pool saved = recorded_pool;
    memset(&c, 0, sizeof c); c.x[0] = (uintptr_t)&tables[0]; nvnp_nvnTexturePoolFinalize(&c);
    CHECK(!tables[0].pool && saved.pool.addr == refs[0].addr && saved.pool.count == refs[0].count);
    tables[0] = tables[1]; apply_pool_case(&ex, OP_SET_TEX_POOL, &saved);
    CHECK(ex.tex_pool.addr == refs[0].addr && ex.tex_pool.addr != refs[1].addr);
    dpool_t bad = wrapper; dpool_ref empty;
    bad.off = -1; CHECK(!nvn_dpool_snapshot(&bad, 1u << 20, &empty));
    bad = wrapper, bad.off = 1; CHECK(!nvn_dpool_snapshot(&bad, 1u << 20, &empty));
    bad = wrapper, bad.n = (1 << 20) + 1; CHECK(!nvn_dpool_snapshot(&bad, 1u << 20, &empty));
    bad = wrapper, bad.off = backing[0].size; CHECK(!nvn_dpool_snapshot(&bad, 1u << 20, &empty));
    pool_t unaligned = backing[0]; unaligned.mem++;
    bad = wrapper, bad.pool = &unaligned; CHECK(!nvn_dpool_snapshot(&bad, 1u << 20, &empty));
    pool_t overflow = backing[0]; overflow.mem = (uint8_t *)(uintptr_t)(UINT64_MAX - 31); overflow.size = UINT64_MAX;
    bad = (dpool_t){&overflow, 64, count}; CHECK(!nvn_dpool_snapshot(&bad, 1u << 20, &empty));
    CHECK(nvn_dpool_snapshot(NULL, 1u << 20, &empty) && !empty.addr && !empty.count && !empty.pad);

    // Imported aliases overlap in host bytes, independent of pool-wrapper identity or GPU import/mirror buffers.
    PoolRec alias = {.kind = POOL_IMPORT, .lo = imported[0].lo + 32, .size = imported[0].size - 32};
    provide(&alias); VkDescKey host_key;
    CHECK(vk_mem_descriptor_key(refs[0].addr + (uint64_t)id * 32, 32, &host_key));
    CHECK(!host_key.storage && !imported[0].used && !alias.used && !alias.stouched);
    CHECK(pidx_remove(&pool_index, &imported[0], NULL));
    vk_texture_descriptors_retire(imported[0].lo, imported[0].size, 0);
    VkSlotStamp alias_stamp;
    CHECK(lookup(&refs[0], id) && vk_texture_slot_token_stamped(&refs[0], id, &alias_stamp) == alias_token);
    CHECK(pidx_remove(&pool_index, &alias, NULL));
    vk_texture_descriptors_retire(alias.lo, alias.size, 0);
    CHECK(!lookup(&refs[0], id) && !vk_texture_slot_token_again(&alias_stamp));   // retired: no token without the lock
    provide(&imported[0]); CHECK(!lookup(&refs[0], id)); // reused host range cannot inherit the retired registration

    // Physical and two virtual aliases reach the same descriptor bytes. Remapping one alias selects the other backing.
    PoolRec physical[2] = {{.kind = POOL_PHYS, .storage_serial = 101, .lo = RT_GPU_VA_BASE + 0x100000,
        .size = 2 * NVN_VPOOL_PAGE}, {.kind = POOL_PHYS, .storage_serial = 102,
        .lo = RT_GPU_VA_BASE + 0x200000, .size = 2 * NVN_VPOOL_PAGE}};
    VPage pages[2][2] = {{{&physical[0], 0}, {&physical[0], NVN_VPOOL_PAGE}},
                        {{&physical[0], 0}, {&physical[0], NVN_VPOOL_PAGE}}};
    PoolRec virtuals[2] = {{.kind = POOL_VIRT, .lo = RT_GPU_VA_BASE + 0x300000,
        .size = 2 * NVN_VPOOL_PAGE, .npages = 2, .pt = pages[0]}, {.kind = POOL_VIRT,
        .lo = RT_GPU_VA_BASE + 0x400000, .size = 2 * NVN_VPOOL_PAGE, .npages = 2, .pt = pages[1]}};
    provide(&physical[0]); provide(&physical[1]); provide(&virtuals[0]); provide(&virtuals[1]);
    dpool_ref phys[2] = {{physical[0].lo, (uint32_t)count, 0}, {physical[1].lo, (uint32_t)count, 0}};
    dpool_ref virt[2] = {{virtuals[0].lo, (uint32_t)count, 0}, {virtuals[1].lo, (uint32_t)count, 0}};
    vk_texture_register(&phys[0], id, &textures[0], NULL); vk_texture_register(&phys[1], id, &textures[1], NULL);
    CHECK(vk_texture_slot_token(&phys[0], id) == vk_texture_slot_token(&virt[0], id));
    CHECK(lookup(&virt[1], id)->addr == resident[0].pub.addr);
    pages[0][0].phys = pages[0][1].phys = &physical[1];
    CHECK(lookup(&virt[0], id)->addr == resident[1].pub.addr && lookup(&virt[1], id)->addr == resident[0].pub.addr);
    VkDescKey key; CHECK(vk_mem_descriptor_key(virtuals[0].lo + NVN_VPOOL_PAGE - 16, 32, &key));
    pages[0][1].phys = &physical[0];
    CHECK(!vk_mem_descriptor_key(virtuals[0].lo + NVN_VPOOL_PAGE - 16, 32, &key));
    CHECK(!physical[0].used && !physical[1].used && !virtuals[0].used && !virtuals[1].stouched);
    CHECK(pidx_remove(&pool_index, &physical[0], NULL));
    for (unsigned v = 0; v < 2; v++)
        for (unsigned p = 0; p < 2; p++) if (pages[v][p].phys == &physical[0]) pages[v][p].phys = NULL;
    vk_texture_descriptors_retire(0, 0, physical[0].storage_serial);
    CHECK(!lookup(&virt[1], id));
    physical[0].storage_serial = 103; provide(&physical[0]); CHECK(!lookup(&phys[0], id));
    vk_texture_register(&phys[0], id, &textures[0], NULL); CHECK(lookup(&phys[0], id)->addr == resident[0].pub.addr);

    // Sampler metadata is pool-relative too. No VkSampler is created; overlapping descriptor kinds replace bytes.
    dpool_ref samplers_ref[2] = {refs[0], refs[1]}; samplers_ref[0].addr += 0x4000; samplers_ref[1].addr += 0x4000;
    samp_par repeat = {.ws = NVN_WRAP_MODE_REPEAT}, clamp = {.ws = NVN_WRAP_MODE_CLAMP_TO_EDGE}, got;
    vk_sampler_register(&samplers_ref[0], id, &repeat); vk_sampler_register(&samplers_ref[1], id, &clamp);
    for (unsigned i = 0; i < 3; i++) {
        CHECK(vk_sampler_params(&samplers_ref[order[i]], id, &got));
        CHECK(got.ws == (order[i] ? clamp.ws : repeat.ws));
    }
    vk_sampler_register(&phys[0], id, &repeat);
    CHECK(!lookup(&phys[0], id) && vk_sampler_params(&phys[0], id, &got) && got.ws == repeat.ws);
    vk_texture_register(&phys[0], id, &textures[0], NULL);
    CHECK(lookup(&phys[0], id) && !vk_sampler_params(&phys[0], id, &got));
    key = (VkDescKey){7, 9}; CHECK(!vk_mem_descriptor_key(UINT64_MAX - 15, 32, &key) && !key.storage && !key.offset);
    CHECK(!vk_mem_descriptor_key(physical[0].lo, 0, &key));
    CHECK(!dependencies && !tex_lock.depth && !pool_lock.p);
    printf("descriptor registry/resolver: %u checks passed (index %d, handle %#llx; CPU only)\n",
        checks, id, (unsigned long long)handle);
    for (uint32_t i = 0; i < descriptors.cap; i++) free(descriptors.vals[i]);
    free(descriptors.keys); free(descriptors.vals); free(by_key.keys); free(by_key.vals);
    free(pool_index.e); free(pool_index.maxhi);
    return 0;
}
'''


def source():
    files = {name: (ROOT / 'runtime' / name).read_text() for name in
             ('nvn_api_core.c', 'nvn_api_obj.c', 'nvn_api_cmd.c', 'vk_texture.c', 'vk_memory.c', 'vk_exec.c',
              'platform/common.c')}
    core, obj, cmd, tex, memory, replay, common = files.values()
    parts = [PRELUDE, statement(memory, 'enum { POOL_IMPORT'), typedef(memory, 'VPage'), typedef(memory, 'PoolRec')]
    parts += [typedef(tex, name) for name in ('TexKey', 'Tex', 'TexSlot', 'SampSlot', 'DescSlot')]
    parts += [statement(tex, 'enum { DESC_TEXTURE')]
    for text, names in ((tex, ('static PlatRMutex tex_lock', 'static PlatOnce tex_once', 'static nvn_hmap by_key',
            'static nvn_hmap by_addr', 'static nvn_hmap descriptors', 'static uint32_t slots_gen',
            'static uint32_t samp_gen', 'static uint32_t reg_serial', 'static uint32_t agen[AGEN_N]', 'static uint32_t rgen[RGEN_N]',
            'static uint32_t submit_epoch', 'static volatile int32_t tex_ready')),
            (memory, ('static PlatMutex pool_lock', 'static pidx pool_index')),
            (common, ('static PlatMutex once_lock', 'static PlatCond once_done'))):
        parts += [statement(text, name) for name in names]
    parts += [r'''
static Tex *entry_of(const tex_t *t) { (void)t; dependencies++; rt_fatal("unexpected image creation\n"); }
static int make_fresh_snap(Tex *t) { (void)t; dependencies++; rt_fatal("unexpected upload\n"); }
''']
    for text, starts in ((common, ('void plat_rmutex_lock(', 'void plat_rmutex_unlock(', 'void plat_once(')),
            (core, ('void nvn_called(',)),
            (tex, ('static void tex_init(', 'static void lock(', 'static void unlock(', 'static void log_limited(',
                'static uint32_t agen_idx(', 'static void make_key(', 'static int same_but_swizzle(',
                'static uint32_t rgen_idx(', 'static void rgen_bump(')),
            (memory, ('int vk_mem_descriptor_key(',)),
            (tex, ('static int descriptor_key(', 'static DescSlot *descriptor_slot(', 'static void descriptor_registered(',
                'void vk_texture_register(', 'static Tex *slot_resolve(', 'uint64_t vk_texture_slot_token(', 'uint64_t vk_texture_slot_token_stamped(',
                'uint64_t vk_texture_slot_token_again(',
                'const VkTexImage *vk_texture_by_id_locked(', 'void vk_sampler_register(', 'int vk_sampler_params(',
                'void vk_texture_descriptors_retire('))):
        parts += [function(text, start) for start in starts]
    parts += [r'''
static const GfxBackend backend = {.name = "CPU registry test", .texture_register = vk_texture_register,
    .sampler_register = vk_sampler_register};
const GfxBackend *gfx_backend(void) { return &backend; }
static p_set_pool recorded_pool; static uint32_t recorded_op;
static void *cmd_of(const Cpu *c) { (void)c; return NULL; }
static void *record_pool(void *cmd, uint32_t op, size_t size) {
    (void)cmd; if (size != sizeof recorded_pool) rt_fatal("unexpected record size\n");
    recorded_op = op; memset(&recorded_pool, 0, sizeof recorded_pool); return &recorded_pool;
}
#define REC(cmd, op, T) ((T *)record_pool((cmd), (op), sizeof(T)))
''', function(core, 'static const int devinfo[]') + ';']
    for text, names in ((core, ('nvnDeviceInitialize', 'nvnDeviceGetInteger', 'nvnDeviceGetTextureHandle')),
            (obj, ('nvnTexturePoolInitialize', 'nvnTexturePoolFinalize', 'nvnTexturePoolRegisterTexture')),
            (cmd, ('nvnCommandBufferSetTexturePool', 'nvnCommandBufferSetSamplerPool'))):
        parts += [function(text, 'NVN_API(' + name + ')') for name in names]
    cases = [next(line.strip() for line in replay.splitlines() if line.strip().startswith('case ' + name + ':'))
             for name in ('OP_SET_TEX_POOL', 'OP_SET_SAMP_POOL')]
    parts += ['static void apply_pool_case(VkExec *ex, uint32_t op, const void *a) { switch(op) {\n' +
              '\n'.join(cases) + '\ndefault: rt_fatal("unexpected pool op\\n"); } }', FIXTURE]
    return '\n\n'.join(parts), files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--deps', type=Path, default=ROOT / 'runtime/deps')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with the project Clang executable')
    include = args.deps / 'include' if (args.deps / 'include').is_dir() else args.deps
    if not (include / 'vulkan/vulkan.h').is_file():
        parser.error('--deps must contain the Vulkan headers (directly or under include/)')
    out = ROOT / 'build/tests/descriptor-pool-registry'
    out.mkdir(parents=True, exist_ok=True)
    text, files = source()
    fixture = out / 'fixture.c'
    fixture.write_text(text)
    report = {'sources': {name: hashlib.sha256((ROOT / 'runtime' / name).read_bytes()).hexdigest() for name in files},
              'fixture_sha256': hashlib.sha256(text.encode()).hexdigest(), 'runs': []}
    for opt in ('O0', 'O2'):
        exe = out / ('registry-' + opt + ('.exe' if os.name == 'nt' else ''))
        command = [args.cc, '-' + opt, '-Wall', '-Werror=format', '-fno-strict-aliasing', '-I', str(ROOT / 'runtime'),
                   '-I', str(include), '-I', str(include / 'vma'), str(fixture), '-o', str(exe)]
        built = subprocess.run(command, capture_output=True, text=True)
        (out / (opt + '-compile.log')).write_text(built.stdout + built.stderr + f'exit={built.returncode}\n')
        if built.returncode:
            print(built.stdout + built.stderr)
            return built.returncode
        # No Vulkan library is linked. When LLVM's companion tool is available, additionally inspect PE imports.
        compiler = Path(shutil.which(args.cc) or args.cc)
        objdump = compiler.with_name('llvm-objdump.exe' if os.name == 'nt' else 'llvm-objdump')
        if objdump.is_file():
            imports = subprocess.check_output([str(objdump), '-p', str(exe)], text=True)
            (out / (opt + '-imports.log')).write_text(imports)
            if re.search(r'DLL Name:.*vulkan', imports, re.IGNORECASE):
                raise RuntimeError('unexpected Vulkan loader import; refusing execution')
        ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
        (out / (opt + '-run.log')).write_text(ran.stdout + ran.stderr + f'exit={ran.returncode}\n')
        print(opt + ': ' + ran.stdout.strip())
        report['runs'].append({'optimization': opt, 'exit': ran.returncode})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        if ran.returncode:
            print(ran.stderr)
            return ran.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
