"""CPU checks of passive Vulkan-pool metadata inspection; no Vulkan loader or device.

Usage: python tests/vk_memory_inspect_test.py --cc /path/to/clang --deps /path/to/runtime/deps
The fixture uses actual VkMemInspect/PoolRec/VPage declarations, pidx code, rec_class and vk_mem_inspect.
Original POD pools and single-thread shared-lock providers check full-range bounds, selected-overlap identity,
zeroed failures and unchanged pool/page/index state. Resolve, cache, mirror and shadow dependencies abort.
This does not test concurrent lifetime changes or establish future descriptor/content/visibility validity.
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
from descriptor_pool_registry_test import typedef, statement

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk.h"
#include "nvn_api.h"
#include "nvn_pool_index.h"
#include "platform/platform.h"
static unsigned checks, failures, locks, unlocks;
static unsigned resolve_calls, shadow_calls, mirror_calls, cache_calls, use_calls, pick_calls;
#define CHECK(c) do { checks++; if (!(c)) { failures++; fprintf(stderr,"inspect check line %d\n",__LINE__); } } while(0)
void plat_mutex_lock_shared(PlatMutex *m) { CHECK(!m->p); m->p=(void*)1; locks++; }
void plat_mutex_unlock_shared(PlatMutex *m) { CHECK(m->p==(void*)1); m->p=NULL; unlocks++; }
// The try fails, so every shared acquire goes through the counted plat_mutex_lock_shared (pool_lock_shared).
int plat_mutex_trylock_shared(PlatMutex *m) { (void)m; return 0; }
unsigned rt_trace;
uint64_t kern_now_ns(void) { return 0; }
void perf_lock_wait(int lock, uint64_t ns) { (void)lock; (void)ns; }
'''

FIXTURE = r'''
int vk_mem_resolve(uint64_t a,uint64_t n,VkMemSpan *o) { (void)a;(void)n;(void)o;resolve_calls++;abort(); }
void shadow_touch(PoolRec *p) { (void)p;shadow_calls++;abort(); }
void mirror_create(PoolRec *p) { (void)p;mirror_calls++;abort(); }
int resolve_check_on(void) { cache_calls++;abort(); }
void *rhit_slot(uint64_t a) { (void)a;cache_calls++;abort(); }
void vk_mem_use(VkMemSpan *p) { (void)p;use_calls++;abort(); }
void vk_mem_pick(VkMemSpan *p) { (void)p;pick_calls++;abort(); }
static PoolRec pools[6];
static VPage pages[4];
static void provide(PoolRec *r) {
    pidx_ent e={.lo=r->lo,.hi=r->lo+r->size,.buf=r,.key=r};
    CHECK(pidx_insert(&pool_index,&e));
}
static int inspect(uint64_t a,uint64_t n,VkMemInspect *out) {
    PoolRec before[6];VPage old_pages[4];pidx_ent index[16];uintptr_t high[16];
    memcpy(before,pools,sizeof pools);memcpy(old_pages,pages,sizeof pages);
    CHECK(pool_index.n<=16);
    memcpy(index,pool_index.e,(size_t)pool_index.n*sizeof *index);
    memcpy(high,pool_index.maxhi,(size_t)pool_index.n*sizeof *high);
    unsigned l=locks,u=unlocks;
    if(out)memset(out,0xa5,sizeof *out);
    int ok=vk_mem_inspect(a,n,out);
    CHECK(!memcmp(before,pools,sizeof pools));
    CHECK(!memcmp(old_pages,pages,sizeof pages));
    CHECK(!memcmp(index,pool_index.e,(size_t)pool_index.n*sizeof *index));
    CHECK(!memcmp(high,pool_index.maxhi,(size_t)pool_index.n*sizeof *high));
    CHECK(!pool_lock.p && locks-l==unlocks-u);
    CHECK(!resolve_calls && !shadow_calls && !mirror_calls && !cache_calls && !use_calls && !pick_calls);
    if(!ok && out) { VkMemInspect zero={0};CHECK(!memcmp(out,&zero,sizeof zero)); }
    return ok;
}
int main(void) {
    const uint64_t P=NVN_VPOOL_PAGE;
    PoolRec *im=&pools[0],*ph=&pools[1],*ph2=&pools[2],*v=&pools[3],*other=&pools[4];
    *im=(PoolRec){.kind=POOL_IMPORT,.flags=NVN_MEMORY_POOL_FLAGS_CPU_CACHED,.lo=0x1000,.size=0x1000,
        .import_lo=0,.import_hi=0x4000,.buf=(VkBuffer)(uintptr_t)1,.used=9,.stouched=7,.bufbound=3,.acc_sub=13};
    *ph=(PoolRec){.kind=POOL_PHYS,.flags=NVN_MEMORY_POOL_FLAGS_PHYSICAL,.lo=0x100000,.size=4*P,
        .buf=(VkBuffer)(uintptr_t)2,.alloc=(VmaAllocation)(uintptr_t)3,.used=5};
    *ph2=(PoolRec){.kind=POOL_PHYS,.lo=0x180000,.size=2*P,.buf=(VkBuffer)(uintptr_t)4};
    *v=(PoolRec){.kind=POOL_VIRT,.flags=NVN_MEMORY_POOL_FLAGS_VIRTUAL,.lo=0x200000,.size=4*P,.pt=pages,.npages=4,.used=11};
    for(unsigned i=0;i<3;i++)pages[i]=(VPage){ph,i*P};
    pages[3]=(VPage){ph2,0};
    *other=(PoolRec){.kind=99,.lo=0x300000,.size=P,.buf=(VkBuffer)(uintptr_t)5};
    for(unsigned i=0;i<5;i++)provide(&pools[i]);
    VkMemInspect m;
    CHECK(!inspect(im->lo,1,NULL));
    CHECK(!inspect(0,1,&m));CHECK(!inspect(im->lo,0,&m));
    CHECK(!inspect(UINT64_MAX,1,&m));CHECK(!inspect(UINT64_MAX-15,16,&m));
    CHECK(!inspect(0x800000,1,&m));CHECK(!inspect(other->lo,1,&m));
    CHECK(inspect(im->lo+0x100,0x100,&m));
    CHECK(m.pool_kind==VK_MEM_INSPECT_POOL_IMPORT && m.pool_flags==im->flags && m.cls==VK_MCLS_HOST_CACHED);
    CHECK(m.offset==0x1100 && m.avail==0xf00 && m.buffer_present && !m.mirror_present && !m.shadow_storage);
    CHECK(m.mirror_state==VK_MEM_INSPECT_MIRROR_NONE);
    CHECK(!inspect(im->lo+im->size-8,9,&m));
    CHECK(inspect(im->lo+im->size-8,8,&m) && m.avail==8);
    im->mstate=MS_ON;im->mbuf=(VkBuffer)(uintptr_t)6;im->smap=(uint8_t*)(uintptr_t)7;
    CHECK(inspect(im->lo,1,&m) && m.mirror_state==VK_MEM_INSPECT_MIRROR_ON && m.mirror_present && m.shadow_storage);
    CHECK(m.cls==VK_MCLS_HOST_CACHED); // snapshot precedes optional mirror picking
    im->mstate=MS_REFUSED;im->mbuf=VK_NULL_HANDLE;im->smap=NULL;
    CHECK(inspect(im->lo,1,&m) && m.mirror_state==VK_MEM_INSPECT_MIRROR_REFUSED && !m.mirror_present);
    im->flags=NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS|NVN_MEMORY_POOL_FLAGS_CPU_CACHED;
    CHECK(inspect(im->lo,1,&m) && m.cls==VK_MCLS_HOST_NOACC);
    im->flags=NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED;
    CHECK(inspect(im->lo,1,&m) && m.cls==VK_MCLS_HOST_UNC);
    im->buf=VK_NULL_HANDLE;im->mbuf=(VkBuffer)(uintptr_t)6;
    CHECK(!inspect(im->lo,1,&m));im->buf=(VkBuffer)(uintptr_t)1;
    im->import_hi=im->lo+7;CHECK(!inspect(im->lo,8,&m));im->import_hi=0x4000;
    im->import_lo=im->lo+1;CHECK(!inspect(im->lo,1,&m));im->import_lo=0;
    CHECK(inspect(ph->lo+P+32,64,&m));
    CHECK(m.pool_kind==VK_MEM_INSPECT_POOL_PHYSICAL && m.pool_flags==ph->flags && m.cls==VK_MCLS_DEVICE);
    CHECK(m.offset==P+32 && m.avail==3*P-32 && m.buffer_present && !m.mirror_present && !m.shadow_storage);
    CHECK(inspect(ph->lo+ph->size-4,4,&m) && m.avail==4);
    CHECK(!inspect(ph->lo+ph->size-4,5,&m));
    ph->buf=VK_NULL_HANDLE;CHECK(!inspect(ph->lo,1,&m));ph->buf=(VkBuffer)(uintptr_t)2;
    CHECK(inspect(v->lo+P-16,32,&m));
    CHECK(m.pool_kind==VK_MEM_INSPECT_POOL_VIRTUAL && m.pool_flags==v->flags && m.cls==VK_MCLS_DEVICE);
    CHECK(m.offset==P-16 && m.avail==2*P+16 && m.buffer_present && !m.mirror_present && !m.shadow_storage);
    CHECK(inspect(v->lo+3*P,1,&m) && m.offset==0 && m.avail==P);
    pages[1].off=2*P;
    CHECK(!inspect(v->lo+P-16,32,&m));
    CHECK(inspect(v->lo+16,32,&m) && m.avail==P-16);pages[1].off=P;
    pages[1].phys=NULL;CHECK(!inspect(v->lo+P,1,&m));CHECK(!inspect(v->lo+P-16,32,&m));pages[1].phys=ph;
    pages[1].phys=ph2;CHECK(!inspect(v->lo+P-16,32,&m));pages[1].phys=ph;
    v->pt=NULL;CHECK(!inspect(v->lo,1,&m));v->pt=pages;
    v->npages=0;CHECK(!inspect(v->lo,1,&m));v->npages=1;
    CHECK(!inspect(v->lo+P-16,32,&m));CHECK(inspect(v->lo,1,&m) && m.avail==P);v->npages=4;
    pages[0].off=31;CHECK(!inspect(v->lo,1,&m));pages[0].off=0;
    ph->buf=VK_NULL_HANDLE;CHECK(!inspect(v->lo,1,&m));ph->buf=(VkBuffer)(uintptr_t)2;
    ph->kind=POOL_IMPORT;CHECK(!inspect(v->lo,1,&m));ph->kind=POOL_PHYS;
    ph->size=P-1;CHECK(!inspect(v->lo,1,&m));ph->size=4*P;
    ph->size=UINT64_MAX;pages[0].off=UINT64_MAX-(P-1);
    CHECK(!inspect(v->lo,1,&m));ph->size=4*P;pages[0].off=0;
    uint64_t saved_lo=im->lo,saved_size=im->size;
    im->lo=UINT64_MAX-15;im->size=32;CHECK(!inspect(saved_lo,1,&m));im->lo=saved_lo;im->size=saved_size;
    CHECK(!inspect(v->lo+v->size,1,&m));
    // Match bind_range's one-byte resolution candidate; do not switch to another overlapping backing for size.
    PoolRec *small=&pools[5];
    *small=(PoolRec){.kind=POOL_IMPORT,.flags=NVN_MEMORY_POOL_FLAGS_CPU_CACHED,
        .lo=im->lo+32,.size=16,.import_lo=im->lo,.import_hi=im->lo+0x1000,.buf=(VkBuffer)(uintptr_t)8};
    provide(small);
    CHECK(pidx_find(&pool_index,small->lo,1)->buf==small);
    CHECK(pidx_find(&pool_index,small->lo,32)->buf==im);
    CHECK(!inspect(small->lo,32,&m));
    CHECK(inspect(small->lo,16,&m) && m.avail==16 && m.pool_flags==small->flags);
    CHECK(!inspect(small->lo,17,&m));
    CHECK(pidx_remove(&pool_index,small,NULL));
    // An unbacked one-byte candidate also stays unbacked when a containing pool could cover the full request.
    pidx_ent nested={.lo=im->lo+32,.hi=im->lo+48,.buf=NULL,.key=&m};CHECK(pidx_insert(&pool_index,&nested));
    CHECK(!inspect(im->lo+32,32,&m));
    CHECK(!inspect(im->lo+32,1,&m));CHECK(pidx_remove(&pool_index,&m,NULL));
    CHECK(locks==unlocks && !pool_lock.p);
    printf("passive memory inspection: %u checks, %u failures\n",checks,failures);
    free(pool_index.e);free(pool_index.maxhi);
    return failures!=0;
}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    memory = (ROOT / 'runtime/vk_memory.c').read_text()
    inspect = function(memory, 'int vk_mem_inspect(')
    pieces = [PRELUDE, statement(memory, 'enum { POOL_IMPORT'), statement(memory, 'enum { MS_NONE'),
              typedef(memory, 'VPage'), typedef(memory, 'PoolRec'),
              function(memory, 'static inline int ms_get('),   # the mirror state's atomic load
              statement(memory, 'static PlatMutex pool_lock'), function(memory, 'static void pool_lock_shared('),
              statement(memory, 'static pidx pool_index'),
              function(memory, 'static int rec_class('), inspect, FIXTURE]
    out = ROOT / 'build/tests/vk-memory-inspect'
    out.mkdir(parents=True, exist_ok=True)
    fixture = out / 'fixture.c'
    fixture.write_text('\n\n'.join(pieces))
    report = {'memory_sha256': sha(ROOT / 'runtime/vk_memory.c'), 'header_sha256': sha(ROOT / 'runtime/vk.h'),
              'index_header_sha256': sha(ROOT / 'runtime/nvn_pool_index.h'), 'fixture_sha256': sha(fixture),
              'providers': 'Original single-thread lock/provider counters and metadata-only pools; actual vk.h, PoolRec, VPage, pidx header, rec_class and vk_mem_inspect. No Vulkan library.',
              'runs': []}
    for opt in ('O0', 'O2'):
        exe = out / ('inspect-' + opt + ('.exe' if os.name == 'nt' else ''))
        command = [args.cc, '-' + opt, '-std=gnu17', '-Wall', '-Wextra', '-Werror', '-fno-strict-aliasing',
                   '-I', str(ROOT / 'runtime'), '-I', str(include), '-I', str(include / 'vma'), str(fixture), '-o', str(exe)]
        built = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (out / (opt + '-compile.log')).write_text(built.stdout + built.stderr + f'exit={built.returncode}\n')
        assert built.returncode == 0, built.stdout + built.stderr
        compiler = Path(shutil.which(args.cc) or args.cc)
        objdump = compiler.with_name('llvm-objdump.exe' if os.name == 'nt' else 'llvm-objdump')
        import_check = 'unavailable; no graphics library linked'
        if objdump.is_file():
            imports = subprocess.check_output([str(objdump), '-p', str(exe)], text=True)
            (out / (opt + '-imports.log')).write_text(imports)
            assert not re.search(r'(?:DLL Name:|NEEDED).*?(?:vulkan|nvoglv|amdvlk)', imports, re.I), 'unexpected GPU import'
            import_check = 'PASS: no Vulkan/driver import'
        ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
        (out / (opt + '-run.log')).write_text(ran.stdout + ran.stderr + f'exit={ran.returncode}\n')
        count = re.search(r'(\d+) checks, (\d+) failures', ran.stdout)
        assert count and int(count[2]) == 0, ran.stdout + ran.stderr
        assert ran.returncode == 0, ran.stdout + ran.stderr
        report['runs'].append({'optimization': opt, 'checks': int(count[1]), 'failures': int(count[2]),
                               'executable_exit': ran.returncode,
                               'command': command, 'executable_sha256': sha(exe), 'import_check': import_check})
        print(opt + ': ' + ran.stdout.strip())
    path = out / 'manifest.json'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
