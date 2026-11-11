"""CPU-only guarded descriptor resolution through actual replay functions and types.

Vulkan handles are inert tokens. No loader or GPU API is linked or called. The
memory provider, scratch allocator and accounting hooks are bounded test stubs;
packing, pointer selection, range clipping and inline-update lookup are real.
"""
import argparse
from pathlib import Path
import subprocess

from nvn2_shader_control_test import function

ROOT = Path(__file__).resolve().parents[1]


def source():
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    before = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#define ZERO_BYTES 65536u
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_CS 5
enum { W_SSBO_ALIGN, W_SSBO_SLOT };
static const Nvn2C0Layout *c0_lay;
static const uint32_t (*c0_push)[NVN2_ROOT_WORDS];
static uint32_t c0_dim[3], c0_binst;
static struct { VkBuffer zbuf; } Z;
static struct { int volatile_; } RS;
static int c0_enabled = 1;
static int c0_on(void) { return c0_enabled; }
VkCaps vk_caps;
unsigned rt_trace;
int vk_mirror_on;
static unsigned checks, failures, wrote, dropped, noted, copies;
static uint64_t last_note_addr, last_note_size;
#define CHECK(c, message) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL: %s\n", message); } } while (0)
void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
static void warn_once(int id, const char *fmt, ...) { (void)id; (void)fmt; }
void vk_mem_count(int cls, int use, uint64_t bytes) { (void)cls; (void)use; (void)bytes; }
static void count_use(int cls, int use, uint64_t bytes) { vk_mem_count(cls, use, bytes); }
// Profiling is disabled in this CPU fixture; reaching either profiling-only hook is an unexpected dependency.
int vk_sync_noting(void) { return 0; }
uint64_t nvn2_origin(const uint8_t *code) { (void)code; abort(); }
void vk_sync_note_write(uint64_t addr, uint64_t size, uint64_t stage) {
    (void)addr; (void)size; (void)stage; abort();
}
void vk_mem_use(VkMemSpan *span) { (void)span; }
void vk_mem_bound_note(void *pool) { (void)pool; }
void vk_mem_gpu_wrote(const VkMemSpan *span, uint64_t n) { (void)span; (void)n; wrote++; }
void vk_usnap_drop(uint64_t addr, uint64_t n) { (void)addr; (void)n; dropped++; }
static void note_written(VkExec *ex, uint64_t addr, uint64_t n) {
    (void)ex; noted++; last_note_addr=addr; last_note_size=n;
}
static uint8_t scratch_data[4096];
static uint8_t *scratch(VkExec *ex, uint64_t n, VkBuffer *buffer, VkDeviceSize *offset) {
    (void)ex; if (n > sizeof scratch_data) abort();
    copies++;
    *buffer=(VkBuffer)(uintptr_t)99; *offset=0; return scratch_data;
}
// Original deterministic backing provider. Addresses are tokens unless explicitly
// registered from an owned CPU object for the actual user-UBO pointer read.
static struct { uint64_t addr, size; VkDeviceSize offset; int device; } pools[4];
static unsigned npools;
static void pool(uint64_t addr, uint64_t size) {
    if (npools == 4) abort();
    pools[npools++] = (typeof(pools[0])){addr,size,0,0};
}
int vk_mem_resolve(uint64_t addr, uint64_t n, VkMemSpan *span) {
    for (unsigned i=0;i<npools;i++) {
        if (addr < pools[i].addr || addr-pools[i].addr > pools[i].size ||
            n > pools[i].size-(addr-pools[i].addr)) continue;
        uint64_t delta=addr-pools[i].addr;
        *span=(VkMemSpan){.buffer=(VkBuffer)(uintptr_t)(i+1),.offset=pools[i].offset+delta,
            .avail=pools[i].size-delta,.alloc=pools[i].device?(VmaAllocation)(uintptr_t)1:NULL,
            .cls=VK_MCLS_HOST_CACHED};
        return 1;
    }
    return 0;
}
'''
    limits = function(draw, 'static struct { uint64_t ubo_max,') + ' LIM;\n'
    limits += '''static void limits_init(void) {
    LIM.ubo_max=65536; LIM.ubo_align=256; LIM.sbo_max=65536; LIM.sbo_align=256;
}\n'''
    updates = draw[draw.index('#define UPD_N '):draw.index('static void upd_note(')]
    pieces = [function(draw, start) for start in (
        'static VkDescriptorBufferInfo zero_info(', 'static int bind_range(',
        'void vk_c0_fill(', 'static int find_stage_scan(',
        'static int ssbo_slot(', 'static int ubo_ptr_res(')]
    pieces += [updates, function(draw, 'static void upd_note('), function(draw, 'static int upd_read('),
               function(draw, 'static uint64_t ubo_ptr('),
               '#define find_stage find_stage_scan\n',
               function(draw, 'static p_bind_buf guarded_storage_binding('),
               function(draw, 'static VkDescriptorBufferInfo ssbo_info(')]
    after = r'''
int main(void) {
    const uint64_t a=UINT64_C(0x200000000), b=UINT64_C(0x500000000);
    pool(a,512); pool(b,256);
    Z.zbuf=(VkBuffer)(uintptr_t)77;
    limits_init(); vk_caps.sbuf_align=256;
    VkExec ex={0}; ex.fence=10;
    Nvn2C0Layout layout={.ubos=1,.ssbos=1,.roots={8}};
    c0_lay=&layout; c0_push=ex.push[1];
    memcpy(&ex.push[1][0][0],&a,8);
    ex.ubo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=a,.size=16};
    ex.ssbo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=b,.size=32};
    GfxShaderRes root={.cbuf_slot=0,.handle=4,.flags=GFX_RES_SSBO_GUARDED};
    GfxShaderRes typed=root; typed.handle=(uint16_t)(nvn2_c0_part(&layout,1)/4);
    GfxShaderRes storage=root; storage.handle=(uint16_t)(nvn2_c0_part(&layout,2)/4);
    VkDescriptorBufferInfo info=ssbo_info(&ex,NVN_STAGE_CS,&root,0);
    CHECK(info.buffer==(VkBuffer)(uintptr_t)1 && info.offset==0 && info.range==512,
          "raw c0 root pointer resolves only its actual backing");
    info=ssbo_info(&ex,NVN_STAGE_CS,&typed,0);
    CHECK(info.buffer==(VkBuffer)(uintptr_t)1 && info.range==16,"typed UBO keeps its declared byte extent");
    info=ssbo_info(&ex,NVN_STAGE_CS,&storage,0);
    CHECK(info.buffer==(VkBuffer)(uintptr_t)2 && info.range==32,"typed SSBO keeps its declared byte extent");
    uint64_t scalar=32; memcpy(&ex.push[1][0][0],&scalar,8);
    info=ssbo_info(&ex,NVN_STAGE_CS,&root,0);
    CHECK(info.buffer==Z.zbuf && info.range==1,"unbacked scalar candidate exposes zero uint elements");
    GfxShaderRes legacy=root; legacy.flags=0;
    info=ssbo_info(&ex,NVN_STAGE_CS,&legacy,0);
    CHECK(info.buffer==Z.zbuf && info.range==ZERO_BYTES,"unguarded invalid resource retains the 64-KiB fallback");
    memcpy(&ex.push[1][0][0],&a,8);
    c0_enabled=0;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&root,0).range==1,"disabled c0 candidate stays invalid");
    c0_enabled=1;
    CHECK(ssbo_info(&ex,NVN_STAGE_FS,&root,0).range==1,"guarded resolution requires compute stage");
    c0_lay=NULL;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&root,0).range==1,"guarded resolution requires the NVN2 layout");
    c0_lay=&layout;
    GfxShaderRes outside=root; outside.handle=VK_C0_BYTES/4-1;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&outside,0).range==1,"last c0 word cannot supply an eight-byte pointer");

    // Exact mode has a naturally four-aligned pointer ending in4. Explicit
    // rounded modes bind the preceding bytes too, exactly as the compiler guard.
    uint64_t unaligned=a+4;
    ex.ubo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=unaligned,.size=16};
    const uint32_t aligns[]={0,4,16,256};
    for (unsigned i=0;i<4;i++) {
        vk_caps.sbuf_align=aligns[i]; LIM.sbo_align=aligns[i]?aligns[i]:4;
        info=ssbo_info(&ex,NVN_STAGE_CS,&typed,0);
        uint64_t lead=aligns[i]?unaligned&(aligns[i]-1):0;
        CHECK(info.offset==4-lead && info.range==16+lead,"exact/rounded descriptor offset and extent agree");
    }
    vk_caps.sbuf_align=256; LIM.sbo_align=256;
    ex.ubo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=a,.size=16};
    wrote=dropped=noted=0;
    info=ssbo_info(&ex,NVN_STAGE_CS,&typed,1);
    CHECK(info.range==16 && wrote==1 && dropped==1 && noted==1,
          "valid guarded output keeps mirror/snapshot/write accounting");
    CHECK(last_note_addr==a && last_note_size==16,"written range is the exact bound range");
    wrote=dropped=noted=0;
    memcpy(&ex.push[1][0][0],&scalar,8);
    info=ssbo_info(&ex,NVN_STAGE_CS,&root,1);
    CHECK(info.range==1 && !wrote && !dropped && !noted,"invalid output does not publish a guest write");
    memcpy(&ex.push[1][0][0],&a,8);
    LIM.sbo_max=64;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&root,0).range==64,"device storage range limit is preserved");
    LIM.sbo_max=65536;

    // Bounded inline c1 uses the actual program stage scanner and control fields.
    uint8_t ctl[0x880]={0}, code[32]={0};
    uint32_t c1=16,len=24,stage=NVN_STAGE_CS;
    memcpy(ctl+0x700,&c1,4); memcpy(ctl+0x704,&len,4); memcpy(ctl+0x714,&stage,4);
    memcpy(code+16,&b,8);
    memcpy(code+24,&a,8); // physically readable, valid-looking pointer beyond the logical c1 tail
    prog_t program={.nstage=1}; program.sd[0].control=ctl;
    program.sd[0].data=(typeof(program.sd[0].data))(uintptr_t)code; ex.prog=&program;
    GfxShaderRes inline_pointer={.cbuf_slot=1,.flags=GFX_RES_SSBO_GUARDED};
    info=ssbo_info(&ex,NVN_STAGE_CS,&inline_pointer,0);
    CHECK(info.buffer==(VkBuffer)(uintptr_t)2 && info.range==256,"bounded program c1 pointer resolves");
    inline_pointer.handle=2;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&inline_pointer,0).range==1,"c1 read cannot cross its declared tail");
    inline_pointer.handle=0; c1=25; memcpy(ctl+0x700,&c1,4);
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&inline_pointer,0).range==1,"c1 offset beyond code length is invalid");
    c1=24; memcpy(ctl+0x700,&c1,4);
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&inline_pointer,0).range==1,"empty c1 remains invalid");
    c1=16; memcpy(ctl+0x700,&c1,4); program.nstage=0;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&inline_pointer,0).range==1,"missing program stage remains invalid");
    program.nstage=1; ex.prog=NULL;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&inline_pointer,0).range==1,"missing program remains invalid");

    // Guarded aliases must share the actual backing, not independent transient
    // copies whose writes never reach the guest. Keep the legacy read fallback.
    _Alignas(256) uint8_t alias_data[256]={0};
    uint64_t alias_addr=(uint64_t)(uintptr_t)alias_data;
    pool(alias_addr,sizeof alias_data); pools[npools-1].offset=1;
    memcpy(&ex.push[1][0][0],&alias_addr,8);
    copies=wrote=dropped=noted=0;
    info=ssbo_info(&ex,NVN_STAGE_CS,&root,0);
    CHECK(info.buffer==Z.zbuf && info.range==1 && !copies,
          "misaligned guarded input never fabricates an independent scratch alias");
    info=ssbo_info(&ex,NVN_STAGE_CS,&root,1);
    CHECK(info.buffer==Z.zbuf && info.range==1 && !copies && !wrote && !dropped && !noted,
          "misaligned guarded output cannot publish a lost scratch write");
    legacy.cbuf_slot=3; legacy.handle=0;
    uint64_t alias_pointer=alias_addr;
    ex.ubo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=(uint64_t)(uintptr_t)&alias_pointer,.size=8};
    UPD.fence=ex.fence; UPD.n=1;
    UPD.u[0]=(typeof(UPD.u[0])){(uint64_t)(uintptr_t)&alias_pointer,
        (uint64_t)(uintptr_t)&alias_pointer+8,(const uint8_t *)&alias_pointer};
    copies=0;
    info=ssbo_info(&ex,NVN_STAGE_CS,&legacy,0);
    CHECK(info.buffer==(VkBuffer)(uintptr_t)99 && info.range==256 && copies==1,
          "legacy imported input retains its transient copy fallback");
    UPD.n=0;
    memcpy(&ex.push[1][0][0],&a,8);

    // User-UBO pointers retain the newest whole inline update from this fence.
    uint64_t pointer_block[2]={a,b};
    pool((uint64_t)(uintptr_t)pointer_block,sizeof pointer_block);
    ex.ubo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=(uint64_t)(uintptr_t)pointer_block,.size=8};
    GfxShaderRes user={.cbuf_slot=3,.flags=GFX_RES_SSBO_GUARDED};
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&user,0).buffer==(VkBuffer)(uintptr_t)1,"user UBO reads its owned pointer block");
    p_update_ub *update=malloc(sizeof *update+8);
    if (!update) return 2;
    update->dst=(uint64_t)(uintptr_t)pointer_block; update->size=8;
    memcpy(update->data,&b,8);
    upd_note(&ex,update);
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&user,0).buffer==(VkBuffer)(uintptr_t)2,"same-fence inline update wins over old CPU bytes");
    ex.fence++;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&user,0).buffer==(VkBuffer)(uintptr_t)1,"another fence cannot reuse stale update data");
    user.handle=2; // valid-looking second pointer exists physically, outside the declared UBO
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&user,0).range==1,"user UBO pointer pair must fit the declared block");
    user.cbuf_slot=2; user.handle=0;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&user,0).range==1,"unsupported cbuf never fabricates a backing");

    // ubo_info reserves c17 for the zero-filled compute push bank, not UBO14.
    GfxShaderRes pushed={.cbuf_slot=17,.flags=GFX_RES_SSBO_GUARDED};
    uint64_t pushed_addr=b+4;
    memcpy(ex.push[1][0],&pushed_addr,8); ex.push_set[1][0]=3;
    ex.ubo[NVN_STAGE_CS][14]=(p_bind_buf){.addr=(uint64_t)(uintptr_t)pointer_block,.size=16};
    CHECK(guarded_storage_binding(&ex,NVN_STAGE_CS,&pushed).addr==pushed_addr &&
          ssbo_info(&ex,NVN_STAGE_CS,&pushed,0).buffer==(VkBuffer)(uintptr_t)2,
          "c17 sees the actual compute pushes rather than unrelated user UBO14");
    ex.ubo[NVN_STAGE_CS][14]=(p_bind_buf){0};
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&pushed,0).buffer==(VkBuffer)(uintptr_t)2,
          "c17 push pointer does not require a bound user UBO14");
    ex.push_set[1][0]=2;
    CHECK(guarded_storage_binding(&ex,NVN_STAGE_CS,&pushed).addr==b,
          "unset low push word reads zero even if stale raw storage is nonzero");
    ex.push_set[1][0]=1;
    CHECK(guarded_storage_binding(&ex,NVN_STAGE_CS,&pushed).addr==4 &&
          ssbo_info(&ex,NVN_STAGE_CS,&pushed,0).range==1,
          "unset high push word reads zero rather than stale pointer bits");
    ex.push_set[1][0]=0;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&pushed,0).range==1,"unwritten c17 has no pointer backing");
    pushed.handle=62; memcpy(ex.push[1][0]+62,&pushed_addr,8);
    ex.push_set[1][0]=UINT64_C(3)<<62;
    CHECK(guarded_storage_binding(&ex,NVN_STAGE_CS,&pushed).addr==pushed_addr,
          "last complete c17 pair retains both pushed words");
    pushed.handle=63;
    CHECK(ssbo_info(&ex,NVN_STAGE_CS,&pushed,0).range==1,"c17 pointer pair cannot cross its bank end");
    free(update);
    printf("storage runtime: %u checks, %u failures\n",checks,failures);
    return failures!=0;
}
'''
    return before + limits + '\n'.join(pieces) + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', required=True)
    parser.add_argument('--deps', type=Path, required=True, help='runtime dependency root with include/vulkan and include/vma')
    args = parser.parse_args()
    out = ROOT / 'build/test/storage-candidates-runtime'
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'runtime.c'
    src.write_text(source())
    for opt in ('-O0', '-O2'):
        exe = out / ('runtime' + opt + '.exe')
        command = [args.cc, opt, '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                   '-I', str(args.deps / 'include'), '-I', str(args.deps / 'include/vma'), str(src),
                   str(ROOT / 'runtime/gfx_c0.c'), '-o', str(exe)]   # vk_c0_fill's bank is gfx_c0_fill's
        run = subprocess.run(command, capture_output=True, text=True)
        (out / (opt + '-compile.log')).write_text(run.stdout + run.stderr)
        if run.returncode:
            print(run.stdout + run.stderr)
            return run.returncode
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        (out / (opt + '-run.log')).write_text(run.stdout + run.stderr)
        print(opt, run.stdout + run.stderr)
        if run.returncode:
            return run.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
