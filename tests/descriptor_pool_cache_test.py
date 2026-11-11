"""Exercise descriptor-table switches through actual replay cache functions.

The original providers return inert image/sampler tokens and fixed registration
generations. No Vulkan loader or GPU is linked. JSON/shader compilation, registry
storage, view creation and pipeline range discovery are outside this CPU seam.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from nvn2_compute_c0_test import function

ROOT = Path(__file__).resolve().parents[1]


def source():
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    replay = (ROOT / 'runtime/vk_exec.c').read_text()
    prelude = r'''
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_GS 2
#define NVN_STAGE_CS 5
#define FM_N 8192
#define TD_N 8192
#define RS_PTRS 4
static struct { unsigned ac_miss; } BS;
static uint64_t bind_gen=1;
static uint32_t c0_binst;
static uint8_t geometry_input;
static uint64_t memory_generation=1;
static unsigned remapped, lookup_count, full_count, locked;
static unsigned checks, failures;
#define CHECK(c,message) do { checks++; if (!(c)) { failures++; fprintf(stderr,"FAIL: %s\n",message); } } while (0)
#define HANDLE ((UINT64_C(257)<<20)|UINT64_C(257))
static const dpool_ref table_a={UINT64_C(0x10000000),259,0};
static const dpool_ref table_b={UINT64_C(0x20000000),259,0};
static const dpool_ref sampler_a={UINT64_C(0x30000000),259,0};
static const dpool_ref sampler_b={UINT64_C(0x40000000),259,0};
static const VkTexImage images[]={
    {.image=(VkImage)(uintptr_t)101,.view_type=VK_IMAGE_VIEW_TYPE_1D,.levels=1,.layers=1,.samples=1},
    {.image=(VkImage)(uintptr_t)202,.view_type=VK_IMAGE_VIEW_TYPE_2D,.levels=1,.layers=1,.samples=1}
};
static const GfxShaderRes resource={.slot=0,.type=GFX_TEX_2D};
void rt_fatal(const char *fmt,...) { (void)fmt; abort(); }
uint64_t vk_mem_gen(void) { return memory_generation; }
uint64_t vk_mem_pool_gen(void) { return memory_generation; }   // a remap below moves the pool mappings
int rt_knob(const char *name,const char *def) { (void)name; return atoi(def); }   // every knob at its default
int rt_ab(void) { return 0; }
uint64_t vk_usnap_gen(void) { return 1; }
uint32_t vk_texture_slots_gen(void) { return 7; }
int vk_exec_par_slice(void) { return 0; }
uint32_t vk_texture_epoch(void) { return 1; }
uint32_t vk_sampler_gen(void) { return 9; }
void vk_texture_lock(void) { if (locked++) abort(); }
void vk_texture_unlock(void) { if (!locked--) abort(); }
uint64_t vk_texture_slot_token(const dpool_ref *pool,int id) {
    if (id!=257 || id>=(int)pool->count) return 0;
    return (pool->addr+32u*(unsigned)id)^memory_generation;
}
const VkTexImage *vk_texture_by_id_locked(const dpool_ref *pool,int id,view_t *view,int *has_view) {
    if (!locked) abort();
    lookup_count++;
    if (view) memset(view,0,sizeof *view);
    if (has_view) *has_view=0;
    if (id!=257 || id>=(int)pool->count) return NULL;
    if (pool->addr==table_a.addr) return &images[remapped];
    if (pool->addr==table_b.addr) return &images[1];
    return NULL;
}
static VkSampler sample_token(const dpool_ref *pool) {
    if (pool->count<=257) return VK_NULL_HANDLE;
    if (pool->addr==sampler_a.addr) return (VkSampler)(uintptr_t)(remapped?505:303);
    if (pool->addr==sampler_b.addr) return (VkSampler)(uintptr_t)404;
    return VK_NULL_HANDLE;
}
int vk_pipeline_tex_type(int target) { (void)target; abort(); } // No view target overrides.
void vk_pipeline_answers(const VkPipeState *s,GfxShaderAnswers *vs,GfxShaderAnswers *fs) {
    (void)s; (void)vs; (void)fs;
}
int gfx_shader_resources(const GfxShader *s,int kind,const GfxShaderRes **out) {
    if (s!=(const GfxShader *)&resource || kind!=GFX_RES_TEX) abort();
    *out=&resource; return 1;
}
int gfx_shader_stage(const GfxShader *s) {
    if (s!=(const GfxShader *)&resource) abort();
    return GFX_STAGE_FS;   // this original provider represents one fragment texture binding
}
static int shader_memo_on(void) { return 1; }
static int td_mode(void) { return 1; }
// Full descriptor construction is a bounded provider: this test checks whether
// the production TD lookup reuses or rebuilds it for the active tables.
static void tex_info_full(VkExec *ex,const GfxShaderRes *r,uint64_t h,int storage,int separate,
                          VkDescriptorImageInfo *out,VkImage *image) {
    (void)r;
    if (storage || separate) abort();
    const VkTexImage *img=vk_texture_by_id_locked(&ex->tex_pool,(int)(h&0xfffff),NULL,NULL);
    *image=img?img->image:VK_NULL_HANDLE;
    *out=(VkDescriptorImageInfo){sample_token(&ex->samp_pool),(VkImageView)*image,VK_IMAGE_LAYOUT_GENERAL};
    full_count++;
}
static void support_in(const VkExec *ex,float in[3]) { (void)ex; (void)in; abort(); }
static uint64_t ubo_ptr(const VkExec *ex,int st,const GfxShaderRes *r) { (void)ex; (void)st; (void)r; abort(); }
'''
    ac = function(draw, 'static struct { uint64_t context, tex[32],') + ' AC[6];\n'
    serial = next(line for line in draw.splitlines() if line.startswith(('static uint64_t descriptor_serial', 'static uint64_t _Thread_local descriptor_serial')))
    sub = function(draw, 'typedef struct { VkImage image; uint32_t l0,') + ' SubRange;\n'
    fm = function(draw, 'static struct { uint64_t context, h, tok; uint32_t gen; SubRange r;') + ' FM[FM_N];\n'
    td_start = draw.index('#define TD_N ')
    td_begin = draw.index('static struct {', td_start)
    td = draw[td_begin:draw.index('} TD_m[TD_N];', td_begin)] + '} TD[TD_N];'
    memo_end = draw.index('} MemoKey;')
    memo = draw[draw.rfind('typedef struct {', 0, memo_end):memo_end + len('} MemoKey;')]
    sm = function(draw, 'static struct { int on, valid, has_fs') + ' SM = {.on=-1};\n'
    range_types = '\n'.join(function(draw, prefix) + suffix for prefix, suffix in (
        ('typedef struct { uint32_t off, len;', ' RsRange;'),
        ('typedef struct { int st; const GfxShaderRes *res;', ' RsPtr;'),
        ('typedef struct { uint32_t n, bytes; uint8_t sup, c0,', ' RsList;')))
    # TOTK_BIND_STATS's counters, which bind_sig updates only while they are on.
    bsf_begin = draw.index('enum { BSF_INFO,')
    bsf = draw[bsf_begin:draw.index('} BSF;', bsf_begin) + len('} BSF;')]
    # Only the fields consumed by bind_sig are needed; pipeline range discovery
    # supplies one original fragment texture handle range, without a GPU pipeline.
    adapters = r'''
static int bind_stats_on(void) { return 0; }   // TOTK_BIND_STATS off
static int rs_class(const VkExec *ex,const RsRange *r) { (void)ex; (void)r; abort(); }
static struct {
    uint64_t h[32],gen;
    uint32_t hn;
    const VkPipeInfo *info;
    int on;
    uint8_t body[16];
} RS;
static const RsList *rs_list(const VkExec *ex,const VkPipeInfo *info) {
    (void)ex; (void)info;
    static RsList *list;
    if (!list) {
        list=calloc(1,sizeof *list+sizeof *list->r);
        if (!list) abort();
        list->ok=1; list->n=1; list->bytes=8;
        list->r[0]=(RsRange){offsetof(VkExec,tex[NVN_STAGE_FS][0]),8};
    }
    return list;
}
'''
    pieces = [function(draw, prefix) for prefix in (
        'static int nvn_stage_of(', 'static int gfx_type_of(', 'static uint64_t token_of(', 'static int slot_still(',
        'static uint8_t type_of_handle(', 'static int in_fbl(', 'static SubRange sub_of(',
        'static uint32_t attached_as(', 'static void descriptor_state(VkExec *ex) {', 'static void answers_for(',
        'static SubRange sampled_sub(', 'static uint32_t feedback_scan(', 'static void tex_info_lookup(',
        'static void memo_key(', 'static int memo_match(', 'static int bind_sig(')]
    # These are the exact production replay cases, not a replacement pool setter.
    cases = '\n'.join(line for line in replay.splitlines()
                      if line.strip().startswith(('case OP_SET_TEX_POOL:', 'case OP_SET_SAMP_POOL:')))
    selection = 'static void select_pool(VkExec *ex,int op,const p_set_pool *a) {\n switch(op) {\n' + cases + '\n default: abort();\n }\n}\n'
    tests = r'''
static void select_table(VkExec *ex,int op,dpool_ref pool) {
    p_set_pool payload={pool};
    select_pool(ex,op,&payload);
    payload.pool=(dpool_ref){0};
}
static void expect(VkExec *ex,unsigned type,unsigned image,unsigned sampler,unsigned feedback) {
    GfxShaderAnswers answers;
    answers_for(ex,NVN_STAGE_FS,NULL,1,&answers);
    CHECK(answers.tex_type[0]==type && answers.img_type[0]==type,"unchanged handles select active-table texture/image types");
    CHECK(!answers.tex_type[1] && !answers.img_type[1],"unqueried slots remain zero");
    VkDescriptorImageInfo out;
    int held=0;   // tex_info_lookup takes the texture lock only when it looks up, and leaves it to the caller
    tex_info_lookup(ex,&resource,HANDLE,0,0,vk_texture_slots_gen(),vk_texture_epoch(),vk_sampler_gen(),&held,&out);
    if (held) vk_texture_unlock();
    CHECK((uintptr_t)out.imageView==image && (uintptr_t)out.sampler==sampler &&
          out.imageLayout==VK_IMAGE_LAYOUT_GENERAL,"sampled descriptor uses active texture and sampler tables");
    CHECK(feedback_scan(ex,(const GfxShader *)&resource)==feedback,"feedback resolves the active image's attachment");
}
int main(void) {
    VkExec ex={0}; prog_t prog={.serial=17}; VkPipeInfo info={0};
    ex.prog=&prog; ex.tex[NVN_STAGE_FS][0]=HANDLE; ex.img[NVN_STAGE_FS][0]=HANDLE;
    ex.pass_ncolor=2;
    ex.color[0]=(VkExecTarget){.image=images[0].image,.layers=1};
    ex.color[1]=(VkExecTarget){.image=images[1].image,.layers=1};
    select_table(&ex,OP_SET_TEX_POOL,table_a);
    select_table(&ex,OP_SET_SAMP_POOL,sampler_a);
    CHECK(ex.tex_pool.addr==table_a.addr && ex.samp_pool.addr==sampler_a.addr,"replay owns table snapshots by value");
    expect(&ex,GFX_TEX_1D,101,303,1);
    GfxShaderAnswers vertex;
    ex.tex[NVN_STAGE_VS][0]=HANDLE; ex.img[NVN_STAGE_VS][0]=HANDLE;
    answers_for(&ex,NVN_STAGE_VS,NULL,1,&vertex);
    CHECK(vertex.tex_type[0]==GFX_TEX_1D && vertex.img_type[0]==GFX_TEX_1D,"vertex stage warms its independent texture-type cache");
    unsigned lookups=lookup_count,builds=full_count;
    expect(&ex,GFX_TEX_1D,101,303,1);
    CHECK(lookup_count==lookups && full_count==builds,"unchanged tables reuse AC, TD and feedback memos");
    CHECK(!bind_sig(&ex,&info) && bind_sig(&ex,&info),"graphics signature warms then reuses unchanged inputs");
    memo_key(&ex,7,&SM.k); SM.valid=1;
    CHECK(memo_match(&ex,7,&SM.k),"shader memo reuses unchanged table state");

    select_table(&ex,OP_SET_TEX_POOL,table_b);
    CHECK(!memo_match(&ex,7,&SM.k),"table B invalidates shader memo despite unchanged numeric handles");
    CHECK(!bind_sig(&ex,&info) && bind_sig(&ex,&info),"table B rebuilds graphics bindings once");
    expect(&ex,GFX_TEX_2D,202,303,2);
    answers_for(&ex,NVN_STAGE_VS,NULL,1,&vertex);
    CHECK(vertex.tex_type[0]==GFX_TEX_2D && vertex.img_type[0]==GFX_TEX_2D,"table B also refreshes the previously warmed vertex cache");
    memo_key(&ex,7,&SM.k); SM.valid=1;
    select_table(&ex,OP_SET_TEX_POOL,table_a);
    CHECK(!memo_match(&ex,7,&SM.k),"return to table A cannot reuse table B shader memo");
    CHECK(!bind_sig(&ex,&info),"return to table A changes graphics signature");
    expect(&ex,GFX_TEX_1D,101,303,1);

    memo_key(&ex,7,&SM.k); SM.valid=1;
    select_table(&ex,OP_SET_SAMP_POOL,sampler_b);
    CHECK(!memo_match(&ex,7,&SM.k),"sampler table switch invalidates combined-descriptor shader memo");
    CHECK(!bind_sig(&ex,&info),"sampler table switch changes graphics signature");
    expect(&ex,GFX_TEX_1D,101,404,1);
    select_table(&ex,OP_SET_SAMP_POOL,sampler_a);
    expect(&ex,GFX_TEX_1D,101,303,1);

    memo_key(&ex,7,&SM.k); SM.valid=1;
    dpool_ref short_table=table_a; short_table.count=257;
    select_table(&ex,OP_SET_TEX_POOL,short_table);
    CHECK(!memo_match(&ex,7,&SM.k),"table count change invalidates shader memo at the same address");
    CHECK(!bind_sig(&ex,&info),"table count change alters graphics signature");
    expect(&ex,0,0,303,0);
    select_table(&ex,OP_SET_TEX_POOL,table_a);
    expect(&ex,GFX_TEX_1D,101,303,1);

    memo_key(&ex,7,&SM.k); SM.valid=1;
    CHECK(!bind_sig(&ex,&info) && bind_sig(&ex,&info),"sampler A warms graphics signature before remapping");
    remapped=1; memory_generation++;
    CHECK(!memo_match(&ex,7,&SM.k),"same table addresses with new backing invalidate shader memo");
    CHECK(!bind_sig(&ex,&info),"mapping generation changes graphics signature");
    expect(&ex,GFX_TEX_2D,202,505,2);
    lookups=lookup_count; builds=full_count;
    expect(&ex,GFX_TEX_2D,202,505,2);
    CHECK(lookup_count==lookups && full_count==builds,"remapped tables can warm caches again");
    CHECK(ex.tex[NVN_STAGE_FS][0]==HANDLE && ex.img[NVN_STAGE_FS][0]==HANDLE &&
          vk_texture_slots_gen()==7 && vk_sampler_gen()==9,"handles and registration generations stayed unchanged");
    CHECK(!locked,"texture lock balanced");
    printf("descriptor pool caches: %u checks, %u failures\n",checks,failures);
    return failures!=0;
}
'''
    return '\n'.join((prelude, ac, serial, sub, fm, td, memo, sm, range_types, bsf, adapters,
                      'static void descriptor_state(VkExec *ex);', *pieces, selection, tests))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc'))
    parser.add_argument('--deps', type=Path, required=True, help='runtime dependency root with include/vulkan and include/vma')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    out = ROOT / 'build/test/descriptor-pool-cache'
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'cache.c'
    src.write_text(source())
    failed = 0
    for opt in ('-O0', '-O2'):
        exe = out / ('cache' + opt + ('.exe' if os.name == 'nt' else ''))
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
