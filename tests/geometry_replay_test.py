"""CPU checks of actual ordinary-GS replay guards, stage lookup, caches and write publication.

Reuses the original bounded descriptor providers; no Vulkan loader, GPU or game
is linked. Shader compilation and rendering remain integration-probe concerns.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

import descriptor_pool_cache_test as descriptor
from nvn2_compute_c0_test import function

ROOT = Path(__file__).resolve().parents[1]


def record(text, name):
    end = text.index('} ' + name + ';') + len('} ' + name + ';')
    return text[text.rfind('typedef struct {', 0, end):end]


def source():
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    shader = (ROOT / 'runtime/gfx_shader.c').read_text()
    replay = (ROOT / 'runtime/vk_exec.c').read_text()
    pipeline = (ROOT / 'runtime/vk_pipeline.c').read_text()
    base = descriptor.source().replace('int main(void)', 'static int descriptor_checks(void)', 1)
    base = base.replace('#include "vk_exec.h"', '#include "vk_exec.h"\n#include "nvn_hmap.h"\n#include "nvn_geometry_control.h"')
    base = base.replace('uint8_t body[16];', 'uint8_t body[8192]; int volatile_;')
    old_stage = function(base, 'int gfx_shader_stage(')
    old_res = function(base, 'int gfx_shader_resources(')
    providers = r'''
typedef struct { int stage; const GfxShaderRes *res; } OriginalShader;
static const GfxShaderRes gs_texture={.slot=0,.type=GFX_TEX_2D};
static const OriginalShader gs_shader={GFX_STAGE_GS,&gs_texture}, vs_shader={GFX_STAGE_VS,&gs_texture};
int gfx_shader_stage(const GfxShader *s) {
    return s==(const GfxShader *)&resource?GFX_STAGE_FS:((const OriginalShader *)s)->stage;
}
int gfx_shader_resources(const GfxShader *s,int kind,const GfxShaderRes **out) {
    *out=s==(const GfxShader *)&resource?&resource:((const OriginalShader *)s)->res;
    return kind==GFX_RES_TEX && *out?1:0;
}
static int enabled,fix_c0,warnings,async_on,translate_fail,translate_pending,translations,awaits,last_stage;
static GfxShaderAnswers last_answers;
static const uint8_t *native_code;
int cfg_bool(const char *name) { return !strcmp(name,"TOTK_NATIVE_GS")?0:!strcmp(name,"TOTK_FIX_C0")?fix_c0:enabled; }
static int c0_on(void) { return fix_c0; }
static void warn_once(int key,const char *format,...) { (void)key;(void)format;warnings++; }
uint64_t nvn2_origin(const uint8_t *code) { return code==native_code?17:0; }
int vk_geometry_supported(void) { return 1; }
GfxStats gfx_counters;
int vk_pipeline_async_on(void) { return async_on; }
uint32_t gfx_shader_slot_mask(const GfxShader *s) {(void)s;return last_answers.geometry_input==GFX_GEOMETRY_POINTS?0:1;}
const GfxShader *gfx_shader_get(int stage,const uint8_t *ctl,const uint8_t *code,uint64_t len,
                              const uint8_t *c1,const GfxShaderAnswers *a) {
    (void)ctl;(void)code;(void)len;(void)c1;translations++;last_stage=stage;last_answers=*a;
    return translate_fail?NULL:(const GfxShader *)&gs_shader;
}
const GfxShader *gfx_shader_get_async(int stage,const uint8_t *ctl,const uint8_t *code,uint64_t len,
                                    const GfxShaderAnswers *a,int *pending) {
    *pending=translate_pending;
    return *pending?NULL:gfx_shader_get(stage,ctl,code,len,NULL,a);
}
const GfxShader *gfx_shader_await(int stage,const uint8_t *ctl,const uint8_t *code,uint64_t len,const GfxShaderAnswers *a) {
    awaits++;return gfx_shader_get(stage,ctl,code,len,NULL,a);
}
@WARNINGS@
'''
    warning = draw.index('enum { W_PROGRAM')
    providers = providers.replace('@WARNINGS@', draw[warning:draw.index(';', warning) + 1])
    base = base.replace(old_stage, '').replace(old_res, providers)
    stage_start = draw.index('#define SHADER_SETS ')
    stage_end = draw.index('// Texture and image types of a stage', stage_start)
    slot_start = draw.index('#define SLOTM_N ')
    slot_end = draw.index('static uint8_t type_of_handle(', slot_start)
    where = base.index('static int nvn_stage_of(')
    base = base[:where] + draw[stage_start:stage_end] + draw[slot_start:slot_end] + base[where:]
    # The native path stays off here: tests/geometry_feature_flags_test.py and geometry_native_control_cache_test.py cover it.
    native = (function(draw, 'static int native_gs_on(') + '\nstatic const GfxShader *native_gs_vs,*native_gs_fs;\n'
              'static const uint8_t *native_gs_control(const uint8_t *c,const uint8_t *code,uint32_t len) '
              '{(void)c;(void)code;(void)len;abort();}\n')
    extra = [function(shader, 'int gfx_shader_is_nvn2('), function(pipeline, 'uint8_t vk_geometry_input('), native,
             function(draw, 'static const GfxShader *shader_for('), function(draw, 'static int experimental_geometry_on('),
             function(draw, 'static int geometry_draw('), function(draw, 'static int geometry_feedback_safe(')]

    # Use actual resource-range discovery and pointer resolution, not a stage-specific replacement signature.
    upd_start = draw.index('#define UPD_N ')
    upd_end = draw.index('static void upd_note(', upd_start)
    base = base.replace(function(base, 'static uint64_t ubo_ptr('), draw[upd_start:upd_end] +
                        function(draw, 'static int upd_read(') + '\n' + function(draw, 'static uint64_t ubo_ptr('))
    old_list = function(base, 'static const RsList *rs_list(')
    ranges = '\n'.join((
        '#define MAX_DESC 1024\n#define MAX_BINDINGS 128\n' +
        next(line for line in draw.splitlines() if line.startswith('#define RS_BYTES ')),
        'static nvn_hmap rs_lists; static const VkPipeInfo *rs_last; static const RsList *rs_last_l;',
        'static int nvn_stage_of(int);', function(draw, 'static int ssbo_slot('), function(draw, 'static int ubo_ptr_res('),
        function(draw, 'static void rs_add('), function(draw, 'static const RsList *rs_list(')))
    base = base.replace(old_list, ranges)
    write_types = '\n'.join(record(draw, name) for name in ('Chunk', 'Range', 'CbRes'))
    write_providers = r'''
_Alignas(256) static uint64_t pointer_words[4];
_Alignas(256) static uint32_t output_words[64];
static CbRes callback_resources;
static unsigned uses,gpu_writes,drops,pending_writes,counts,barriers,passes;
static uint64_t expected_count_bytes=sizeof output_words;
static uint64_t pending_addr,pending_size;
static int last_use;
static struct {VkBuffer zbuf;} Z;
static const Nvn2C0Layout *c0_lay;
static uint32_t c0_dim[3];
int vk_mirror_on=1;
uint32_t rt_trace;
static void *grow(void *p,int *cap,size_t bytes) {
    *cap=*cap?*cap*2:8;void *n=realloc(p,bytes*(size_t)*cap);if(!n)abort();return n;
}
static CbRes *cb_res(VkExec *ex) { (void)ex;return &callback_resources; }
int vk_mem_resolve(uint64_t addr,uint64_t bytes,VkMemSpan *out) {
    const uint64_t begin[]={ (uintptr_t)pointer_words,(uintptr_t)output_words };
    const uint64_t size[]={ sizeof pointer_words,sizeof output_words };
    for(unsigned i=0;i<2;i++) if(addr>=begin[i] && addr-begin[i]<=size[i] && bytes<=size[i]-(addr-begin[i])) {
        *out=(VkMemSpan){.buffer=(VkBuffer)(uintptr_t)11,.offset=addr-begin[i],.avail=size[i]-(addr-begin[i]),
                         .pool=(void *)(uintptr_t)1,.cls=VK_MCLS_HOST_CACHED};return 1;
    }
    return 0;
}
void vk_mem_use(VkMemSpan *span) {uses++;span->buffer=(VkBuffer)(uintptr_t)22;span->offset+=256;}
void vk_mem_gpu_wrote(const VkMemSpan *span,uint64_t bytes) {CHECK((uintptr_t)span->buffer==22 && bytes,"write uses selected backing");gpu_writes++;}
void vk_usnap_drop(uint64_t addr,uint64_t bytes) {CHECK(addr==(uintptr_t)output_words && bytes==sizeof output_words,"write drops exact output snapshot");drops++;}
void vk_mem_gpu_pending(uint64_t addr,uint64_t bytes) {pending_writes++;pending_addr=addr;pending_size=bytes;}
int vk_sync_noting(void) {return 0;}
int vk_exec_par_slice(void) {return 0;}
void vk_sync_note_write(uint64_t a,uint64_t b,uint64_t c) {(void)a;(void)b;(void)c;abort();}
void vk_mem_count(int cls,int use,uint64_t bytes) {(void)cls;(void)use;(void)bytes;}
void vk_mem_bound_note(void *pool) {(void)pool;abort();}
VkCaps vk_caps;
static void count_use(int cls,int use,uint64_t bytes) {(void)cls;CHECK(bytes==expected_count_bytes,"descriptor range retained");counts++;last_use=use;}
static void *scratch(VkExec *ex,uint64_t n,VkBuffer *b,VkDeviceSize *o) {(void)ex;(void)n;(void)b;(void)o;abort();}
static void limits_init(void) {abort();}
static VkDescriptorBufferInfo c0_info(VkExec *ex,int st) {(void)ex;(void)st;abort();}
// This ordinary-GS fixture must never construct a synthetic compute bank. The
// actual guarded-storage selector below rejects GS before this provider.
void vk_c0_fill(uint8_t bank[VK_C0_BYTES],int st,const p_bind_buf ssbo[16],const p_bind_buf ubo[16],
                const uint32_t dispatch[3],uint32_t base_instance) {
    (void)bank;(void)st;(void)ssbo;(void)ubo;(void)dispatch;(void)base_instance;abort();
}
int vk_usnap_find(uint64_t a,uint64_t n,uint64_t align,VkDescriptorBufferInfo *out) {(void)a;(void)n;(void)align;(void)out;return 0;}
void vk_exec_pass_end(VkExec *ex) {passes++;ex->in_pass=0;}
VkExecFrame vk_exec_frame;
void rc_vkCmdPipelineBarrier2(VkCommandBuffer cb,const VkDependencyInfo *di) {
    (void)cb;barriers++;CHECK(di->memoryBarrierCount==1,"one full memory dependency");
    const VkMemoryBarrier2 *m=di->pMemoryBarriers;
    CHECK(m->srcStageMask==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && m->srcAccessMask==VK_ACCESS_2_MEMORY_WRITE_BIT &&
          m->dstStageMask==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
          (m->dstAccessMask&(VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT))==
           (VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT),"full dependency covers GS stores and later indirect reads");
}
'''
    lim = function(draw, 'static struct { uint64_t ubo_max,') + ' LIM;'
    memory_helpers = '\n'.join(function(draw, prefix) for prefix in ('static VkDescriptorBufferInfo zero_info(',
        'static int bind_range(', 'static uint32_t written_hash(', 'static void note_written(', 'static p_bind_buf guarded_storage_binding(',
        'static VkDescriptorBufferInfo ssbo_info(', 'static VkDescriptorBufferInfo ubo_info('))
    barrier_helpers = '\n'.join(function(replay, prefix) for prefix in ('static int has_image(',
        'static VkImageMemoryBarrier2 layout_barrier(', 'static void barrier_to(', 'void vk_exec_barrier('))
    cases = r'''
static void put32(uint8_t *p,uint32_t v) {memcpy(p,&v,4);}
int main(int argc,char **argv) {
    enabled=argc>1?atoi(argv[1]):0;
    CHECK(!descriptor_checks(),"previous descriptor-cache assertions remain active");
    uint8_t ctl[3][0x880]={{0}},code[3][0xa0]={{0}};
    prog_t program={.nstage=2,.serial=111}; VkExec ex={.prog=&program,.prog_stages=NVN_SHADER_STAGE_ALL_GRAPHICS_BITS};
    for(unsigned i=0;i<3;i++) {
        put32(ctl[i]+0x704,sizeof code[i]);put32(ctl[i]+0x714,i==0?NVN_STAGE_VS:NVN_STAGE_GS);
        put32(code[i]+0x30,1|(3<<5)|(4<<10));
        program.sd[i]=(NVNshaderData){(NVNbufferAddress)(uintptr_t)code[i],ctl[i]};
    }
    if(!enabled) {
        native_code=code[1];fix_c0=1;put32(code[1]+0x30,1<<24);
        CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_PATCHES)==0 && !geometry_input,"default-off preserves legacy path without admitting GS");
        CHECK(!warnings && !translations,"default-off performs no GS translation or opt-in rejection");
        printf("ordinary geometry default-off: %u checks, %u failures\n",checks,failures);return failures!=0;
    }
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_POINTS)==1 && geometry_input==GFX_GEOMETRY_POINTS,"ordinary points accepted");
    ex.prog_stages=NVN_SHADER_STAGE_VERTEX_BIT;
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_POINTS)<0,"partial stage bind cannot execute an unrequested GS");
    ex.prog_stages=NVN_SHADER_STAGE_ALL_GRAPHICS_BITS;
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY)==1 && geometry_input==GFX_GEOMETRY_TRIANGLES_ADJACENCY,"adjacency preserved");
    const int bad_prim[]={NVN_DRAW_PRIMITIVE_QUADS,NVN_DRAW_PRIMITIVE_QUAD_STRIP,NVN_DRAW_PRIMITIVE_LINE_LOOP,
                          NVN_DRAW_PRIMITIVE_POLYGON,NVN_DRAW_PRIMITIVE_PATCHES,999};
    for(unsigned i=0;i<sizeof bad_prim/sizeof *bad_prim;i++) CHECK(geometry_draw(&ex,bad_prim[i])<0,"unsupported primitives rejected");
    fix_c0=1;CHECK(geometry_draw(&ex,0)<0,"synthetic GS c0 rejected");fix_c0=0;
    native_code=code[1];CHECK(geometry_draw(&ex,0)<0,"native lowering origin rejected");native_code=NULL;
    put32(code[1]+0x30,(4<<10)|(1<<24));CHECK(geometry_draw(&ex,0)<0,"passthrough rejected");put32(code[1]+0x30,4<<10);
    // Host-authored portable metadata is distinct from a bare NVN1 passthrough flag.
    put32(code[1]+0x30,1|(3<<5)|(4<<10)|(1<<24));
    put32(code[1]+0x38,1u<<24);put32(code[1]+0x3c,7u<<24);put32(code[1]+0x40,1);
    put32(ctl[1]+0x700,sizeof code[1]);
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_TRIANGLES)<0,"bare passthrough has no host metadata");
    uint32_t skip[8]={0x0fffffff,0xfffffff0,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX};
    CHECK(nvn_geometry_control_pack_portable(ctl[1],sizeof ctl[1],code[1],sizeof code[1],skip),"owned portable metadata packed");
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_TRIANGLES)<0,"portable draw requires the complete VS/GS/FS program");
    put32(ctl[2]+0x714,NVN_STAGE_FS);program.nstage=3;
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_TRIANGLES)==1 && geometry_input==GFX_GEOMETRY_TRIANGLES,"canonical portable triangles accepted");
    ex.prog_stages=7;
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_TRIANGLES)<0,"portable binding follows the verified whole-program mask");
    ex.prog_stages=NVN_SHADER_STAGE_ALL_GRAPHICS_BITS;
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_POINTS)<0,"portable point topology remains unsupported");
    nvn_geometry_control_put(ctl[1]+0x810,NVN_GEOMETRY_CONTROL_NATIVE);
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_TRIANGLES)<0,"native metadata remains unsupported");
    nvn_geometry_control_put(ctl[1]+0x810,NVN_GEOMETRY_CONTROL_PORTABLE);
    native_code=code[1];
    CHECK(geometry_draw(&ex,NVN_DRAW_PRIMITIVE_TRIANGLES)<0,"portable metadata cannot enable native-origin code");native_code=NULL;
    memset(ctl[1]+NVN_GEOMETRY_CONTROL_OFFSET,0,NVN_GEOMETRY_CONTROL_SIZE);
    put32(ctl[1]+0x700,0);put32(code[1]+0x30,4<<10);put32(ctl[2]+0x714,NVN_STAGE_GS);program.nstage=2;
    for(unsigned stage=3;stage<=4;stage++){put32(ctl[1]+0x714,stage);CHECK(geometry_draw(&ex,0)<0,"tessellation rejected");}
    put32(ctl[1]+0x714,99);CHECK(geometry_draw(&ex,0)<0,"unknown stage rejected");put32(ctl[1]+0x714,NVN_STAGE_GS);
    program.nstage=3;CHECK(geometry_draw(&ex,0)<0,"duplicate GS rejected");program.nstage=2;
    put32(ctl[1],0x98761234);put32(ctl[1]+4,0xc9);CHECK(geometry_draw(&ex,0)<0,"raw NVN2 rejected before legacy offsets");put32(ctl[1],0);
    CHECK(geometry_draw(&ex,0)==1,"valid GS restored");
    int present,pending;
    const GfxShader *s=shader_for(&ex,NVN_STAGE_GS,&present,&pending,0);
    CHECK(s && present && !pending && last_stage==GFX_STAGE_GS && last_answers.geometry_input==GFX_GEOMETRY_POINTS &&
          !last_answers.local[0],"GS maps explicitly and uses draw input instead of compute answers");
    int before=translations;CHECK(shader_for(&ex,NVN_STAGE_GS,&present,&pending,0)==s && translations==before,"shader table reuses identical GS answers");
    ex.tex_pool=table_a;ex.tex[NVN_STAGE_GS][0]=HANDLE;remapped=0;memory_generation++;
    geometry_input=GFX_GEOMETRY_LINES;CHECK(shader_for(&ex,NVN_STAGE_GS,&present,&pending,0)==s && translations==before+1 &&
          last_answers.geometry_input==GFX_GEOMETRY_LINES && last_answers.tex_type[0]==GFX_TEX_1D,
          "new topology relearns queried texture slots instead of inheriting the points use mask");
    program.serial++;translate_fail=1;CHECK(!shader_for(&ex,NVN_STAGE_GS,&present,&pending,0) && present && !pending,"failed requested GS remains present, not absent");
    translate_fail=0;program.serial++;async_on=1;translate_pending=1;
    CHECK(!shader_for(&ex,NVN_STAGE_GS,&present,&pending,0) && present && pending,"pending GS is not usable");
    translate_pending=0;CHECK(shader_for(&ex,NVN_STAGE_GS,&present,NULL,1) && awaits==1,"await translates required GS");async_on=0;
    ex.tex_pool=table_a;ex.samp_pool=sampler_a;ex.tex[NVN_STAGE_GS][0]=HANDLE;ex.img[NVN_STAGE_GS][0]=HANDLE;
    remapped=0;memory_generation++;descriptor_state(&ex);
    memo_key(&ex,7,&SM.k);SM.valid=1;CHECK(memo_match(&ex,7,&SM.k),"GS shader memo warms");
    ex.tex[NVN_STAGE_GS][0]=0;ex.tex_serial++;CHECK(!memo_match(&ex,7,&SM.k),"GS texture handle invalidates memo");
    ex.tex[NVN_STAGE_GS][0]=HANDLE;memo_key(&ex,7,&SM.k);SM.valid=1;geometry_input=GFX_GEOMETRY_TRIANGLES;
    CHECK(!memo_match(&ex,7,&SM.k),"GS input topology invalidates shader memo");
    geometry_input=GFX_GEOMETRY_POINTS;ex.pass_ncolor=1;ex.color[0]=(VkExecTarget){.image=images[0].image,.layers=1};
    CHECK(feedback_scan(&ex,(const GfxShader *)&gs_shader)==1,"GS samples its own stage handles when detecting unsafe attachment feedback");
    CHECK(!geometry_feedback_safe(&ex,(const GfxShader *)&vs_shader,(const GfxShader *)&gs_shader),"GS attachment feedback rejects the draw");
    ex.tex[NVN_STAGE_GS][0]=0;ex.tex_serial++;CHECK(!feedback_scan(&ex,(const GfxShader *)&gs_shader),"unbound GS stage cannot inherit FS attachment handles");
    ex.tex[NVN_STAGE_VS][0]=HANDLE;CHECK(feedback_scan(&ex,(const GfxShader *)&vs_shader)==1,"VS attachment dependency detected for GS draw rejection");
    CHECK(!geometry_feedback_safe(&ex,(const GfxShader *)&vs_shader,(const GfxShader *)&gs_shader),"VS attachment feedback also rejects a GS draw");
    ex.tex[NVN_STAGE_VS][0]=0;ex.tex_serial++;
    CHECK(geometry_feedback_safe(&ex,(const GfxShader *)&vs_shader,(const GfxShader *)&gs_shader),"no pre-raster attachment dependency permits normal GS resources");
    CHECK(geometry_feedback_safe(&ex,(const GfxShader *)&vs_shader,NULL),"default VS/FS feedback behavior remains separate");
    GfxShaderRes buffer={.cbuf_slot=3,.handle=0,.flags=1};
    VkPipeBinding binding={.stage=GFX_STAGE_GS,.kind=GFX_RES_SSBO,.count=1,.res=&buffer};
    VkPipeInfo info={.n_bindings=1,.bindings=&binding};
    pointer_words[0]=(uintptr_t)output_words;ex.ubo[NVN_STAGE_GS][0]=(p_bind_buf){.addr=(uintptr_t)pointer_words,.size=sizeof pointer_words};
    CHECK(!bind_sig(&ex,&info) && bind_sig(&ex,&info),"GS pointer-block descriptor signature warms");
    pointer_words[0]+=16;CHECK(!bind_sig(&ex,&info),"GS pointer contents invalidate resource signature without a bind op");pointer_words[0]-=16;
    LIM.ubo_max=65536;LIM.sbo_max=sizeof output_words;LIM.sbo_align=16;LIM.ubo_align=16;
    VkDescriptorBufferInfo out=ssbo_info(&ex,NVN_STAGE_GS,&buffer,1);
    CHECK((uintptr_t)out.buffer==22 && out.offset==256 && out.range==sizeof output_words && uses==1 && gpu_writes==1 &&
          drops==1 && counts==1 && last_use==VK_MUSE_SSBO,"GS storage resolves user-c3 pointer and publishes selected backing");
    CHECK(callback_resources.nwritten==1 && pending_writes==1 && pending_addr==(uintptr_t)output_words &&
          pending_size==sizeof output_words,"GS write range is retained for fence completion");
    note_written(&ex,(uintptr_t)output_words,sizeof output_words);CHECK(pending_writes==1,"same write range retained once");
    Nvn2C0Layout compute_layout={0};c0_lay=&compute_layout;
    p_bind_buf guarded=guarded_storage_binding(&ex,NVN_STAGE_GS,&buffer);
    CHECK(!guarded.addr && !guarded.size,"guarded compute candidates cannot inherit an ordinary GS pointer block");
    c0_lay=NULL;
    ex.ubo[NVN_STAGE_GS][14]=(p_bind_buf){.addr=(uintptr_t)pointer_words,.size=sizeof pointer_words};
    expected_count_bytes=sizeof pointer_words;out=ubo_info(&ex,NVN_STAGE_GS,17);
    CHECK((uintptr_t)out.buffer==22 && out.range==sizeof pointer_words && uses==2 && counts==2 &&
          last_use==VK_MUSE_UBO && gpu_writes==1,"ordinary GS c17 binds user UBO14 without a synthetic blit bank");
    int is_ubo;GfxShaderRes table={.cbuf_slot=0,.handle=0x410/4};
    CHECK(ssbo_slot(&table,NVN_STAGE_GS,&is_ubo)==0 && !is_ubo && ssbo_slot(&table,NVN_STAGE_CS,&is_ubo)<0,"ordinary GS table maps separately from CS");
    ex.recorded=1;ex.barrier_at=-1;ex.nfbl=0;ex.in_pass=1;vk_exec_barrier(&ex);
    CHECK(barriers==1 && passes==1 && !ex.in_pass,"later work receives the existing full memory barrier");
    printf("ordinary geometry enabled: %u checks, %u failures\n",checks,failures);return failures!=0;
}
'''
    return '\n'.join((base, *extra, write_types, '#define ZERO_BYTES 65536u', write_providers, lim, memory_helpers, barrier_helpers, cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--deps', type=Path, required=True)
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc')
    out = ROOT / 'build/test/geometry-replay'
    out.mkdir(parents=True, exist_ok=True)
    fixture = out / 'replay.c'
    fixture.write_text(source())
    failed = 0
    for opt in ('-O0', '-O2'):
        exe = out / ('replay' + opt + ('.exe' if os.name == 'nt' else ''))
        command = [args.cc, opt, '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                   '-I', str(args.deps / 'include'), '-I', str(args.deps / 'include/vma'), str(fixture), '-o', str(exe)]
        run = subprocess.run(command, capture_output=True, text=True)
        (out / (opt + '-compile.log')).write_text(run.stdout + run.stderr)
        if run.returncode:
            print(run.stdout + run.stderr)
            return run.returncode
        for mode in ('0', '1'):
            run = subprocess.run([str(exe), mode], capture_output=True, text=True)
            (out / (opt + '-' + mode + '.log')).write_text(run.stdout + run.stderr)
            print(opt, mode, run.stdout + run.stderr)
            failed |= run.returncode
    return failed


if __name__ == '__main__':
    raise SystemExit(main())
