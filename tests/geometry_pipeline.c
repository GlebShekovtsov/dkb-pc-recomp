// Original CPU contract for real vk_pipeline.c functions; no loader/device is linked.
// Included after actual GfxShader structure, reflector and accessor definitions.
#include <stdarg.h>

static unsigned checks, failures, made, destroyed, pipeline_calls, fail_module;
static VkShaderStageFlags captured_stages;
static VkPrimitiveTopology captured_topology;
static uint32_t captured_count, captured_binding[8][128], captured_nbinding[8];
static uint32_t captured_outputs[8];
static unsigned captured_clip_declarations[8], captured_cull_declarations[8];
static VkCtx fixture_context;
static const char *output_directory;
static int capture_plain_module;
static const char *capture_prefix;
VkCaps vk_caps;
#define CHECK(c) do { checks++; if (!(c)) { failures++; fprintf(stderr,"line %d: %s\n",__LINE__,#c); } } while (0)

VkCtx *vk_ctx(void) { return &fixture_context; }
uint64_t perf_begin(void) { return 0; }
void perf_end(int kind, uint64_t start) { (void)kind; (void)start; }
void perf_count(int kind) { (void)kind; }
void rt_log(const char *format, ...) { (void)format; }
// gfx_spirv_final reads the FMZ switches; every other key stays forbidden.
const char *cfg_get(const char *name) { if (!strcmp(name, "TOTK_FMZ") || !strcmp(name, "TOTK_FMZ_SEL")) return NULL; abort(); }
int cfg_int(const char *name) { (void)name; abort(); }
int tex_vk_format(int format, TexVkFormat *out) { (void)format; (void)out; abort(); }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice device,
    VkFormat format, VkFormatProperties *out) { (void)device; (void)format; (void)out; abort(); }

VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device,
    const VkShaderModuleCreateInfo *info, const VkAllocationCallbacks *alloc, VkShaderModule *out) {
    (void)device; (void)alloc;
    unsigned index = ++made;
    if (fail_module && index == fail_module) return VK_ERROR_OUT_OF_HOST_MEMORY;
    CHECK(index < 8);
    captured_clip_declarations[index]=captured_cull_declarations[index]=0;
    if(!pipeline_calls) {
        char path[4096];
        if(capture_plain_module)snprintf(path,sizeof path,"%s/plain-vs.spv",output_directory);
        else if(capture_prefix)snprintf(path,sizeof path,"%s/%s-module%u.spv",output_directory,capture_prefix,index);
        else snprintf(path,sizeof path,"%s/module%u.spv",output_directory,index);
        FILE *f=fopen(path,"wb"); if(!f)abort();
        if(fwrite(info->pCode,1,info->codeSize,f)!=info->codeSize || fclose(f))abort();
    }
    const uint32_t *w = info->pCode;
    GfxShader reflected={.spirv=(uint32_t*)w,.words=info->codeSize/4,.stage=GFX_STAGE_VS};
    for (size_t i=5; i<info->codeSize/4;) {
        uint32_t len=w[i]>>16, op=w[i]&65535;
        CHECK(len && len<=info->codeSize/4-i);
        if (!len || len>info->codeSize/4-i) break;
        if(op==15 && len>=3) reflected.stage=w[i+1]==3?GFX_STAGE_GS:w[i+1]==4?GFX_STAGE_FS:GFX_STAGE_VS;
        if(op==72 && len==5 && w[i+3]==11) {
            captured_clip_declarations[index]+=w[i+4]==3;
            captured_cull_declarations[index]+=w[i+4]==4;
        }
        if (op==71 && len==4 && w[i+2]==33) {
            CHECK(captured_nbinding[index]<128);
            captured_binding[index][captured_nbinding[index]++]=w[i+3];
        }
        i+=len;
    }
    reflect_spirv(&reflected); captured_outputs[index]=reflected.out_mask;
    *out=(VkShaderModule)(uintptr_t)index;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice device, VkShaderModule module,
    const VkAllocationCallbacks *alloc) { (void)device; (void)module; (void)alloc; destroyed++; }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache cache,
    uint32_t count, const VkGraphicsPipelineCreateInfo *info, const VkAllocationCallbacks *alloc, VkPipeline *out) {
    (void)device; (void)cache; (void)alloc;
    CHECK(count==1); pipeline_calls++;
    captured_count=info->stageCount; captured_stages=0;
    for (uint32_t i=0;i<info->stageCount;i++) {
        CHECK(!(captured_stages & info->pStages[i].stage));
        captured_stages|=info->pStages[i].stage;
        CHECK(!strcmp(info->pStages[i].pName,"main"));
    }
    captured_topology=info->pInputAssemblyState->topology;
    CHECK(!info->flags || !(info->flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR));
    *out=(VkPipeline)(uintptr_t)99; return VK_SUCCESS;
}

static void load_shader(GfxShader *s, const char *dir, const char *name, int stage) {
    memset(s,0,sizeof *s); s->stage=stage;
    char path[4096]; snprintf(path,sizeof path,"%s/%s.spv",dir,name);
    FILE *f=fopen(path,"rb"); if(!f)abort();
    if(fseek(f,0,SEEK_END))abort(); long size=ftell(f); rewind(f);
    if(size<20 || size%4)abort();
    s->spirv=malloc((size_t)size); if(!s->spirv)abort(); s->words=(uint64_t)size/4;
    if(fread(s->spirv,1,(size_t)size,f)!=(size_t)size || fclose(f))abort();
    memset(s->iid,stage==GFX_STAGE_FS?2:stage+1,sizeof s->iid);
    if(stage==GFX_STAGE_GS)s->geometry=(GfxShaderGeometry){1,1,1,3,1,3,0};
    fixture_resources(s,name);
    reflect_spirv(s);
}

static void capabilities(void) {
    P.dyn_vi=1; P.indep=1;
    P.f_geometry=P.f_vertex_stores=P.f_gs_point_size=P.f_clip=P.f_cull=1;
#ifndef GEOMETRY_SUBGROUP_BASELINE
    P.subgroup_stages=VK_SHADER_STAGE_GEOMETRY_BIT;
    P.subgroup_operations=VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_VOTE_BIT|
        VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
#endif
    VkPhysicalDeviceLimits *l=&fixture_context.props.limits;
    l->maxGeometryShaderInvocations=32; l->maxGeometryInputComponents=128;
    l->maxGeometryOutputComponents=128; l->maxGeometryOutputVertices=256;
    l->maxGeometryTotalOutputComponents=1024; l->maxVertexOutputComponents=128;
    l->maxFragmentInputComponents=128; l->maxClipDistances=8;
    l->maxCullDistances=8; l->maxCombinedClipAndCullDistances=8;
}

static void topology_tests(void) {
    const unsigned prim[]={NVN_DRAW_PRIMITIVE_POINTS,NVN_DRAW_PRIMITIVE_LINES,NVN_DRAW_PRIMITIVE_LINE_STRIP,
        NVN_DRAW_PRIMITIVE_LINES_ADJACENCY,NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY,
        NVN_DRAW_PRIMITIVE_TRIANGLES,NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP,NVN_DRAW_PRIMITIVE_TRIANGLE_FAN,
        NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY,NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY,
        NVN_DRAW_PRIMITIVE_LINE_LOOP,NVN_DRAW_PRIMITIVE_QUADS,NVN_DRAW_PRIMITIVE_QUAD_STRIP,
        NVN_DRAW_PRIMITIVE_POLYGON,NVN_DRAW_PRIMITIVE_PATCHES,0xffffffff};
    const uint8_t input[]={1,2,2,3,3,4,4,4,5,5,0,0,0,0,0,0};
    for(unsigned i=0;i<sizeof prim/sizeof *prim;i++)CHECK(vk_geometry_input(prim[i])==input[i]);
    CHECK(stage_bit(GFX_STAGE_GS)==VK_SHADER_STAGE_GEOMETRY_BIT);
    CHECK(stage_bit(GFX_STAGE_CS)==VK_SHADER_STAGE_COMPUTE_BIT);
    CHECK(!stage_bit(-1));
}

static void cull_tests(const char *dir) {
    static const char *uses[]={"unused","direct","alias","whole","derived",
        "clip","clip_alias","clip_derived","clip_dynamic","cull_dynamic","ext_literal"};
    for(unsigned k=0;k<sizeof uses/sizeof *uses;k++) {
        char name[64]; snprintf(name,sizeof name,"cull-%s",uses[k]);
        GfxShader s; load_shader(&s,dir,name,GFX_STAGE_GS);
        uint64_t count=0; uint32_t *fixed=spirv_fixup(s.spirv,s.words,&count,clip_cull_limit());
        if(!k || k==5 || k==8 || k==10) { CHECK(fixed && count<s.words); }
        else CHECK(!fixed); // Static, aliased or whole-struct use retains CullDistance.
        if(fixed && (!k || k==5 || k==10)) {
            unsigned clip=0,cull=0;
            for(uint64_t i=5;i<count;i+=fixed[i]>>16)
                if((fixed[i]&65535)==72 && fixed[i]>>16==5 && fixed[i+3]==11) {
                    clip+=fixed[i+4]==3; cull+=fixed[i+4]==4;
                }
            CHECK(!cull && clip==(k==5));
            char path[4096];snprintf(path,sizeof path,"%s/fixed-%s.spv",dir,name);
            FILE *f=fopen(path,"wb");if(!f)abort();
            if(fwrite(fixed,4,(size_t)count,f)!=(size_t)count || fclose(f))abort();
        }
        free(fixed); free(s.spirv);
    }
}

static void interface_tests(VkPipeState *s) {
    VkPhysicalDeviceLimits *l=&fixture_context.props.limits;
    const char *why=geometry_validate(s,l);
    if(why)fprintf(stderr,"positive interface rejected: %s\n",why);
    CHECK(!why);
    GeometryInterface vo,gi,go,fi;
    CHECK(geometry_interface(s->vs,3,0,&vo)); CHECK(geometry_interface(s->gs,1,1,&gi));
    CHECK(geometry_interface(s->gs,3,0,&go)); CHECK(geometry_interface(s->fs,1,0,&fi));
    if(why) for(unsigned i=0;i<64;i++) if((vo.builtin_mask|gi.builtin_mask|go.builtin_mask)>>i&1)
        fprintf(stderr,"builtin%u: VS=%llx GI=%llx GO=%llx\n",i,(unsigned long long)vo.builtin[i],
            (unsigned long long)gi.builtin[i],(unsigned long long)go.builtin[i]);
    CHECK(vo.mask==1 && gi.mask==1 && go.mask==1 && fi.mask==1);
    CHECK(vo.builtin_mask & 2); // Extra VS PointSize is legal with triangle-emitting GS.
    CHECK(geometry_builtin_link(&vo,&gi));
    GeometryInterface changed=gi;
    changed.block_members--; CHECK(!geometry_builtin_link(&vo,&changed));
    changed=gi; changed.block_invariant^=1; CHECK(!geometry_builtin_link(&vo,&changed));
    changed=gi; changed.block_builtin[0]=1; CHECK(!geometry_builtin_link(&vo,&changed));
    changed=gi; changed.builtin[0]^=UINT64_C(1)<<62; CHECK(!geometry_builtin_link(&vo,&changed));
    changed=gi; changed.builtin[0]^=UINT64_C(1)<<8; CHECK(!geometry_builtin_link(&vo,&changed));
    CHECK(!geometry_interface(s->gs,1,3,&gi));
    GfxShader *gs=(GfxShader*)s->gs;
    for(unsigned bits=0;bits<=64;bits+=16)if(bits!=32) {
        gs->in[0].scalar_bits=(uint8_t)bits; CHECK(geometry_validate(s,l));
    }
    gs->in[0].scalar_bits=32;
    gs->in[0].packed=1; CHECK(geometry_validate(s,l)); gs->in[0].packed=0;
    gs->geometry.input=2; CHECK(geometry_validate(s,l)); gs->geometry.input=1;
    gs->geometry.passthrough=1; CHECK(geometry_validate(s,l)); gs->geometry.passthrough=0;
    gs->geometry.output=GFX_GEOMETRY_OUTPUT_POINTS; CHECK(geometry_validate(s,l));
    gs->geometry.output=GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP;
    s->poly.mode=NVN_POLYGON_MODE_POINT; CHECK(geometry_validate(s,l)); s->poly.mode=NVN_POLYGON_MODE_FILL;
    gs->geometry.invocations=l->maxGeometryShaderInvocations+1; CHECK(geometry_validate(s,l)); gs->geometry.invocations=1;
    gs->geometry.max_output_vertices=l->maxGeometryOutputVertices+1; CHECK(geometry_validate(s,l)); gs->geometry.max_output_vertices=3;
    uint32_t old=l->maxGeometryTotalOutputComponents;
    l->maxGeometryTotalOutputComponents=go.components*3; CHECK(!geometry_validate(s,l));
    l->maxGeometryTotalOutputComponents--; CHECK(geometry_validate(s,l)); l->maxGeometryTotalOutputComponents=old;
    old=l->maxGeometryOutputComponents;
    l->maxGeometryOutputComponents=go.components-1; CHECK(geometry_validate(s,l)); l->maxGeometryOutputComponents=old;
    old=l->maxFragmentInputComponents;
    l->maxFragmentInputComponents=fi.components-1; CHECK(geometry_validate(s,l)); l->maxFragmentInputComponents=old;
    P.f_geometry=0; CHECK(geometry_validate(s,l)); P.f_geometry=1;
    P.f_vertex_stores=0; CHECK(geometry_validate(s,l)); P.f_vertex_stores=1;
    CHECK(!geometry_validate(s,l));
}

static void descriptor_tests(VkPipeState *s) {
    made=destroyed=pipeline_calls=fail_module=0;
    memset(captured_nbinding,0,sizeof captured_nbinding);
    Bindings b={0};
    uint32_t gs_off=binding_span(s->vs)-1,fs_off=gs_off+binding_span(s->gs)-1;
    CHECK(add_stage(&b,s->vs,0)); CHECK(add_stage(&b,s->gs,gs_off)); CHECK(add_stage(&b,s->fs,fs_off));
    int support=0,storage=0;
    for(uint32_t i=0;i<b.n;i++) {
        if(!b.b[i].binding) { support++; CHECK(b.b[i].stageFlags==(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_GEOMETRY_BIT|VK_SHADER_STAGE_FRAGMENT_BIT)); }
        if(b.pb[i].stage==GFX_STAGE_GS && b.pb[i].kind==GFX_RES_SSBO) {
            storage++; CHECK(b.b[i].binding==gs_off+1); CHECK(b.b[i].stageFlags==VK_SHADER_STAGE_GEOMETRY_BIT);
        }
    }
    CHECK(support==1 && storage==1);
    Layout *l=calloc(1,sizeof *l+b.n*sizeof *l->b); if(!l)abort();
    l->n=b.n; memcpy(l->b,b.b,b.n*sizeof *b.b); l->layout=(VkPipelineLayout)(uintptr_t)88;
    char why[256]; CHECK(layout_covers(s,l,fs_off,why,sizeof why));
    l->set1=(VkDescriptorSetLayout)(uintptr_t)9;
    CHECK(!layout_covers(s,l,fs_off,why,sizeof why)); l->set1=VK_NULL_HANDLE;
    GKey k; gkey_fill(s,&k);
    CHECK(k.gs==s->gs && k.geometry_input==1 && !k.indep);
    GKey changed=k; changed.gs=s->vs; CHECK(memcmp(&k,&changed,sizeof k));
    changed=k; changed.geometry_input=4; CHECK(memcmp(&k,&changed,sizeof k));
    CHECK(create_graphics(s,&k,l,fs_off));
    CHECK(captured_count==3 && captured_stages==(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_GEOMETRY_BIT|VK_SHADER_STAGE_FRAGMENT_BIT));
    CHECK(captured_topology==VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    CHECK(made==3 && destroyed==3 && pipeline_calls==1);
    // These original programs never read or write distance arrays. Both GS
    // input/output boilerplate arrays must be absent from the uploaded module.
    CHECK(captured_clip_declarations[2]==0 && captured_cull_declarations[2]==0);
    CHECK(captured_clip_declarations[3]==captured_clip_declarations[2] &&
        captured_cull_declarations[3]==captured_cull_declarations[2]);
    CHECK(captured_outputs[3]==s->vs->out_mask); // GS path must not repair VS→FS.
    for(unsigned module=1;module<=3;module++) {
        unsigned expected=module==1?fs_off:module==2?gs_off:0;
        int resource=0;
        for(unsigned i=0;i<captured_nbinding[module];i++) if(captured_binding[module][i]) {
            CHECK(captured_binding[module][i]==expected+1); resource++;
        }
        CHECK(resource==(module==2 || (module==1?s->fs->nres[0]:s->vs->nres[0])));
    }
    made=destroyed=0; fail_module=2;
    CHECK(!create_graphics(s,&k,l,fs_off)); CHECK(destroyed==2 && pipeline_calls==1);
    made=destroyed=fail_module=0;
    const GfxShader *fs=s->fs; s->fs=NULL; gkey_fill(s,&k);
    CHECK(create_graphics(s,&k,l,fs_off)); CHECK(captured_count==2);
    s->fs=fs;
    free(l);
}

static void serialization_tests(VkPipeState *s) {
    VkPipeState seeded; memset(&seeded,0x5a,sizeof seeded);
    seeded.vs=s->vs; seeded.fs=s->fs; seeded.gs=NULL;
    KeyRec legacy; rec_fill(&legacy,&seeded);
    char path[4096]; snprintf(path,sizeof path,"%s/legacy-key.bin",output_directory);
    FILE *f=fopen(path,"wb"); if(!f)abort();
    if(fwrite(&legacy,1,sizeof legacy,f)!=sizeof legacy || fclose(f))abort();
    A.on=1; strcpy(K.path,"owned-key-fixture");
    keys_note(s); CHECK(!K.pend_n && !K.seen_n);
    const GfxShader *gs=s->gs; s->gs=NULL;
    keys_note(s); CHECK(K.pend_n==1 && K.seen_n==1);
    keys_note(s); CHECK(K.pend_n==1 && K.seen_n==1);
    CHECK(KEYS_VERSION==3);
    CHECK(sizeof ((KeyRec*)0)->state==offsetof(VkPipeState,gs));
    s->gs=gs; free(K.pend); free(K.seen);
}

static void portable_tests(const char *dir) {
    GfxShader vs,gs,fs;
    load_shader(&vs,dir,"vs_portable",GFX_STAGE_VS);
    load_shader(&gs,dir,"gs_portable",GFX_STAGE_GS);
    load_shader(&fs,dir,"ps_portable",GFX_STAGE_FS);
    CHECK(gs.geometry.version==GFX_GEOMETRY_PASSTHROUGH_VERSION);
    CHECK(gs.geometry.input==GFX_GEOMETRY_TRIANGLES && gs.geometry.input_vertices==3);
    CHECK(gs.geometry.output==GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP && !gs.geometry.passthrough);
    CHECK(gs.geometry.invocations==1 && gs.geometry.max_output_vertices==3);
    CHECK(gs.nres[GFX_RES_SSBO]==1 && gs.res[GFX_RES_SSBO][0].cbuf_slot==3 &&
        !gs.res[GFX_RES_SSBO][0].handle && gs.res[GFX_RES_SSBO][0].flags==1);
    VkPipeState s={.vs=&vs,.gs=&gs,.fs=&fs,.prim=NVN_DRAW_PRIMITIVE_TRIANGLES,.samples=1};
    s.poly.mode=NVN_POLYGON_MODE_FILL;
    VkPhysicalDeviceLimits *limits=&fixture_context.props.limits;
    const char *why=geometry_validate(&s,limits);
    if(why)fprintf(stderr,"portable interface rejected: %s\n",why);
    CHECK(!why);
    GeometryInterface vo,gi,go,fi;
    CHECK(geometry_interface(&vs,3,0,&vo)); CHECK(geometry_interface(&gs,1,3,&gi));
    CHECK(geometry_interface(&gs,3,0,&go)); CHECK(geometry_interface(&fs,1,0,&fi));
    CHECK(vo.mask==1 && gi.mask==1 && go.mask==1 && fi.mask==1);
    CHECK(geometry_builtin_link(&vo,&gi));
    CHECK(!geometry_interface(&gs,1,1,&gi));
    GfxShaderGeometry saved=gs.geometry;
    made=destroyed=pipeline_calls=fail_module=0;
    s.prim=NVN_DRAW_PRIMITIVE_POINTS; CHECK(geometry_validate(&s,limits));
    s.prim=NVN_DRAW_PRIMITIVE_LINES; gs.geometry.input=GFX_GEOMETRY_LINES; gs.geometry.input_vertices=2;
    why=geometry_validate(&s,limits); CHECK(why && strstr(why,"unsupported portable geometry contract"));
    s.prim=NVN_DRAW_PRIMITIVE_TRIANGLES; gs.geometry=saved;
    for(unsigned output=GFX_GEOMETRY_OUTPUT_POINTS;output<=GFX_GEOMETRY_OUTPUT_LINE_STRIP;output++) {
        gs.geometry.output=output; why=geometry_validate(&s,limits);
        CHECK(why && strstr(why,"unsupported portable geometry contract"));
    }
    gs.geometry=saved; gs.geometry.invocations=2;
    why=geometry_validate(&s,limits); CHECK(why && strstr(why,"unsupported portable geometry contract"));
    gs.geometry=saved; gs.geometry.max_output_vertices=4;
    why=geometry_validate(&s,limits); CHECK(why && strstr(why,"unsupported portable geometry contract"));
    gs.geometry=saved; s.fs=NULL;
    why=geometry_validate(&s,limits); CHECK(why && strstr(why,"unsupported portable geometry contract"));
    s.fs=&fs; gs.geometry.passthrough=1; CHECK(geometry_validate(&s,limits));
    gs.geometry=saved; gs.geometry.version=3; CHECK(geometry_validate(&s,limits));
    gs.geometry=saved; P.f_vertex_stores=0; CHECK(geometry_validate(&s,limits)); P.f_vertex_stores=1;
    CHECK(!made && !pipeline_calls);
    CHECK(!geometry_validate(&s,limits));

    Bindings b={0};
    uint32_t gs_off=binding_span(&vs)-1,fs_off=gs_off+binding_span(&gs)-1;
    CHECK(add_stage(&b,&vs,0)); CHECK(add_stage(&b,&gs,gs_off)); CHECK(add_stage(&b,&fs,fs_off));
    unsigned support=0,storage=0;
    for(uint32_t i=0;i<b.n;i++) {
        if(!b.b[i].binding) {
            support++; CHECK(b.b[i].stageFlags==(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_GEOMETRY_BIT|VK_SHADER_STAGE_FRAGMENT_BIT));
        }
        if(b.pb[i].stage==GFX_STAGE_GS && b.pb[i].kind==GFX_RES_SSBO) {
            storage++; CHECK(b.b[i].stageFlags==VK_SHADER_STAGE_GEOMETRY_BIT);
            CHECK(b.b[i].binding==gs.res[GFX_RES_SSBO][0].binding+gs_off);
        }
    }
    CHECK(support==1 && storage==1);
    Layout *layout=calloc(1,sizeof *layout+b.n*sizeof *layout->b);if(!layout)abort();
    layout->n=b.n;memcpy(layout->b,b.b,b.n*sizeof *b.b);layout->layout=(VkPipelineLayout)(uintptr_t)88;
    char reason[256];CHECK(layout_covers(&s,layout,fs_off,reason,sizeof reason));
    GKey key;gkey_fill(&s,&key);
    CHECK(key.gs==&gs && key.geometry_input==GFX_GEOMETRY_TRIANGLES && !key.indep);
    memset(captured_nbinding,0,sizeof captured_nbinding);
    capture_prefix="portable";
    CHECK(create_graphics(&s,&key,layout,fs_off));
    capture_prefix=NULL;
    CHECK(made==3 && destroyed==3 && pipeline_calls==1 && captured_count==3);
    CHECK(captured_stages==(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_GEOMETRY_BIT|VK_SHADER_STAGE_FRAGMENT_BIT));
    CHECK(captured_topology==VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    CHECK(!captured_clip_declarations[2] && !captured_cull_declarations[2]);
    CHECK(captured_clip_declarations[3]==captured_clip_declarations[2] &&
        captured_cull_declarations[3]==captured_cull_declarations[2]);
    const GfxShader *stages[]={&fs,&gs,&vs};
    const uint32_t offsets[]={fs_off,gs_off,0};
    for(unsigned module=1;module<=3;module++) {
        const GfxShader *stage=stages[module-1];unsigned expected=0,seen=0;
        for(unsigned kind=0;kind<GFX_RES_KINDS;kind++)expected+=(unsigned)stage->nres[kind];
        for(unsigned j=0;j<captured_nbinding[module];j++)if(captured_binding[module][j]) {
            int found=0;seen++;
            for(unsigned kind=0;kind<GFX_RES_KINDS;kind++)for(int r=0;r<stage->nres[kind];r++)
                found|=captured_binding[module][j]==stage->res[kind][r].binding+offsets[module-1];
            CHECK(found);
        }
        CHECK(seen==expected);
    }
    free(layout);free(vs.spirv);free(gs.spirv);free(fs.spirv);
}

int main(int argc,char **argv) {
    if(argc!=2)return 2;
    output_directory=argv[1];
    capabilities(); topology_tests(); cull_tests(argv[1]);
    GfxShader vs,gs,fs; load_shader(&vs,argv[1],"vs",GFX_STAGE_VS);
    load_shader(&gs,argv[1],"gs",GFX_STAGE_GS); load_shader(&fs,argv[1],"ps",GFX_STAGE_FS);
    // The plain wrapper keeps the legacy VS Clip[8] declaration. The GS-chain
    // create path below must trim it without changing this default behavior.
    capture_plain_module=1; CHECK(make_module(&vs,vs.spirv,vs.words)); capture_plain_module=0;
    CHECK(captured_clip_declarations[1]==1 && captured_cull_declarations[1]==0);
    VkPipeState s={.vs=&vs,.fs=&fs,.gs=&gs,.prim=NVN_DRAW_PRIMITIVE_POINTS,.samples=1};
    s.poly.mode=NVN_POLYGON_MODE_FILL;
    interface_tests(&s); descriptor_tests(&s); serialization_tests(&s);
    GfxShader live_vs; load_shader(&live_vs,argv[1],"vs_live_clip",GFX_STAGE_VS);
    s.vs=&live_vs;
    const char *mismatch=geometry_validate(&s,&fixture_context.props.limits);
    CHECK(mismatch && strstr(mismatch,"built-in block mismatch"));
    made=destroyed=pipeline_calls=fail_module=0;
    CHECK(make_module_linked(&live_vs,live_vs.spirv,live_vs.words,1));
    CHECK(captured_clip_declarations[1]==1 && captured_cull_declarations[1]==0);
    s.vs=&vs;free(live_vs.spirv);
    GfxShader reordered; load_shader(&reordered,argv[1],"gs_reordered",GFX_STAGE_GS);
    s.gs=&reordered; CHECK(geometry_validate(&s,&fixture_context.props.limits));
    s.gs=&gs;free(reordered.spirv);
    GfxShader flat_builtin; load_shader(&flat_builtin,argv[1],"gs_flat_builtin",GFX_STAGE_GS);
    s.gs=&flat_builtin; CHECK(geometry_validate(&s,&fixture_context.props.limits));
    s.gs=&gs;free(flat_builtin.spirv);
    GfxShader live_clip; load_shader(&live_clip,argv[1],"gs_live_clip",GFX_STAGE_GS);
    s.gs=&live_clip;
    const char *limit=geometry_validate(&s,&fixture_context.props.limits);
    CHECK(limit && strstr(limit,"combined input/output distance device limit"));
    s.gs=&gs;free(live_clip.spirv);
    GfxShader primitive; load_shader(&primitive,argv[1],"ps_primitive",GFX_STAGE_FS);
    s.fs=&primitive; CHECK(geometry_validate(&s,&fixture_context.props.limits));
    s.fs=&fs; free(primitive.spirv);
    GfxShader vr,fr; load_shader(&vr,argv[1],"vs_ubo",GFX_STAGE_VS);
    load_shader(&fr,argv[1],"ps_ubo",GFX_STAGE_FS);
    CHECK(vr.nres[GFX_RES_UBO]==1 && fr.nres[GFX_RES_UBO]==1);
    s.vs=&vr; s.fs=&fr;
    CHECK(!geometry_validate(&s,&fixture_context.props.limits));
    descriptor_tests(&s);
    static const char *names[]={"gs_lines","gs_lines_adjacency","gs_triangles","gs_triangles_adjacency"};
    const uint8_t prim[]={NVN_DRAW_PRIMITIVE_LINES,NVN_DRAW_PRIMITIVE_LINES_ADJACENCY,
        NVN_DRAW_PRIMITIVE_TRIANGLES,NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY};
    const VkPrimitiveTopology native[]={VK_PRIMITIVE_TOPOLOGY_LINE_LIST,VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY};
    s.vs=&vs; s.fs=&fs;
    for(unsigned i=0;i<4;i++) {
        GfxShader alternate; load_shader(&alternate,argv[1],names[i],GFX_STAGE_GS);
        s.gs=&alternate; s.prim=prim[i]; CHECK(!geometry_validate(&s,&fixture_context.props.limits));
        GKey k; gkey_fill(&s,&k); CHECK(k.geometry_input==i+2 && !k.indep);
        Bindings b={0}; CHECK(add_stage(&b,&vs,0) && add_stage(&b,&alternate,0) && add_stage(&b,&fs,1));
        Layout *l=calloc(1,sizeof *l+b.n*sizeof *l->b);if(!l)abort();
        l->n=b.n;memcpy(l->b,b.b,b.n*sizeof *b.b);
        made=destroyed=fail_module=0;
        CHECK(create_graphics(&s,&k,l,1)); CHECK(captured_topology==native[i]); CHECK(made==3 && destroyed==3);
        free(l); free(alternate.spirv);
    }
    GfxShader go,fi; load_shader(&go,argv[1],"gs_output1",GFX_STAGE_GS);
    load_shader(&fi,argv[1],"ps_input1",GFX_STAGE_FS);
    s.gs=&go;s.fs=&fi;s.prim=NVN_DRAW_PRIMITIVE_POINTS;
    CHECK(vs.out_mask==1 && go.out_mask==2 && fi.in_mask==2);
    CHECK(!geometry_validate(&s,&fixture_context.props.limits)); descriptor_tests(&s);
    free(go.spirv);free(fi.spirv);
    portable_tests(argv[1]);
    printf("%u checks, %u failures (CPU spies only)\n",checks,failures);
    free(vs.spirv); free(gs.spirv); free(fs.spirv); free(vr.spirv); free(fr.spirv);
    return failures?1:0;
}
