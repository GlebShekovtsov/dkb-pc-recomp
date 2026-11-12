// Original CPU packing controls. @-markers are filled with unchanged production
// source by descriptor_empty_set_test.py; all remaining helpers are providers.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_hmap.h"
#include "vk_exec.h"
#define MAX_BINDINGS 128
#define MAX_DESC 1024
#define SUPPORT_BYTES 1408u
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_GS 2
#define NVN_STAGE_CS 5
#define A8(x) (((x) + 7) & ~(size_t)7)
#define DP_MARK(i) ((void)(i))   // the draw profile's marks time nothing here
@CONTEXT@
static DrawCtx ctx;   // the replay thread's draw context; TOTK_REPLAY_PAR=check's digest (DIG) is off

static unsigned checks, failures, pushed[2], written[2], allocated[2], bound[2];
static const char *case_name;
static int use_template, push_set, validation;
static VkCtx context;
static VkDescSlot expected[2][8];
static uint32_t expected_count[2], expected_bindings[2];
static VkDescriptorBufferInfo buffers[6][32];
static struct { VkBufferView zview; } standins;
#define TOKEN(type,n) ((type)(uintptr_t)(n))
#define CHECK(c,message) do { checks++; if (!(c)) { failures++; \
    fprintf(stderr,"FAIL %s template=%d push=%d validation=%d: %s\n", \
            case_name,use_template,push_set,validation,message); } } while (0)

void rt_fatal(const char *fmt,...) { (void)fmt; abort(); }
VkCtx *vk_ctx(void) { return &context; }
int rt_knob(const char *name,const char *def) { (void)name; (void)def; return 0; } // dynamic uniform buffers off
int rt_ab(void) { return 0; }
static void bind_stats(DrawCtx *dc,const VkPipeInfo *info) { (void)dc; (void)info; }
static int bind_skip_on(DrawCtx *dc) { (void)dc; return 0; } // exercise every descriptor build
static int push_skip_on(void) { return 0; } // exercise every push
static int bind_sig(const VkExec *ex,const VkPipeInfo *info) { (void)ex; (void)info; abort(); }
void vk_mem_count(int cls,int use,uint64_t bytes) { (void)cls; (void)use; (void)bytes; abort(); }
static void support_in(const VkExec *ex,float in[3]) { (void)ex; (void)in; abort(); }
void gfx_support_fill(uint8_t *out,float x,float y,float z) { (void)out; (void)x; (void)y; (void)z; abort(); }
static void *scratch(VkExec *ex,uint32_t n,VkBuffer *buffer,VkDeviceSize *offset) {
    (void)ex; (void)n; (void)buffer; (void)offset; abort();
}
static VkDescriptorBufferInfo ubo_info(VkExec *ex,int stage,int slot) {
    (void)ex; if (stage<0 || stage>=6 || slot<0 || slot>=32) abort();
    return buffers[stage][slot];
}
static VkDescriptorBufferInfo ssbo_info(VkExec *ex,int stage,const GfxShaderRes *r,int written_) {
    (void)ex; (void)stage; (void)r; (void)written_; abort();
}
void vk_texture_lock(void) { abort(); }
void vk_texture_unlock(void) { abort(); }
static void tex_info(VkExec *ex,const GfxShaderRes *r,uint64_t h,int image,int separate,uint32_t gen,uint32_t ep,
                     uint32_t sg,int *locked,VkDescriptorImageInfo *out) {
    (void)ex; (void)r; (void)h; (void)image; (void)separate; (void)gen; (void)ep; (void)sg; (void)locked; (void)out;
    abort();
}
// The descriptor memos' keys: one unchanged descriptor context for every case.
static void descriptor_state(VkExec *ex) { (void)ex; }
uint32_t vk_texture_slots_gen(void) { return 0; }
uint32_t vk_texture_epoch(void) { return 0; }
uint32_t vk_sampler_gen(void) { return 0; }
// TOTK_REPLAY_PAR=check's digest of what a draw bound: off.
static void dig_bound(DrawCtx *dc) { (void)dc; abort(); }
static void dig_desc(DrawCtx *dc,const VkPipeInfo *info,const VkDescSlot *d) { (void)dc; (void)info; (void)d; abort(); }
static int tex_repeat(VkExec *ex,const VkPipeInfo *info,const TexPend *tp,int n) { (void)ex; (void)info; (void)tp; (void)n; return 0; }
static int tex_info_hit(VkExec *ex,const GfxShaderRes *r,uint64_t h,int separate,VkDescriptorImageInfo *out) {
    (void)ex; (void)r; (void)h; (void)separate; (void)out; abort();
}

// Bounded malloc storage supplies alignment without imposing an array's declared
// effective type. The real producers decide packet sizes and copy every payload.
enum { R_PushTpl=1, R_Push=2, MAX_PACKET_BYTES=4096 };
typedef struct { void *data; uint32_t op; size_t bytes; int published; } Packet;
static Packet packets[4], *pending, *consuming;
static unsigned packet_count;
static VkCommandBuffer rec_cb;
static int rec_live=1;
#define OWNED(cb) (rec_live && (cb)==rec_cb)
#define FALLBACK() abort()
static void *put(uint32_t op,size_t bytes) {
    if (packet_count>=4 || !bytes || bytes>MAX_PACKET_BYTES || pending) abort();
    pending=&packets[packet_count++];
    *pending=(Packet){.data=malloc(bytes),.op=op,.bytes=bytes};
    if (!pending->data) abort();
    memset(pending->data,0xa5,bytes); return pending->data;
}
static void pub(void) { if (!pending) abort(); pending->published=1; pending=NULL; }
@RECORDER@

static int packet_range(const void *data,size_t bytes) {
    uintptr_t start=(uintptr_t)consuming->data, at=(uintptr_t)data;
    return at>=start && at-start<=consuming->bytes && bytes<=consuming->bytes-(at-start);
}
static void expect_slots(uint32_t set,const void *data) {
    CHECK(set<2,"valid descriptor set number"); if (set>=2) return;
    CHECK(!memcmp(data,expected[set],expected_count[set]*sizeof(VkDescSlot)),"all descriptor values and order match the set");
}
static VKAPI_ATTR void VKAPI_CALL push_template(VkCommandBuffer cb,VkDescriptorUpdateTemplate tpl,
                                               VkPipelineLayout layout,uint32_t set,const void *data) {
    CHECK(cb==rec_cb && layout==TOKEN(VkPipelineLayout,91),"template dispatch preserves command buffer and layout");
    CHECK(set<2 && tpl==TOKEN(VkDescriptorUpdateTemplate,101+set),"template dispatch selects the right set template");
    if (set>=2) return;
    pushed[set]++;
    size_t bytes=expected_count[set]*sizeof(VkDescSlot);
    CHECK(consuming->bytes==sizeof(A_PushTpl)+bytes,"template packet copies the complete set descriptor span");
    int valid=bytes==0 || (data && packet_range(data,bytes));
    CHECK(valid,"template payload belongs to its complete recorder packet");
    if (valid && bytes) expect_slots(set,data); // never read outside a failing baseline packet
}
static void expect_writes(uint32_t set,uint32_t n,const VkWriteDescriptorSet *writes) {
    CHECK(n==expected_bindings[set],"write path retains the set's binding count");
    uint32_t at=0;
    for (uint32_t i=0;i<n;i++) {
        const VkWriteDescriptorSet *w=&writes[i];
        CHECK(w->sType==VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET && !w->pNext &&
              w->dstBinding==i+3 && !w->dstArrayElement && w->descriptorType==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              "write path preserves binding fields");
        CHECK(w->descriptorCount && w->descriptorCount<=expected_count[set]-at,"write array count is in its set");
        if (!w->descriptorCount || w->descriptorCount>expected_count[set]-at) return;
        CHECK(w->pBufferInfo && !w->pImageInfo && !w->pTexelBufferView,"write selects only buffer descriptors");
        if (consuming) CHECK(packet_range(w->pBufferInfo,w->descriptorCount*sizeof *w->pBufferInfo),"write payload is copied into recorder storage");
        if (w->pBufferInfo) CHECK(!memcmp(w->pBufferInfo,&expected[set][at],w->descriptorCount*sizeof *w->pBufferInfo),"write array values match the set");
        at+=w->descriptorCount;
    }
    CHECK(at==expected_count[set],"write path covers every descriptor in the set");
}
static VKAPI_ATTR void VKAPI_CALL push_writes(VkCommandBuffer cb,VkPipelineBindPoint bp,VkPipelineLayout layout,
                                             uint32_t set,uint32_t n,const VkWriteDescriptorSet *writes) {
    CHECK(cb==rec_cb && layout==TOKEN(VkPipelineLayout,91),"write dispatch preserves command buffer and layout");
    CHECK(bp==VK_PIPELINE_BIND_POINT_GRAPHICS || bp==VK_PIPELINE_BIND_POINT_COMPUTE,"write dispatch keeps a supported bind point");
    CHECK(set<2,"write dispatch selects a valid set"); if (set>=2) return;
    pushed[set]++;
    CHECK(packet_range(writes,n*sizeof *writes),"write structs belong to recorder storage");
    expect_writes(set,n,writes);
}
PFN_vkCmdPushDescriptorSetWithTemplateKHR vk_pipeline_push_tpl_fn(void) { return push_template; }
PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void) { return push_writes; }
static VkDescriptorSet alloc_set(VkExec *ex,VkDescriptorSetLayout layout) {
    (void)ex; unsigned set=layout==TOKEN(VkDescriptorSetLayout,82);
    if (layout!=TOKEN(VkDescriptorSetLayout,81+set)) abort();
    allocated[set]++; return TOKEN(VkDescriptorSet,201+set);
}
static VKAPI_ATTR void VKAPI_CALL update_template(VkDevice device,VkDescriptorSet set,
                                                 VkDescriptorUpdateTemplate tpl,const void *data) {
    (void)device; unsigned s=set==TOKEN(VkDescriptorSet,202);
    CHECK(set==TOKEN(VkDescriptorSet,201+s) && tpl==TOKEN(VkDescriptorUpdateTemplate,101+s),"allocated template selects its set");
    written[s]++; expect_slots(s,data);
}
PFN_vkUpdateDescriptorSetWithTemplate vk_pipeline_update_tpl_fn(void) { return update_template; }
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device,uint32_t n,const VkWriteDescriptorSet *writes,
                                                  uint32_t copies,const VkCopyDescriptorSet *copy) {
    (void)device; CHECK(!copies && !copy && n,"allocated writes use only write structs");
    if (!n) return;
    unsigned s=writes[0].dstSet==TOKEN(VkDescriptorSet,202);
    for (uint32_t i=0;i<n;i++) CHECK(writes[i].dstSet==TOKEN(VkDescriptorSet,201+s),"each allocated write targets the same set");
    written[s]++; expect_writes(s,n,writes);
}
void rc_vkCmdBindDescriptorSets(VkCommandBuffer cb,VkPipelineBindPoint bp,VkPipelineLayout layout,uint32_t first,
                                uint32_t n,const VkDescriptorSet *sets,uint32_t nd,const uint32_t *offsets) {
    (void)bp; CHECK(cb==rec_cb && layout==TOKEN(VkPipelineLayout,91) && first<2 && n==1 && !nd && !offsets,
                    "allocated bind preserves dispatch arguments");
    if (first<2) { CHECK(*sets==TOKEN(VkDescriptorSet,201+first),"allocated bind selects the right set"); bound[first]++; }
}
@BINDER@

static void consume(void) {
    for (unsigned i=0;i<packet_count;i++) {
        consuming=&packets[i]; CHECK(consuming->published,"only published packets reach the consumer");
        uint32_t op=consuming->op; void *p=consuming->data; VkCommandBuffer cb=rec_cb;
        switch (op) {
@CONSUMERS@
        default: abort();
        }
        free(consuming->data); consuming->data=NULL;
    }
    consuming=NULL; packet_count=0;
}
static void run(const char *name,uint32_t vs,uint32_t fs,int compute,int default_sets) {
    case_name=name;
    memset(&ctx,0,sizeof ctx);
    memset(pushed,0,sizeof pushed); memset(written,0,sizeof written);
    memset(allocated,0,sizeof allocated); memset(bound,0,sizeof bound);
    memset(expected,0,sizeof expected); memset(expected_count,0,sizeof expected_count);
    memset(expected_bindings,0,sizeof expected_bindings); packet_count=0; pending=consuming=NULL;
    context.validation=validation;
    VkExec ex={.cb=TOKEN(VkCommandBuffer,71),.draw=&ctx}; rec_cb=ex.cb;
    GfxShaderRes resources[4]={{0}}; VkPipeBinding bindings[4]={{0}};
    VkPipeInfo info={.layout=TOKEN(VkPipelineLayout,91),.set_layout=TOKEN(VkDescriptorSetLayout,81),
        .set_layout1=TOKEN(VkDescriptorSetLayout,82),.bind_point=compute?VK_PIPELINE_BIND_POINT_COMPUTE:VK_PIPELINE_BIND_POINT_GRAPHICS,
        .n_sets=compute?(default_sets?0:1):2,.bindings=bindings};
    uint32_t counts[2]={vs,fs};
    for (unsigned s=0;s<(compute?1u:2u);s++) {
        uint32_t left=counts[s],slot=3;
        while (left) {
            // A singleton followed by an array distinguishes descriptor count
            // from binding count at the set boundary.
            unsigned count=slot==3?1:left, i=info.n_bindings++, b=expected_bindings[s]++;
            unsigned stage=compute?NVN_STAGE_CS:s;
            resources[i]=(GfxShaderRes){.slot=(int16_t)slot};
            bindings[i]=(VkPipeBinding){.binding=(uint16_t)(b+3),.stage=compute?GFX_STAGE_CS:s?GFX_STAGE_FS:GFX_STAGE_VS,
                .kind=GFX_RES_UBO,.type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.count=(uint16_t)(count==1?0:count),.res=&resources[i],.set=(uint8_t)s};
            for (unsigned j=0;j<count;j++) expected[s][expected_count[s]++].buffer=buffers[stage][slot+j];
            slot+=count; left-=count;
        }
        if (expected_count[s] && use_template) info.tpl[s]=TOKEN(VkDescriptorUpdateTemplate,101+s);
    }
    info.push=push_set==0 && expected_count[0]; info.push1=push_set==1 && expected_count[1];
    bind_resources(&ex,&info);
    consume();
    CHECK(ex.recorded==1,"one descriptor build is recorded");
    for (unsigned s=0;s<(compute?1u:2u);s++) {
        unsigned is_push=s?info.push1:info.push;
        CHECK(pushed[s]==(is_push?1u:0u),"only the chosen nonempty push set is pushed");
        CHECK(written[s]==(!is_push && expected_count[s]?1u:0u),"every allocated nonempty set is written once");
        unsigned alloc=!is_push && (!s || expected_count[s] || validation);
        CHECK(allocated[s]==alloc && bound[s]==alloc,"empty fragment set binds only for the validation workaround");
    }
}
int main(void) {
    _Static_assert(sizeof(VkDescSlot)==sizeof(VkDescriptorBufferInfo),"packed buffer array stride");
    for (unsigned st=0;st<6;st++) for (unsigned slot=0;slot<32;slot++)
        buffers[st][slot]=(VkDescriptorBufferInfo){TOKEN(VkBuffer,1000+st*100+slot),32*slot,256+st*16+slot};
    for (use_template=0;use_template<2;use_template++) for (validation=0;validation<2;validation++) {
        for (push_set=0;push_set<2;push_set++) {
            run("vertex only",1,0,0,0); run("vertex array",4,0,0,0);
            run("fragment only",0,1,0,0); run("fragment array",0,4,0,0);
            run("both stages",4,3,0,0); run("empty graphics",0,0,0,0);
        }
        push_set=0; run("compute single set",4,0,1,0); run("compute default set count",4,0,1,1);
    }
    printf("descriptor empty set: %u checks, %u failures\n",checks,failures);
    return failures!=0;
}
