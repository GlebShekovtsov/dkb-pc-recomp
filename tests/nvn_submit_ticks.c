// Original synthetic lifetime cases. Named markers contain actual production code, not copies of its decisions.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif
#include "vk_exec.h"
#include "platform/platform.h"
#define BY_VALUE_API @API@
const int rt_game_version = TEST_GAME_VERSION;
unsigned rt_trace;
static unsigned checks, failures;
#define CHECK(c) do { checks++; if (!(c)) { failures++; fprintf(stderr,"FAIL line%d: %s\n",__LINE__,#c); } } while (0)
static uint64_t clock_tick, now_ns=3200000000ull, issued, reserved, last_fence;
static int async_mode, early, inside_frontend, nested, nesting;
@GLOBALS@
@REPLAY_TYPES@
static ReplayItem replay[4096];
static unsigned replay_head,replay_count;
typedef struct { void (*fn)(void*); void *arg; uint64_t fence; } Job;
static Job callbacks[16384];
static unsigned jobs, consumed;
static uint64_t callback_ns[16384];
static VkExec X;
static int x_ready,x_here,split_draws,part_start;
static uint64_t part_t0,n_subs,n_parts;
VkExecFrame vk_exec_frame;
typedef struct { VkImageView view; } ViewEnt;
typedef struct { int n; VkImageView v[]; } ViewList;
static void state_reset(VkExec *ex) { memset(ex,0,sizeof *ex); }
void vk_dump_init(void) {}
void vk_exec_pass_end(VkExec *ex) { ex->in_pass=0; }
static void par_flush(VkExec *ex) { (void)ex; }
static int par_mode;
static int par_threads(void) { return par_mode; }
static int npiece;
static int par_collecting(void) { return 0; }
static uint64_t par_end_op[OP_COUNT_ + 1];
void vk_exec_barrier(VkExec *ex) { ex->recorded++; }
void vk_mem_mirror_begin(void) {}
int vk_mem_mirror_end(VkCommandBuffer cb,uint64_t f) { (void)cb;(void)f;return 0; }
void vk_usnap_begin(const uint64_t *h,int n,uint64_t f) { (void)h;(void)n;(void)f; }
void vk_usnap_end(uint64_t f) { (void)f; }
static void free_views(void *p) { (void)p;abort(); }
uint64_t kern_now_ns(void) { return now_ns; }
uint64_t rt_ticks(void) { return clock_tick; }
uint64_t rt_ns_to_ticks(uint64_t ns) { return ns/32; }
void rt_log(const char *fmt,...) { (void)fmt; }
void rt_fatal(const char *fmt,...) { (void)fmt;abort(); }
void nvn_called(uint8_t *p,const char *n) { (void)p;(void)n; }
int kern_core_away(void) { return 1; }
void kern_core_back(int *p) { *p=0; }
#ifdef _WIN32
_Static_assert(sizeof(PlatMutex)==sizeof(SRWLOCK),"mutex storage");
void plat_mutex_lock(PlatMutex *m) { AcquireSRWLockExclusive((PSRWLOCK)m); }
void plat_mutex_unlock(PlatMutex *m) { ReleaseSRWLockExclusive((PSRWLOCK)m); }
#else
static pthread_mutex_t *mutex_of(PlatMutex *m) {
    if (!m->p) {
        pthread_mutex_t *p=malloc(sizeof *p);
        if (!p || pthread_mutex_init(p,NULL)) abort();
        m->p=p; // fixture participants are deterministic/sequential; no concurrent initialization
    }
    return m->p;
}
void plat_mutex_lock(PlatMutex *m) { pthread_mutex_lock(mutex_of(m)); }
void plat_mutex_unlock(PlatMutex *m) { pthread_mutex_unlock(mutex_of(m)); }
#endif
int gfx_replay_async(void) { return async_mode; }
uint64_t nvn_submit_count(void) { return lod_entered; }
void vk_sync_start(void) {}
void vk_present_drain(void) {}
void vk_texture_new_submit(void) {}
void vk_mem_count(int cls,int use,uint64_t size) { (void)cls;(void)use;(void)size; }
int vk_mem_resolve(uint64_t addr,uint64_t size,VkMemSpan *m) { memset(m,0,sizeof *m);return addr && size==sizeof(NVNcounterData); }
static int oq_ready(void) { abort(); }
static void oq_report(VkExec *ex,uint64_t addr) { (void)ex;(void)addr;abort(); }
#define PENDING 256
static struct { uint64_t addr,fence;int unknown; } pending[PENDING];
static unsigned pending_head;
void vk_completion_add(uint64_t f,void(*fn)(void*),void *arg) {
    if (jobs==16384) abort(); callbacks[jobs++]=(Job){fn,arg,f};
}
@COUNTERS@
typedef struct { int count; p_report_counter reports[2]; } Recording;
// Command decoding adapter: only the fixture's original report descriptors exist, no GPU operations.
static void walk(VkExec *ex,const op_hdr *h,int depth) {
    (void)depth;CHECK(x_here && !split_draws && part_t0==now_ns);
    const Recording *r=(const Recording*)h;
    for (int i=0;i<r->count;i++) report_counter(ex,&r->reports[i]);
}
// The helper thread is off here (vk_spec.c, the par_* functions of vk_exec.c): a submission is recorded by its one
// thread. A checkout from before the helper uses none of these.
enum { PS_SCAN };
static struct { const op_hdr *next; int state; uint64_t rec; } P __attribute__((unused));
int vk_spec_on(void) { return 0; }
int vk_mem_shadow_on(void) { return 0; }
static void par_close(VkExec *ex) { (void)ex; }
static void par_log(void) {}
@RECORD@
static void complete_one(void) {
    if (consumed==jobs) abort();unsigned i=consumed++;
    callback_ns[i]=now_ns;callbacks[i].fn(callbacks[i].arg);
}
// Vulkan queue adapter. The real vk_sync_submit forwarding hop is also checked by vk_resource_frontier_test.py.
#if BY_VALUE_API
uint64_t vk_sync_submit(const uint64_t *h,int n,uint64_t stamp) {
    uint64_t f=++issued;vk_exec_record((VkCommandBuffer)1,h,n,f,stamp);
#else
uint64_t vk_sync_submit(const uint64_t *h,int n) {
    uint64_t f=++issued;vk_exec_record((VkCommandBuffer)1,h,n,f);
#endif
    if (early && consumed<jobs) { CHECK(inside_frontend);complete_one(); }
    return f;
}
static uint64_t replay_enqueue(ReplayItem *it) {
    if (replay_count==4096) abort();it->value=++reserved;replay[replay_count++]=*it;return it->value;
}
@BACKEND@
static const GfxBackend backend={.submit=vk_be_submit};
const GfxBackend *gfx_backend(void) { return &backend; }
void nvnp_nvnQueueSubmitCommands(Cpu *c);
static Recording empty,nested_record;
static void submit(const uint64_t *h,int n,uint64_t stamp) {
    queue_t q={0};Cpu c={0};clock_tick=stamp;c.x[0]=(uintptr_t)&q;c.x[1]=(uint64_t)n;c.x[2]=(uintptr_t)h;
    int outer=inside_frontend;inside_frontend=1;nvnp_nvnQueueSubmitCommands(&c);inside_frontend=outer;
    CHECK(q.submits==1);
}
void nvn_submit_prepare(const uint64_t *h,int n) {
    (void)h;(void)n;
    if (nested && !nesting) { nesting=1;uint64_t e=(uintptr_t)&nested_record;submit(&e,1,777);nesting=0; }
}
void nvn_submit_prepare_async(const uint64_t *h,uint64_t *out,int n) { memcpy(out,h,(size_t)n*sizeof *out); }
@FRONTEND@
static void pump(void) {
    while (replay_head<replay_count) {
        ReplayItem it=replay[replay_head++];
        @SUBMIT_BRANCH@
    }
}
static void finish_callbacks(void) { while(consumed<jobs) { now_ns+=3200;complete_one(); } }
static void check_reports(NVNcounterData *out,int n,uint64_t stamp,unsigned begin) {
    for (int i=0;i<n;i++) {
        uint64_t want=rt_game_version?callback_ns[begin+i]/32:stamp;
        CHECK(out[i].counter==0 && out[i].timestamp==want);
    }
}
int main(void) {
    for (int mode=0;mode<2;mode++) {
        async_mode=mode;reserved=issued;early=!mode;
        for (unsigned trial=0;trial<3;trial++) {
            uint64_t stamp=trial==0?0:trial==1?123:UINT64_MAX;
            // Windows may place a stack in the emulated GPU-only address window. Static owned records avoid that
            // invalid fixture input; the production nvn_gpu_only guard must remain active.
            static NVNcounterData out[4];memset(out,0,sizeof out);CHECK(!nvn_gpu_only((uintptr_t)out));Recording r[2];
            for(int k=0;k<2;k++) {
                r[k].count=2;r[k].reports[0]=(p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP_TOP,(uintptr_t)&out[k*2]};
                r[k].reports[1]=(p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP,(uintptr_t)&out[k*2+1]};
            }
            uint64_t h[2]={(uintptr_t)&r[0],(uintptr_t)&r[1]};unsigned begin=jobs;
            submit(h,2,stamp);clock_tick=999999; // caller state changes before deferred replay
            if(mode) { CHECK(jobs==begin);pump(); }
            CHECK(jobs==begin+4);
            early=0; // retain the remaining payloads while later submissions replace the recorder context
            uint64_t e=(uintptr_t)&empty;
            for(int i=0;i<1026;i++) { submit(&e,1,5000+(uint64_t)i);if(mode)pump(); }
            finish_callbacks();check_reports(out,4,stamp,begin);early=!mode;
        }
    }
    // The later entry submits first: preserve each captured value, without claiming global timestamp monotonicity.
    async_mode=0;early=0;nested=1;static NVNcounterData out[2];CHECK(!nvn_gpu_only((uintptr_t)out));
    nested_record=(Recording){.count=1,.reports={{NVN_COUNTER_TYPE_TIMESTAMP,(uintptr_t)&out[0]}}};
    Recording r={.count=1,.reports={{NVN_COUNTER_TYPE_TIMESTAMP,(uintptr_t)&out[1]}}};uint64_t h=(uintptr_t)&r;unsigned begin=jobs;
    submit(&h,1,321);nested=0;CHECK(jobs==begin+2);finish_callbacks();
    check_reports(&out[0],1,777,begin);check_reports(&out[1],1,321,begin+1);
    printf("submitted-tick lifetime: %u checks, %u failures, legacy%d\n",checks,failures,rt_game_version);
    return failures?1:0;
}
