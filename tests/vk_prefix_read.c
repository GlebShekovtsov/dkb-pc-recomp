// Original CPU provider fixture for the actual prefix-split/readback functions. No Vulkan loader is linked.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "host.h"
#include "vk_exec.h"
#include "platform/platform.h"

#define H(type, n) ((type)(uintptr_t)(n))
#define NEED(c) do { checks++; if (!(c)) rt_fatal("fixture line %d: %s", __LINE__, #c); } while (0)
static unsigned checks, submits, waits, begins, destroys, frees, invalidates, binds, copies, barriers, resolutions, signals;
static int recorder, has_prologue, create_failure, map_failure, invalidate_failure, submit_failure, wait_failure;
static int resolved = 1, bad_offset, short_span, queue_held, copied, completed, selected;
static unsigned step, end_step, upload_step, pro_step, submit_step, wait_step, begin_step;
static VkCtx context;
static VkExec X;
static int x_ready;
static _Thread_local int x_here;
static int split_draws, part_start;
static uint64_t part_t0, n_parts, n_subs;
static PlatMutex sub_lock, st_lock;
static PlatCond cv_work;
static unsigned mirror_begin, mirror_end, snapshot_begin, snapshot_end, walked, worked;
static unsigned expect_recorded;
VkExecFrame vk_exec_frame;
/* PRODUCTION_TYPES */
typedef struct { VkCommandBuffer cb; uint64_t fence; } CmdSlot;
static CmdSlot slot_storage[3], *slots = slot_storage;
static int cur_slot;
static uint64_t cur_fence, issued, done, reserved;
static VkBuffer copy_source;
static VkBufferCopy region;
static uint8_t gpu[3][256], readback[264];
unsigned rt_trace;
int vk_mirror_on;

void rt_fatal(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\nfrees=%u waits=%u begins=%u signals=%u\n", frees, waits, begins, signals);
    exit(70);
}
void rt_log(const char *fmt, ...) { (void)fmt; }
void plat_mutex_lock(PlatMutex *m) { NEED(!m->p); m->p = (void *)1; }
void plat_mutex_unlock(PlatMutex *m) { NEED(m->p); m->p = NULL; }
void plat_cond_broadcast(PlatCond *c) { NEED(c == &cv_work); }
VkCtx *vk_ctx(void) { return &context; }
uint64_t perf_begin(void) { return 123; }
void perf_end(int k, uint64_t t) { (void)k; (void)t; }
void perf_gpu_wait(uint64_t t) { (void)t; }
void perf_replay(uint64_t t) { (void)t; }
void perf_replay_totals(uint64_t d, uint64_t t, uint64_t c) { (void)d; (void)t; (void)c; }
uint64_t vk_exec_draws_total(void) { return 0; }
void vk_exec_pool_counts(uint64_t *c, uint64_t *t) { *c = *t = 0; }
uint64_t kern_now_ns(void) { return 456; }
void vk_sync_start(void) {}
void vk_mem_dirty_submit(void) {}
void vk_mem_protect_until(uint64_t f) { NEED(f == cur_fence); }
void rec_stats_log(void) {}
static void ts_drop(void) {}
static void state_reset(VkExec *ex) { memset(ex, 0, sizeof *ex); }
void vk_dump_init(void) {}
void vk_mem_mirror_begin(void) { NEED(x_here); mirror_begin++; }
int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t f) { NEED(!x_here && cb == X.cb && f == X.fence); mirror_end++; return 2; }
void vk_usnap_begin(const uint64_t *handles, int n, uint64_t f) {
    NEED(x_here && f == X.fence && (!n || handles)); snapshot_begin++;
}
void vk_usnap_end(uint64_t f) { NEED(!x_here && f == X.fence); snapshot_end++; }
void vk_exec_pass_end(VkExec *ex) { NEED(ex == &X); ex->in_pass = 0; }
static void walk(VkExec *ex, const op_hdr *h, int depth) {
    NEED(x_here && ex == &X && h->op == OP_BEGIN && !depth); ex->recorded += 2; walked++;
}
static void free_views(void *arg) { (void)arg; rt_fatal("unexpected views in provider fixture"); }
void vk_completion_add(uint64_t f, void (*fn)(void *), void *arg) {
    (void)f; (void)fn; (void)arg; rt_fatal("unexpected completion callback in provider fixture");
}
void vk_device_fault_report(VkResult r) { NEED(r != VK_SUCCESS); }
void vk_queue_lock(void) { NEED(!queue_held); queue_held = 1; }
void vk_queue_unlock(void) { NEED(queue_held); queue_held = 0; }
static void ts_end(VkCommandBuffer cb, uint64_t fence) { NEED(cb == slots[cur_slot].cb && fence == cur_fence); }
static void ts_begin(VkCommandBuffer cb) { NEED(cb == slots[cur_slot].cb); }
int rec_end_cb(VkCommandBuffer cb, int crumb, uint64_t n) {
    NEED(cb == slots[cur_slot].cb && ((!crumb && !n) || (crumb == VK_CRUMB_SUBMIT_END && n == cur_fence)));
    end_step = ++step; return recorder;
}
void rec_begin(VkCommandBuffer cb) { NEED(cb == slots[cur_slot].cb && begin_step > submit_step); }
void vk_crumb(VkCommandBuffer cb, int k, uint64_t n) { NEED(cb == slots[cur_slot].cb && k == VK_CRUMB_SUBMIT && n == cur_fence); }
uint64_t vk_upload_flush(int wait) { NEED(!wait && end_step); upload_step = ++step; return 1; }
static VkCommandBuffer prologue(VkCtx *c, uint64_t fence) {
    NEED(c == &context && fence == cur_fence && upload_step > end_step); pro_step = ++step;
    return has_prologue ? H(VkCommandBuffer, 2) : VK_NULL_HANDLE;
}
static CmdSlot *take_slot(VkCtx *c, uint64_t fence) {
    NEED(c == &context && fence == cur_fence && cur_slot < 2);
    cur_slot++; slots[cur_slot] = (CmdSlot){H(VkCommandBuffer, 10 + cur_slot), fence}; return &slots[cur_slot];
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice d, const VkFenceCreateInfo *i, const VkAllocationCallbacks *a, VkFence *f) {
    (void)d; NEED(i->sType == VK_STRUCTURE_TYPE_FENCE_CREATE_INFO && !i->flags && !a); *f = H(VkFence, 7); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer cb) { NEED(cb == slots[cur_slot].cb && end_step); return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f) {
    (void)q; NEED(queue_held && n == 1 && pro_step > upload_step);
    NEED(!s->waitSemaphoreInfoCount);
    if (s->signalSemaphoreInfoCount) {
        signals++;
        NEED(s->signalSemaphoreInfoCount == 1 && f == VK_NULL_HANDLE);
        NEED(s->pSignalSemaphoreInfos->semaphore == context.timeline && s->pSignalSemaphoreInfos->value == cur_fence);
        NEED(s->pSignalSemaphoreInfos->stageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        NEED(s->commandBufferInfoCount == (unsigned)(has_prologue + expect_recorded));
        if (expect_recorded) NEED(s->pCommandBufferInfos[s->commandBufferInfoCount - 1].commandBuffer == slots[cur_slot].cb);
    } else {
        NEED(s->commandBufferInfoCount == (has_prologue ? 2u : 1u));
        NEED(s->pCommandBufferInfos[s->commandBufferInfoCount - 1].commandBuffer == slots[cur_slot].cb);
    }
    if (has_prologue) NEED(s->pCommandBufferInfos[0].commandBuffer == H(VkCommandBuffer, 2));
    NEED(f == H(VkFence, 7) || f == VK_NULL_HANDLE); submits++; submit_step = ++step;
    return submit_failure ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice d, uint32_t n, const VkFence *f, VkBool32 all, uint64_t timeout) {
    (void)d; NEED(!queue_held && n == 1 && *f == H(VkFence, 7) && all && timeout == UINT64_MAX);
    NEED(submit_step > pro_step); waits++; wait_step = ++step;
    if (wait_failure) return VK_ERROR_DEVICE_LOST;
    if (copied) {
        unsigned source = (unsigned)(uintptr_t)copy_source - 20;
        NEED(source < 3 && region.srcOffset + region.size <= sizeof gpu[0]);
        memcpy(readback + 4 + region.dstOffset, gpu[source] + region.srcOffset, (size_t)region.size);
    }
    completed = 1; return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice d, VkFence f, const VkAllocationCallbacks *a) {
    (void)d; NEED(f == H(VkFence, 7) && !a && completed && wait_step > submit_step); destroys++;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *i) {
    NEED(cb == slots[cur_slot].cb && i->flags == VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    begins++; begin_step = ++step; return VK_SUCCESS;
}
int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *s) {
    NEED(addr == 0x2000 && size && size <= (8ull << 20)); resolutions++;
    *s = (VkMemSpan){.buffer = H(VkBuffer, 20), .offset = bad_offset ? 1 : 16,
        .avail = short_span ? size - 1 : size, .pool = (void *)1, .cls = VK_MCLS_HOST_CACHED};
    return resolved;
}
void vk_mem_pick(VkMemSpan *s) { NEED(s->buffer == H(VkBuffer, 20)); s->buffer = H(VkBuffer, 20 + selected); }
void vk_mem_count(int cls, int use, uint64_t n) { NEED(cls == VK_MCLS_HOST_CACHED && use == VK_MUSE_COPY_SRC && n); }
void vk_mem_bound_note(void *p) { NEED(p == (void *)1); }
void vk_exec_barrier(VkExec *ex) { NEED(ex == &X); ex->in_pass = 0; ex->recorded++; barriers++; }
void vk_exec_draw_new_cb(void) { NEED(X.cb == slots[cur_slot].cb && !X.in_pass); binds++; }
VkResult vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *b, const VmaAllocationCreateInfo *i,
                         VkBuffer *out, VmaAllocation *allocation, VmaAllocationInfo *mapped) {
    (void)a; NEED(b->size && b->size <= (8ull << 20) && b->usage == VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    NEED((i->flags & (VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) ==
         (VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    if (create_failure) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *out = H(VkBuffer, 30); *allocation = H(VmaAllocation, 31); mapped->pMappedData = map_failure ? NULL : readback + 4;
    return VK_SUCCESS;
}
void rc_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy *r) {
    NEED(cb == X.cb && !X.in_pass && barriers && n == 1 && dst == H(VkBuffer, 30));
    NEED(!r->dstOffset && !(r->srcOffset & 3) && !(r->size & 3)); copied = 1; copies++; copy_source = src; region = *r;
}
void rc_vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *di) {
    NEED(cb == X.cb && copied && di->bufferMemoryBarrierCount == 1);
    const VkBufferMemoryBarrier2 *b = di->pBufferMemoryBarriers;
    NEED(b->srcStageMask == VK_PIPELINE_STAGE_2_COPY_BIT && b->srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT);
    NEED(b->dstStageMask == VK_PIPELINE_STAGE_2_HOST_BIT && b->dstAccessMask == VK_ACCESS_2_HOST_READ_BIT);
    NEED(b->buffer == H(VkBuffer, 30) && !b->offset && b->size == region.size);
}
VkResult vmaInvalidateAllocation(VmaAllocator a, VmaAllocation p, VkDeviceSize off, VkDeviceSize n) {
    (void)a; NEED(p == H(VmaAllocation, 31) && !off && n == region.size && completed && begins && binds);
    invalidates++; return invalidate_failure ? VK_ERROR_MEMORY_MAP_FAILED : VK_SUCCESS;
}
void vmaDestroyBuffer(VmaAllocator a, VkBuffer b, VmaAllocation p) {
    (void)a; NEED(b == H(VkBuffer, 30) && p == H(VmaAllocation, 31) && (!copied || completed)); frees++;
}

/* PRODUCTION_FUNCTIONS */

static void reset(void) {
    submits = waits = begins = destroys = frees = invalidates = binds = copies = barriers = resolutions = signals = 0;
    step = end_step = upload_step = pro_step = submit_step = wait_step = begin_step = 0;
    copied = completed = 0; cur_slot = 0;
    slot_storage[0] = (CmdSlot){H(VkCommandBuffer, 10), 99}; cur_fence = 99;
    issued = 98; done = 97; reserved = 100;
    X = (VkExec){.cb = slots[0].cb, .fence = 99, .recorded = 9, .in_pass = 1, .barrier_at = 9};
    x_here = 1; split_draws = 17; part_start = 3; part_t0 = 0; n_parts = 0;
    memset(readback, 0xa5, sizeof readback);
    mirror_begin = mirror_end = snapshot_begin = snapshot_end = walked = worked = 0;
}
static void original_work(VkExec *ex, void *arg) {
    NEED(x_here && ex == &X && ex->fence == cur_fence && ex->submitted_ticks == 12345 && !ex->recorded);
    NEED(mirror_begin == 1 && snapshot_begin == 1); worked++;
    if (arg) NEED(vk_exec_read_buffer(ex, 0x2000, 20, arg));
    ex->recorded++;
}
static int original_record(VkCommandBuffer cb, uint64_t f, uint64_t ticks, void *arg) {
    return vk_exec_record_work(cb, f, ticks, original_work, arg);
}
int main(int argc, char **argv) {
    for (unsigned a = 0; a < 3; a++) for (unsigned b = 0; b < sizeof gpu[0]; b++) gpu[a][b] = (uint8_t)(11 + a * 53 + b);
    if (argc > 1) {
        reset(); submit_failure = !strcmp(argv[1], "submit-fail"); wait_failure = !strcmp(argv[1], "wait-fail");
        uint8_t out[20]; (void)vk_exec_read_buffer(&X, 0x2000, sizeof out, out); return 3;
    }
    for (recorder = 0; recorder < 2; recorder++) for (has_prologue = 0; has_prologue < 2; has_prologue++) {
        reset(); VkCommandBuffer cb = vk_sync_split();
        NEED(cb == slots[cur_slot].cb && submits == 1 && !waits && !destroys && begins == 1 && !signals);
        NEED(issued == 98 && done == 97 && reserved == 100 && cur_fence == 99 && slots[0].fence == 99);
        for (selected = 0; selected < 3; selected++) {
            reset(); vk_mirror_on = selected != 0;
            uint8_t out[28]; memset(out, 0xcd, sizeof out);
            NEED(vk_exec_read_buffer(&X, 0x2000, 20, out + 4));
            NEED(!memcmp(out + 4, gpu[selected] + 16, 20));
            for (unsigned k = 0; k < 4; k++) NEED(out[k] == 0xcd && out[24 + k] == 0xcd && readback[k] == 0xa5 && readback[24 + k] == 0xa5);
            NEED(submits == 1 && waits == 1 && destroys == 1 && frees == 1 && invalidates == 1 && copies == 1 && !signals);
            NEED(wait_step > submit_step && begin_step > wait_step && binds == 1);
            NEED(X.fence == 99 && issued == 98 && done == 97 && reserved == 100 && slots[0].fence == 99);
            NEED(X.cb == slots[1].cb && X.recorded == 12 && part_start == 12 && !split_draws && part_t0 == 456);
            NEED(X.barrier_at == -1 && X.dirty == VK_EXEC_DIRTY_ALL && n_parts == 1);
        }
    }
    selected = 0; vk_mirror_on = 0; recorder = 1; has_prologue = 1;
    reset(); uint8_t out[20]; memset(out, 0x7a, sizeof out);
    NEED(!vk_exec_read_buffer(&X, 0, 20, out));
    NEED(!vk_exec_read_buffer(&X, 0x2001, 20, out));
    NEED(!vk_exec_read_buffer(&X, 0x2000, 19, out));
    NEED(!vk_exec_read_buffer(&X, UINT64_MAX - 3, 8, out));
    NEED(!vk_exec_read_buffer(&X, 0x2000, (8ull << 20) + 4, out));
    NEED(!vk_exec_read_buffer(&X, 0x2000, 0, out));
    NEED(!vk_exec_read_buffer(&X, 0x2000, 20, NULL));
    VkExec other = X; NEED(!vk_exec_read_buffer(&other, 0x2000, 20, out));
    x_here = 0; NEED(!vk_exec_read_buffer(&X, 0x2000, 20, out)); x_here = 1;
    NEED(!resolutions && !copies && !submits);
    resolved = 0; NEED(!vk_exec_read_buffer(&X, 0x2000, 20, out)); resolved = 1;
    bad_offset = 1; NEED(!vk_exec_read_buffer(&X, 0x2000, 20, out)); bad_offset = 0;
    short_span = 1; NEED(!vk_exec_read_buffer(&X, 0x2000, 20, out)); short_span = 0;
    NEED(!copies && !submits);
    create_failure = 1; NEED(!vk_exec_read_buffer(&X, 0x2000, 20, out)); create_failure = 0;
    map_failure = 1; NEED(!vk_exec_read_buffer(&X, 0x2000, 20, out)); map_failure = 0;
    NEED(!copies && !submits && frees == 1);
    reset(); invalidate_failure = 1; NEED(!vk_exec_read_buffer(&X, 0x2000, 20, out));
    NEED(frees == 1 && waits == 1 && begins == 1); for (unsigned k = 0; k < sizeof out; k++) NEED(out[k] == 0x7a);
    invalidate_failure = 0;
    for (unsigned route = 0; route < 4; route++) {
        reset(); cur_slot = -1; x_here = 0; expect_recorded = route != 0;
        static op_hdr empty = {OP_BEGIN, sizeof(op_hdr)};
        const uint64_t handles[2] = {(uintptr_t)&empty, (uintptr_t)&empty};
        uint64_t f = route < 2 ? vk_sync_submit(route ? handles : NULL, route ? 2 : 0, 12345) :
            vk_sync_submit_record(original_record, route == 3 ? out : NULL, 12345);
        NEED(f == 99 && issued == 99 && done == 97 && reserved == 100 && !x_here && X.cb == VK_NULL_HANDLE);
        NEED(walked == (route == 1 ? 2u : 0u) && worked == (route >= 2 ? 1u : 0u));
        NEED(mirror_begin == expect_recorded && mirror_end == expect_recorded);
        NEED(snapshot_begin == expect_recorded && snapshot_end == expect_recorded);
        NEED(submits == (route == 3 ? 2u : 1u) && waits == (route == 3 ? 1u : 0u) && signals == 1);
        NEED(!sub_lock.p && !st_lock.p && slots[cur_slot].fence == (route ? 99u : 0u));
        if (route == 3) NEED(!memcmp(out, gpu[0] + 16, sizeof out) && frees == 1);
    }
    printf("vk prefix CPU: %u checks passed; provider-only, no Vulkan execution\n", checks);
    return 0;
}
