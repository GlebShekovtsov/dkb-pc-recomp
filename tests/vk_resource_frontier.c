// CPU fixture for vk_resource_frontier_test.py. Production functions/types are inserted at the markers below.
// Every driver entry is redirected to an explicitly synthetic adapter. No window, loader, Vulkan device or GPU runs.
#include <windows.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk.h"
#include "tex.h"
#include "nvn_hmap.h"
#include "platform/platform.h"

#define vkGetSemaphoreCounterValue synthetic_counter
#define vkGetFenceStatus synthetic_fence_status
#define vkWaitForFences synthetic_wait_fences
#define vkDestroyImage synthetic_destroy_image
#define vkDestroyImageView synthetic_destroy_view
#define vmaDestroyImage synthetic_vma_destroy_image
#define vmaDestroyBuffer synthetic_vma_destroy_buffer
#define vkBeginCommandBuffer synthetic_begin
#define vkEndCommandBuffer synthetic_end
#define vkAllocateCommandBuffers synthetic_allocate
#define vkQueueSubmit2 synthetic_submit
#define vkDeviceWaitIdle synthetic_device_idle
#define vkDestroyFence synthetic_destroy_fence
#define vkCreateFence synthetic_create_fence
#define vkDestroyCommandPool synthetic_destroy_pool
#define vmaDestroyPool synthetic_vma_destroy_pool
#define vkGetPhysicalDeviceQueueFamilyProperties synthetic_queue_families
#define vkCreateQueryPool synthetic_create_query_pool
#define vkGetQueryPoolResults synthetic_query_results

enum { ALIAS, SPLIT, EMPTY, SKIPPED, SHUTDOWN };
static int scenario, referenced, image_dead, view_dead, old_image_dead, old_view_dead, premature;
static int partials, finals, waits, sub_depth, in_recording, driver_pending;
static uint64_t gpu = 1, next_cb = 0xC000;
static VkCtx context;
static CRITICAL_SECTION command_mutex;
static SRWLOCK latch = SRWLOCK_INIT;
static CONDITION_VARIABLE latch_cv = CONDITION_VARIABLE_INIT;
static int phase;
static void check(int yes, const char *message) {
    if (!yes) { fprintf(stderr, "FAIL: %s\n", message); ExitProcess(20); }
}
_Noreturn void rt_fatal(const char *format, ...) {
    va_list args; va_start(args, format); vfprintf(stderr, format, args); va_end(args);
    fputc('\n', stderr); ExitProcess(21);
}
void rt_log(const char *format, ...) { (void)format; }
unsigned rt_trace;
static void wb_async_forget(const void *e) { (void)e; }
void vk_dlss_image_gone(VkImage image) { (void)image; }
void perf_count(int kind) { (void)kind; }
uint64_t perf_begin(void) { return 0; }
void perf_replay(uint64_t start) { (void)start; }
void perf_replay_totals(uint64_t d, uint64_t t, uint64_t c) { (void)d; (void)t; (void)c; }
uint64_t vk_exec_draws_total(void) { return 0; }
void vk_exec_pool_counts(uint64_t *c, uint64_t *t) { *c = *t = 0; }
// The actual shared split helper also contains the ordered-readback branch. Retirement scenarios must keep using
// the nonwaiting branch; reaching any of these new dependencies is a fixture failure, not a simulated success.
void perf_end(int kind, uint64_t start) { (void)kind; (void)start; check(0, "retirement never times a prefix wait"); }
void perf_gpu_wait(uint64_t start) { (void)start; check(0, "retirement never waits for a prefix"); }
int cfg_bool(const char *key) { check(!strcmp(key, "TOTK_TEX_ALIAS"), "default texture alias setting"); return 1; }
VkCtx *vk_ctx(void) { return &context; }
// Upload batch timing (vk_memory.c, inserted with the memory types): the synthetic device has no queue family with timestamps
// and no query pool, so uploads stay untimed.
static void synthetic_queue_families(VkPhysicalDevice phys, uint32_t *n, VkQueueFamilyProperties *props) {
    (void)phys; (void)props; *n = 0;
}
static VkResult synthetic_create_query_pool(VkDevice device, const VkQueryPoolCreateInfo *info,
                                            const VkAllocationCallbacks *alloc, VkQueryPool *pool) {
    (void)device; (void)info; (void)alloc; *pool = VK_NULL_HANDLE; return VK_ERROR_FEATURE_NOT_PRESENT;
}
static VkResult synthetic_query_results(VkDevice device, VkQueryPool pool, uint32_t first, uint32_t count, size_t size,
                                        void *data, VkDeviceSize stride, VkQueryResultFlags flags) {
    (void)device; (void)pool; (void)first; (void)count; (void)size; (void)data; (void)stride; (void)flags;
    check(0, "an untimed upload batch never reads timestamps");
    return VK_NOT_READY;
}

@TEXTURE_TYPES@
@VIEW_TYPES@
#define cmd_pool memory_cmd_pool
@MEMORY_TYPES@
#undef cmd_pool

typedef struct { VkCommandBuffer cb; uint64_t fence; } CmdSlot;
static CmdSlot *slots;
static int nslots, cur_slot;
static uint64_t issued = 1, done = 1, cur_fence;
static PlatMutex sub_lock, st_lock;
static PlatCond cv_work;
static VkCommandPool cmd_pool = (VkCommandPool)(uintptr_t)0xC100;
static nvn_hmap by_key = {.klen = sizeof(TexKey)}, by_addr = {.klen = 8};
static nvn_hmap view_of = {.klen = sizeof(ViewKey)}, views_by_image = {.klen = 8};
static Tex *rel_head, *rel_tail;
static int nreleased, tex_alias = -1;
static uint64_t n_alias_copied, n_alias_upload, n_alias_uninit;
static tex_t texture_b;
static TexSlot slot_a;

// Empty non-retirement subsystems used by the actual shutdown function. No pools, mirrors or staging work exist.
typedef struct { int unused; } PoolRec;
static struct { int n; struct { PoolRec *buf; } *e; } pool_index;
static PlatMutex pool_lock, shadow_lock, gen_lock;
static VmaPool stg_pool;
static int stg_pool_tried, nm, npend, nwbl, nstouch, nspend, nsblocked, spend_dirty, s_live_n, nretiring, nphys;
static uint64_t s_live_bytes;
static PoolRec **retiring;
static uint32_t pool_gen;
#define GEN_CHUNKS 2
static void *gen_chunk[GEN_CHUNKS];
static void destroy_rec(VkCtx *c, PoolRec *r) { (void)c; (void)r; check(0, "fixture has no pool records"); }

void plat_mutex_lock(PlatMutex *m) {
    AcquireSRWLockExclusive((PSRWLOCK)&m->p);
    if (m == &sub_lock) sub_depth++;
}
void plat_mutex_unlock(PlatMutex *m) {
    if (m == &sub_lock) sub_depth--;
    ReleaseSRWLockExclusive((PSRWLOCK)&m->p);
}
void plat_rmutex_lock(PlatRMutex *m) { (void)m; EnterCriticalSection(&command_mutex); }
void plat_rmutex_unlock(PlatRMutex *m) { (void)m; LeaveCriticalSection(&command_mutex); }
void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) {
    if (once->state != 2) { check(once->state == 0, "once initialized before threads"); fn(arg); once->state = 2; }
}
void plat_cond_broadcast(PlatCond *condition) { (void)condition; }
void vk_queue_lock(void) {}
void vk_queue_unlock(void) {}
static VkResult synthetic_counter(VkDevice d, VkSemaphore s, uint64_t *value) {
    check(d == context.device && s == context.timeline, "synthetic timeline identity"); *value = gpu; return VK_SUCCESS;
}
static VkResult synthetic_fence_status(VkDevice d, VkFence f) {
    (void)d; (void)f; check(0, "no synthetic staging fences exist"); return VK_NOT_READY;
}
static VkResult synthetic_wait_fences(VkDevice d, uint32_t n, const VkFence *f, VkBool32 all, uint64_t timeout) {
    (void)d; (void)n; (void)f; (void)all; (void)timeout; waits++;
    check(0, "retirement must not wait for the unsubmitted current fence"); return VK_TIMEOUT;
}
static void destroyed(uintptr_t handle, int view) {
    if (handle == 0xA001 || handle == 0xA002) {
        if (referenced && gpu < 2) premature++;
        if (view) view_dead++; else image_dead++;
    } else if (handle == 0xD001) old_image_dead++;
    else if (handle == 0xD002) old_view_dead++;
    else check(0, "known synthetic resource destroyed");
}
static void synthetic_destroy_image(VkDevice d, VkImage i, const VkAllocationCallbacks *a) {
    (void)d; (void)a; destroyed((uintptr_t)i, 0);
}
static void synthetic_destroy_view(VkDevice d, VkImageView v, const VkAllocationCallbacks *a) {
    (void)d; (void)a; destroyed((uintptr_t)v, 1);
}
static void synthetic_vma_destroy_image(VmaAllocator a, VkImage i, VmaAllocation m) {
    (void)a; (void)m; destroyed((uintptr_t)i, 0);
}
static void synthetic_vma_destroy_buffer(VmaAllocator a, VkBuffer b, VmaAllocation m) {
    (void)a; (void)b; (void)m; check(0, "fixture retires images and views only");
}
static void synthetic_destroy_fence(VkDevice d, VkFence f, const VkAllocationCallbacks *a) { (void)d; (void)f; (void)a; check(0, "no staging fences"); }
static VkResult synthetic_create_fence(VkDevice d, const VkFenceCreateInfo *i, const VkAllocationCallbacks *a, VkFence *f) {
    (void)d; (void)i; (void)a; (void)f; check(0, "retirement never creates a prefix fence"); return VK_ERROR_UNKNOWN;
}
static void synthetic_destroy_pool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks *a) { (void)d; (void)p; (void)a; check(0, "no upload command pool"); }
static void synthetic_vma_destroy_pool(VmaAllocator a, VmaPool p) { (void)a; (void)p; check(0, "no staging pool"); }
static VkResult synthetic_device_idle(VkDevice d) {
    check(d == context.device && !driver_pending && !in_recording, "synthetic device idle without a future timeline signal"); return VK_SUCCESS;
}
void vk_texture_shutdown(void) { check(scenario == SHUTDOWN, "shutdown uses an empty synthetic texture cache"); }

#define cmd_pool memory_cmd_pool
@MEMORY_FUNCTIONS@
#undef cmd_pool

static void own_remove(Tex *e) { check(!e->listed, "fixture has no GPU writeback-list entry"); }
static void tex_gen_bump(uint64_t addr) { (void)addr; }
void vk_exec_bc_gone(VkImage image) { (void)image; }
@VIEW_FUNCTION@
static Tex *create_tex(const tex_t *t, const TexKey *key, int deferred) {
    check(deferred == 1, "default alias handling enabled");
    Tex *e = calloc(1, sizeof *e); check(e != NULL, "synthetic image B metadata");
    e->t = *t; e->key = *key; e->pub.image = (VkImage)(uintptr_t)0xB001; e->pub.addr = key->addr; return e;
}
static int alias_compatible(const Tex *a, const Tex *b) { (void)a; (void)b; check(0, "no live alias source"); return 0; }
static int alias_has_gpu_content(const Tex *a) { (void)a; check(0, "no live alias source"); return 0; }
static int alias_copy(Tex *a, const Tex *b) { (void)a; (void)b; check(0, "no live alias source"); return 0; }
static void init_image(Tex *image, VkCommandBuffer cb) { (void)image; (void)cb; }
VkCommandBuffer vk_upload_begin(void) { return (VkCommandBuffer)(uintptr_t)0xC200; }
void vk_upload_end(void) {}
int vk_mem_changed(uint64_t a, uint64_t n, uint32_t before, uint32_t *now) {
    (void)a; (void)n; (void)before; (void)now; check(0, "not a mirrored image"); return 0;
}
@TEXTURE_FUNCTIONS@

void vk_sync_start(void) { check(context.timeline != VK_NULL_HANDLE, "synthetic context initialized"); }
void vk_mem_dirty_submit(void) {}
void vk_mem_shadow_sync(void) {}
static void ts_begin(VkCommandBuffer cb) { (void)cb; }
static void ts_end(VkCommandBuffer cb, uint64_t fence) { (void)cb; check(fence == 2, "split keeps its final fence"); }
static void ts_drop(void) {}
void rec_begin(VkCommandBuffer cb) { (void)cb; }
int rec_end_cb(VkCommandBuffer cb, int crumb, uint64_t fence) { (void)cb; (void)crumb; (void)fence; return 0; }
void rec_stats_log(void) {}
static VkCommandBuffer prologue(VkCtx *c, uint64_t fence) { (void)c; check(fence == 2, "prologue fence"); return VK_NULL_HANDLE; }
void vk_crumb(VkCommandBuffer cb, int kind, uint64_t value) { (void)cb; (void)kind; check(value == 2, "command crumb fence"); }
void vk_device_fault_report(VkResult result) { fprintf(stderr, "synthetic submit rejected an invalid resource (%d)\n", result); }
uint64_t vk_upload_flush(int wait) {
    check(!wait || !in_recording, "recording never waits for an unsubmitted frontier"); vk_mem_collect(); return 0;
}
static VkResult synthetic_begin(VkCommandBuffer cb, const VkCommandBufferBeginInfo *info) {
    check(sub_depth == 1 && cb && info->flags == VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, "real submit/split entry holds submission lock");
    in_recording = 1; return VK_SUCCESS;
}
static VkResult synthetic_end(VkCommandBuffer cb) { check(cb && in_recording, "synthetic command recording ends"); in_recording = 0; return VK_SUCCESS; }
static VkResult synthetic_allocate(VkDevice d, const VkCommandBufferAllocateInfo *info, VkCommandBuffer *cb) {
    check(d == context.device && info->commandBufferCount == 1, "synthetic command allocation"); *cb = (VkCommandBuffer)(uintptr_t)++next_cb; return VK_SUCCESS;
}
static VkResult synthetic_submit(VkQueue q, uint32_t count, const VkSubmitInfo2 *info, VkFence fence) {
    check(q == context.gfx_queue && count == 1 && !fence && sub_depth == 1 && issued == 1 && done == 1, "actual issued/done unchanged until submit returns");
    if (referenced && (image_dead || view_dead)) return VK_ERROR_DEVICE_LOST;
    if (info->signalSemaphoreInfoCount) {
        check(info->signalSemaphoreInfoCount == 1 && info->pSignalSemaphoreInfos[0].value == 2, "final timeline signal"); finals++;
        check(info->commandBufferInfoCount == ((scenario == EMPTY || scenario == SKIPPED) ? 0u : 1u), "empty/failed recording still signals its fence");
    } else { check(scenario == SPLIT && info->commandBufferInfoCount == 1, "partial submit has no final signal"); partials++; }
    driver_pending = 1; return VK_SUCCESS;
}
int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks);
@SYNC_FUNCTIONS@

static void await_phase(int wanted) {
    while (phase < wanted) if (!SleepConditionVariableSRW(&latch_cv, &latch, 5000, 0)) ExitProcess(22);
}
int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks) {
    check(submitted_ticks == 123, "actual submit forwards immutable entry tick to recorder");
    (void)handles; check(cb && fence == 2 && n == (scenario == EMPTY ? 0 : 1), "production recorder entry");
    if (scenario != EMPTY) {
        view_t view; int has_view; Tex *a = slot_resolve(&slot_a, &view, &has_view);
        check(a && a->released && (uintptr_t)a->pub.image == 0xA001, "actual finalized-slot fallback");
        referenced = scenario != SKIPPED;
        Tex *b = entry_of(&texture_b); check(b && (uintptr_t)b->pub.image == 0xB001, "actual alias eviction");
    }
    AcquireSRWLockExclusive(&latch); phase = 1; WakeAllConditionVariable(&latch_cv); await_phase(2); ReleaseSRWLockExclusive(&latch);
    if (scenario == SPLIT) {
        (void)vk_sync_split(); vk_mem_collect();
        check(!image_dead && !view_dead && gpu == 1 && issued == 1, "partial submit cannot retire current resources");
    }
    return scenario == EMPTY || scenario == SKIPPED ? 0 : 1;
}
static DWORD WINAPI submit_thread(void *arg) {
    (void)arg; uint64_t handle = 1;
    check(vk_sync_submit(&handle, scenario == EMPTY ? 0 : 1, 123) == 2, "actual submit returns current fence"); return 0;
}
static void seed_texture(void) {
    tex_t a = {0}; a.mem = (void *)(uintptr_t)0x100000; a.d.format = 1; a.d.width = a.d.height = 64; a.d.depth = a.d.levels = 1;
    Tex *entry = calloc(1, sizeof *entry); check(entry != NULL, "retained image metadata");
    entry->t = a; make_key(&a, &entry->key); entry->pub.image = (VkImage)(uintptr_t)0xA001; entry->pub.addr = entry->key.addr; entry->released = 1;
    if (scenario == SPLIT) entry->alloc = (VmaAllocation)(uintptr_t)0xA003;   // cover both pool-bound and owned images
    nreleased = 1; rel_head = rel_tail = entry; nvn_hm_put(&by_key, &entry->key, entry); nvn_hm_put(&by_addr, &entry->key.addr, entry);
    slot_a.t = a; slot_a.used = 1; texture_b = a; texture_b.d.format = 2;
    ViewNode *node = calloc(1, sizeof *node); check(node != NULL, "retained view metadata");
    node->k.image = entry->pub.image; node->view = (VkImageView)(uintptr_t)0xA002;
    nvn_hm_put(&view_of, &node->k, (void *)node->view); nvn_hm_put(&views_by_image, &entry->pub.image, node);
}
int main(int argc, char **argv) {
    check(argc == 2, "scenario argument"); const char *names[] = {"alias", "split", "empty", "skipped", "shutdown"};
    for (; scenario < 5 && strcmp(argv[1], names[scenario]); scenario++) {}
    check(scenario < 5, "known scenario"); InitializeCriticalSection(&command_mutex);
    context.device = (VkDevice)(uintptr_t)0xE001; context.timeline = (VkSemaphore)(uintptr_t)0xE002; context.gfx_queue = (VkQueue)(uintptr_t)0xE003;
    plat_once(&cmd_once, cmd_init, NULL); inflight = calloc(1, sizeof *inflight); cap_inflight = 1;
    @FRONTIER@(1); vk_defer_image((VkImage)(uintptr_t)0xD001, NULL); vk_defer_view((VkImageView)(uintptr_t)0xD002);
    if (scenario == SHUTDOWN) {
        @FRONTIER@(2); vk_defer_image((VkImage)(uintptr_t)0xA001, NULL); vk_defer_view((VkImageView)(uintptr_t)0xA002);
        vk_mem_collect(); check(!image_dead && !view_dead && old_image_dead == 1 && old_view_dead == 1, "future frontier retains only newly retired objects");
        vk_mem_shutdown(); check(image_dead == 1 && view_dead == 1 && issued == 1 && gpu == 1 && waits == 0, "idle shutdown clears an unsubmitted frontier without waiting for it");
        check(!stg_pool_tried && !nm && !nstouch && !s_live_n && !nphys, "empty shutdown bookkeeping cleared");
    } else {
        if (scenario != EMPTY) seed_texture();
        HANDLE thread = CreateThread(NULL, 0, submit_thread, NULL, 0, NULL); check(thread != NULL, "CPU submit thread");
        AcquireSRWLockExclusive(&latch); await_phase(1); ReleaseSRWLockExclusive(&latch);
        vk_mem_collect();
        check(old_image_dead == 1 && old_view_dead == 1, "older completed resources collect during current recording");
        check(!image_dead && !view_dead, "current unsubmitted image/view must survive collection");
        AcquireSRWLockExclusive(&latch); phase = 2; WakeAllConditionVariable(&latch_cv); ReleaseSRWLockExclusive(&latch);
        check(WaitForSingleObject(thread, 5000) == WAIT_OBJECT_0, "bounded submission completion"); CloseHandle(thread);
        check(finals == 1 && partials == (scenario == SPLIT) && issued == 2 && done == 1, "real submission bookkeeping and split count");
        vk_mem_collect(); check(!image_dead && !view_dead, "submitted resources wait for their actual timeline completion");
        gpu = 2; driver_pending = 0; vk_mem_collect(); vk_mem_collect();
        check(image_dead == (scenario != EMPTY) && view_dead == (scenario != EMPTY) && ndead == 0 && !premature && !waits, "exactly-once retirement after the current fence");
    }
    check(cap_inflight == 1 && (!rel_head || scenario == SHUTDOWN) && n_alias_copied == 0 && n_alias_upload == 0 && n_alias_uninit == 0, "isolated metadata adapters");
    printf("PASS %s: actual submit/split/retirement code; synthetic driver; no Vulkan library linked\n", names[scenario]); return 0;
}
