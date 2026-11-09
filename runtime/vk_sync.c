// Submission, fences and the completion thread of the Vulkan back end.
// One timeline semaphore (VkCtx.timeline) orders everything: each submission records its command handles into a
// primary command buffer (vk_exec_record) and signals the next timeline value, which is the NVN fence value
// the front end hands to syncs. The completion thread waits on the timeline and, in submit order, runs the CPU side
// effects queued for each value (event writes, counter reports, sync state), then frees the memory and texture objects whose users
// completed. A value counts as done only after its side effects ran, so a guest woken by fence_wait sees them.
//
// The submission shape (one timeline value per submission, signalled at ALL_COMMANDS, vkWaitSemaphores for waits, a
// failed wait treated as a lost device) follows LAS Recomp runtime/nvn/vulkan/exec.c:39-110, 355-397
// (https://github.com/peiikai/LASRecomp @ da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
// Side effects written after GPU completion, as MacNativeGames does: games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c:7-9
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"

#define WEAK __attribute__((weak))

typedef struct { uint64_t fence; void (*fn)(void *); void *arg; } Pending;
typedef struct { VkCommandBuffer cb; uint64_t fence; } CmdSlot;   // reusable once the timeline reached fence

static PlatMutex start_lock = PLAT_MUTEX_INIT;
static int started;
static PlatThread *completion_thread;

// Submissions: one at a time, so timeline values are signalled in the order they are handed out.
static PlatMutex sub_lock = PLAT_MUTEX_INIT;
static VkCommandPool cmd_pool;
static CmdSlot *slots;
static int nslots;
static int cur_slot;          // the slot vk_sync_submit records into now (an index: take_slot may move slots)
static uint64_t cur_fence;    // the value of the submission being recorded

// Shared with the completion thread (st_lock).
static PlatMutex st_lock = PLAT_MUTEX_INIT;
static PlatCond cv_work = PLAT_COND_INIT;   // a submission, a callback or stop
static PlatCond cv_done = PLAT_COND_INIT;   // done advanced
static uint64_t issued, done;     // newest submitted value; newest value whose side effects ran
static uint64_t reserved;         // asynchronous replay: newest value handed out before its submission (>= issued)
static int stop;
static Pending *queue;            // side effects in the order they were added
static int qhead, qn, qcap;

// ---- GPU time (TOTK_TRACE=perf): each submission part (vk_sync_split) writes a timestamp when it starts and when it
// ends, into a ring of query pairs. The completion thread reads a pair once the part's value is reached and adds the
// part's interval to the busy time, counting overlap with the previous part once. Off (one log line) when the queue
// family has no timestamp bits or the period is 0. Upload batches (vk_memory.c) are not timed.
#define TS_PAIRS 1024
static VkQueryPool ts_pool;
static int ts_state;                       // 0 not set up, 1 on, -1 off
static double ts_period;                   // ns per tick
static uint64_t ts_mask;                   // valid bits of a timestamp
static uint32_t ts_next;                   // submitting thread (sub_lock): next pair
static volatile uint8_t ts_busy[TS_PAIRS]; // written until the completion thread read the pair
static int ts_cur = -1;                    // the pair of the part being recorded, -1 untimed
static uint64_t ts_last_end;               // completion thread: end of the latest part read (ticks)
static volatile int64_t ts_busy_ns, ts_parts, ts_missed;

// On the first submission that wants timing (perf_gpu_timing: TOTK_TRACE=perf or the full performance overlay).
static void ts_setup(VkCtx *c) {
    ts_state = -1;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    if (!qp) return;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    free(qp);
    float period = c->props.limits.timestampPeriod;
    if (!bits || period <= 0) {
        rt_log("vk: no GPU timestamps on this queue (valid bits %u, period %.3f ns): the perf line shows no GPU time", bits, period);
        return;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * TS_PAIRS};
    VkResult r = vkCreateQueryPool(c->device, &qi, NULL, &ts_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: timestamp query pool failed (%d): the perf line shows no GPU time", (int)r);
        return;
    }
    ts_period = period;
    ts_mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    ts_state = 1;
}

// What the replay part being timed draws and dispatches (vk_sync_note), kept with its pair: a part that runs on the GPU
// for at least TOTK_PERF_SLOW_GPU_MS (default 100) gets a line naming its work, at most 200 lines. A respawn in the
// Ingot Isle Cave-In section (#10) spent about 16.8 s of GPU time per frame inside the replay's parts.
#define NOTE_PAIRS 6
#define NOTE_ARGS 65536   // indirect argument addresses and storage writes kept per part
// size 0: an indirect draw's arguments at addr, drawn by stages a/b; else a storage write of [addr, addr+size) by
// stage a, in recording order with the draws.
typedef struct { uint64_t addr, a, b, size; } NoteArg;
typedef struct {
    uint32_t draws, draws_ind, dispatches, dispatches_ind, pairs;
    uint64_t max_work, max_a, max_b;               // the largest direct draw (vertices x instances) or dispatch (groups)
    uint64_t a[NOTE_PAIRS], b[NOTE_PAIRS];         // the first distinct stage names: vertex/fragment, or compute/0
    uint64_t last_a, last_b;                       // the stages of the last noted draw or dispatch
    NoteArg *args;                                 // indirect draws whose arguments the host can read (malloc'd)
    uint32_t nargs, cap;
} PartNote;
static PartNote note;   // the part being recorded (sub_lock, like ts_cur)

int vk_sync_noting(void) { return ts_cur >= 0; }

static void note_push(NoteArg e) {
    if (ts_cur < 0 || note.nargs >= NOTE_ARGS) return;
    if (note.nargs == note.cap) {
        uint32_t cap = note.cap ? 2 * note.cap : 256;
        NoteArg *p = realloc(note.args, cap * sizeof *p);
        if (!p) return;
        note.args = p, note.cap = cap;
    }
    note.args[note.nargs++] = e;
}

void vk_sync_note_args(uint64_t addr) { note_push((NoteArg){addr, note.last_a, note.last_b, 0}); }

void vk_sync_note_write(uint64_t addr, uint64_t size, uint64_t stage) {
    if (size) note_push((NoteArg){addr, stage, 0, size});
}

void vk_sync_note(int kind, uint64_t a, uint64_t b, uint64_t work) {
    if (ts_cur < 0) return;
    note.last_a = a, note.last_b = b;
    if (kind == 0) note.draws++;
    else if (kind == 1) note.draws_ind++;
    else if (kind == 2) note.dispatches++;
    else note.dispatches_ind++;
    if (work > note.max_work) note.max_work = work, note.max_a = a, note.max_b = b;
    for (uint32_t i = 0; i < note.pairs; i++)
        if (note.a[i] == a && note.b[i] == b) return;
    if (note.pairs < NOTE_PAIRS) note.a[note.pairs] = a, note.b[note.pairs] = b, note.pairs++;
}

typedef struct { uint32_t pair, noted; uint64_t fence; PartNote note; } TsRead;

static void slow_part(const TsRead *t, double ms) {
    static double slow_ms = -1;
    static int lines;
    if (slow_ms < 0) {
        const char *e = cfg_get("TOTK_PERF_SLOW_GPU_MS");
        slow_ms = e && atof(e) > 0 ? atof(e) : 100;
    }
    if (!t->noted || ms < slow_ms || lines >= 200) return;
    lines++;
    const PartNote *n = &t->note;
    char st[NOTE_PAIRS * 36 + 8] = "", *p = st;
    for (uint32_t i = 0; i < n->pairs; i++)
        p += snprintf(p, (size_t)(st + sizeof st - p), " %016llx/%016llx", (unsigned long long)n->a[i], (unsigned long long)n->b[i]);
    rt_log("perf: a GPU part of submission %llu ran %.1f ms: %u draws (%u indirect), %u dispatches (%u indirect); largest "
           "direct one %llu (%016llx/%016llx); stages%s%s", (unsigned long long)t->fence, ms, n->draws + n->draws_ind,
           n->draws_ind, n->dispatches + n->dispatches_ind, n->dispatches_ind, (unsigned long long)n->max_work,
           (unsigned long long)n->max_a, (unsigned long long)n->max_b, n->pairs ? st : " none",
           n->pairs == NOTE_PAIRS ? " ..." : "");
    // The indirect draws' arguments as they are now, after the GPU used them: the game may already have written the
    // next frame's, and a pool freed meanwhile is skipped (plat_mem_readable).
    uint64_t total = 0, max = 0;
    uint32_t read = 0, best = 0, w[5] = {0}, bw[5] = {0};
    for (uint32_t i = 0; i < n->nargs; i++) {
        if (n->args[i].size || !plat_mem_readable(n->args[i].addr, sizeof w)) continue;
        memcpy(w, (const void *)(uintptr_t)n->args[i].addr, sizeof w);
        uint64_t work = (uint64_t)w[0] * w[1];   // indices x instances
        read++, total += work;
        if (work >= max) max = work, best = i, memcpy(bw, w, sizeof w);
    }
    if (n->draws_ind)
        rt_log("perf:   its indirect draws: %u of %u arguments read back, %llu indices x instances in all; the largest "
               "%llu (%u indices from %u, vertex offset %d, %u instances from %u; %016llx/%016llx)", read, n->draws_ind,
               (unsigned long long)total, (unsigned long long)max, bw[0], bw[2], (int32_t)bw[3], bw[1], bw[4],
               (unsigned long long)(read ? n->args[best].a : 0), (unsigned long long)(read ? n->args[best].b : 0));
    if (!read) return;
    // Who stored over those arguments earlier in this part: the last shader write that covers them before the draw.
    // None means the CPU or an earlier part wrote them, or a shader whose storage the host bound to a stand-in.
    uint64_t at = n->args[best].addr, by = 0;
    uint32_t writes = 0;
    for (uint32_t i = 0; i < best; i++)
        if (n->args[i].size && at + sizeof w > n->args[i].addr && at < n->args[i].addr + n->args[i].size)
            writes++, by = n->args[i].a;
    if (writes)
        rt_log("perf:   the largest one's arguments at %#llx were written in this part by %u shader stores, the last by %016llx",
               (unsigned long long)at, writes, (unsigned long long)by);
    else
        rt_log("perf:   the largest one's arguments at %#llx were not written by a shader earlier in this part",
               (unsigned long long)at);
}

static void ts_read_cb(void *arg) {   // completion thread, in submission order
    TsRead *t = arg;
    VkCtx *c = vk_ctx();
    uint64_t v[2];
    if (vkGetQueryPoolResults(c->device, ts_pool, 2 * t->pair, 2, sizeof v, v, sizeof v[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        uint64_t s = v[0] & ts_mask, e = v[1] & ts_mask;
        if (e >= s) {
            slow_part(t, (double)(e - s) * ts_period / 1e6);
            uint64_t from = s > ts_last_end ? s : ts_last_end;
            if (e > from) {
                int64_t ns = (int64_t)((double)(e - from) * ts_period);
                plat_add64(&ts_busy_ns, ns);
                perf_gpu_busy((uint64_t)ns);
            }
            if (e > ts_last_end) ts_last_end = e;
            plat_inc64(&ts_parts);
        } else plat_inc64(&ts_missed);   // the counter wrapped
    } else plat_inc64(&ts_missed);
    __atomic_store_n(&ts_busy[t->pair], 0, __ATOMIC_RELEASE);
    free(t->note.args);
    free(t);
}

// At the start of a command buffer (sub_lock held): the pair it times, -1 untimed.
static int ts_open(VkCommandBuffer cb) {
    if (ts_state != 1 || !perf_gpu_timing()) {
        if (ts_state || !perf_gpu_timing()) return -1;
        ts_setup(vk_ctx());   // the overlay turned full: timestamps from now on
        if (ts_state != 1) return -1;
    }
    uint32_t p = ts_next % TS_PAIRS;
    if (__atomic_load_n(&ts_busy[p], __ATOMIC_ACQUIRE)) { plat_inc64(&ts_missed); return -1; }
    ts_next++;
    ts_busy[p] = 1;
    vkCmdResetQueryPool(cb, ts_pool, 2 * p, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts_pool, 2 * p);
    return (int)p;
}

// At its end, before vkEndCommandBuffer; fence: the value whose completion covers it. The note's argument list moves
// to the reader.
static void ts_close(VkCommandBuffer cb, int pair, uint64_t fence, PartNote *n) {
    if (pair < 0) {
        if (n) free(n->args), n->args = NULL;
        return;
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool, 2 * (uint32_t)pair + 1);
    TsRead *t = calloc(1, sizeof *t);
    if (!t) rt_fatal("vk: out of memory");
    t->pair = (uint32_t)pair, t->fence = fence;
    if (n) t->note = *n, t->noted = 1, n->args = NULL;
    vk_completion_add(fence, ts_read_cb, t);
}

// The pair of the part being recorded.
static void ts_begin(VkCommandBuffer cb) {
    ts_cur = ts_open(cb);
    memset(&note, 0, sizeof note);
}

static void ts_end(VkCommandBuffer cb, uint64_t fence) {
    ts_close(cb, ts_cur, fence, &note);
    ts_cur = -1;
}

// An empty submission records no command buffer: its pair is never written.
static void ts_drop(void) {
    free(note.args), note.args = NULL;
    if (ts_cur < 0) return;
    __atomic_store_n(&ts_busy[ts_cur], 0, __ATOMIC_RELEASE);
    ts_cur = -1;
}

static void gpu_stats(PerfGpu *g) {
    g->have_gpu = ts_state == 1;
    g->gpu_ns = (uint64_t)plat_xchg64(&ts_busy_ns, 0);
    g->gpu_parts = (uint64_t)plat_xchg64(&ts_parts, 0);
    g->gpu_missed = (uint64_t)plat_xchg64(&ts_missed, 0);
    g->have_vram = vk_mem_vram(&g->vram_used, &g->vram_budget, &g->vram_exact);
    vk_upload_perf(&g->up_gpu_ns, &g->up_timed, &g->up_batches, &g->up_wait_ns, &g->up_bytes);
}

// Weak stand-in for vk_exec.c's replay: only empty recordings ({OP_BEGIN, OP_END}) can be submitted.
WEAK int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks) {
    (void)cb, (void)fence, (void)submitted_ticks;
    for (int i = 0; i < n; i++) {
        const op_hdr *h = handles[i] ? nvn_op_first(handles[i]) : NULL;
        if (!h || h->op != OP_BEGIN || h->size < sizeof *h || nvn_op_next(h)->op != OP_END)
            rt_fatal("vk: op replay is not linked");
    }
    return 0;
}

static void completion_main(void *arg) {
    VkCtx *c = arg;
    plat_mutex_lock(&st_lock);
    for (;;) {
        while (!stop && done == issued && !(qn && queue[qhead].fence <= done))
            plat_cond_wait(&cv_work, &st_lock);
        if (stop) break;
        uint64_t reached = done;
        if (issued > done) {
            uint64_t want = done + 1;
            plat_mutex_unlock(&st_lock);
            VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                                      .pSemaphores = &c->timeline, .pValues = &want};
            VkResult r = vkWaitSemaphores(c->device, &wi, 100000000ull);   // 100 ms: stop is checked between waits
            if (r != VK_SUCCESS && r != VK_TIMEOUT) {
                vk_device_fault_report(r);
                rt_fatal("vk: vkWaitSemaphores failed (%d; device lost?)", (int)r);
            }
            vkGetSemaphoreCounterValue(c->device, c->timeline, &reached);
            plat_mutex_lock(&st_lock);
            if (reached > issued) reached = issued;
        }
        while (qn && queue[qhead].fence <= reached) {
            Pending p = queue[qhead++];
            qn--;
            plat_mutex_unlock(&st_lock);
            p.fn(p.arg);
            plat_mutex_lock(&st_lock);
        }
        if (reached > done) {
            plat_mutex_unlock(&st_lock);
            vk_mem_collect();
            plat_mutex_lock(&st_lock);
            __atomic_store_n(&done, reached, __ATOMIC_RELEASE);
            plat_cond_broadcast(&cv_done);
        }
    }
    plat_mutex_unlock(&st_lock);
    return;
}

void vk_sync_start(void) {
    if (__atomic_load_n(&started, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&start_lock);
    if (!started) {
        if (!vk_init()) rt_fatal("vk: no usable Vulkan device (see the vk: lines above; TOTK_GPU=null runs without a GPU)");
        VkCtx *c = vk_ctx();
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &cmd_pool) != VK_SUCCESS) rt_fatal("vk: submission command pool");
        if (!ts_state && perf_gpu_timing()) ts_setup(c);
        perf_gpu_stats = gpu_stats;
        uint64_t now = 0;
        vkGetSemaphoreCounterValue(c->device, c->timeline, &now);
        if (issued > now) {   // a new device after vk_shutdown: values keep increasing for the front end
            VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = c->timeline,
                                        .value = issued};
            if (vkSignalSemaphore(c->device, &si) != VK_SUCCESS) rt_fatal("vk: timeline restart at %llu", (unsigned long long)issued);
            now = issued;
        }
        plat_mutex_lock(&st_lock);
        issued = done = now;
        if (reserved < now) reserved = now;
        stop = 0;
        plat_mutex_unlock(&st_lock);
        vk_mem_protect_until(now);
        completion_thread = plat_thread_new(completion_main, c, "vk completion");
        if (!completion_thread) rt_fatal("vk: cannot start the completion thread");
        __atomic_store_n(&started, 1, __ATOMIC_RELEASE);
        vk_astc_mode();   // decides TOTK_ASTC and builds the ASTC decoder now, not at the first ASTC upload
    }
    plat_mutex_unlock(&start_lock);
}

// The index of a command buffer the GPU no longer uses, claimed for value `fence`. Called with sub_lock held; may move
// slots[].
static int claim_slot(VkCtx *c, uint64_t fence) {
    uint64_t gpu = 0;
    vkGetSemaphoreCounterValue(c->device, c->timeline, &gpu);
    for (int i = 0; i < nslots; i++)
        if (slots[i].fence <= gpu) {
            slots[i].fence = fence;
            return i;
        }
    CmdSlot *n = realloc(slots, sizeof *n * (size_t)(nslots + 1));
    if (!n) rt_fatal("vk: out of memory (command buffers)");
    slots = n;
    CmdSlot *s = &slots[nslots];
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(c->device, &ai, &s->cb) != VK_SUCCESS) rt_fatal("vk: command buffer allocation");
    s->fence = fence;
    return nslots++;
}

static CmdSlot *take_slot(VkCtx *c, uint64_t fence) {
    cur_slot = claim_slot(c, fence);
    return &slots[cur_slot];
}

// The copies the replay's first accesses to VRAM mirrors queued (vk_memory.c), in a command buffer of their own that is
// submitted right before the part that needs them and timed like a part; VK_NULL_HANDLE when there are none. After the
// part was recorded, sub_lock held; may move slots[].
// First brings the shadow storage up to date with the CPU's writes, which needs no command buffer.
static VkCommandBuffer prologue(VkCtx *c, uint64_t fence) {
    vk_mem_shadow_sync();   // shadow storage: the CPU's writes go in before the part is submitted
    if (!vk_mem_mirror_pending()) return VK_NULL_HANDLE;
    int i = claim_slot(c, fence);   // before reading slots: the claim may move it
    VkCommandBuffer cb = slots[i].cb;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    int pair = ts_open(cb);
    vk_mem_mirror_prologue(cb);
    ts_close(cb, pair, fence, NULL);
    vkEndCommandBuffer(cb);
    return cb;
}

static VkCommandBuffer split_impl(int wait) {
    VkCtx *c = vk_ctx();
    VkFence prefix = VK_NULL_HANDLE;
    if (wait) {
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(c->device, &fi, NULL, &prefix) != VK_SUCCESS) rt_fatal("vk: prefix fence allocation");
    }
    VkCommandBuffer cb = slots[cur_slot].cb;
    ts_end(cb, cur_fence);
    if (!rec_end_cb(cb, 0, 0)) vkEndCommandBuffer(cb);   // every command of the part is in cb, ended by its recorder
    vk_upload_flush(0);   // the uploads this part asked for precede it in queue order
    VkCommandBuffer pro = prologue(c, cur_fence);
    VkCommandBufferSubmitInfo cbi[2] = {{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = pro},
                                        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb}};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = pro ? 2 : 1,
                        .pCommandBufferInfos = pro ? cbi : cbi + 1};
    vk_queue_lock();
    VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, prefix);
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueSubmit2 of a part of value %llu failed (%d)", (unsigned long long)cur_fence, (int)r);
    }
    if (wait) {
        // The full submission's timeline value is not signalled yet. Wait only for this prefix, without running its
        // final completion callbacks or releasing the resources/command slots protected by cur_fence.
        uint64_t t0 = perf_begin();
        r = vkWaitForFences(c->device, 1, &prefix, VK_TRUE, UINT64_MAX);
        perf_end(PERF_FENCE, t0);
        perf_gpu_wait(t0);
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: wait for a prefix of value %llu failed (%d)", (unsigned long long)cur_fence, (int)r);
        }
        vkDestroyFence(c->device, prefix, NULL);
    }
    CmdSlot *s = take_slot(c, cur_fence);   // the submitted part keeps its slot until the value is reached
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(s->cb, &bi);
    vk_crumb(s->cb, VK_CRUMB_SUBMIT, cur_fence);
    ts_begin(s->cb);
    rec_begin(s->cb);
    return s->cb;
}

VkCommandBuffer vk_sync_split(void) { return split_impl(0); }

VkCommandBuffer vk_sync_split_wait(void) { return split_impl(1); }

// Asynchronous replay (vk_backend.c): the value the next queued submission or present will signal. Values are handed
// out in queue order and the replay thread submits them in that order, so each vk_sync_submit gets exactly its value.
uint64_t vk_sync_reserve(void) {
    vk_sync_start();
    plat_mutex_lock(&st_lock);
    if (reserved < issued) reserved = issued;
    uint64_t v = ++reserved;
    plat_mutex_unlock(&st_lock);
    return v;
}

uint64_t vk_sync_submit_record(VkSyncRecordFn record, void *context, uint64_t submitted_ticks) {
    if (!record) rt_fatal("vk: submission without a recorder");
    vk_sync_start();
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&sub_lock);
    vk_mem_dirty_submit();   // TOTK_TRACE=perf: what a mirror would upload before this submit
    uint64_t fence = __atomic_load_n(&issued, __ATOMIC_ACQUIRE) + 1;
    cur_fence = fence;
    // A texture alias can retire an image or view used earlier in this recording. Collection must wait for this
    // submission, including every split part, before the first command can reference those objects.
    vk_mem_protect_until(fence);
    CmdSlot *s = take_slot(c, fence);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    uint64_t t0 = perf_begin();
    vkBeginCommandBuffer(s->cb, &bi);
    vk_crumb(s->cb, VK_CRUMB_SUBMIT, fence);
    ts_begin(s->cb);
    rec_begin(s->cb);
    int recorded = record(s->cb, fence, submitted_ticks, context);
    s = &slots[cur_slot];   // the last part (vk_sync_split)
    if (recorded) ts_end(s->cb, fence);
    else ts_drop();
    // The recorder is done with the part: the end-of-submission crumb, then the end, by the recorder when it records.
    if (!rec_end_cb(s->cb, VK_CRUMB_SUBMIT_END, fence)) vkEndCommandBuffer(s->cb);
    perf_replay(t0);
    if (rt_trace & RT_TRACE_PERF) {
        uint64_t created, taken;
        vk_exec_pool_counts(&created, &taken);
        perf_replay_totals(vk_exec_draws_total(), taken, created);
    }
    // After the recording: the uploads it asked for (and every earlier one) precede it in queue order.
    vk_upload_flush(0);
    VkCommandBuffer pro = prologue(c, fence);
    s = &slots[cur_slot];
    VkCommandBufferSubmitInfo cbi[2] = {{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = pro},
                                        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->cb}};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = c->timeline,
                                 .value = fence, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                        .commandBufferInfoCount = (pro ? 1u : 0u) + (recorded ? 1u : 0u),
                        .pCommandBufferInfos = pro ? cbi : cbi + 1, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sig};
    vk_queue_lock();
    VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, VK_NULL_HANDLE);
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueSubmit2 of value %llu failed (%d)", (unsigned long long)fence, (int)r);
    }
    s->fence = recorded ? fence : 0;
    plat_mutex_lock(&st_lock);
    __atomic_store_n(&issued, fence, __ATOMIC_RELEASE);
    plat_cond_broadcast(&cv_work);
    plat_mutex_unlock(&st_lock);
    static unsigned subs;
    if (++subs % 4096 == 0) rec_stats_log();
    plat_mutex_unlock(&sub_lock);
    return fence;
}

typedef struct { const uint64_t *handles; int n; } ReplayHandles;

static int record_handles(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks, void *context) {
    const ReplayHandles *r = context;
    return vk_exec_record(cb, r->handles, r->n, fence, submitted_ticks);
}

uint64_t vk_sync_submit(const uint64_t *handles, int n, uint64_t submitted_ticks) {
    ReplayHandles r = {handles, n};
    return vk_sync_submit_record(record_handles, &r, submitted_ticks);
}

void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg) {
    plat_mutex_lock(&st_lock);
    if (qhead + qn == qcap) {
        if (qhead) {
            memmove(queue, queue + qhead, sizeof *queue * (size_t)qn);
            qhead = 0;
        } else {
            int cap = qcap ? qcap * 2 : 256;
            Pending *q = realloc(queue, sizeof *q * (size_t)cap);
            if (!q) rt_fatal("vk: out of memory (completion queue)");
            queue = q, qcap = cap;
        }
    }
    queue[qhead + qn++] = (Pending){fence, fn, arg};
    // The completion thread sleeps only while nothing is issued beyond done: a callback for a later value (the replay's,
    // for the submission it records: counter reports, freed resources) becomes due through the submit, which wakes it.
    if (fence <= done) plat_cond_broadcast(&cv_work);
    plat_mutex_unlock(&st_lock);
}

static int fence_wait_impl(uint64_t value, uint64_t timeout_ns);

int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns) {
    uint64_t p0 = perf_begin();
    int ok = fence_wait_impl(value, timeout_ns);
    perf_end(PERF_FENCE, p0);
    perf_gpu_wait(p0);
    return ok;
}

static int fence_wait_impl(uint64_t value, uint64_t timeout_ns) {
    vk_sync_start();
    plat_mutex_lock(&st_lock);
    int ok;
    if (value > issued && value > reserved) {
        ok = 0;   // never submitted nor queued: never reached (as in the null back end)
    } else {
        uint64_t t0 = kern_now_ns();
        while (done < value) {
            if (timeout_ns < (1ull << 62)) {
                uint64_t el = kern_now_ns() - t0;
                if (el >= timeout_ns) break;
                plat_cond_wait_ns(&cv_done, &st_lock, timeout_ns - el);
            } else {
                plat_cond_wait(&cv_done, &st_lock);
            }
        }
        ok = done >= value;
    }
    plat_mutex_unlock(&st_lock);
    return ok;
}

uint64_t vk_sync_fence_done(void) { return __atomic_load_n(&done, __ATOMIC_ACQUIRE); }

uint64_t vk_sync_issued(void) { return __atomic_load_n(&issued, __ATOMIC_ACQUIRE); }

void vk_sync_finish(void) {
    vk_sync_start();
    vk_sync_fence_wait(__atomic_load_n(&issued, __ATOMIC_ACQUIRE), UINT64_MAX);
    vk_upload_flush(1);
}

void vk_sync_shutdown(void) {
    plat_mutex_lock(&start_lock);
    if (started) {
        vk_sync_finish();
        plat_mutex_lock(&st_lock);
        stop = 1;
        plat_cond_broadcast(&cv_work);
        plat_mutex_unlock(&st_lock);
        plat_thread_join(completion_thread);
        completion_thread = NULL;
        plat_mutex_lock(&st_lock);
        qhead = qn = 0;   // callbacks of values never submitted
        plat_mutex_unlock(&st_lock);
        plat_mutex_lock(&sub_lock);
        VkCtx *c = vk_ctx();
        if (c && cmd_pool) vkDestroyCommandPool(c->device, cmd_pool, NULL);
        perf_gpu_stats = NULL;
        if (c && ts_pool) vkDestroyQueryPool(c->device, ts_pool, NULL);
        ts_pool = VK_NULL_HANDLE;
        ts_state = 0;
        ts_last_end = 0;
        memset((void *)ts_busy, 0, sizeof ts_busy);
        cmd_pool = VK_NULL_HANDLE;
        free(slots);
        slots = NULL;
        nslots = 0;
        plat_mutex_unlock(&sub_lock);
        __atomic_store_n(&started, 0, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&start_lock);
}
