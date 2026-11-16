// The recorder thread: see vk_rec.h. One producer (the submitting thread, between rec_begin and rec_end) and one
// consumer (the recorder) share a ring of 64-bit words. A packet is a header word (op, length in words) and the
// call's arguments; its pointers point into the packet itself, so the recorder passes the payload to the driver as
// it is. A packet never wraps: one that does not fit before the end is preceded by a pad. The producer publishes head
// after each packet, or every TOTK_REC_BATCH packets when that is set; the recorder makes the calls and advances tail, which the producer waits on when the ring is
// full and in rec_drain. While a part is recorded the recorder spins on head; idle, it sleeps on an event.
// Design: Savage Wind Recomp #189 and LAS Recomp runtime/nvn/vulkan/record.c (https://github.com/peiikai/LASRecomp
// @ a4b53a3, MIT, Copyright (c) 2026 Peiikai); the code is ours.
#define VK_REC_IMPL
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "platform/platform.h"

#define RING_WORDS (1u << 21)   // 16 MiB: a part of a submission is ~1-2 MiB of packets in the town
static uint64_t ring[RING_WORDS];
// head and tail each own a cache line: the recorder spins on head, so head must be written only when published and
// share its line with nothing the producer writes per packet (put_h, unpub, RS_).
typedef struct { _Alignas(64) volatile uint64_t v; } CacheLine;   // sizeof 64: nothing else shares the line
static CacheLine head_line, tail_line;
#define head head_line.v   // words published by the producer
#define tail tail_line.v   // words consumed by the recorder
static volatile int32_t sleeping;        // the recorder waits for wake_ev
static PlatEvent wake_ev;              // auto-reset
static uint64_t put_h;                 // producer: words written (published up to head)
static VkCommandBuffer rec_cb;         // the part being recorded: stored by the producer before its first packet
static int rec_live;                   // producer: rec_begin started the recorder for rec_cb (stored atomically: OWNED
                                       // is also asked from the helper translators, vk_spec.h)
static PlatThread *rec_thread;
static int rec_prio;                   // the submitting thread's priority, which the recorder takes

// Producer statistics, logged by rec_stats_log.
static struct { uint64_t packets, words, drains, fallbacks, drain_ns, full_waits, parts, parts_on; } RS_;

enum {
    R_PAD, R_BeginQuery, R_BeginRendering, R_BindDescriptorSets, R_BindIndexBuffer, R_BindPipeline, R_BindVertexBuffers2,
    R_BlitImage, R_ClearAttachments, R_ClearColorImage, R_ClearDepthStencilImage, R_CopyBuffer, R_CopyBufferToImage,
    R_CopyImage, R_CopyImageToBuffer, R_Dispatch, R_DispatchIndirect, R_Draw, R_DrawIndexed, R_DrawIndexedIndirect,
    R_EndQuery, R_EndRendering, R_FillBuffer, R_PipelineBarrier2, R_PushConstants, R_ResetQueryPool, R_ResolveImage,
    R_SetBlendConstants, R_SetCullMode, R_SetDepthBias, R_SetDepthBiasEnable, R_SetDepthCompareOp, R_SetDepthTestEnable,
    R_SetDepthWriteEnable, R_SetFrontFace, R_SetPrimitiveRestartEnable, R_SetPrimitiveTopology,
    R_SetRasterizerDiscardEnable, R_SetScissorWithCount, R_SetStencilCompareMask, R_SetStencilOp, R_SetStencilReference,
    R_SetStencilTestEnable, R_SetStencilWriteMask, R_SetViewportWithCount, R_WriteTimestamp, R_WriteTimestamp2,
    R_PushTpl, R_Push, R_VertexInput, R_Bool, R_PolygonMode, R_BlendEnable, R_BlendEquation, R_WriteMask, R_EndCb, R_Crumb, R_Checkpoint, R_BeginCb,
    R_Submit, R_Call,
};

static uint64_t now_ns(void) { return plat_now_ns(); }

// ---- consumer

static void exec_cb(VkCommandBuffer cb, uint32_t op, void *p);

// Publishes how far the recorder got. The driver may write command buffers through a write-combined mapping (RADV
// does, and AMD's Windows driver is the other place the recorder lost the device), and on x86 a release store does not
// wait for write-combined stores: the producer could see tail, end and submit the part while its last commands are
// still in the CPU's write-combining buffers. SFENCE drains them first; on ARM64 the release store already orders them.
// Lead from mkw-recomp de006fc (vk_rec.c publish_tail), reported in #204.
static inline void publish_tail(uint64_t t) {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_sfence();
#endif
    __atomic_store_n(&tail, t, __ATOMIC_RELEASE);
}

static void rec_main(void *arg) {
    (void)arg;
    plat_thread_priority(rec_prio);
    uint64_t t = 0;
    for (;;) {
        uint64_t h = __atomic_load_n(&head, __ATOMIC_ACQUIRE);
        if (t == h) {
            for (int i = 0; i < 4096 && t == h; i++) plat_pause(), h = __atomic_load_n(&head, __ATOMIC_ACQUIRE);
            if (t == h) {
                __atomic_store_n(&sleeping, 1, __ATOMIC_SEQ_CST);
                h = __atomic_load_n(&head, __ATOMIC_SEQ_CST);
                if (t == h) plat_event_wait_ns(&wake_ev, 2000000);   // a missed wake costs at most 2 ms
                __atomic_store_n(&sleeping, 0, __ATOMIC_RELAXED);
                continue;
            }
        }
        for (int k = 0; t != h; k++) {
            uint64_t w = ring[t % RING_WORDS];
            uint32_t op = (uint32_t)(w & 0xffff), n = (uint32_t)(w >> 32);
            if (op != R_PAD) exec_cb(__atomic_load_n(&rec_cb, __ATOMIC_RELAXED), op, &ring[t % RING_WORDS + 1]);
            t += n;
            if ((k & 63) == 63) publish_tail(t);
        }
        publish_tail(t);
    }
    return;
}

// ---- producer

// TOTK_REC_BATCH=N (0 off): head is published every N packets, not after each, so the recorder's spin does not read a
// cache line the producer rewrites constantly. Every wait or handoff (wait_space, rec_drain) publishes the rest.
static int rec_batch;     // read once in rec_begin
static int unpub;         // producer: packets written since head was last stored

static void wake(void) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(&sleeping, __ATOMIC_RELAXED)) plat_event_set(&wake_ev);
}

static void wait_space(uint64_t need) {
    if (put_h + need - __atomic_load_n(&tail, __ATOMIC_ACQUIRE) <= RING_WORDS) return;
    RS_.full_waits++;
    unpub = 0;
    __atomic_store_n(&head, put_h, __ATOMIC_RELEASE);
    wake();
    while (put_h + need - __atomic_load_n(&tail, __ATOMIC_ACQUIRE) > RING_WORDS) plat_pause();
}

// Room for a packet of bytes of arguments; the caller fills it and calls pub().
// ---- capture (vk_exec.c: the end of a sliced pass whose slices are still being recorded)
// Between rec_capture_begin and rec_capture_end the packets of the part being recorded go to cap instead of the ring,
// after the ones captured before, and rec_capture_play makes the words [lo, hi) later into a command buffer on the
// calling thread, while the recorder is idle; rec_capture_reset empties cap once nothing captured is held. The packets'
// pointers point into cap itself, which never moves. A call without a packet (FALLBACK) cannot be held back, nor a
// packet that no longer fits: the capture spills (spill_fn, vk_exec.c par_spill), which makes what it held into the part
// at once and ends the capture, and the call goes on as without one.
#define CAP_WORDS (1u << 15)
static uint64_t cap[CAP_WORDS];
static uint64_t cap_n;
static int capturing;
static void (*spill_fn)(void);
// A capture without the recorder (rec_capture_begin_for): the command buffer whose calls it takes, NULL otherwise.
// OWNED reads it from every thread that records (the slice workers record their own secondaries), so atomically.
static VkCommandBuffer cap_cb;

void rec_capture_spill(void (*fn)(void)) { spill_fn = fn; }
int rec_capture_room(uint32_t words) { return cap_n + words <= CAP_WORDS; }

static void spill(void) {
    if (!spill_fn) rt_fatal("vk_rec: a held-back pass end cannot be captured and nothing can spill it");
    spill_fn();   // it ends the capture (rec_capture_end)
    if (capturing) rt_fatal("vk_rec: a spilled pass end is still captured");
}

uint32_t rec_capture_begin(void) {
    capturing = 1;
    return (uint32_t)cap_n;
}
// Without the recorder, the submitting thread records cb itself: the pass end's calls go to the capture through the
// same wrappers, and rec_capture_play makes them on this thread, which recorded the rest of the part. vk_sync.c
// vk_sync_split_defer and vk_sync_split_finish need no recorder (rec_call declines and the end is made here).
// Only without a recorder thread at all: vk_sync_split_finish hands the held part's end to rec_call, which takes any
// recorder thread there is, so with TOTK_VK_RECORDER=ab a part the submitting thread recorded in a B block would be
// ended on the recorder, the pattern of issue #5.
int rec_capture_alone(VkCommandBuffer cb) { return cb && !rec_thread; }
uint32_t rec_capture_begin_for(VkCommandBuffer cb) {
    if (!rec_capture_alone(cb)) return rec_capture_begin();
    capturing = 1;
    __atomic_store_n(&cap_cb, cb, __ATOMIC_RELAXED);
    return (uint32_t)cap_n;
}
uint32_t rec_capture_end(void) {
    capturing = 0;
    __atomic_store_n(&cap_cb, (VkCommandBuffer)VK_NULL_HANDLE, __ATOMIC_RELAXED);
    return (uint32_t)cap_n;
}
void rec_capture_reset(void) { cap_n = 0; }

void rec_capture_play(VkCommandBuffer cb, uint32_t lo, uint32_t hi) {
    for (uint64_t t = lo; t < hi;) {
        uint32_t op = (uint32_t)(cap[t] & 0xffff), n = (uint32_t)(cap[t] >> 32);
        exec_cb(cb, op, &cap[t + 1]);
        t += n;
    }
}

int rec_owns(VkCommandBuffer cb) { return rec_live && cb == rec_cb; }

// A call whose packet no longer fits a capture without the recorder: spill ends the capture, and nothing reads the
// ring, so the packet goes to solo and pub makes it into the command buffer at once, after what the spill made.
static uint64_t solo[RING_WORDS / 4 + 1];
static VkCommandBuffer solo_cb;
static uint32_t solo_op;

static void *put(uint32_t op, size_t bytes) {
    uint64_t n = 1 + (bytes + 7) / 8, pos = put_h % RING_WORDS;
    if (capturing && cap_n + n > CAP_WORDS) {
        VkCommandBuffer alone = __atomic_load_n(&cap_cb, __ATOMIC_RELAXED);
        spill();
        if (alone) {
            if (n > RING_WORDS / 4) rt_fatal("vk_rec: a packet of %llu bytes", (unsigned long long)bytes);
            solo_cb = alone, solo_op = op;
            return &solo[1];
        }
    }
    if (capturing) {
        cap[cap_n] = (uint64_t)op | n << 32;
        cap_n += n;
        RS_.packets++, RS_.words += n;
        return &cap[cap_n - n + 1];
    }
    if (n > RING_WORDS / 4) rt_fatal("vk_rec: a packet of %llu bytes", (unsigned long long)bytes);
    if (pos + n > RING_WORDS) {
        uint64_t pad = RING_WORDS - pos;
        wait_space(pad);
        ring[pos] = (uint64_t)R_PAD | pad << 32;
        put_h += pad, pos = 0;
    }
    wait_space(n);
    ring[pos] = (uint64_t)op | n << 32;
    put_h += n;
    RS_.packets++, RS_.words += n;
    return &ring[pos + 1];
}

static void pub(void) {
    if (solo_cb) {   // the packet put kept after an alone capture spilled
        VkCommandBuffer cb = solo_cb;
        solo_cb = VK_NULL_HANDLE;
        exec_cb(cb, solo_op, &solo[1]);
        return;
    }
    if (capturing) return;
    if (rec_batch) {
        if (++unpub < rec_batch) return;
        unpub = 0;
        __atomic_store_n(&head, put_h, __ATOMIC_RELEASE);
        wake();   // the recorder sleeps more often when batched; a missed wake costs up to 2 ms
        return;
    }
    __atomic_store_n(&head, put_h, __ATOMIC_RELEASE);
    if (__atomic_load_n(&sleeping, __ATOMIC_RELAXED)) plat_event_set(&wake_ev);
}

void rec_drain(void) {
    if (!rec_thread || __atomic_load_n(&tail, __ATOMIC_ACQUIRE) == put_h) return;
    uint64_t t0 = now_ns();
    unpub = 0;
    __atomic_store_n(&head, put_h, __ATOMIC_RELEASE);
    wake();
    for (int i = 0; __atomic_load_n(&tail, __ATOMIC_ACQUIRE) != put_h; i++)
        if (i < 20000) plat_pause();
        else plat_yield();
    RS_.drains++, RS_.drain_ns += now_ns() - t0;
}

void rec_begin(VkCommandBuffer cb) {
    static int mode = -1;
    if (mode < 0) {
        // Default on, except on AMD's own Windows driver. AMD drivers once lost the device with the recorder (issues
        // #5 and #40); after vk_mem_note_recording kept objects alive until their submission completed and the
        // recorder began ending its own parts (rec_end_cb), RDNA 1 and 2 cards ran with it on and off without a loss.
        // Bananza still lost the device on AMD's proprietary driver, and TOTK_VK_RECORDER=0 stopped it for three
        // players on three GPU generations: a Radeon 610M (#84), a ROG Xbox Ally X (#155) and an RX 9070 XT (#142).
        // RADV keeps the recorder: its hangs (#122, #204) were not tested with it off.
        mode = rt_knob("VK_RECORDER", vk_caps.amd && !vk_caps.radv ? "0" : "1");
        if (mode) {
            rec_batch = cfg_int("TOTK_REC_BATCH");
            if (rec_batch < 0) rec_batch = 0;
            if (rec_batch > 64) rec_batch = 64;
            if (rec_batch) rt_log("vk_rec: batched head publish every %d packets (TOTK_REC_BATCH=%d)", rec_batch, rec_batch);
            plat_event_init(&wake_ev, 0, 0);
            rec_prio = plat_thread_priority_get();
            rec_thread = plat_thread_new(rec_main, NULL, "vk recorder");
            if (!rec_thread) {
                rt_log("vk_rec: no recorder thread: the replay makes its Vulkan calls itself");
                mode = 0;
            } else {
                rt_log("vk_rec: the replay's Vulkan commands are made by a recorder thread%s",
                       mode == 2 ? " in the B blocks (TOTK_AB)" : "");
            }
        }
    }
    RS_.parts++;
    if (!RT_KNOB_ON(mode)) return;
    RS_.parts_on++;
    // The recorder made every call of the part before and at most still submits it (rec_submit), which reads no
    // command buffer from here; it loads rec_cb for each packet, and cb's first packet is published after this store.
    __atomic_store_n(&rec_cb, cb, __ATOMIC_RELAXED);
    __atomic_store_n(&rec_live, 1, __ATOMIC_RELAXED);
}

void rec_end(void) {
    rec_drain();   // also when this part was not the recorder's: it may still submit the one before (rec_submit)
    __atomic_store_n(&rec_live, 0, __ATOMIC_RELAXED);
}

// TOTK_REC_END (default 1): the thread that recorded the part also ends it, after the breadcrumb crumb (kind 0: none)
// that vk_sync.c writes at the end of a submission. On an RX 6700 XT (driver 26.10.1, issue #5) parts the recorder
// recorded and the submitting thread ended broke when a submission was split (vk_sync.c vk_sync_split): the GPU hung
// or the scene turned green, then white. The handoff in rec_drain orders the two threads, so Vulkan allows the old
// split; the driver did not cope with it. TOTK_REC_END=0 brings the old split back.
typedef struct { int kind; uint64_t n; } A_EndCb;
int rec_end_cb(VkCommandBuffer cb, int crumb, uint64_t n) {
    static int mode = -1;
    if (mode < 0) {
        mode = rt_knob("REC_END", "1");
        rt_log("vk_rec: a part the recorder records is ended by %s (TOTK_REC_END=%d)",
               mode ? "the recorder" : "the submitting thread", mode);
    }
    if (!rec_live || cb != rec_cb || !mode) {
        rec_end();
        if (crumb) vk_crumb(cb, crumb, n);
        return 0;
    }
    A_EndCb *a = put(R_EndCb, sizeof *a);
    *a = (A_EndCb){crumb, n};
    pub();
    rec_drain();
    __atomic_store_n(&rec_live, 0, __ATOMIC_RELAXED);
    return 1;
}

// TOTK_REC_SUBMIT=1 (default 0): the recorder also submits a part it ended when more of the submission follows
// (vk_sync.c split_impl), and the replay goes on with the next part meanwhile. vkQueueSubmit2 was 37 us of the 77 us a
// split cost the replaying thread on an RTX 3060 Ti (Bananza, 7 to 8 parts a frame), and that thread is what the guest's
// fence wait is waiting for. Queue order is as before: the packet follows the part's last call, the end of every later
// part and of the submission drains the ring first (rec_end_cb, rec_end), and an upload batch waits for a submit in
// flight (rec_submit_wait). 0, the default: the replaying thread submits every part; ab: switches with the A/B blocks.
// The recorder calls fn(arg) after every call queued before it (1); 0 without a recorder thread or during a capture,
// when the caller calls it. vk_sync.c vk_sync_split_finish ends a held part with it, on the thread that recorded it.
typedef struct { void (*fn)(void *); void *arg; } A_Call;
int rec_call(void (*fn)(void *), void *arg) {
    if (!rec_thread) return 0;
    // A captured packet runs when the capture is replayed, after the caller has gone on; vk_sync_split_finish, the
    // one caller, runs between captures, so a call here while capturing is a bug, not something to make in place.
    if (capturing) rt_fatal("vk_rec: rec_call inside a capture");
    A_Call *a = put(R_Call, sizeof *a);
    *a = (A_Call){fn, arg};
    unpub = 0;
    __atomic_store_n(&head, put_h, __ATOMIC_RELEASE);
    wake();
    return 1;
}

typedef struct { VkCommandBuffer pro, cb; uint64_t fence; } A_Submit;
static volatile uint64_t sub_put, sub_done;   // part submits queued by the producer, made by the recorder

int rec_submit(VkCommandBuffer pro, VkCommandBuffer cb, uint64_t fence) {
    static int mode = -1;
    if (mode < 0) {
        mode = rt_knob("REC_SUBMIT", "0");
        rt_log("vk_rec: a part the recorder ended is submitted by %s (TOTK_REC_SUBMIT=%s)",
               mode ? "the recorder" : "the replaying thread", mode == 2 ? "ab" : mode ? "1" : "0");
    }
    if (!rec_thread || !RT_KNOB_ON(mode)) return 0;
    A_Submit *a = put(R_Submit, sizeof *a);
    *a = (A_Submit){pro, cb, fence};
    __atomic_store_n(&sub_put, sub_put + 1, __ATOMIC_RELEASE);
    unpub = 0;   // at once, whatever TOTK_REC_BATCH says: the GPU waits for this packet
    __atomic_store_n(&head, put_h, __ATOMIC_RELEASE);
    wake();
    return 1;
}

void rec_submit_wait(void) {
    uint64_t want = __atomic_load_n(&sub_put, __ATOMIC_ACQUIRE);
    for (int i = 0; __atomic_load_n(&sub_done, __ATOMIC_ACQUIRE) < want; i++)
        if (i < 20000) plat_pause();
        else plat_yield();
}

// TOTK_REC_BEGIN (default 0): the thread that records a part also begins it: vkBeginCommandBuffer, the submission crumb
// and the start timestamp pair go into the ring ahead of the part's commands, so on the AMD proprietary driver on
// Windows no call into the buffer is made by the submitting thread (the same class as TOTK_REC_END, issue #5).
// The timestamp pair (ts_pair, -1 none) is picked by the submitting thread, which owns that state.
typedef struct { uint64_t fence; VkQueryPool pool; int pair; } A_BeginCb;
static void begin_cb(VkCommandBuffer cb, const A_BeginCb *a) {
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    vk_crumb(cb, VK_CRUMB_SUBMIT, a->fence);
    if (a->pair < 0) return;
    vkCmdResetQueryPool(cb, a->pool, 2 * (uint32_t)a->pair, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, a->pool, 2 * (uint32_t)a->pair);
}

int rec_begin_part(VkCommandBuffer cb, uint64_t fence, int ts_pair, VkQueryPool ts_pool) {
    static int mode = -1;
    if (mode < 0) {
        mode = rt_knob("REC_BEGIN", "0");
        rt_log("vk_rec: a part the recorder records is begun by %s (TOTK_REC_BEGIN=%d)",
               mode ? "the recorder" : "the submitting thread", mode);
    }
    if (!mode) return 0;
    A_BeginCb a = {fence, ts_pool, ts_pair};
    rec_begin(cb);
    if (!rec_live) {   // no recorder: begin it here, in the same order
        begin_cb(cb, &a);
        return 1;
    }
    A_BeginCb *p = put(R_BeginCb, sizeof *p);
    *p = a;
    pub();
    return 1;
}

void rec_stats_log(void) {
    if (!(rt_trace & RT_TRACE_PERF) || !RS_.parts) return;
    rt_log("vk_rec: %llu of %llu parts recorded by the recorder: %llu packets (%.1f MiB), %llu drains (%llu for a call "
           "without a packet) waited %.2f ms in all, ring full %llu times",
           (unsigned long long)RS_.parts_on, (unsigned long long)RS_.parts, (unsigned long long)RS_.packets,
           RS_.words * 8 / 1048576.0, (unsigned long long)RS_.drains, (unsigned long long)RS_.fallbacks,
           RS_.drain_ns / 1e6, (unsigned long long)RS_.full_waits);
    memset(&RS_, 0, sizeof RS_);
}

// A helper translator asks this too, for its own command buffer, which is never rec_cb: the loads are atomic so that
// reading them while the producer moves to another part is not a data race.
// A capture without the recorder (cap_cb) owns its command buffer the same way: its calls become packets in cap.
#define OWNED(cb) ((__atomic_load_n(&rec_live, __ATOMIC_RELAXED) && (cb) == __atomic_load_n(&rec_cb, __ATOMIC_RELAXED)) || \
                   ((cb) && (cb) == __atomic_load_n(&cap_cb, __ATOMIC_RELAXED)))
// A call the ring takes no packet for: everything queued first, then the call here.
#define FALLBACK() (capturing ? spill() : (void)0, rec_drain(), RS_.fallbacks++)

// Copies n elements of arr after the fixed part at *at and returns where they went (NULL for none).
static void *tail_copy(uint8_t **at, const void *arr, size_t bytes) {
    if (!arr || !bytes) return NULL;
    void *d = *at;
    memcpy(d, arr, bytes);
    *at += (bytes + 7) & ~(size_t)7;
    return d;
}
#define A8(x) (((x) + 7) & ~(size_t)7)

// ---- fixed-size commands: the arguments after cb, as a struct

#define FIXED(name, params, call_args, ...)                                                                             \
    typedef struct { __VA_ARGS__; } A_##name;                                                                          \
    void rc_vkCmd##name params {                                                                                       \
        if (!OWNED(cb)) { vkCmd##name call_args; return; }                                                             \
        A_##name *a = put(R_##name, sizeof *a);                                                                        \
        PACK_##name;                                                                                                    \
        pub();                                                                                                          \
    }

#define PACK_BeginQuery *a = (A_BeginQuery){p, q, f}
FIXED(BeginQuery, (VkCommandBuffer cb, VkQueryPool p, uint32_t q, VkQueryControlFlags f), (cb, p, q, f),
      VkQueryPool p; uint32_t q; VkQueryControlFlags f)
#define PACK_BindIndexBuffer *a = (A_BindIndexBuffer){b, o, t}
FIXED(BindIndexBuffer, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t), (cb, b, o, t), VkBuffer b;
      VkDeviceSize o; VkIndexType t)
// DKB_GPU_CHECKPOINTS=1: a checkpoint in front of a dispatch, copy, blit or clear (vk_crumbs.c vk_checkpoint_tag), so a
// lost device names that command and not only the pipeline bound before it. The dispatch checkpoints carry the compute
// pipeline last bound here, which is why the binds are written by hand. Nothing happens with the checkpoints off.
static void mark(VkCommandBuffer cb, const char *kind) {
    if (vk_checkpoints_on()) rec_checkpoint(cb, vk_checkpoint_tag(kind));
}

typedef struct { VkPipelineBindPoint bp; VkPipeline p; } A_BindPipeline;
void rc_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE && vk_checkpoints_on()) vk_checkpoint_compute(p);
    if (!OWNED(cb)) { vkCmdBindPipeline(cb, bp, p); return; }
    A_BindPipeline *a = put(R_BindPipeline, sizeof *a);
    *a = (A_BindPipeline){bp, p};
    pub();
}
typedef struct { uint32_t x, y, z; } A_Dispatch;
void rc_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    mark(cb, "dispatch");
    if (!OWNED(cb)) { vkCmdDispatch(cb, x, y, z); return; }
    A_Dispatch *a = put(R_Dispatch, sizeof *a);
    *a = (A_Dispatch){x, y, z};
    pub();
}
typedef struct { VkBuffer b; VkDeviceSize o; } A_DispatchIndirect;
void rc_vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o) {
    mark(cb, "dispatch indirect");
    if (!OWNED(cb)) { vkCmdDispatchIndirect(cb, b, o); return; }
    A_DispatchIndirect *a = put(R_DispatchIndirect, sizeof *a);
    *a = (A_DispatchIndirect){b, o};
    pub();
}
#define PACK_Draw *a = (A_Draw){n, i, f, fi}
FIXED(Draw, (VkCommandBuffer cb, uint32_t n, uint32_t i, uint32_t f, uint32_t fi), (cb, n, i, f, fi), uint32_t n, i, f, fi)
#define PACK_DrawIndexed *a = (A_DrawIndexed){n, i, f, v, fi}
FIXED(DrawIndexed, (VkCommandBuffer cb, uint32_t n, uint32_t i, uint32_t f, int32_t v, uint32_t fi), (cb, n, i, f, v, fi),
      uint32_t n, i, f; int32_t v; uint32_t fi)
#define PACK_DrawIndexedIndirect *a = (A_DrawIndexedIndirect){b, o, n, s}
FIXED(DrawIndexedIndirect, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s), (cb, b, o, n, s),
      VkBuffer b; VkDeviceSize o; uint32_t n, s)
#define PACK_EndQuery *a = (A_EndQuery){p, q}
FIXED(EndQuery, (VkCommandBuffer cb, VkQueryPool p, uint32_t q), (cb, p, q), VkQueryPool p; uint32_t q)
#define PACK_EndRendering (void)a
FIXED(EndRendering, (VkCommandBuffer cb), (cb), int unused)
#define PACK_FillBuffer *a = (A_FillBuffer){b, o, n, v}
FIXED(FillBuffer, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, uint32_t v), (cb, b, o, n, v),
      VkBuffer b; VkDeviceSize o, n; uint32_t v)
#define PACK_ResetQueryPool *a = (A_ResetQueryPool){p, f, n}
FIXED(ResetQueryPool, (VkCommandBuffer cb, VkQueryPool p, uint32_t f, uint32_t n), (cb, p, f, n), VkQueryPool p;
      uint32_t f, n)
#define PACK_SetBlendConstants memcpy(a->c, c, sizeof a->c)
FIXED(SetBlendConstants, (VkCommandBuffer cb, const float c[4]), (cb, c), float c[4])
#define PACK_SetCullMode a->v = m
FIXED(SetCullMode, (VkCommandBuffer cb, VkCullModeFlags m), (cb, m), VkCullModeFlags v)
#define PACK_SetDepthBias *a = (A_SetDepthBias){c, cl, s}
FIXED(SetDepthBias, (VkCommandBuffer cb, float c, float cl, float s), (cb, c, cl, s), float c, cl, s)
#define PACK_SetDepthBiasEnable a->v = v
FIXED(SetDepthBiasEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v), VkBool32 v)
#define PACK_SetDepthCompareOp a->v = v
FIXED(SetDepthCompareOp, (VkCommandBuffer cb, VkCompareOp v), (cb, v), VkCompareOp v)
#define PACK_SetDepthTestEnable a->v = v
FIXED(SetDepthTestEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v), VkBool32 v)
#define PACK_SetDepthWriteEnable a->v = v
FIXED(SetDepthWriteEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v), VkBool32 v)
#define PACK_SetFrontFace a->v = v
FIXED(SetFrontFace, (VkCommandBuffer cb, VkFrontFace v), (cb, v), VkFrontFace v)
#define PACK_SetPrimitiveRestartEnable a->v = v
FIXED(SetPrimitiveRestartEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v), VkBool32 v)
#define PACK_SetPrimitiveTopology a->v = v
FIXED(SetPrimitiveTopology, (VkCommandBuffer cb, VkPrimitiveTopology v), (cb, v), VkPrimitiveTopology v)
#define PACK_SetRasterizerDiscardEnable a->v = v
FIXED(SetRasterizerDiscardEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v), VkBool32 v)
#define PACK_SetStencilCompareMask *a = (A_SetStencilCompareMask){f, v}
FIXED(SetStencilCompareMask, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v), (cb, f, v), VkStencilFaceFlags f;
      uint32_t v)
#define PACK_SetStencilOp *a = (A_SetStencilOp){f, x, y, z, d}
FIXED(SetStencilOp, (VkCommandBuffer cb, VkStencilFaceFlags f, VkStencilOp x, VkStencilOp y, VkStencilOp z, VkCompareOp d),
      (cb, f, x, y, z, d), VkStencilFaceFlags f; VkStencilOp x, y, z; VkCompareOp d)
#define PACK_SetStencilReference *a = (A_SetStencilReference){f, v}
FIXED(SetStencilReference, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v), (cb, f, v), VkStencilFaceFlags f;
      uint32_t v)
#define PACK_SetStencilTestEnable a->v = v
FIXED(SetStencilTestEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v), VkBool32 v)
#define PACK_SetStencilWriteMask *a = (A_SetStencilWriteMask){f, v}
FIXED(SetStencilWriteMask, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v), (cb, f, v), VkStencilFaceFlags f;
      uint32_t v)
#define PACK_WriteTimestamp *a = (A_WriteTimestamp){s, p, q}
FIXED(WriteTimestamp, (VkCommandBuffer cb, VkPipelineStageFlagBits s, VkQueryPool p, uint32_t q), (cb, s, p, q),
      VkPipelineStageFlagBits s; VkQueryPool p; uint32_t q)
#define PACK_WriteTimestamp2 *a = (A_WriteTimestamp2){s, p, q}
FIXED(WriteTimestamp2, (VkCommandBuffer cb, VkPipelineStageFlags2 s, VkQueryPool p, uint32_t q), (cb, s, p, q),
      VkPipelineStageFlags2 s; VkQueryPool p; uint32_t q)

// ---- commands with arrays: a struct with the call's arguments, the arrays after it, pointers into the packet

typedef struct { uint32_t n; const VkViewport *v; } A_SetViewportWithCount;
void rc_vkCmdSetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport *v) {
    if (!OWNED(cb)) { vkCmdSetViewportWithCount(cb, n, v); return; }
    A_SetViewportWithCount *a = put(R_SetViewportWithCount, sizeof *a + sizeof *v * n);
    uint8_t *at = (uint8_t *)(a + 1);
    a->n = n, a->v = tail_copy(&at, v, sizeof *v * n);
    pub();
}

typedef struct { uint32_t n; const VkRect2D *r; } A_SetScissorWithCount;
void rc_vkCmdSetScissorWithCount(VkCommandBuffer cb, uint32_t n, const VkRect2D *r) {
    if (!OWNED(cb)) { vkCmdSetScissorWithCount(cb, n, r); return; }
    A_SetScissorWithCount *a = put(R_SetScissorWithCount, sizeof *a + sizeof *r * n);
    uint8_t *at = (uint8_t *)(a + 1);
    a->n = n, a->r = tail_copy(&at, r, sizeof *r * n);
    pub();
}

typedef struct {
    VkPipelineBindPoint bp;
    VkPipelineLayout l;
    uint32_t first, n, nd;
    const VkDescriptorSet *s;
    const uint32_t *d;
} A_BindDescriptorSets;
void rc_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first,
                                uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {
    if (!OWNED(cb)) { vkCmdBindDescriptorSets(cb, bp, l, first, n, s, nd, d); return; }
    A_BindDescriptorSets *a = put(R_BindDescriptorSets, sizeof *a + A8(sizeof *s * n) + A8(sizeof *d * nd));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_BindDescriptorSets){bp, l, first, n, nd, NULL, NULL};
    a->s = tail_copy(&at, s, sizeof *s * n), a->d = tail_copy(&at, d, sizeof *d * nd);
    pub();
}

typedef struct { uint32_t first, n; const VkBuffer *b; const VkDeviceSize *o, *sz, *st; } A_BindVertexBuffers2;
void rc_vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o,
                                const VkDeviceSize *sz, const VkDeviceSize *st) {
    if (!OWNED(cb)) { vkCmdBindVertexBuffers2(cb, first, n, b, o, sz, st); return; }
    size_t e = sizeof(VkDeviceSize) * n;
    A_BindVertexBuffers2 *a = put(R_BindVertexBuffers2, sizeof *a + A8(sizeof *b * n) + 3 * A8(e));
    uint8_t *at = (uint8_t *)(a + 1);
    a->first = first, a->n = n;
    a->b = tail_copy(&at, b, sizeof *b * n), a->o = tail_copy(&at, o, e);
    a->sz = sz ? tail_copy(&at, sz, e) : NULL, a->st = st ? tail_copy(&at, st, e) : NULL;
    pub();
}

typedef struct { VkImage s; VkImageLayout sl; VkImage d; VkImageLayout dl; uint32_t n; const void *r; VkFilter f; } A_Img;
#define IMG_CMD(name, rtype, params, call, ...)                                                                         \
    void rc_vkCmd##name params {                                                                                       \
        mark(cb, #name);                                                                                                \
        if (!OWNED(cb)) { vkCmd##name call; return; }                                                                  \
        A_Img *a = put(R_##name, sizeof *a + sizeof(rtype) * n);                                                       \
        uint8_t *at = (uint8_t *)(a + 1);                                                                              \
        *a = (A_Img){__VA_ARGS__};                                                                                      \
        a->r = tail_copy(&at, r, sizeof(rtype) * n);                                                                   \
        pub();                                                                                                          \
    }
IMG_CMD(BlitImage, VkImageBlit, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
        const VkImageBlit *r, VkFilter f), (cb, s, sl, d, dl, n, r, f), s, sl, d, dl, n, NULL, f)
IMG_CMD(CopyImage, VkImageCopy, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
        const VkImageCopy *r), (cb, s, sl, d, dl, n, r), s, sl, d, dl, n, NULL, 0)
IMG_CMD(ResolveImage, VkImageResolve, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
        uint32_t n, const VkImageResolve *r), (cb, s, sl, d, dl, n, r), s, sl, d, dl, n, NULL, 0)

typedef struct { uint64_t s, d; VkImageLayout l; uint32_t n; const void *r; } A_Xfer;   // buffer or image handles
#define XFER_CMD(name, rtype, params, call, S, D, L)                                                                    \
    void rc_vkCmd##name params {                                                                                       \
        mark(cb, #name);                                                                                                \
        if (!OWNED(cb)) { vkCmd##name call; return; }                                                                  \
        A_Xfer *a = put(R_##name, sizeof *a + sizeof(rtype) * n);                                                      \
        uint8_t *at = (uint8_t *)(a + 1);                                                                              \
        *a = (A_Xfer){(uint64_t)(S), (uint64_t)(D), L, n, NULL};                                                        \
        a->r = tail_copy(&at, r, sizeof(rtype) * n);                                                                   \
        pub();                                                                                                          \
    }
XFER_CMD(CopyBuffer, VkBufferCopy, (VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r),
         (cb, s, d, n, r), s, d, 0)
XFER_CMD(CopyBufferToImage, VkBufferImageCopy, (VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout dl, uint32_t n,
         const VkBufferImageCopy *r), (cb, s, d, dl, n, r), s, d, dl)
XFER_CMD(CopyImageToBuffer, VkBufferImageCopy, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkBuffer d, uint32_t n,
         const VkBufferImageCopy *r), (cb, s, sl, d, n, r), s, d, sl)

typedef struct { uint32_t na, nr; const VkClearAttachment *a; const VkClearRect *r; } A_ClearAttachments;
void rc_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *ca, uint32_t nr, const VkClearRect *r) {
    if (!OWNED(cb)) { vkCmdClearAttachments(cb, na, ca, nr, r); return; }
    A_ClearAttachments *a = put(R_ClearAttachments, sizeof *a + A8(sizeof *ca * na) + A8(sizeof *r * nr));
    uint8_t *at = (uint8_t *)(a + 1);
    a->na = na, a->nr = nr;
    a->a = tail_copy(&at, ca, sizeof *ca * na), a->r = tail_copy(&at, r, sizeof *r * nr);
    pub();
}

typedef struct { VkImage i; VkImageLayout l; union { VkClearColorValue c; VkClearDepthStencilValue ds; } v; uint32_t n;
                 const VkImageSubresourceRange *r; } A_ClearImage;
void rc_vkCmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *c, uint32_t n,
                             const VkImageSubresourceRange *r) {
    mark(cb, "ClearColorImage");
    if (!OWNED(cb)) { vkCmdClearColorImage(cb, i, l, c, n, r); return; }
    A_ClearImage *a = put(R_ClearColorImage, sizeof *a + sizeof *r * n);
    uint8_t *at = (uint8_t *)(a + 1);
    a->i = i, a->l = l, a->v.c = *c, a->n = n, a->r = tail_copy(&at, r, sizeof *r * n);
    pub();
}
void rc_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue *c,
                                    uint32_t n, const VkImageSubresourceRange *r) {
    if (!OWNED(cb)) { vkCmdClearDepthStencilImage(cb, i, l, c, n, r); return; }
    A_ClearImage *a = put(R_ClearDepthStencilImage, sizeof *a + sizeof *r * n);
    uint8_t *at = (uint8_t *)(a + 1);
    a->i = i, a->l = l, a->v.ds = *c, a->n = n, a->r = tail_copy(&at, r, sizeof *r * n);
    pub();
}

typedef struct { VkPipelineLayout l; VkShaderStageFlags st; uint32_t o, n; const void *v; } A_PushConstants;
void rc_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t o, uint32_t n,
                           const void *v) {
    if (!OWNED(cb)) { vkCmdPushConstants(cb, l, st, o, n, v); return; }
    A_PushConstants *a = put(R_PushConstants, sizeof *a + n);
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_PushConstants){l, st, o, n, NULL};
    a->v = tail_copy(&at, v, n);
    pub();
}

void rc_vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *di) {
    if (!OWNED(cb)) { vkCmdPipelineBarrier2(cb, di); return; }
    int chained = di->pNext != NULL;
    for (uint32_t i = 0; i < di->memoryBarrierCount; i++) chained |= di->pMemoryBarriers[i].pNext != NULL;
    for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; i++) chained |= di->pBufferMemoryBarriers[i].pNext != NULL;
    for (uint32_t i = 0; i < di->imageMemoryBarrierCount; i++) chained |= di->pImageMemoryBarriers[i].pNext != NULL;
    if (chained) { FALLBACK(); vkCmdPipelineBarrier2(cb, di); return; }
    size_t m = sizeof *di->pMemoryBarriers * di->memoryBarrierCount;
    size_t b = sizeof *di->pBufferMemoryBarriers * di->bufferMemoryBarrierCount;
    size_t im = sizeof *di->pImageMemoryBarriers * di->imageMemoryBarrierCount;
    VkDependencyInfo *a = put(R_PipelineBarrier2, sizeof *a + A8(m) + A8(b) + A8(im));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = *di;
    a->pMemoryBarriers = tail_copy(&at, di->pMemoryBarriers, m);
    a->pBufferMemoryBarriers = tail_copy(&at, di->pBufferMemoryBarriers, b);
    a->pImageMemoryBarriers = tail_copy(&at, di->pImageMemoryBarriers, im);
    pub();
}

void rc_vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) {
    if (!OWNED(cb)) { vkCmdBeginRendering(cb, ri); return; }
    int chained = ri->pNext != NULL || (ri->pDepthAttachment && ri->pDepthAttachment->pNext) ||
                  (ri->pStencilAttachment && ri->pStencilAttachment->pNext);
    for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) chained |= ri->pColorAttachments[i].pNext != NULL;
    if (chained) { FALLBACK(); vkCmdBeginRendering(cb, ri); return; }
    size_t c = sizeof *ri->pColorAttachments * ri->colorAttachmentCount, one = sizeof(VkRenderingAttachmentInfo);
    VkRenderingInfo *a = put(R_BeginRendering, sizeof *a + A8(c) + 2 * A8(one));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = *ri;
    a->pColorAttachments = tail_copy(&at, ri->pColorAttachments, c);
    a->pDepthAttachment = ri->pDepthAttachment ? tail_copy(&at, ri->pDepthAttachment, one) : NULL;
    a->pStencilAttachment = ri->pStencilAttachment ? tail_copy(&at, ri->pStencilAttachment, one) : NULL;
    pub();
}

// ---- extension commands through function pointers

typedef struct { int kind; uint64_t n; } A_Crumb;
void rec_crumb(VkCommandBuffer cb, int kind, uint64_t n) {
    if (!OWNED(cb)) { vk_crumb(cb, kind, n); return; }
    A_Crumb *a = put(R_Crumb, sizeof *a);
    *a = (A_Crumb){kind, n};
    pub();
}

typedef struct { const void *marker; } A_Checkpoint;
void rec_checkpoint(VkCommandBuffer cb, const void *marker) {
    if (!OWNED(cb)) { vk_checkpoint(cb, marker); return; }
    A_Checkpoint *a = put(R_Checkpoint, sizeof *a);
    a->marker = marker;
    pub();
}

typedef struct { PFN_vkCmdPushDescriptorSetWithTemplateKHR fn; VkDescriptorUpdateTemplate tpl; VkPipelineLayout l;
                 uint32_t set; const void *data; } A_PushTpl;
void rec_push_tpl(PFN_vkCmdPushDescriptorSetWithTemplateKHR fn, VkCommandBuffer cb, VkDescriptorUpdateTemplate tpl,
                  VkPipelineLayout layout, uint32_t set, const void *data, uint32_t bytes) {
    if (!OWNED(cb)) { fn(cb, tpl, layout, set, data); return; }
    A_PushTpl *a = put(R_PushTpl, sizeof *a + bytes);
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_PushTpl){fn, tpl, layout, set, NULL};
    a->data = tail_copy(&at, data, bytes);
    pub();
}

typedef struct { PFN_vkCmdPushDescriptorSetKHR fn; VkPipelineBindPoint bp; VkPipelineLayout l; uint32_t set, n;
                 VkWriteDescriptorSet *w; } A_Push;
void rec_push(PFN_vkCmdPushDescriptorSetKHR fn, VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout,
              uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    if (!OWNED(cb)) { fn(cb, bp, layout, set, n, w); return; }
    size_t bytes = A8(sizeof *w * n);
    int chained = 0;
    for (uint32_t i = 0; i < n; i++) {
        chained |= w[i].pNext != NULL;
        if (w[i].pBufferInfo) bytes += A8(sizeof *w[i].pBufferInfo * w[i].descriptorCount);
        if (w[i].pImageInfo) bytes += A8(sizeof *w[i].pImageInfo * w[i].descriptorCount);
        if (w[i].pTexelBufferView) bytes += A8(sizeof *w[i].pTexelBufferView * w[i].descriptorCount);
    }
    if (chained) { FALLBACK(); fn(cb, bp, layout, set, n, w); return; }
    A_Push *a = put(R_Push, sizeof *a + bytes);
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_Push){fn, bp, layout, set, n, NULL};
    a->w = tail_copy(&at, w, sizeof *w * n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t k = w[i].descriptorCount;
        a->w[i].pBufferInfo = tail_copy(&at, w[i].pBufferInfo, w[i].pBufferInfo ? sizeof *w[i].pBufferInfo * k : 0);
        a->w[i].pImageInfo = tail_copy(&at, w[i].pImageInfo, w[i].pImageInfo ? sizeof *w[i].pImageInfo * k : 0);
        a->w[i].pTexelBufferView =
            tail_copy(&at, w[i].pTexelBufferView, w[i].pTexelBufferView ? sizeof *w[i].pTexelBufferView * k : 0);
    }
    pub();
}

typedef struct { PFN_vkCmdSetVertexInputEXT fn; uint32_t nb, na; const VkVertexInputBindingDescription2EXT *b;
                 const VkVertexInputAttributeDescription2EXT *v; } A_VertexInput;
void rec_vertex_input(PFN_vkCmdSetVertexInputEXT fn, VkCommandBuffer cb, uint32_t nb,
                      const VkVertexInputBindingDescription2EXT *b, uint32_t na,
                      const VkVertexInputAttributeDescription2EXT *v) {
    if (!OWNED(cb)) { fn(cb, nb, b, na, v); return; }
    A_VertexInput *a = put(R_VertexInput, sizeof *a + A8(sizeof *b * nb) + A8(sizeof *v * na));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_VertexInput){fn, nb, na, NULL, NULL};
    a->b = tail_copy(&at, b, sizeof *b * nb), a->v = tail_copy(&at, v, sizeof *v * na);   // pNext: always NULL there
    pub();
}

typedef struct { PFN_vkCmdSetDepthClampEnableEXT fn; VkBool32 v; } A_Bool;
void rec_bool(PFN_vkCmdSetDepthClampEnableEXT fn, VkCommandBuffer cb, VkBool32 v) {
    if (!OWNED(cb)) { fn(cb, v); return; }
    A_Bool *a = put(R_Bool, sizeof *a);
    *a = (A_Bool){fn, v};
    pub();
}

typedef struct { PFN_vkCmdSetPolygonModeEXT fn; VkPolygonMode m; } A_PolygonMode;
void rec_polygon_mode(PFN_vkCmdSetPolygonModeEXT fn, VkCommandBuffer cb, VkPolygonMode m) {
    if (!OWNED(cb)) { fn(cb, m); return; }
    A_PolygonMode *a = put(R_PolygonMode, sizeof *a);
    *a = (A_PolygonMode){fn, m};
    pub();
}

typedef struct { void *fn; uint32_t first, n; const void *v; } A_Targets;
#define TARGET_CMD(fname, R, PFN, T)                                                                                    \
    void fname(PFN fn, VkCommandBuffer cb, uint32_t first, uint32_t n, const T *v) {                                    \
        if (!OWNED(cb)) { fn(cb, first, n, v); return; }                                                               \
        A_Targets *a = put(R, sizeof *a + sizeof *v * n);                                                              \
        uint8_t *at = (uint8_t *)(a + 1);                                                                              \
        *a = (A_Targets){(void *)fn, first, n, NULL};                                                                   \
        a->v = tail_copy(&at, v, sizeof *v * n);                                                                       \
        pub();                                                                                                          \
    }
TARGET_CMD(rec_blend_enable, R_BlendEnable, PFN_vkCmdSetColorBlendEnableEXT, VkBool32)
TARGET_CMD(rec_blend_equation, R_BlendEquation, PFN_vkCmdSetColorBlendEquationEXT, VkColorBlendEquationEXT)
TARGET_CMD(rec_write_mask, R_WriteMask, PFN_vkCmdSetColorWriteMaskEXT, VkColorComponentFlags)

// ---- the recorder's side

static void exec_cb(VkCommandBuffer cb, uint32_t op, void *p) {
    switch (op) {
#define F(name, ...) case R_##name: { A_##name *a = p; vkCmd##name(cb, __VA_ARGS__); break; }
    F(BeginQuery, a->p, a->q, a->f)
    F(BindIndexBuffer, a->b, a->o, a->t)
    F(BindPipeline, a->bp, a->p)
    F(Dispatch, a->x, a->y, a->z)
    F(DispatchIndirect, a->b, a->o)
    F(Draw, a->n, a->i, a->f, a->fi)
    F(DrawIndexed, a->n, a->i, a->f, a->v, a->fi)
    F(DrawIndexedIndirect, a->b, a->o, a->n, a->s)
    F(EndQuery, a->p, a->q)
    F(FillBuffer, a->b, a->o, a->n, a->v)
    F(ResetQueryPool, a->p, a->f, a->n)
    F(SetBlendConstants, a->c)
    F(SetCullMode, a->v)
    F(SetDepthBias, a->c, a->cl, a->s)
    F(SetDepthBiasEnable, a->v)
    F(SetDepthCompareOp, a->v)
    F(SetDepthTestEnable, a->v)
    F(SetDepthWriteEnable, a->v)
    F(SetFrontFace, a->v)
    F(SetPrimitiveRestartEnable, a->v)
    F(SetPrimitiveTopology, a->v)
    F(SetRasterizerDiscardEnable, a->v)
    F(SetStencilCompareMask, a->f, a->v)
    F(SetStencilOp, a->f, a->x, a->y, a->z, a->d)
    F(SetStencilReference, a->f, a->v)
    F(SetStencilTestEnable, a->v)
    F(SetStencilWriteMask, a->f, a->v)
    F(WriteTimestamp, a->s, a->p, a->q)
    F(WriteTimestamp2, a->s, a->p, a->q)
    F(SetViewportWithCount, a->n, a->v)
    F(SetScissorWithCount, a->n, a->r)
    F(BindDescriptorSets, a->bp, a->l, a->first, a->n, a->s, a->nd, a->d)
    F(BindVertexBuffers2, a->first, a->n, a->b, a->o, a->sz, a->st)
    F(ClearAttachments, a->na, a->a, a->nr, a->r)
    case R_ClearColorImage: { A_ClearImage *a = p; vkCmdClearColorImage(cb, a->i, a->l, &a->v.c, a->n, a->r); break; }
    case R_ClearDepthStencilImage: {
        A_ClearImage *a = p;
        vkCmdClearDepthStencilImage(cb, a->i, a->l, &a->v.ds, a->n, a->r);
        break;
    }
    F(PushConstants, a->l, a->st, a->o, a->n, a->v)
#undef F
    case R_EndRendering: vkCmdEndRendering(cb); break;
    case R_BlitImage: { A_Img *a = p; vkCmdBlitImage(cb, a->s, a->sl, a->d, a->dl, a->n, a->r, a->f); break; }
    case R_CopyImage: { A_Img *a = p; vkCmdCopyImage(cb, a->s, a->sl, a->d, a->dl, a->n, a->r); break; }
    case R_ResolveImage: { A_Img *a = p; vkCmdResolveImage(cb, a->s, a->sl, a->d, a->dl, a->n, a->r); break; }
    case R_CopyBuffer: { A_Xfer *a = p; vkCmdCopyBuffer(cb, (VkBuffer)a->s, (VkBuffer)a->d, a->n, a->r); break; }
    case R_CopyBufferToImage: {
        A_Xfer *a = p;
        vkCmdCopyBufferToImage(cb, (VkBuffer)a->s, (VkImage)a->d, a->l, a->n, a->r);
        break;
    }
    case R_CopyImageToBuffer: {
        A_Xfer *a = p;
        vkCmdCopyImageToBuffer(cb, (VkImage)a->s, a->l, (VkBuffer)a->d, a->n, a->r);
        break;
    }
    case R_PipelineBarrier2: vkCmdPipelineBarrier2(cb, p); break;
    case R_BeginRendering: vkCmdBeginRendering(cb, p); break;
    case R_PushTpl: { A_PushTpl *a = p; a->fn(cb, a->tpl, a->l, a->set, a->data); break; }
    case R_Push: { A_Push *a = p; a->fn(cb, a->bp, a->l, a->set, a->n, a->w); break; }
    case R_VertexInput: { A_VertexInput *a = p; a->fn(cb, a->nb, a->b, a->na, a->v); break; }
    case R_Bool: { A_Bool *a = p; a->fn(cb, a->v); break; }
    case R_PolygonMode: { A_PolygonMode *a = p; a->fn(cb, a->m); break; }
    case R_BlendEnable: { A_Targets *a = p; ((PFN_vkCmdSetColorBlendEnableEXT)a->fn)(cb, a->first, a->n, a->v); break; }
    case R_BlendEquation: { A_Targets *a = p; ((PFN_vkCmdSetColorBlendEquationEXT)a->fn)(cb, a->first, a->n, a->v); break; }
    case R_WriteMask: { A_Targets *a = p; ((PFN_vkCmdSetColorWriteMaskEXT)a->fn)(cb, a->first, a->n, a->v); break; }
    case R_Checkpoint: { A_Checkpoint *a = p; vk_checkpoint(cb, a->marker); break; }
    case R_Crumb: { A_Crumb *a = p; vk_crumb(cb, a->kind, a->n); break; }
    case R_BeginCb: begin_cb(cb, p); break;
    case R_Call: { A_Call *a = p; a->fn(a->arg); break; }
    case R_EndCb: {
        A_EndCb *a = p;
        if (a->kind) vk_crumb(cb, a->kind, a->n);
        vkEndCommandBuffer(cb);
        break;
    }
    case R_Submit: {   // the mirror prologue, when the part has one, then the part: vk_sync.c split_impl
        A_Submit *a = p;
        VkCommandBufferSubmitInfo cbi[2] = {{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = a->pro},
                                            {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = a->cb}};
        VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = a->pro ? 2 : 1,
                            .pCommandBufferInfos = a->pro ? cbi : cbi + 1};
        vk_queue_lock();
        VkResult r = vkQueueSubmit2(vk_ctx()->gfx_queue, 1, &si, VK_NULL_HANDLE);
        vk_queue_unlock();
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: vkQueueSubmit2 of a part of value %llu failed (%d)", (unsigned long long)a->fence, (int)r);
        }
        __atomic_store_n(&sub_done, sub_done + 1, __ATOMIC_RELEASE);
        break;
    }
    default: rt_fatal("vk_rec: unknown packet %u", op);
    }
}
