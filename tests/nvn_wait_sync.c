// Handwritten NVN commands and a condition-latched completion backend. No game, Vulkan loader or GPU.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "nvn_api.h"
#include "platform/platform.h"

// Use the actual null replay to check that queued waits never dereference their submit-time source.
#define gfx_backend unchosen_null_backend
#include "../runtime/gfx_null.c"
#undef gfx_backend

#define PROC(n) void nvnp_##n(Cpu *c)
PROC(nvnCommandBufferInitialize);
PROC(nvnCommandBufferFinalize);
PROC(nvnCommandBufferAddCommandMemory);
PROC(nvnCommandBufferAddControlMemory);
PROC(nvnCommandBufferBeginRecording);
PROC(nvnCommandBufferEndRecording);
PROC(nvnCommandBufferGetCommandMemoryUsed);
PROC(nvnCommandBufferWaitSync);
PROC(nvnCommandBufferPushGraphicsConstantsDeferred);
PROC(nvnQueueSubmitCommands);
PROC(nvnQueueFenceSync);
PROC(nvnSyncInitialize);
#undef PROC
#define U(p) ((uint64_t)(uintptr_t)(p))
static uint64_t call(void (*fn)(Cpu *), const uint64_t args[8]) {
    Cpu c = {0};
    memcpy(c.x, args, 8 * sizeof *args);
    fn(&c);
    return c.x[0];
}
#define CALL(n, ...) call(nvnp_##n, (const uint64_t[8]){__VA_ARGS__})
static int checks, failures;
#define CHECK(c, text) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL: %s\n", text); } } while (0)

unsigned rt_trace;
// Explicit off diagnostic provider; ordering assertions exercise the real recorder/submit path.
void nvn_mode_trace(unsigned kind, uint64_t cb, uint64_t handle, uint32_t value, uint64_t submit) {
    (void)kind; (void)cb; (void)handle; (void)value; (void)submit;
}
void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_fatal(const char *fmt, ...) {
    va_list args; va_start(args, fmt); vfprintf(stderr, fmt, args); va_end(args);
    fputc('\n', stderr); exit(70);
}
uint64_t rt_call(Cpu *c, uint64_t fn, int n, ...) {
    (void)c; (void)fn; (void)n; rt_fatal("unexpected guest callback"); return 0;
}
int kern_core_away(void) { return 1; }
void kern_core_back(int *away) { *away = 0; }
uint64_t kern_now_ns(void) { return 0; }
uint64_t rt_ticks(void) { return 0; } // synthetic entry clock; this fixture checks ordering, not timestamps
uint64_t rt_ns_to_ticks(uint64_t ns) {
    (void)ns; rt_fatal("unexpected counter report in WaitSync fixture"); return 0;
}
// Unrelated object/observer entry points are linked but must never run in this fixture.
const RtConfig *rt_config(void) { rt_fatal("unexpected configuration lookup"); return NULL; }
const GfxBackend gfx_vulkan_backend = {0};
const char rt_game_target[] = "dkb100";
void nvn_pool_flushed(const pool_t *p, uint64_t off, uint64_t size) {
    (void)p; (void)off; (void)size; rt_fatal("unexpected pool flush");
}
uint64_t rt_original(Cpu *c) { (void)c; rt_fatal("unexpected guest observer"); return 0; }
int cfg_bool(const char *name) { (void)name; rt_fatal("unexpected setting lookup"); return 0; }
void plat_sleep_ms(uint32_t ms) { (void)ms; rt_fatal("unexpected sleep"); }

// Actual concurrent locks, not no-op stubs: the completion thread also acquires the recording-copy lock.
#ifdef _WIN32
_Static_assert(sizeof(PlatMutex) == sizeof(SRWLOCK) && sizeof(PlatCond) == sizeof(CONDITION_VARIABLE), "lock storage");
void plat_mutex_lock(PlatMutex *m) { AcquireSRWLockExclusive((PSRWLOCK)m); }
void plat_mutex_unlock(PlatMutex *m) { ReleaseSRWLockExclusive((PSRWLOCK)m); }
void plat_mutex_lock_shared(PlatMutex *m) { AcquireSRWLockShared((PSRWLOCK)m); }
void plat_mutex_unlock_shared(PlatMutex *m) { ReleaseSRWLockShared((PSRWLOCK)m); }
void plat_cond_wait(PlatCond *c, PlatMutex *m) {
    if (!SleepConditionVariableSRW((PCONDITION_VARIABLE)c, (PSRWLOCK)m, INFINITE, 0)) abort();
}
void plat_cond_broadcast(PlatCond *c) { WakeAllConditionVariable((PCONDITION_VARIABLE)c); }
typedef HANDLE Thread;
#define THREAD_RETURN DWORD WINAPI
#define THREAD_DONE 0
static Thread start(LPTHREAD_START_ROUTINE fn, void *arg) {
    Thread t = CreateThread(NULL, 0, fn, arg, 0, NULL); if (!t) abort(); return t;
}
static void join(Thread t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
#else
static void *lazy(void *volatile *slot, int cond) {
    void *p = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (p) return p;
    void *fresh = malloc(cond ? sizeof(pthread_cond_t) : sizeof(pthread_mutex_t));
    if (!fresh) abort();
    if (cond) pthread_cond_init(fresh, NULL); else pthread_mutex_init(fresh, NULL);
    if (__atomic_compare_exchange_n(slot, &p, fresh, 0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) return fresh;
    if (cond) pthread_cond_destroy(fresh); else pthread_mutex_destroy(fresh);
    free(fresh); return p;
}
void plat_mutex_lock(PlatMutex *m) { pthread_mutex_lock(lazy(&m->p, 0)); }
void plat_mutex_unlock(PlatMutex *m) { pthread_mutex_unlock(m->p); }
void plat_mutex_lock_shared(PlatMutex *m) { plat_mutex_lock(m); }
void plat_mutex_unlock_shared(PlatMutex *m) { plat_mutex_unlock(m); }
void plat_cond_wait(PlatCond *c, PlatMutex *m) { pthread_cond_wait(lazy(&c->p, 1), m->p); }
void plat_cond_broadcast(PlatCond *c) { pthread_cond_broadcast(lazy(&c->p, 1)); }
typedef pthread_t Thread;
#define THREAD_RETURN void *
#define THREAD_DONE NULL
static Thread start(void *(*fn)(void *), void *arg) {
    Thread t; if (pthread_create(&t, NULL, fn, arg)) abort(); return t;
}
static void join(Thread t) { pthread_join(t, NULL); }
#endif

static PlatMutex latch = PLAT_MUTEX_INIT;
static PlatCond changed = PLAT_COND_INIT;
static int async_mode, producer_call, callback_release, callback_done, wait_entered, consumer_entered;
static uint64_t issued, gpu_reached, done, replayed, target, waited, producer_no;
static uint32_t source, next_source, captured[2];
static uint64_t batch[2], queued[2];
static int backend_copy_lock_available;
static const p_push_const *push(uint64_t h) {
    for (const op_hdr *op = nvn_op_first(h); op->op != OP_END; op = nvn_op_next(op))
        if (op->op == OP_PUSH_CONST) return nvn_op_payload(op);
    return NULL;
}
int gfx_replay_async(void) { return async_mode; }
uint64_t gfx_replayed_submits(void) {
    plat_mutex_lock(&latch); uint64_t n = replayed; plat_mutex_unlock(&latch); return n;
}
static uint64_t backend_done(void) {
    plat_mutex_lock(&latch); uint64_t n = done; plat_mutex_unlock(&latch); return n;
}
static int backend_wait(uint64_t value, uint64_t timeout) {
    if (timeout != UINT64_MAX) rt_fatal("WaitSync must not invent a timeout");
    plat_mutex_lock(&latch);
    wait_entered = 1; waited = value; plat_cond_broadcast(&changed);
    while (done < value) plat_cond_wait(&changed, &latch);
    plat_mutex_unlock(&latch);
    return 1;
}
static uint64_t backend_submit(const uint64_t *handles, int n, uint64_t submitted_ticks) {
    (void)submitted_ticks;
    plat_mutex_lock(&latch);
    uint64_t f = ++issued;
    if (producer_call) {
        producer_no = nvn_submit_count();
        gpu_reached = f;              // GPU signal precedes the deliberately latched CPU completion callback
    } else {
        if (n != 2) abort();
        for (int i = 0; i < 2; i++) {
            queued[i] = handles[i];
            const p_push_const *p = push(handles[i]);
            captured[i] = p ? p->w[0] : 0;
        }
        consumer_entered = 1;
        plat_cond_broadcast(&changed);
    }
    plat_mutex_unlock(&latch);
    return f;
}
static const GfxBackend backend = {.submit=backend_submit, .fence_done=backend_done, .fence_wait=backend_wait};
const GfxBackend *gfx_backend(void) { return &backend; }

static THREAD_RETURN complete_producer(void *arg) {
    (void)arg;
    plat_mutex_lock(&latch);
    while (!callback_release) plat_cond_wait(&changed, &latch);
    plat_mutex_unlock(&latch);
    // This models an already-enqueued replay/completion path, which never takes the front submit_lock.
    // It must be able to acquire copy_lock while the consumer waits.
    int live = nvn_handle_live(batch[0]);
    plat_mutex_lock(&latch);
    backend_copy_lock_available = live;
    source = next_source;             // producer CPU completion callback
    callback_done = 1;
    replayed = producer_no;
    done = target;                   // publish only after callback, as vk_sync completion_main does
    plat_cond_broadcast(&changed);
    plat_mutex_unlock(&latch);
    return THREAD_DONE;
}
static THREAD_RETURN submit_consumer(void *arg) {
    CALL(nvnQueueSubmitCommands, U(arg), 2, U(batch));
    return THREAD_DONE;
}

static size_t page_size;
static sync_t *sync_page(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); page_size = si.dwPageSize;
    void *p = VirtualAlloc(NULL, page_size, MEM_RESERVE|MEM_COMMIT, PAGE_NOACCESS);
    if (!p) abort();
#else
    page_size = (size_t)sysconf(_SC_PAGESIZE);
    void *p = mmap(NULL, page_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) abort();
#endif
    return p;
}
static void protect(sync_t *s, int readable) {
#ifdef _WIN32
    DWORD old; if (!VirtualProtect(s, page_size, readable?PAGE_READWRITE:PAGE_NOACCESS, &old)) abort();
#else
    if (mprotect(s, page_size, readable?(PROT_READ|PROT_WRITE):PROT_NONE)) abort();
#endif
}
static void free_sync(sync_t *s) {
#ifdef _WIN32
    VirtualFree(s, 0, MEM_RELEASE);
#else
    munmap(s, page_size);
#endif
}
static void begin(NVNcommandBuffer *cmd, void *ctl) {
    CALL(nvnCommandBufferInitialize, U(cmd), 0);
    CALL(nvnCommandBufferAddCommandMemory, U(cmd), 0, 0, 0x10000);
    CALL(nvnCommandBufferAddControlMemory, U(cmd), U(ctl), 0x1000);
    CALL(nvnCommandBufferBeginRecording, U(cmd));
}
static void scenario(int async) {
    async_mode = async;
    NVNqueue queue = {0};
    NVNcommandBuffer cmds[2] = {0};
    _Alignas(16) uint8_t control[2][0x1000];
    sync_t *s = sync_page();
    source = 0x11111111;
    // A late wait must precede source reads in *all* handles, even the first handle's push.
    for (int i = 0; i < 2; i++) {
        begin(&cmds[i], control[i]);
        if (i == 1) {
            CALL(nvnCommandBufferWaitSync, U(&cmds[i]), U(s)); // source is NOACCESS
            CHECK(CALL(nvnCommandBufferGetCommandMemoryUsed, U(&cmds[i])) == 0,
                  "CPU control wait does not invent record-time GPU command memory");
        }
        CALL(nvnCommandBufferPushGraphicsConstantsDeferred, U(&cmds[i]), 0, 0, 4, U(&source));
        batch[i] = CALL(nvnCommandBufferEndRecording, U(&cmds[i]));
    }
    protect(s, 1);
    CALL(nvnSyncInitialize, U(s), 0);
    uint64_t previous_queued[2] = {0};
    uint32_t previous_value = 0;
    for (int round = 0; round < 2; round++) {
        // Real QueueFenceSync associates the sync with the backend value returned for this producer.
        producer_call = 1;
        CALL(nvnQueueSubmitCommands, U(&queue), 0, 0);
        CALL(nvnQueueFenceSync, U(&queue), U(s), NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
        producer_call = 0;
        target = s->fence;
        uint32_t before[2] = {push(batch[0])->w[0], push(batch[1])->w[0]};
        plat_mutex_lock(&latch);
        callback_release = callback_done = wait_entered = consumer_entered = backend_copy_lock_available = 0;
        waited = 0;
        next_source = 0x22222222u + (unsigned)round;
        plat_mutex_unlock(&latch);
        Thread completion = start(complete_producer, NULL);
        Thread consumer = start(submit_consumer, &queue);
        plat_mutex_lock(&latch);
        while (!wait_entered && !consumer_entered) plat_cond_wait(&changed, &latch);
        CHECK(gpu_reached >= target && done < target, "GPU timeline is reached while host completion is still pending");
        CHECK(wait_entered && waited == target, "wait reads the current QueueFenceSync value at submit");
        CHECK(!consumer_entered && !callback_done, "raw GPU reach does not release consumer before CPU callback");
        CHECK(push(batch[0])->w[0] == before[0] && push(batch[1])->w[0] == before[1],
              "no deferred snapshot in an earlier handle before the late wait");
        callback_release = 1;
        plat_cond_broadcast(&changed);
        plat_mutex_unlock(&latch);
        join(completion); join(consumer);
        CHECK(backend_copy_lock_available, "producer completion can acquire copy_lock while consumer waits");
        CHECK(consumer_entered && captured[0] == next_source && captured[1] == next_source,
              "all consumer snapshots observe completed producer writes");
        if (round && async) {
            CHECK(queued[0] != previous_queued[0], "pending asynchronous resubmit clones deferred data");
            CHECK(push(previous_queued[0])->w[0] == previous_value, "earlier queued snapshot stays immutable");
        }
        previous_queued[0] = queued[0]; previous_queued[1] = queued[1];
        previous_value = captured[0];
        protect(s, 0);
        gfx_null_backend.submit(queued, 2, 0); // actual backend traversal must never reread the pointer
        protect(s, 1);
    }
    // An initialized but unfenced sync emits no SDK wait; window acquire's fence0 is already complete.
    for (int acquired = 0; acquired < 2; acquired++) {
        CALL(nvnSyncInitialize, U(s), 0);
        if (acquired) s->fenced = 3;
        wait_entered = 0;
        source = 0x33333333;
        if (async) nvn_submit_prepare_async(batch, queued, 2); else nvn_submit_prepare(batch, 2);
        CHECK(!wait_entered, "unfenced and acquired syncs need no backend wait");
        CHECK(push(async?queued[0]:batch[0])->w[0] == source, "satisfied wait still refreshes deferred data");
    }
    plat_mutex_lock(&latch); replayed = nvn_submit_count(); plat_mutex_unlock(&latch);
    for (int i = 0; i < 2; i++) {
        CALL(nvnCommandBufferAddControlMemory, U(&cmds[i]), U(control[i]), sizeof control[i]);
        CALL(nvnCommandBufferFinalize, U(&cmds[i]));
    }
    free_sync(s);
}
int main(void) {
    scenario(0);
    scenario(1);
    printf("WaitSync: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
