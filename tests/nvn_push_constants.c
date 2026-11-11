// Synthetic command-recorder checks. No modules, renderer or game data are needed.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "nvn_api.h"
#include "platform/platform.h"

#define PROC(n) void nvnp_##n(Cpu *c)
PROC(nvnCommandBufferInitialize);
PROC(nvnCommandBufferFinalize);
PROC(nvnCommandBufferAddCommandMemory);
PROC(nvnCommandBufferAddControlMemory);
PROC(nvnCommandBufferBeginRecording);
PROC(nvnCommandBufferEndRecording);
PROC(nvnCommandBufferPushGraphicsConstants);
PROC(nvnCommandBufferPushComputeConstants);
PROC(nvnCommandBufferPushGraphicsConstantsDeferred);
PROC(nvnCommandBufferPushComputeConstantsDeferred);
#undef PROC

static int checks, failures;
#define CHECK(cond, text) do { checks++; if (!(cond)) { failures++; fprintf(stderr, "FAIL: %s\n", text); } } while (0)
#define U(p) ((uint64_t)(uintptr_t)(p))
static uint64_t call(void (*fn)(Cpu *), const uint64_t args[8]) {
    Cpu c = {0};
    memcpy(c.x, args, sizeof c.x[0] * 8);
    fn(&c);
    return c.x[0];
}
#define CALL(n, ...) call(nvnp_##n, (const uint64_t[8]){__VA_ARGS__})

// Only the recorder and submit resolver are linked. These replace unrelated host services; the test is single-threaded.
unsigned rt_trace;
static uint64_t replayed;
void nvn_called(uint8_t *seen, const char *name) { (void)seen; (void)name; }
// This fixture tests recording, not diagnostics; the real trace has its own concurrent packet.
void nvn_mode_trace(unsigned kind, uint64_t cb, uint64_t handle, uint32_t value, uint64_t submit) {
    (void)kind; (void)cb; (void)handle; (void)value; (void)submit;
}
// DKB_HDR_TRACE is off in this fixture.
int hdr_trace_hit(const Cpu *c, const char *name) { (void)c; (void)name; return 0; }
void hdr_trace_dump(const char *name, const void *p) { (void)name; (void)p; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }
void plat_mutex_lock_shared(PlatMutex *m) { (void)m; }
void plat_mutex_unlock_shared(PlatMutex *m) { (void)m; }
uint64_t gfx_replayed_submits(void) { return replayed; }
const GfxBackend *gfx_backend(void) { rt_fatal("unexpected sync wait in push-constant fixture"); return NULL; }
void rt_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}
void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    abort();
}
uint64_t rt_call(Cpu *c, uint64_t fn, int n, ...) {
    (void)c; (void)fn; (void)n;
    rt_fatal("unexpected guest callback in synthetic recorder test");
}

static size_t page_size;
static uint8_t *guarded_source(void) {
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    page_size = info.dwPageSize;
    void *p = VirtualAlloc(NULL, 2 * page_size, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    if (!p) rt_fatal("VirtualAlloc failed");
#else
    page_size = (size_t)sysconf(_SC_PAGESIZE);
    void *p = mmap(NULL, 2 * page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) rt_fatal("mmap failed");
#endif
    return p;
}
static void source_writable(uint8_t *p) {
#ifdef _WIN32
    DWORD old;
    if (!VirtualProtect(p, page_size, PAGE_READWRITE, &old)) rt_fatal("VirtualProtect failed");
#else
    if (mprotect(p, page_size, PROT_READ | PROT_WRITE)) rt_fatal("mprotect failed");
#endif
}
static void source_free(uint8_t *p) {
#ifdef _WIN32
    if (!VirtualFree(p, 0, MEM_RELEASE)) rt_fatal("VirtualFree failed");
#else
    if (munmap(p, 2 * page_size)) rt_fatal("munmap failed");
#endif
}

static void begin(NVNcommandBuffer *cmd, void *control) {
    CALL(nvnCommandBufferInitialize, U(cmd), 0);
    CALL(nvnCommandBufferAddCommandMemory, U(cmd), 0, 0, 0x10000);
    CALL(nvnCommandBufferAddControlMemory, U(cmd), U(control), 0x1000);
    CALL(nvnCommandBufferBeginRecording, U(cmd));
}
static const p_push_const *push_at(uint64_t handle, int index) {
    for (const op_hdr *h = nvn_op_first(handle); h->op != OP_END; h = nvn_op_next(h))
        if (h->op == OP_PUSH_CONST && index-- == 0) return nvn_op_payload(h);
    return NULL;
}
static void finish(NVNcommandBuffer *cmd, void *control, uint64_t handle) {
    replayed = nvn_submit_count();
    CALL(nvnCommandBufferAddControlMemory, U(cmd), U(control), 0x1000);
    CHECK(!nvn_handle_live(handle), "control-memory reuse retires the recording");
    CALL(nvnCommandBufferFinalize, U(cmd));
}

static void deferred_snapshot(int compute) {
    NVNcommandBuffer cmd = {0};
    _Alignas(16) uint8_t control[0x1000];
    uint8_t *pages = guarded_source();
    uint32_t *source = (uint32_t *)(void *)(pages + page_size - 4);
    uint32_t immediate = 0x12345678;
    begin(&cmd, control);
    // Neither page is readable yet. Recording must retain the direct pointer without reading it.
    if (compute) CALL(nvnCommandBufferPushComputeConstantsDeferred, U(&cmd), 9, 0x10, 7, U(source));
    else CALL(nvnCommandBufferPushGraphicsConstantsDeferred, U(&cmd), 9, 0x10, 7, U(source));
    if (compute) CALL(nvnCommandBufferPushComputeConstants, U(&cmd), 0, 0, 4, U(&immediate));
    else CALL(nvnCommandBufferPushGraphicsConstants, U(&cmd), 0, 0, 4, U(&immediate));
    uint64_t handle = CALL(nvnCommandBufferEndRecording, U(&cmd));
    source_writable(pages);
    *source = 0x11112222;
    immediate = 0xffffffff;

    uint64_t first, second;
    nvn_submit_prepare_async(&handle, &first, 1);
    const p_push_const *a = push_at(first, 0), *imm = push_at(first, 1);
    CHECK(a && a->compute == compute && a->table == 1 && a->offset == 0x10 && a->bytes == 4 && a->w[0] == *source,
          "deferred five-argument push reads whole words at submit");
    CHECK(imm && imm->compute == compute && imm->table == 0 && imm->offset == 0 && imm->bytes == 4 &&
          imm->w[0] == 0x12345678, "immediate push keeps its record-time value");
    *source = 0x33334444;
    nvn_submit_prepare_async(&handle, &second, 1);
    const p_push_const *b = push_at(second, 0);
    CHECK(first != second, "resubmitting a pending deferred recording makes a snapshot");
    CHECK(a && a->w[0] == 0x11112222, "next submit preserves the first queued submission");
    CHECK(b && b->compute == compute && b->table == 1 && b->w[0] == 0x33334444,
          "second submit reads the new source value");
    *source = 0x55556666;
    CHECK(b && b->w[0] == 0x33334444, "changing source after submit preserves its snapshot");
    replayed = nvn_submit_count();
    nvn_submit_prepare(&handle, 1);
    a = push_at(handle, 0);
    CHECK(a && a->w[0] == 0x55556666, "synchronous resubmit also refreshes deferred data");
    finish(&cmd, control, handle);
    source_free(pages);
}

static void bounds(int compute) {
    // Compute's deferred driver helper does not clamp: these checks preserve the host's bounded root-table policy.
    NVNcommandBuffer cmd = {0};
    _Alignas(16) uint8_t control[0x1000];
    uint8_t *pages = guarded_source();
    uint32_t *source = (uint32_t *)(void *)(pages + page_size - 256);
    begin(&cmd, control);
    void (*fn)(Cpu *) = compute ? nvnp_nvnCommandBufferPushComputeConstantsDeferred :
                                 nvnp_nvnCommandBufferPushGraphicsConstantsDeferred;
    call(fn, (const uint64_t[8]){U(&cmd), 0, 0, 300, U(source)});
    // These must not create commands or access their deliberately unreadable source.
    call(fn, (const uint64_t[8]){U(&cmd), 0, 0, 3, U(source)});
    call(fn, (const uint64_t[8]){U(&cmd), 0, 2, 4, U(source)});
    call(fn, (const uint64_t[8]){U(&cmd), 0, 252, 8, U(source)});
    uint64_t handle = CALL(nvnCommandBufferEndRecording, U(&cmd));
    source_writable(pages);
    for (unsigned i = 0; i < 64; i++) source[i] = 0xa0000000u + i;
    nvn_submit_prepare(&handle, 1);
    const p_push_const *p = push_at(handle, 0);
    CHECK(p && p->compute == compute && p->bytes == 256 && !memcmp(p->w, source, 256),
          "deferred copy clamps to the 256-byte table before reading source");
    CHECK(!push_at(handle, 1), "sub-word, unaligned and out-of-table pushes are dropped");
    finish(&cmd, control, handle);
    source_free(pages);
}

int main(void) {
    for (int compute = 0; compute <= 1; compute++) {
        deferred_snapshot(compute);
        bounds(compute);
    }
    printf("push constants: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
