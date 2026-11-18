// --probe-vkreset: original pointer lists and indirect records through the active replay and recorder lifecycle.
// GPU copies replace host-decoy list entries before reset; full-window readbacks check reset, later counts and canaries.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "vk_exec.h"

enum { POOL_BYTES = 0x10000, WINDOW_OFF = 0x1000, WINDOW_BYTES = 512, LIST_ENTRIES = 7, LIST_BYTES = 56 };
#define CANARY UINT32_C(0xa5c35a7e)
#define U(p) ((uint64_t)(uintptr_t)(p))
static unsigned checks, failures;

typedef struct { uint64_t address, size; } WriteRange;
typedef struct {
    const char *name;
    uint64_t list, target, read_target;
    VkBuffer producer, readback;
    uint64_t pointers[LIST_ENTRIES];
    WriteRange writes[4];
    unsigned nwrites;
    int reject, warm, device, applied, failed, warmed_mirror;
} ResetCase;

#define CHECK(c, text) do { checks++; if (!(c)) { failures++; printf("probe-vkreset: FAIL %s\n",text); } } while (0)

static void written(void *context) {
    const WriteRange *range = context;
    vk_gpu_written(range->address, range->size);
}

static void publish(VkExec *ex, ResetCase *test, const VkMemSpan *span, uint64_t address, uint64_t size) {
    vk_mem_gpu_wrote(span, size);
    vk_usnap_drop(address, size);
    if (nvn_gpu_only(address)) return;
    if (test->nwrites == sizeof test->writes / sizeof *test->writes) rt_fatal("probe-vkreset: publication capacity");
    WriteRange *range = &test->writes[test->nwrites++];
    *range = (WriteRange){address, size};
    vk_mem_gpu_pending(address, size);
    vk_completion_add(ex->fence, written, range);
}

static int span_of(ResetCase *test, uint64_t address, uint64_t bytes, VkMemSpan *span) {
    if (!vk_mem_resolve(address, bytes, span) || span->offset % 4) {
        printf("probe-vkreset: FAIL %s: original range is not backed/aligned\n", test->name);
        test->failed = 1;
        return 0;
    }
    return 1;
}

// VK_MEM_USE, and whether the span has a buffer after it (a staged pool with neither a mirror nor an import has none).
static int use_of(ResetCase *test, VkMemSpan *span) {
    VK_MEM_USE(*span);
    if (span->buffer) return 1;
    printf("probe-vkreset: FAIL %s: a staged pool has neither a mirror nor an import\n", test->name);
    test->failed = 1;
    return 0;
}

static int snapshot(VkExec *ex, ResetCase *test, unsigned phase) {
    VkMemSpan target;
    if (!span_of(test, test->read_target, WINDOW_BYTES, &target)) return 0;
    // This is the first indirect-consumer-like USE of the target in the cold case. A reset that wrote only the
    // imported buffer before a mirror existed leaves this consumer on stale bytes, which the exact check detects.
    if (!use_of(test, &target)) return 0;
    vk_exec_barrier(ex);
    VkBufferCopy copy = {target.offset, (VkDeviceSize)phase * WINDOW_BYTES, WINDOW_BYTES};
    vkCmdCopyBuffer(ex->cb, target.buffer, test->readback, 1, &copy);
    ex->recorded++;
    return 1;
}

static void work(VkExec *ex, void *context) {
    ResetCase *test = context;
    VkMemSpan target, list;
    if (test->device) {
        if (!span_of(test, test->read_target, WINDOW_BYTES, &target)) return;
        vk_exec_barrier(ex);
        vkCmdFillBuffer(ex->cb, target.buffer, target.offset, WINDOW_BYTES, CANARY);
        ex->recorded++;
    } else if (test->warm) {
        if (!span_of(test, test->target, WINDOW_BYTES, &target)) return;
        VkBuffer original = target.buffer;
        if (!use_of(test, &target)) return;
        test->warmed_mirror = target.buffer != original;
    }

    // LIST and TARGET are distinct pools. Resolving/using the list cannot create the target's first mirror.
    if (!span_of(test, test->list, LIST_BYTES, &list)) return;
    if (!use_of(test, &list)) return;
    vk_exec_barrier(ex);
    VkBufferCopy producer = {0, list.offset, LIST_BYTES};
    vkCmdCopyBuffer(ex->cb, test->producer, list.buffer, 1, &producer);
    ex->recorded++;
    publish(ex, test, &list, test->list, LIST_BYTES);

    // Entries 0 and 6 are outside the selection; the ordered cases include a duplicate.
    test->applied = vk_exec_reset_indirect(ex, test->list, LIST_BYTES, 1, 5);
    if (!snapshot(ex, test, 0)) return;
    if (!test->reject) {
        for (unsigned i = 0; i < 3; i++) {
            uint64_t address = test->target + 0x40 + 0x40 * i;
            if (!span_of(test, address, 4, &target)) return;
            if (!use_of(test, &target)) return;
            vk_exec_barrier(ex);
            vkCmdFillBuffer(ex->cb, target.buffer, target.offset, 4, 3);
            ex->recorded++;
            publish(ex, test, &target, address, 4);
        }
    }
    if (!snapshot(ex, test, 1)) return;
    VkMemoryBarrier2 host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1, .pMemoryBarriers = &host};
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    ex->recorded++;
}

static int record(VkCommandBuffer cb, uint64_t fence, uint64_t ticks, void *context) {
    return vk_exec_record_work(cb, fence, ticks, work, context);
}

static uint32_t expected_word(unsigned word, unsigned phase, int reject) {
    if (!reject) {
        for (unsigned i = 0; i < 3; i++) {
            unsigned first = (0x40 + 0x40 * i) / 4;
            if (word >= first && word < first + 5) {
                if (word == first + 1) return 1;
                return word == first && phase ? 3 : 0;
            }
        }
    }
    return CANARY;
}

static void compare_window(const char *name, const uint32_t *words, unsigned phase, int reject) {
    for (unsigned i = 0; i < WINDOW_BYTES / 4; i++) {
        uint32_t expected = expected_word(i, phase, reject);
        checks++;
        if (words[i] != expected) {
            if (failures < 12) printf("probe-vkreset: FAIL %s phase %u word %u: %08x expected %08x\n",
                name, phase, i, words[i], expected);
            failures++;
        }
    }
}

static int buffer(VkBufferUsageFlags usage, VkDeviceSize bytes, int write, VkBuffer *out, VmaAllocation *allocation,
                  void **mapping) {
    VkBufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
        .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo memory = {.usage = VMA_MEMORY_USAGE_AUTO,
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | (write ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT :
                                                          VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)};
    VmaAllocationInfo actual;
    VkBuffer created = VK_NULL_HANDLE;
    VmaAllocation storage = NULL;
    if (vmaCreateBuffer(vk_ctx()->vma, &info, &memory, &created, &storage, &actual) != VK_SUCCESS) return 0;
    if (!actual.pMappedData) {
        vmaDestroyBuffer(vk_ctx()->vma, created, storage);
        return 0;
    }
    *out = created;
    *allocation = storage;
    *mapping = actual.pMappedData;
    return 1;
}

static int run_case(uint8_t *heap, const char *name, int warm, int reject, int device, int disjoint) {
    unsigned before = failures;
    pool_t list_pool = {.magic = NVN_POOL_MAGIC, .flags = 0x24, .mem = heap + 0x100000,
        .size = POOL_BYTES, .game_mem = heap + 0x100000};
    pool_t target_pool = {.magic = NVN_POOL_MAGIC, .flags = 0x24, .mem = heap + 0x200000,
        .size = POOL_BYTES, .game_mem = heap + 0x200000};
    pool_t physical = {.magic = NVN_POOL_MAGIC, .flags = 0x189,
        .mem = (uint8_t *)(uintptr_t)(RT_GPU_VA_BASE + UINT64_C(0x100000000)),
        .size = POOL_BYTES, .game_mem = heap + 0x300000, .span = POOL_BYTES};
    pool_t virtual = {.magic = NVN_POOL_MAGIC, .flags = 0x2a1,
        .mem = (uint8_t *)(uintptr_t)(RT_GPU_VA_BASE + UINT64_C(0x200000000)),
        .size = POOL_BYTES, .span = POOL_BYTES};
    ResetCase test = {.name = name, .list = U(list_pool.mem + WINDOW_OFF),
        .target = U(target_pool.mem + WINDOW_OFF), .read_target = U(target_pool.mem + WINDOW_OFF),
        .reject = reject, .warm = warm, .device = device};
    VmaAllocation producer_allocation = NULL, readback_allocation = NULL;
    void *producer_map = NULL, *readback_map = NULL;
    int list_live = 0, target_live = 0, physical_live = 0, virtual_live = 0;
    for (unsigned i = 0; i < POOL_BYTES / 4; i++) {
        ((uint32_t *)list_pool.mem)[i] = CANARY;
        ((uint32_t *)target_pool.mem)[i] = CANARY;
    }
    if (!vk_pool_init(&list_pool)) { CHECK(0, "list pool initialization"); goto done; }
    list_live = 1;
    if (device) {
        if (!vk_pool_init(&physical)) { CHECK(0, "physical pool initialization"); goto done; }
        physical_live = 1;
        if (!vk_pool_init(&virtual)) { CHECK(0, "virtual pool initialization"); goto done; }
        virtual_live = 1;
        nvn_mapping_t map = {(const NVNmemoryPool *)(const void *)&physical, 0, 0, POOL_BYTES, 0, 0};
        if (!vk_pool_map_virtual(&virtual, 1, &map)) { CHECK(0, "virtual target maps physical storage"); goto done; }
        test.target = U(virtual.mem) + WINDOW_OFF;
        test.read_target = U(physical.mem) + WINDOW_OFF;
    } else {
        if (!vk_pool_init(&target_pool)) { CHECK(0, "target pool initialization"); goto done; }
        target_live = 1;
    }
    uint64_t decoys[LIST_ENTRIES] = {test.target + 0x100, test.target + 0x140, test.target + 0x180, 0,
                                    test.target + 0x140, test.target + 0x1c0, test.target + 0x120};
    memcpy((void *)(uintptr_t)test.list, decoys, sizeof decoys);
    test.pointers[0] = test.target + 0x100;
    test.pointers[1] = test.target + 0x40;
    test.pointers[2] = test.target + 0x80;
    test.pointers[3] = 0;
    test.pointers[4] = test.target + 0x40;
    test.pointers[5] = reject ? UINT64_C(0xfffffff000000000) : test.target + 0xc0;
    if (disjoint) {
        test.pointers[4] = test.pointers[5];
        test.pointers[5] = 0;
    }
    test.pointers[6] = test.target + 0x120;
    if (!buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, LIST_BYTES, 1, &test.producer, &producer_allocation, &producer_map) ||
        !buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, 2 * WINDOW_BYTES, 0, &test.readback, &readback_allocation, &readback_map)) {
        CHECK(0, "owned producer/readback allocations"); goto done;
    }
    memcpy(producer_map, test.pointers, sizeof test.pointers);
    if (vmaFlushAllocation(vk_ctx()->vma, producer_allocation, 0, LIST_BYTES) != VK_SUCCESS) {
        CHECK(0, "producer flush"); goto done;
    }
    uint64_t fence = vk_sync_submit_record(record, &test, rt_ticks());
    CHECK(fence != 0 && vk_sync_fence_wait(fence, UINT64_C(10000000000)), "original reset submission completed");
    vk_sync_finish();   // All callbacks that refer to this stack context finish before cleanup, including failures.
    CHECK(!test.failed, "all probe commands use backed aligned ranges");
    CHECK(test.applied == !reject, "reset accepts valid lists and rejects the late-invalid target");
    if (warm && vk_mirror_on) CHECK(test.warmed_mirror, "pre-existing target mirror was exercised");
    if (vmaInvalidateAllocation(vk_ctx()->vma, readback_allocation, 0, 2 * WINDOW_BYTES) != VK_SUCCESS) {
        CHECK(0, "readback invalidation"); goto done;
    }
    compare_window(name, readback_map, 0, reject);
    compare_window(name, (const uint32_t *)((uint8_t *)readback_map + WINDOW_BYTES), 1, reject);
    CHECK(!memcmp((const void *)(uintptr_t)test.list, test.pointers, sizeof test.pointers), "GPU list producer publishes its bytes to guest memory");
    uint32_t prefix, suffix;
    memcpy(&prefix, (const void *)(uintptr_t)(test.list - 4), 4);
    memcpy(&suffix, (const void *)(uintptr_t)(test.list + LIST_BYTES), 4);
    CHECK(prefix == CANARY && suffix == CANARY, "list producer/reset preserve list border canaries");
    if (!device) compare_window(name, (const uint32_t *)(target_pool.mem + WINDOW_OFF), 1, reject);

done:
    if (list_live || target_live || physical_live) vk_sync_finish();
    if (test.readback) vmaDestroyBuffer(vk_ctx()->vma, test.readback, readback_allocation);
    if (test.producer) vmaDestroyBuffer(vk_ctx()->vma, test.producer, producer_allocation);
    if (virtual_live) vk_pool_fini(&virtual);
    if (physical_live) vk_pool_fini(&physical);
    if (target_live) vk_pool_fini(&target_pool);
    if (list_live) vk_pool_fini(&list_pool);
    printf("probe-vkreset: %s: %s\n", name, failures == before ? "passed" : "FAILED");
    return failures != before;
}

void vk_exec_draw_shutdown(void);
void vk_exec_xfer_shutdown(void);

int probe_vkreset(void) {
    checks = failures = 0;
    if (gfx_backend() != &gfx_vulkan_backend) {
        printf("probe-vkreset: requires TOTK_GPU=vulkan\n");
        return 1;
    }
    uint64_t heap = 0;
    CHECK(!mem_set_heap_size(UINT64_C(16) << 20, &heap) && heap, "owned heap allocation");
    if (failures) return 1;
    vk_sync_start();
    int bad = run_case((uint8_t *)(uintptr_t)heap, "first target use", 0, 0, 0, 0);
    if (!bad) bad = run_case((uint8_t *)(uintptr_t)heap, "pre-existing target backing", 1, 0, 0, 0);
    if (!bad) bad = run_case((uint8_t *)(uintptr_t)heap, "late-invalid target is atomic rejection", 0, 1, 0, 0);
    if (!bad) bad = run_case((uint8_t *)(uintptr_t)heap, "virtual reset targets / physical readback", 0, 0, 1, 0);
    if (!bad) bad = run_case((uint8_t *)(uintptr_t)heap, "disjoint reset targets", 0, 0, 0, 1);
    vk_sync_finish();
    vk_pipeline_shutdown();
    vk_exec_draw_shutdown();
    vk_exec_xfer_shutdown();
    vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    printf("probe-vkreset: %s (%u checks, %u failures, %ld validation errors)\n",
        bad || errors ? "FAILED" : "passed", checks, failures, errors);
    printf("probe-vkreset: reset helper and transfer count ordering; shader recognition, geometry and indirect rasterization are not covered\n");
    return bad || errors;
}
