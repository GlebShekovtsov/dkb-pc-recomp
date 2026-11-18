// --probe-gfx-abi: every host object of nvn_obj.h fits the storage nvn.h reserves
// for its NVN type, the layouts the game reads directly (NVNtextureView, the command buffer's inline fast-path
// fields) sit where TOTK's NVN 55 code expects them, and the GPU address window lies above every guest region.
// The same facts are static assertions in nvn_obj.h; this mode prints the sizes so a change shows its margin.
// It also checks the write watch, the frozen regions (memory.c mem_freeze) and, on Windows, the GPU-owned pages of the
// staged mirror against them (check_guards). Needs no game files.
#include <stddef.h>
#include <stdio.h>

#include "gfx.h"
#include "platform/platform.h"

static int failures, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__); putchar('\n'); } } while (0)

typedef struct { const char *host, *nvn; size_t size, reserved; } Fit;
#define FIT(T, H) {#T, #H, sizeof(T), sizeof(H)}
static const Fit fits[] = {
    FIT(dev_builder_t, NVNdeviceBuilder),       FIT(dev_t_, NVNdevice),
    FIT(queue_builder_t, NVNqueueBuilder),      FIT(queue_t, NVNqueue),
    FIT(pool_builder_t, NVNmemoryPoolBuilder),  FIT(pool_t, NVNmemoryPool),
    FIT(dpool_t, NVNtexturePool),               FIT(dpool_t, NVNsamplerPool),
    FIT(tex_builder_t, NVNtextureBuilder),      FIT(tex_t, NVNtexture),
    FIT(view_t, NVNtextureView),
    FIT(samp_builder_t, NVNsamplerBuilder),     FIT(samp_t, NVNsampler),
    FIT(prog_t, NVNprogram),
    FIT(sync_t, NVNsync),
    FIT(ev_builder_t, NVNeventBuilder),         FIT(ev_t, NVNevent),
    FIT(win_builder_t, NVNwindowBuilder),       FIT(win_t, NVNwindow),
    FIT(cmd_t, NVNcommandBuffer),
    FIT(color_st, NVNcolorState),               FIT(chmask_st, NVNchannelMaskState),
    FIT(blend_st, NVNblendState),               FIT(ms_st, NVNmultisampleState),
    FIT(poly_st, NVNpolygonState),              FIT(ds_st, NVNdepthStencilState),
    FIT(va_st, NVNvertexAttribState),           FIT(vs_st, NVNvertexStreamState),
};

_Static_assert(sizeof(NVNtextureView) == 16, "NVNtextureView is 16 bytes in TOTK's NVN 55");
_Static_assert(offsetof(cmd_t, inline_cur) == 0x20 && offsetof(cmd_t, inline_end) == 0x28, "cmd_t inline fields");
_Static_assert(RT_GPU_VA_BASE >= RT_ADDRESS_LIMIT, "GPU address window overlaps the guest address space");

// Write watch: a committed heap page and an alias page, written once, are reported once.
static void check_write_watch(void) {
    const RtGuestLayout *g = mem_layout();
    uint64_t heap = 0, pages[8], n;
    CHECK(mem_set_heap_size(0x200000, &heap) == 0 && heap == g->heap.base, "SetHeapSize 2 MiB");
    // With TOTK_HEAP_LAZY (the Windows default) the new heap is only reserved, and memory.c freeze_locked never freezes
    // a heap region with reserved pages. Commit it the way vk_memory.c does before the driver imports guest memory, so
    // the region below is the kind that gets frozen: committed and idle.
    mem_touch(heap, 0x200000);
    mem_written_pages(heap, 0x200000, pages, 8);   // drop whatever the commit itself marked
    volatile uint8_t *p = (volatile uint8_t *)(uintptr_t)heap;
    p[0x5000] = 1;
    n = mem_written_pages(heap, 0x200000, pages, 8);
    CHECK(n == 1 && pages[0] == heap + 0x5000, "heap write: %llu pages reported (want 1 at %#llx)",
          (unsigned long long)n, (unsigned long long)(heap + 0x5000));
    printf("write-watch: heap page written once: %llu reported at +%#llx\n", (unsigned long long)n,
           n ? (unsigned long long)(pages[0] - heap) : 0ull);
    n = mem_written_pages(heap, 0x200000, pages, 8);
    CHECK(n == 0, "second call reported %llu pages (want 0)", (unsigned long long)n);
    printf("write-watch: second call: %llu reported\n", (unsigned long long)n);
    p[0x7000] = 2;
    p[0x9000] = 3;
    p[0xA000] = 4;
    n = mem_written_pages(heap, 0x200000, pages, 2);
    uint64_t m = mem_written_pages(heap, 0x200000, pages + 2, 6);
    printf("write-watch: three pages written, max 2: %llu then %llu\n", (unsigned long long)n, (unsigned long long)m);
    CHECK(n == 2 && m == 1, "max 2 reported %llu, then %llu (want 2, 1)", (unsigned long long)n, (unsigned long long)m);
    // In address order: vk_memory.c ww_stamp starts its next query after the last page a full answer held.
    CHECK(pages[0] == heap + 0x7000 && pages[1] == heap + 0x9000 && pages[2] == heap + 0xA000,
          "pages not in address order: +%#llx +%#llx then +%#llx", (unsigned long long)(pages[0] - heap),
          (unsigned long long)(pages[1] - heap), (unsigned long long)(pages[2] - heap));

    // Frozen region: read-only until the first write thaws it (the write lands and is reported); mem_thaw (a file
    // read) thaws it and holds it thawed; SetHeapSize below decommits it.
    mem_written_pages(heap, 0x200000, pages, 8);
    CHECK(mem_freeze(heap) == 1 && mem_frozen(heap), "freeze of an idle committed 2 MiB region");
    uint64_t end;
    CHECK(plat_mem_state(heap, heap + 1, &end) == PLAT_PROT_READ, "frozen region is not read-only");
    p[0x3000] = 7;
    CHECK(!mem_frozen(heap) && p[0x3000] == 7, "a write into a frozen region did not thaw it");
    n = mem_written_pages(heap, 0x200000, pages, 8);
    CHECK(n == 1 && pages[0] == heap + 0x3000, "write after a thaw: %llu pages reported (want 1)", (unsigned long long)n);
    CHECK(mem_freeze(heap) == 1, "refreeze after a write thaw");
    mem_thaw(heap + 0x6000, 0x1000);
    CHECK(!mem_frozen(heap) && mem_freeze(heap) == 0, "mem_thaw did not thaw and hold the region");
    p[0x6000] = 8;
    printf("write-watch: frozen region thawed by a write (reported once) and by mem_thaw (held)\n");

    // Frozen regions committed throughout thaw with protection changes only, no state query (memory.c thaw_locked):
    // on Windows a query reports the whole frozen run above the address, and a layer exit's thaw of a 450 MiB pool made
    // 226 of them. (Other hosts count no queries: plat_vm_stats reports zeros there.)
    CHECK(mem_set_heap_size(0x800000, &heap) == 0, "SetHeapSize 8 MiB");
    mem_touch(heap, 0x800000);
    for (uint64_t r = heap + 0x200000; r < heap + 0x800000; r += 0x200000)
        CHECK(mem_freeze(r) == 1 && mem_frozen(r), "freeze of the committed region at +%#llx", (unsigned long long)(r - heap));
    PlatVmStat vm[PLAT_VM_N];
    plat_vm_stats(vm);   // from here on
    mem_thaw(heap + 0x200000, 0x600000);
    plat_vm_stats(vm);
    CHECK(vm[PLAT_VM_STATE].calls == 0, "thawing three whole regions made %llu state queries (want 0)",
          (unsigned long long)vm[PLAT_VM_STATE].calls);
    for (uint64_t r = heap + 0x200000; r < heap + 0x800000; r += 0x200000) {
        CHECK(!mem_frozen(r), "mem_thaw left the region at +%#llx frozen", (unsigned long long)(r - heap));
        // A page left read-only crashes here: its region is no longer frozen, so the write fault is no thaw.
        *(volatile uint8_t *)(uintptr_t)(r + 0x1fffff) = 5;
    }
    printf("write-watch: three frozen regions thawed by one mem_thaw: protection changes %llu, state queries 0\n",
           (unsigned long long)vm[PLAT_VM_PROTECT].calls);

    uint64_t a = g->alias.base + 0x10000;
    CHECK(mem_map_physical(a, 0x4000) == 0, "MapPhysicalMemory");
    mem_written_pages(a, 0x4000, pages, 8);
    *(volatile uint8_t *)(uintptr_t)(a + 0x1000) = 1;
    n = mem_written_pages(a, 0x4000, pages, 8);
    CHECK(n == 1 && pages[0] == a + 0x1000, "alias write: %llu pages reported (want 1)", (unsigned long long)n);
    CHECK(mem_written_pages(a, 0x4000, pages, 8) == 0, "alias: second call reported pages");

    // A region with reserved holes freezes its committed pages only, and a write thaws them without committing a hole.
    uint64_t ra = a & ~0x1fffffull;
    CHECK(mem_freeze(ra) == 1 && mem_frozen(ra), "freeze of a 2 MiB region with reserved holes");
    CHECK(plat_mem_state(a, a + 1, &end) == PLAT_PROT_READ, "its committed pages are not read-only");
    *(volatile uint8_t *)(uintptr_t)(a + 0x2000) = 9;
    CHECK(!mem_frozen(ra) && *(volatile uint8_t *)(uintptr_t)(a + 0x2000) == 9, "a write did not thaw the region with holes");
    CHECK(plat_mem_state(a, a + 0x10000, &end) == PLAT_PROT_RW && end == a + 0x4000, "the thawed pages are not read-write");
    CHECK(plat_mem_state(a + 0x4000, a + 0x4001, &end) == PLAT_MEM_RESERVED, "a hole got committed");
    printf("write-watch: region with reserved holes frozen and thawed by a write\n");
    CHECK(mem_unmap_physical(a, 0x4000) == 0, "UnmapPhysicalMemory");
    CHECK(mem_written_pages(RT_KERNEL_BASE, 0x1000, pages, 8) == UINT64_MAX, "kernel region is not write-watched");
    CHECK(mem_set_heap_size(0, &heap) == 0, "SetHeapSize 0");
    printf("write-watch: alias page written once: reported once, then not\n");
}

#ifdef _WIN32
// GPU-owned pages (memory.c, vk_memory.c staged mirror) against frozen regions. The fetch hook stands in for
// vk_memory.c staged_fetch: it writes the "GPU's" byte into the claimed, read-write page and opens it.
static int fetches;
static uint64_t fetched;
static int fetch_hook(uint64_t page, int write, uint64_t rip, int kernel) {
    (void)write, (void)rip, (void)kernel;
    fetches++, fetched = page;
    *(volatile uint8_t *)(uintptr_t)page = 0x5a;
    mem_guard_clear(page);
    return 1;
}

static int prot_of(uint64_t a) {
    uint64_t end;
    return plat_mem_state(a, a + 0x1000, &end);
}

static void check_guards(void) {
    uint64_t heap = 0, pages[8];
    CHECK(mem_set_heap_size(0x1000000, &heap) == 0, "SetHeapSize 16 MiB");
    mem_touch(heap, 0x1000000);
    // The regions from +8 MiB: those below were thawed by mem_thaw above, which holds them unfrozen for seconds.
    uint64_t r1 = heap + 0x800000, r2 = heap + 0xa00000, r3 = heap + 0xc00000, r4 = heap + 0xe00000;
    mem_written_pages(heap, 0x1000000, pages, 8);
    mem_guard_hook(fetch_hook);

    // A page owned after its region was frozen whole: the whole-region thaw (one protection change for the 2 MiB,
    // memory.c thaw_locked) must leave it PAGE_NOACCESS, and its first read must still fault into the fetch.
    CHECK(mem_freeze(r1) == 1, "freeze of an idle committed region at +8 MiB");
    CHECK(mem_guard_own(r1 + 0x5000) == 1 && mem_guard_state(r1 + 0x5000) == 1, "guard of a page in a frozen region");
    CHECK(prot_of(r1 + 0x5000) == PLAT_PROT_NONE, "the owned page is not PAGE_NOACCESS");
    mem_thaw_keep(r1, 0x200000);
    CHECK(!mem_frozen(r1), "the thaw left the region frozen");
    CHECK(prot_of(r1 + 0x5000) == PLAT_PROT_NONE && mem_guard_state(r1 + 0x5000) == 1,
          "the whole-region thaw opened the owned page (protection %d)", prot_of(r1 + 0x5000));
    CHECK(prot_of(r1 + 0x4000) == PLAT_PROT_RW && prot_of(r1 + 0x6000) == PLAT_PROT_RW, "the thaw left its other pages read-only");
    uint8_t got = *(volatile uint8_t *)(uintptr_t)(r1 + 0x5123);
    CHECK(fetches == 1 && fetched == r1 + 0x5000 && got == 0 && *(volatile uint8_t *)(uintptr_t)(r1 + 0x5000) == 0x5a,
          "a read of the owned page after the thaw: %d fetches, byte %#x", fetches, got);
    CHECK(mem_guard_state(r1 + 0x5000) == 0 && prot_of(r1 + 0x5000) == PLAT_PROT_RW, "the fetched page is not open");
    printf("guards: a page owned in a region frozen whole stays PAGE_NOACCESS through the region's thaw, and its read "
           "faults into the fetch\n");

    // The same over a run of pages (mem_guard_own_range), the region frozen whole and thawed by a write fault.
    CHECK(mem_freeze(r4) == 1, "freeze of the region at +14 MiB");
    int failed = 0;
    CHECK(mem_guard_own_range(r4 + 0x10000, 0x3000, &failed) == 3 && !failed, "guard of a run of three pages");
    *(volatile uint8_t *)(uintptr_t)(r4 + 0x40000) = 1;   // a write elsewhere in the region thaws it (thaw_fault)
    CHECK(!mem_frozen(r4), "a write did not thaw the region with the owned run");
    for (uint64_t p = r4 + 0x10000; p < r4 + 0x13000; p += 0x1000)
        CHECK(prot_of(p) == PLAT_PROT_NONE && mem_guard_state(p) == 1, "the write's thaw opened owned page +%#llx",
              (unsigned long long)(p - r4));
    for (uint64_t p = r4 + 0x10000; p < r4 + 0x13000; p += 0x1000) (void)*(volatile uint8_t *)(uintptr_t)p;
    CHECK(fetches == 4, "reads of the owned run: %d fetches in all (want 4)", fetches);

    // A region with an owned page is not frozen; once the page is fetched, it is.
    CHECK(mem_guard_own(r2 + 0x3000) == 1, "guard of a page in an open region");
    CHECK(mem_freeze(r2) == 0 && !mem_frozen(r2), "a region with an owned page was frozen");
    (void)*(volatile uint8_t *)(uintptr_t)(r2 + 0x3000);
    CHECK(fetches == 5 && mem_freeze(r2) == 1, "after the fetch the region does not freeze (%d fetches)", fetches);
    CHECK(prot_of(r2) == PLAT_PROT_READ && prot_of(r2 + 0x3000) == PLAT_PROT_READ, "the frozen region is not read-only");
    printf("guards: a region with an owned page is refused by the freeze, and freezes once the page is fetched\n");

    // A write into a page owned in a frozen region: the fetch thaws the region first, so the write lands and the
    // write watch reports it (vk_memory.c skips frozen regions in its queries).
    mem_written_pages(heap, 0x1000000, pages, 8);
    CHECK(mem_freeze(r3) == 1, "freeze of the region at +12 MiB");
    CHECK(mem_guard_own(r3 + 0x2000) == 1, "guard of a page in the frozen region at +12 MiB");
    *(volatile uint8_t *)(uintptr_t)(r3 + 0x2001) = 7;
    CHECK(fetches == 6 && *(volatile uint8_t *)(uintptr_t)(r3 + 0x2000) == 0x5a &&
          *(volatile uint8_t *)(uintptr_t)(r3 + 0x2001) == 7, "a write into the owned page: %d fetches", fetches);
    CHECK(!mem_frozen(r3), "the fetch for a write left its region frozen");
    uint64_t n = mem_written_pages(heap, 0x1000000, pages, 8);
    int seen = 0;
    for (uint64_t i = 0; i < n && i < 8; i++) seen |= pages[i] == r3 + 0x2000;
    CHECK(seen, "the write into the fetched page is not reported (%llu pages)", (unsigned long long)n);
    printf("guards: a write into a page owned in a frozen region thaws the region and is reported\n");

    for (uint64_t a = heap; a < heap + 0x1000000; a += 0x1000)
        if (mem_guard_state(a)) CHECK(0, "page +%#llx is still guarded", (unsigned long long)(a - heap));
    mem_guard_hook(NULL);
    CHECK(mem_set_heap_size(0, &heap) == 0, "SetHeapSize 0");
}
#endif

int probe_gfx_abi(void) {
    size_t tightest = (size_t)-1;
    const char *tight = "";
    for (size_t i = 0; i < sizeof fits / sizeof *fits; i++) {
        const Fit *f = &fits[i];
        CHECK(f->size <= f->reserved, "%s is %zu bytes, %s reserves %zu", f->host, f->size, f->nvn, f->reserved);
        if (f->reserved - f->size < tightest) tightest = f->reserved - f->size, tight = f->host;
    }
    CHECK(sizeof(NVNtextureView) == 16 && sizeof(view_t) == 16, "NVNtextureView %zu bytes, view_t %zu (want 16)",
          sizeof(NVNtextureView), sizeof(view_t));
    CHECK(offsetof(cmd_t, guest_flags) == 0x10, "cmd_t.guest_flags at %#zx (want 0x10)", offsetof(cmd_t, guest_flags));
    CHECK(offsetof(cmd_t, inline_cur) == 0x20, "cmd_t.inline_cur at %#zx (want 0x20)", offsetof(cmd_t, inline_cur));
    CHECK(offsetof(cmd_t, inline_end) == 0x28, "cmd_t.inline_end at %#zx (want 0x28)", offsetof(cmd_t, inline_end));
    CHECK(sizeof(samp_par) == 44, "samp_par is %zu bytes (want 44)", sizeof(samp_par));
    CHECK(RT_GPU_VA_BASE >= RT_ADDRESS_LIMIT && RT_GPU_VA_SIZE > 0, "GPU address window %#llx overlaps the guest space",
          RT_GPU_VA_BASE);
    check_write_watch();
#ifdef _WIN32
    check_guards();   // memory.c routes the faults of guarded pages on Windows only
#endif
    printf("gfx-abi: %d layouts fit (tightest %s, %zu bytes spare), cmd_t inline fields at +0x20/+0x28, "
           "NVNtextureView %zu bytes\n", (int)(sizeof fits / sizeof *fits), tight, tightest, sizeof(NVNtextureView));
    if (failures) {
        printf("probe-gfx-abi: FAILED (%d of %d checks)\n", failures, checks);
        return 1;
    }
    printf("probe-gfx-abi: passed\n");
    return 0;
}
