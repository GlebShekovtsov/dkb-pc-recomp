// Code patch sets (recomp/patches.py, gen.py --patches). The generator lifts both the original and the replacement
// instructions of every patched site, and rt_patch_on[set] picks one when the site runs, so a set is switched at start
// without touching code, and also while the game runs (rt_patch_switch: the frame-limit setting of the menu). The sets,
// their environment switches and defaults, and the patch page come from the generated patches.h; without it (C
// generated without --patches) there are no sets and every site runs the original.
//
// The patch page is the first 4 KiB page after main's image, inside the module window memory.c reserved: replacement
// code reaches it with one adrp, as MacNativeGames' data page after main (games/totk/runtime/game_patches.c). Its
// layout belongs to the sets that use it (frame_rate.c: +0 the 30 Hz ratio, +4 the frame step in seconds, +8 the
// step in 60 Hz ticks).
// Probe modes never call rt_patches_init: every set stays off.
#include "cfg.h"
#include "platform/platform.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dispatch.h"
#include "host.h"

#if __has_include("patches.h")
#include "patches.h"
#endif
#ifndef RT_PATCH_COUNT
#define RT_PATCH_COUNT 0
#endif

uint8_t rt_patch_on[RT_PATCH_COUNT > 0 ? RT_PATCH_COUNT : 1];   // read by the generated C at every patched site

#if RT_PATCH_COUNT
static const struct { const char *name, *env; int on; } sets[] = {
#define X(i, n, e, d) {#n, e, d},
    RT_PATCH_SETS(X)
#undef X
};
static void *page;
#endif

#if RT_PATCH_COUNT
// The switch a set gets from the settings: a set with a row in the settings table takes its default from there; the
// generated default serves the rest.
static int wanted(int i) {
    const char *e = cfg_get(sets[i].env);
    return cfg_meta(sets[i].env) ? cfg_bool(sets[i].env) : e && *e ? strcmp(e, "0") != 0 : sets[i].on;
}
#endif

int rt_patch_wanted(int set) {
#if RT_PATCH_COUNT
    if (set >= 0 && set < RT_PATCH_COUNT) return wanted(set);
#endif
    (void)set;
    return 0;
}

void rt_patches_init(void) {
#if RT_PATCH_COUNT
    if (!plat_mem_commit(RT_PATCH_PAGE, 0x1000))
        rt_fatal("patches: cannot commit the patch page at %#llx", RT_PATCH_PAGE);
    page = (void *)(uintptr_t)RT_PATCH_PAGE;
    char list[512];
    size_t n = 0;
    list[0] = 0;
    for (int i = 0; i < RT_PATCH_COUNT; i++) {
        rt_patch_on[i] = (uint8_t)wanted(i);
        n += (size_t)snprintf(list + n, sizeof list - n, "%s%s %s", i ? ", " : "", sets[i].name,
                              rt_patch_on[i] ? "on" : "off");
        if (n >= sizeof list) n = sizeof list - 1;
    }
    rt_log("patches: %s (page %#llx)", list, RT_PATCH_PAGE);
#endif
}

int rt_patch_set(const char *name) {
#if RT_PATCH_COUNT
    for (int i = 0; i < RT_PATCH_COUNT; i++)
        if (!strcmp(sets[i].name, name)) return i;
#endif
    (void)name;
    return -1;
}

int rt_patch_enabled(int set) { return set >= 0 && set < RT_PATCH_COUNT && rt_patch_on[set]; }

void rt_patch_disable(int set) {
    if (set >= 0 && set < RT_PATCH_COUNT) rt_patch_on[set] = 0;
}

void rt_patch_switch(int set, int on) {
    if (set >= 0 && set < RT_PATCH_COUNT) __atomic_store_n(&rt_patch_on[set], (uint8_t)(on != 0), __ATOMIC_RELAXED);
}

void *rt_patch_page(void) {
#if RT_PATCH_COUNT
    return page;
#else
    return NULL;
#endif
}

// ---- guards (recomp/patches.py `guard`): checks of one register before game code follows it, cheap enough to run
// once per element of a hot loop (recomp/patches-dkb100.txt, poolguard, #126). A failed check sends the code to the
// guard's target instead. They read memory only once it is known to be readable.

// A guest data pointer: inside the guest address space, past the null page and 8-byte aligned. Catches the stale
// list entries whose "pointers" are float data or non-canonical.
static int guard_ok(uint64_t p) { return p >= 0x10000 && p < RT_ADDRESS_LIMIT && !(p & 7); }

// Every failed check is counted, and the first ones are logged with the value: a guard that fires in normal play
// rather than around an area change would show that something else is wrong.
static uint64_t g_guard_fails;
static int guard_fail(const char *kind, uint64_t v, uint64_t w) {
    uint64_t n = __atomic_add_fetch(&g_guard_fails, 1, __ATOMIC_RELAXED);
    if (n <= 32 || !(n & (n - 1)))
        rt_log("patches: guard %s failed (%llu so far): %#llx %#llx", kind, (unsigned long long)n, (unsigned long long)v,
               (unsigned long long)w);
    return 0;
}

// Range checks alone pass an aligned pointer into a page nothing maps (a crash at 410df0 after the first fix), so a
// pointer must also be readable over the bytes the code reads next. The guards run once per list entry (about 20,000
// a frame on Hilltop Layer), so each thread keeps the last few pages it found readable. Those cover most entries but
// not all (60 to 70 misses a frame there), and a miss is answered from memory.c's map of the guest heap and alias
// region (mem_guest_readable) before plat_mem_readable, a system call, is asked: on Windows that VirtualQuery scanned
// the run of like pages around the pointer, up to 308 MiB of committed heap with TOTK_HEAP_LAZY=0, up to 10 ms for one
// call and 23 ms of the game threads' time a frame, and the back end's GetWriteWatch over the same reservation waited
// behind it (#317: 5.2 to 6.5 ms a frame of queries that worked 0.4 to 0.5 ms; 36 and 45 fps on Hilltop, 74 and 71
// with this map). rt_guard_stats counts the checks (in per-thread batches, so the hit path makes no shared write) and
// the misses.
#define GUARD_PAGES 16
#define GUARD_COUNT_BATCH 1024
static _Thread_local uint64_t g_good[GUARD_PAGES];
static _Thread_local unsigned g_good_next, g_checks;
static volatile int64_t g_check_total, g_miss_total;
static int guard_readable(uint64_t p, uint64_t n) {
    if (++g_checks == GUARD_COUNT_BATCH) __atomic_add_fetch(&g_check_total, GUARD_COUNT_BATCH, __ATOMIC_RELAXED), g_checks = 0;
    uint64_t first = p >> 12, last = (p + n - 1) >> 12;
    for (uint64_t pg = first; pg <= last; pg++) {
        int hit = 0;
        for (int i = 0; i < GUARD_PAGES && !hit; i++) hit = g_good[i] == pg + 1;
        if (hit) continue;
        __atomic_add_fetch(&g_miss_total, 1, __ATOMIC_RELAXED);
        int known = mem_guest_readable(pg << 12, 4096);
        if (!known || (known < 0 && !plat_mem_readable(pg << 12, 1))) return 0;
        g_good[g_good_next++ % GUARD_PAGES] = pg + 1;
    }
    return 1;
}

void rt_guard_stats(uint64_t *checks, uint64_t *misses) {
    *checks = (uint64_t)plat_xchg64(&g_check_total, 0);
    *misses = (uint64_t)plat_xchg64(&g_miss_total, 0);
}

// A pointer the code reads from next: a layer record (+0x30, +0x40, +0x44), a cell's +0x48 target, a cell's element
// array. 0x50 bytes cover the furthest field read at the guarded sites.
int rt_guard_ptr(uint64_t p) {
    if (!guard_ok(p)) return guard_fail("ptr", p, 0);
    return guard_readable(p, 0x50) ? 1 : guard_fail("ptr unmapped", p, 0);
}

// An object whose first word is a vtable in main and whose slot +0x10 points into main, as a devirtualised call
// expects before it loads that slot and compares or calls it. A zeroed object (a cell that was never constructed), a
// pointer into main's code, or an object whose vtable slot is gone fails it instead of calling through garbage.
int rt_guard_vcall(uint64_t obj) {
    if (!guard_ok(obj)) return guard_fail("vcall object", obj, 0);
    if (!guard_readable(obj, 0x50)) return guard_fail("vcall object unmapped", obj, 0);   // +0 vtable ... +0x48 (eb8 path)
    const RtLoaded *m = ldr_count() > 0 ? ldr_module(0) : NULL;
    if (!m) return 1;
    uint64_t data = m->base + m->data_off, end = m->base + m->size, text = m->base, text_end = m->base + m->text_end;
    uint64_t vt = *(const volatile uint64_t *)(uintptr_t)obj;
    if (vt < data || vt + 0x18 > end || (vt & 7)) return guard_fail("vcall vtable", obj, vt);   // in main's data
    uint64_t fn = *(const volatile uint64_t *)(uintptr_t)(vt + 0x10);
    return fn >= text && fn < text_end && !(fn & 3) ? 1 : guard_fail("vcall slot", obj, fn);   // code of main
}

// ---- the job graph builder's edge guards (#317; recomp/patches-dkb458752.txt main+0x111674 and main+0x111678,
// recomp/patches-dkb100.txt main+0x11d854 and main+0x11d858)
// The builder makes one node per job of a graph description, records node i in slot base + i of the graph's node
// table, then walks the description's edges (two u16 job indices each) and looks both ends up in that table. An edge
// whose slot the build did not fill reads null there: the description held more jobs when its edges were read than
// when its jobs were. In #317 that was a job added with the edges to and from it, by a loader thread that the console
// keeps off the main thread's core (runtime/kernel.c, svc 0x08). rt_guard_edge checks the source (x13) and the target
// (x12) of each edge before the build follows them; the failing path goes on with the next edge, as for a full link
// pool. The first failures also log what tells who changed the description (edge_report).

static uint64_t rd64(uint64_t a) { uint64_t v = 0; return plat_mem_read(a, &v, 8) ? v : 0; }
static uint32_t rd32(uint64_t a) { uint32_t v = 0; return plat_mem_read(a, &v, 4) ? v : 0; }
static uint16_t rd16(uint64_t a) { uint16_t v = 0; return plat_mem_read(a, &v, 2) ? v : 0; }

// The description the build reads: a pointer next to the graph in one of the callers' objects, whose edge array
// (+0x38, of +0x34 entries; the count is at +0x30) holds the edge being built. The callers keep that object in a
// callee-saved register (f_7000111408: x20, description at +0x20, graph at +0x30), so x19-x28 are searched.
static uint64_t edge_desc(const Cpu *c, uint64_t cur, uint64_t end, int *reg, int *off) {
    for (int r = 19; r <= 28; r++) {
        uint64_t obj = c->x[r];
        if (!guard_ok(obj)) continue;
        for (int o = 0; o <= 0x100; o += 8) {
            uint64_t d = rd64(obj + o), edges = d && guard_ok(d) ? rd64(d + 0x38) : 0;
            if (!edges || cur < edges || end <= cur || end > edges + 8ull * rd32(d + 0x34)) continue;
            *reg = r, *off = o;
            return d;
        }
    }
    return 0;
}

// Read from the guest thread's Cpu at the guard: the builder (f_70001114a0, 1.0.0 f_700011d680) holds the graph in
// x0, the first slot of this build in w8, the edge cursor in x9 and the end of the edge array in x10. Those four are
// current in the Cpu only while recomp/regs.py keeps them out of the function's C locals, as it does today; should
// that change, the report shows stale values, and every read goes through plat_mem_read, so it logs zeros rather than
// faulting.
static void edge_report(uint64_t node) {
    Cpu *c = kern_cpu();
    if (!c) return;
    uint64_t g = c->x[0], cur = c->x[9], end = c->x[10], e = rd64(cur), table = rd64(g + 0x28);
    uint32_t base = (uint32_t)c->x[8], from = rd16(e), to = rd16(e + 2), cap = rd32(g + 0x20);
    uint32_t made = rd32(g + 0xa8) - base;
    uint64_t src = base + from < cap ? rd64(table + 8ull * (base + from)) : 0;
    uint64_t dst = base + to < cap ? rd64(table + 8ull * (base + to)) : 0;
    rt_log("patches: job graph %#llx: edge %u -> %u (slots %u, %u of %u: %#llx, %#llx; guarded %#llx) with %llu edges "
           "left; this build made %u nodes from slot %u, %u slots and %u links taken in all",
           (unsigned long long)g, from, to, base + from, base + to, cap, (unsigned long long)src, (unsigned long long)dst,
           (unsigned long long)node, (unsigned long long)(end > cur ? (end - cur) / 8 : 0), made, base,
           rd32(g + 0xa8), rd32(g + 0xa4));
    int reg = 0, off = 0;
    uint64_t d = edge_desc(c, cur, end, &reg, &off);
    if (!d) {
        rt_log("patches: job graph %#llx: the description was not found next to the graph in x19-x28",
               (unsigned long long)g);
    } else {
        uint32_t nodes = rd32(d + 8), edges = rd32(d + 0x30), seen = (uint32_t)((end - rd64(d + 0x38)) / 8);
        uint32_t next = rd16(d + 0x28);
        rt_log("patches: job graph %#llx: description %#llx (x%d+%#x) has %u jobs (next index %u) and %u edges now; the "
               "build read %u jobs and %u edges: %s", (unsigned long long)g, (unsigned long long)d, reg, off, nodes,
               next, edges, made, seen,
               nodes > made ? "jobs were added while the build ran"
               : next != nodes ? "its index counter and its job count disagree"
                               : "unchanged since the build read it");
        // The last edges the build saw and any added after: the jobs they join tell which system added them.
        uint64_t arr = rd64(d + 0x38);
        char line[512];
        int n = 0;
        for (uint32_t i = seen > 8 ? seen - 8 : 0; i < edges && i < seen + 8 && n < (int)sizeof line - 32; i++) {
            uint64_t ei = rd64(arr + 8ull * i);
            n += snprintf(line + n, sizeof line - n, " %s%u:%u->%u", i == seen ? "| " : "", i, rd16(ei), rd16(ei + 2));
        }
        rt_log("patches: job graph %#llx: edges%s", (unsigned long long)g, n ? line : " -");
    }
    rt_guest_dump(c);
    kern_cores_report("patches: job graph");
}

int rt_guard_edge(uint64_t node) {
    if (guard_ok(node) && guard_readable(node, 0x30)) return 1;   // the build reads the node's +0x10 .. +0x2b
    static int reports;
    guard_fail("edge", node, 0);
    if (__atomic_add_fetch(&reports, 1, __ATOMIC_RELAXED) <= 2) edge_report(node);
    return 0;
}
