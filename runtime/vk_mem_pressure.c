// SPDX-License-Identifier: GPL-3.0-or-later
// Linux memory pressure: the renderer gives the heap's freed pages back to the system when the machine runs short of
// memory, and once after the shader library prebuild. The rule (linux_pressure.h) and this use of it are Hookshot's
// (gpu/vk/vk_memory.cpp, and gpu/vk/vk_draw.cpp pipeline_cache_play for the precompile, commit 58ec158c "give memory
// back under Linux memory pressure and after the precompile", GPL-3.0-or-later), written again in C with what this
// renderer can give back. Hookshot also trims spare capture buffers, idle guest-pool imports and the texture budget,
// which here are either not kept (uploads free their staging when their batch retires), not safe to take back (a guest
// pool is the game's memory and the GPU resolves into it), or another file's (vk_texture.c).
//
// The renderer's own device heap can show room while the system is out of memory: on a machine with 16 GB shared
// between the CPU and the GPU every VMA block, staging block and pinned guest pool is physical memory that cannot be
// swapped. vk_mem_collect calls vk_mem_pressure_check after each collection; once a second at most it reads
// /proc/meminfo (plat_mem_avail) and compares it with the two reserves. In a shortfall it gives the heap's freed pages
// back (plat_mem_trim, glibc malloc_trim, which also reaches the arenas of the translator and prebuild threads that
// came and went), on a thread of its own. Nothing is read or done without a shortfall beyond one clock read per
// collection and the /proc/meminfo read once a second; where plat_mem_avail has nothing to read (every host but Linux)
// the check turns itself off at its first call.
//
// DKB_MEM_AVAIL_RESERVE_MB (1536) and DKB_MEM_COMMIT_RESERVE_MB (2560): the reserves, 0 turns that check off;
// DKB_MEM_TRIM=0 turns all of it off. In dkb.ini: mem_avail_reserve_mb, mem_commit_reserve_mb, mem_trim.
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "linux_pressure.h"
#include "vk.h"
#include "platform/platform.h"

static int trim_off(void) {
    const char *off = cfg_get("TOTK_MEM_TRIM");   // DKB_MEM_TRIM in the environment, or dkb.ini's mem_trim
    return off && !strcmp(off, "0");
}

// ---- under pressure

static PlatMutex press_lock = PLAT_MUTEX_INIT;
static uint64_t press_next_ms;          // under press_lock, read without it as a first look
static int press_state;                 // under press_lock: 0 not set up, 1 on, -1 off
static uint64_t press_avail_res, press_commit_res;
static LpEpisode press_ep;
static uint32_t press_gives;            // times it gave back
static int press_trimming;              // a give-back thread runs

typedef struct { uint64_t over, avail, swap_free; uint32_t n; } GiveBack;

// malloc_trim walks every arena, so it runs on a thread of its own rather than on the caller (the completion thread,
// or a guest thread freeing a pool), and at normal priority: it holds each arena's lock while it trims it.
static void give_back(void *arg) {
    GiveBack g = *(GiveBack *)arg;
    free(arg);
    float used, total, before, after;
    plat_mem_info(&used, &total, &before);   // the last is this process's working set
    uint64_t t0 = plat_now_ns();
    int freed = plat_mem_trim();
    double ms = (double)(plat_now_ns() - t0) / 1e6;
    plat_mem_info(&used, &total, &after);
    rt_log("vk: memory pressure: %.0f MB short of the reserves (%.0f MB available, %.0f MB swap free); the heap's "
           "freed pages %s in %.1f ms, working set %.0f -> %.0f MB (%u times)", (double)g.over / 1048576.0,
           (double)g.avail / 1048576.0, (double)g.swap_free / 1048576.0,
           freed ? "went back to the system" : "had nothing to return", ms, (double)before, (double)after, g.n);
    __atomic_store_n(&press_trimming, 0, __ATOMIC_RELEASE);
}

void vk_mem_pressure_check(void) {
    if (__atomic_load_n(&press_state, __ATOMIC_RELAXED) < 0) return;
    uint64_t now = plat_now_ms();
    if (now < __atomic_load_n(&press_next_ms, __ATOMIC_RELAXED)) return;
    if (!plat_mutex_trylock(&press_lock)) return;   // another thread is sampling
    uint64_t avail, swap_free, total;
    int has_swap = 1;
    if (press_state == 0) {
        // TOTK_ names, so that dkb.ini's keys count as well; DKB_MEM_TRIM and the others in the environment still win
        press_avail_res = lp_reserve_bytes(cfg_get("TOTK_MEM_AVAIL_RESERVE_MB"), LP_AVAIL_RESERVE_MB);
        press_commit_res = lp_reserve_bytes(cfg_get("TOTK_MEM_COMMIT_RESERVE_MB"), LP_COMMIT_RESERVE_MB);
        int on = !trim_off() && (press_avail_res || press_commit_res) &&
                 plat_mem_avail(&avail, &swap_free, &total, &has_swap);
        __atomic_store_n(&press_state, on ? 1 : -1, __ATOMIC_RELAXED);
    }
    if (press_state > 0 && lp_sample_due(now, &press_next_ms) &&
        plat_mem_avail(&avail, &swap_free, &total, &has_swap)) {
        uint64_t over = lp_shortfall(avail, swap_free, press_avail_res, press_commit_res, has_swap);
        if (lp_episode_step(&press_ep, over, now) && !__atomic_exchange_n(&press_trimming, 1, __ATOMIC_ACQ_REL)) {
            GiveBack *g = malloc(sizeof *g);
            if (g) *g = (GiveBack){over, avail, swap_free, ++press_gives};
            if (!g || !plat_thread_spawn(give_back, g, "dkb mem trim", 0)) {
                free(g);
                __atomic_store_n(&press_trimming, 0, __ATOMIC_RELEASE);
            }
        }
    }
    plat_mutex_unlock(&press_lock);
}

// ---- after the precompile

// The precompile built its libraries on threads that are gone or idle now, and freed their shader copies and keys:
// glibc keeps those pages in the threads' arenas, which a 16 GB machine feels as swap. This gives them back once, on
// the thread that finished the precompile (a prebuild worker or --prepare-shaders), Linux only (plat_mem_trim does
// nothing elsewhere). Hookshot also drops the VkShaderModules of built programs there, but here a module lives only
// inside the build that makes it and the shader cache keeps no module to drop.
void vk_mem_trim_after(const char *what) {
    if (trim_off()) return;
    float used, total, before, after;
    plat_mem_info(&used, &total, &before);
    uint64_t t0 = plat_now_ns();
    if (!plat_mem_trim()) return;
    double ms = (double)(plat_now_ns() - t0) / 1e6;
    plat_mem_info(&used, &total, &after);
    rt_log("vk: %s: the heap's freed pages went back to the system in %.1f ms (working set %.0f MB, was %.0f MB)",
           what, ms, (double)after, (double)before);
}
