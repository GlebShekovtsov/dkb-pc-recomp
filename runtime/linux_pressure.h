// SPDX-License-Identifier: GPL-3.0-or-later
// Linux memory pressure, as a rule with nothing of Vulkan or the operating system in it (tests/linux_pressure_test.py
// compiles this file alone): the reading of /proc/meminfo, how far below its reserves the system is, and when the
// renderer gives memory back (vk_mem_pressure.c vk_mem_pressure_check). The rule and its default reserves are
// Hookshot's (gpu/vk/linux_pressure.h, commit 58ec158c "give memory back under Linux memory pressure and after the
// precompile", GPL-3.0-or-later), written again in C; the texture budget hold and the guest-pool import accounting of
// that file belong to Hookshot's renderer and have no counterpart here.
//
// Two reserves: MemAvailable (physical memory that can be had without swapping) under 1536 MB, and MemAvailable plus
// SwapFree (what can still be committed) under 2560 MB when swap exists. Either short is a shortfall; the larger of the
// two is its size. A machine that keeps both is never read more than once a second and is not touched.
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LP_AVAIL_RESERVE_MB 1536u    // DKB_MEM_AVAIL_RESERVE_MB
#define LP_COMMIT_RESERVE_MB 2560u   // DKB_MEM_COMMIT_RESERVE_MB
#define LP_CLEAR_MARGIN (256ull << 20)   // a deeper shortfall by this much is news inside an episode
#define LP_SAMPLE_MS 1000u           // /proc/meminfo is read at most this often
#define LP_RETRY_MS 30000u           // inside one episode, give back again after this long at the latest

// MemAvailable, SwapFree, MemTotal in bytes, and whether swap is configured (SwapTotal or SwapFree above 0; with no
// SwapTotal line at all it is taken as configured, which only makes the commit check stricter). 0 without MemAvailable
// (an old kernel): there is nothing to compare then. total and has_swap may be NULL.
static inline int lp_parse_meminfo(const char *text, uint64_t *avail, uint64_t *swap_free, uint64_t *total,
                                   int *has_swap) {
    uint64_t swap_total = 0;
    int have_avail = 0, saw_swap_total = 0;
    *avail = *swap_free = 0;
    if (total) *total = 0;
    for (const char *p = text; p && *p;) {
        const char *end = strchr(p, '\n');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        unsigned long long kb = 0;
        if (len >= 13 && !memcmp(p, "MemAvailable:", 13) && sscanf(p + 13, "%llu", &kb) == 1)
            *avail = (uint64_t)kb << 10, have_avail = 1;
        else if (len >= 9 && !memcmp(p, "SwapFree:", 9) && sscanf(p + 9, "%llu", &kb) == 1)
            *swap_free = (uint64_t)kb << 10;
        else if (len >= 10 && !memcmp(p, "SwapTotal:", 10) && sscanf(p + 10, "%llu", &kb) == 1)
            swap_total = (uint64_t)kb << 10, saw_swap_total = 1;
        else if (total && len >= 9 && !memcmp(p, "MemTotal:", 9) && sscanf(p + 9, "%llu", &kb) == 1)
            *total = (uint64_t)kb << 10;
        if (!end) break;
        p = end + 1;
    }
    if (has_swap) *has_swap = !saw_swap_total || swap_total > 0 || *swap_free > 0;
    return have_avail;
}

// A reserve in bytes from a setting's text (megabytes): unset or empty gives def_mb, 0 turns the check off, a value
// that is not a non-negative whole number gives def_mb.
static inline uint64_t lp_reserve_bytes(const char *text, uint32_t def_mb) {
    if (!text || !*text) return (uint64_t)def_mb << 20;
    uint64_t v = 0;
    for (const char *p = text; *p; p++) {
        if (*p < '0' || *p > '9' || v > (UINT64_MAX >> 21)) return (uint64_t)def_mb << 20;
        v = v * 10 + (uint64_t)(*p - '0');
    }
    return v << 20;
}

// How many bytes the system is short of its reserves: the larger of the physical shortfall (MemAvailable under
// avail_reserve) and, with swap, the commit shortfall (MemAvailable + SwapFree under commit_reserve). 0 when both are
// kept, or turned off with a reserve of 0.
static inline uint64_t lp_shortfall(uint64_t avail, uint64_t swap_free, uint64_t avail_reserve, uint64_t commit_reserve,
                                    int has_swap) {
    uint64_t a = avail_reserve > avail ? avail_reserve - avail : 0;
    uint64_t total = avail <= UINT64_MAX - swap_free ? avail + swap_free : UINT64_MAX;
    uint64_t c = has_swap && commit_reserve > total ? commit_reserve - total : 0;
    return a > c ? a : c;
}

// A pressure episode runs from the first sample with a shortfall to the first without. Within it the renderer gives
// memory back at the start, again when the shortfall has grown by LP_CLEAR_MARGIN past its deepest, and at the latest
// every LP_RETRY_MS, so a system that hovers just under its reserve is not trimmed at every sample.
typedef struct {
    int active;
    uint64_t deepest;     // bytes
    uint64_t last_ms;     // when it last gave back
} LpEpisode;

// 1: give back now.
static inline int lp_episode_step(LpEpisode *e, uint64_t shortfall, uint64_t now_ms) {
    if (!shortfall) {
        e->active = 0, e->deepest = 0;
        return 0;
    }
    if (!e->active) {
        e->active = 1, e->deepest = shortfall, e->last_ms = now_ms;
        return 1;
    }
    int grew = shortfall >= e->deepest + LP_CLEAR_MARGIN;
    if (shortfall > e->deepest) e->deepest = shortfall;
    if (grew || now_ms - e->last_ms >= LP_RETRY_MS) {
        e->last_ms = now_ms;
        return 1;
    }
    return 0;
}

// 1 when a sample is due (and *next_ms moves on); the first call is always due.
static inline int lp_sample_due(uint64_t now_ms, uint64_t *next_ms) {
    if (*next_ms && now_ms < *next_ms) return 0;
    *next_ms = now_ms + LP_SAMPLE_MS;
    return 1;
}
