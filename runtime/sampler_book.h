// SPDX-License-Identifier: GPL-3.0-or-later
// The renderer's book of Vulkan samplers (vk_texture.c sampler_admit), without Vulkan in it so a test can drive it
// (tests/sampler_book_test.c).
//
// A driver makes only so many samplers (maxSamplerAllocationCount: NVIDIA about 4000, RADV and NVK more) and the
// renderer used to make one for each distinct game sampler state and never destroy one, then ended the run when
// vkCreateSampler failed. The book instead:
// - keeps one sampler for each distinct create info (the key; the caller builds it from the whole VkSamplerCreateInfo
//   with every field no sample reaches left at zero), so game states that differ only in fields the GPU ignores share
//   one;
// - holds at most `room` samplers, the driver's limit less a reserve for the renderer's other samplers (the
//   upscalers');
// - from three quarters of the room, retires the least recently resolved sampler before making a new one. A retired
//   sampler leaves the book at once (generation bumped, so every table that remembers a handle asks again) but is
//   destroyed only once the GPU timeline reaches the value taken when it was retired: the submission being recorded and
//   everything before it, so nothing in flight loses a sampler it draws with;
// - never retires a pinned sampler (one a caller keeps for good, the stand-ins');
// - when the samplers made and the retired ones the GPU may still use fill the room, makes nothing and answers 0: the
//   caller uses a sampler it has.
//
// The design follows Hookshot (GPL-3.0), docs/sampler-admission.md and gpu/vk/vk_draw.cpp (sampler_make) at commit
// 073e5ca8: reuse by create info, least-recently-used retirement, destruction behind the GPU timeline. This is a C
// version over nvn_hmap.h with the same numbers (a reserve of 64 or a quarter of a small limit, retirement from three
// quarters of the room).
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nvn_hmap.h"

#define SB_KEY_BYTES 80

typedef struct { uint8_t b[SB_KEY_BYTES]; } SbKey;

typedef struct SbEnt {
    SbKey key;
    uint64_t handle;
    uint64_t used;        // tick of the last resolution through the book
    uint32_t at;          // index in SamplerBook.ents
    uint8_t pinned;
} SbEnt;

typedef struct { uint64_t handle, value; } SbDead;

typedef struct SamplerBook {
    nvn_hmap by_key;      // SbKey -> SbEnt *
    SbEnt **ents;         // every live entry
    uint32_t n, cap;
    SbDead *dead;         // retired, waiting for the GPU
    uint32_t ndead, capd;
    uint32_t room, high;  // most samplers (live and retired) the book holds; the live count that starts retirement
    uint32_t gen;         // bumped by every retirement: a handle remembered under an older one is asked for again
    uint64_t tick;
    uint64_t n_made, n_reused, n_retired, n_destroyed, n_full;
} SamplerBook;

// create: makes a sampler for key, 0 when the driver refuses. destroy: frees one. Both under the caller's lock.
typedef uint64_t (*SbCreate)(void *ctx, const SbKey *key);
typedef void (*SbDestroy)(void *ctx, uint64_t handle);

// room from the driver's limit: the limit less a reserve of 64 (a quarter of a small limit), at least 1.
static inline uint32_t sb_room_of(uint32_t limit) {
    uint32_t reserve = limit / 4 < 64 ? limit / 4 : 64;
    uint32_t room = limit > reserve ? limit - reserve : 1;
    return room ? room : 1;
}

static inline void sb_init(SamplerBook *b, uint32_t limit) {
    memset(b, 0, sizeof *b);
    nvn_hm_init(&b->by_key, sizeof(SbKey));
    b->room = sb_room_of(limit);
    b->high = b->room - b->room / 4;   // three quarters; a room of 1 retires from 1
    if (!b->high) b->high = 1;
}

static inline void sb_free(SamplerBook *b, SbDestroy destroy, void *ctx) {
    for (uint32_t i = 0; i < b->n; i++) { if (destroy) destroy(ctx, b->ents[i]->handle); free(b->ents[i]); }
    for (uint32_t i = 0; i < b->ndead; i++) if (destroy) destroy(ctx, b->dead[i].handle);
    free(b->by_key.keys), free(b->by_key.vals), free(b->ents), free(b->dead);
    memset(b, 0, sizeof *b);
}

// Destroys the retired samplers whose value the GPU has reached (done = the highest completed timeline value).
static inline void sb_collect(SamplerBook *b, uint64_t done, SbDestroy destroy, void *ctx) {
    uint32_t k = 0;
    for (uint32_t i = 0; i < b->ndead; i++) {
        if (b->dead[i].value <= done) {
            if (destroy) destroy(ctx, b->dead[i].handle);
            b->n_destroyed++;
        } else b->dead[k++] = b->dead[i];
    }
    b->ndead = k;
}

// Retires the least recently resolved entry that is not pinned; value: the timeline value its destruction waits for.
// 0 when there is none, or when noting it cannot allocate (it then stays made).
static inline int sb_retire_oldest(SamplerBook *b, uint64_t value) {
    SbEnt *v = NULL;
    for (uint32_t i = 0; i < b->n; i++)
        if (!b->ents[i]->pinned && (!v || b->ents[i]->used < v->used)) v = b->ents[i];
    if (!v) return 0;
    if (b->ndead == b->capd) {
        uint32_t cap = b->capd ? b->capd * 2 : 64;
        SbDead *d = realloc(b->dead, sizeof *d * cap);
        if (!d) return 0;
        b->dead = d, b->capd = cap;
    }
    b->dead[b->ndead++] = (SbDead){v->handle, value};
    nvn_hm_del(&b->by_key, &v->key);
    SbEnt *last = b->ents[--b->n];
    b->ents[v->at] = last, last->at = v->at;
    free(v);
    b->gen++, b->n_retired++;
    return 1;
}

// The sampler for key: the one made for the same create info, or a new one. done: the highest completed timeline value;
// retire_value: a value the timeline reaches only after every recording made so far (the submission being recorded and
// the ones before it). 0 when the room is full of samplers the GPU may still use and none can be made.
static inline uint64_t sb_get(SamplerBook *b, const SbKey *key, uint64_t done, uint64_t retire_value, SbCreate create,
                              SbDestroy destroy, void *ctx) {
    SbEnt *e = nvn_hm_get(&b->by_key, key);
    if (e) {
        e->used = ++b->tick;
        b->n_reused++;
        return e->handle;
    }
    sb_collect(b, done, destroy, ctx);
    if (b->n >= b->high) sb_retire_oldest(b, retire_value);
    if (b->n + b->ndead >= b->room) { b->n_full++; return 0; }
    uint64_t h = create(ctx, key);
    if (!h && b->n && sb_retire_oldest(b, retire_value)) {   // the driver's limit was below the room: one less, once
        sb_collect(b, done, destroy, ctx);
        h = create(ctx, key);
    }
    if (!h) { b->n_full++; return 0; }
    e = calloc(1, sizeof *e);
    uint32_t newcap = b->n == b->cap ? (b->cap ? b->cap * 2 : 256) : b->cap;
    SbEnt **grown = newcap != b->cap ? realloc(b->ents, sizeof *b->ents * newcap) : b->ents;
    if (!e || !grown) {
        if (destroy) destroy(ctx, h);   // out of memory noting it: not kept
        free(e);
        return 0;
    }
    b->ents = grown, b->cap = newcap;
    e->key = *key, e->handle = h, e->used = ++b->tick, e->at = b->n;
    b->ents[b->n++] = e;
    nvn_hm_put(&b->by_key, &e->key, e);
    b->n_made++;
    return h;
}

// Keeps the sampler h for good: a caller holds its handle (the draw stand-ins).
static inline void sb_pin(SamplerBook *b, uint64_t h) {
    for (uint32_t i = 0; i < b->n; i++)
        if (b->ents[i]->handle == h) b->ents[i]->pinned = 1;
}

// The handle of the entry resolved last, 0 when the book is empty: what a caller uses when sb_get answers 0.
static inline uint64_t sb_newest(const SamplerBook *b) {
    const SbEnt *v = NULL;
    for (uint32_t i = 0; i < b->n; i++)
        if (!v || b->ents[i]->used > v->used) v = b->ents[i];
    return v ? v->handle : 0;
}

// sb_get, and when it answers 0 the newest sampler in its place (*stood_in 1): a draw cannot wait for the GPU from
// inside its own recording, so it samples with a sampler the book has until the GPU frees room. 0 only when the book
// is empty as well.
static inline uint64_t sb_admit(SamplerBook *b, const SbKey *key, uint64_t done, uint64_t retire_value, SbCreate create,
                                SbDestroy destroy, void *ctx, int *stood_in) {
    *stood_in = 0;
    uint64_t h = sb_get(b, key, done, retire_value, create, destroy, ctx);
    if (h) return h;
    h = sb_newest(b);
    *stood_in = h != 0;
    return h;
}
