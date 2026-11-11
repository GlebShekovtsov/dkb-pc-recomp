// The pure part of the Bananza upscaler size note (resolution.c res_note_upscaler, res_sizes): four 16-bit sizes in one
// 64-bit word, and the rule that decides whether a note is recent enough to show. Kept apart so a unit test can include
// it without the rest of resolution.c.
#pragma once
#include <stdint.h>

#define RES_UP_FRESH_FRAMES 120   // a note older than this many presents is stale: the game's upscaler no longer runs

// Sizes above 65535 are clamped; 0 in any field means "no note".
static inline uint64_t res_up_pack(unsigned in_w, unsigned in_h, unsigned out_w, unsigned out_h) {
    unsigned v[4] = {in_w, in_h, out_w, out_h};
    uint64_t p = 0;
    for (int i = 0; i < 4; i++) p |= (uint64_t)(v[i] > 0xffff ? 0xffff : v[i]) << (16 * i);
    return p;
}

static inline void res_up_unpack(uint64_t p, unsigned in[2], unsigned out[2]) {
    in[0] = p & 0xffff, in[1] = p >> 16 & 0xffff, out[0] = p >> 32 & 0xffff, out[1] = p >> 48 & 0xffff;
}

// Whether a note packed as p and stamped with frame `stamp` still counts at frame `now`.
static inline int res_up_fresh(uint64_t p, uint64_t stamp, uint64_t now) {
    return p && now >= stamp && now - stamp <= RES_UP_FRESH_FRAMES;
}
