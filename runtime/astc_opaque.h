// SPDX-License-Identifier: GPL-3.0-or-later
// Whether ASTC blocks decode with every texel's alpha at 255, told from their headers alone (no decode). A texture of
// such blocks is kept as BC1 instead of BC3 (vk_texture.c upload, vk_astc.c vk_astc_record): BC3's colour half alone,
// the same texels in half the video memory. tests/astc_opaque_test.c.
//
// Ported from Hookshot (GPL-3.0-or-later), gpu/vk/astc_opaque.h at commit 97024ef5, from C++ to C; the block rules are
// unchanged. The choice per upload (astc_bc_choice) is Hookshot's too (gpu/vk/vk_texture.cpp, the same commit).
//
// A block is opaque when:
// - it is a void-extent block (one colour) in LDR whose alpha's high byte is 0xFF (the decoder scales the 16-bit alpha
//   to 8);
// - or its colour endpoint mode carries no alpha: 0-3, 6-9 and 11. The modes with alpha are 4 and 5
//   (luminance + alpha), 10 (RGB scale + two alphas) and 12, 13, 14 and 15 (two layers, each with its own
//   endpoints). The mode is in bits 13-16 for one partition, and in bits 25-28 for partitions that share one
//   (bits 23-24 zero). 14 and 15 are counted with the alpha modes although the compute decoder writes magenta
//   for them (its endpoint unpacker handles 0, 1, 4, 5, 6, 8, 9, 10, 12 and 13 and magentas the rest, alpha 255
//   included): the rule then rests on the block header alone, not on what the decoder happens to write.
// Partitions of different modes would need the weights' size to read their modes whole: they count as maybe-alpha, as
// do an HDR void extent and an alpha mode (which could still decode to 255 everywhere: a decode would tell; here BC3
// stays). A block of zeros is counted apart: storage before the game wrote the texture, or padding between levels,
// says nothing about its texels.
#pragma once
#include <stdint.h>
#include <string.h>

enum { ASTC_KIND_ALPHA = 0, ASTC_KIND_OPAQUE = 1, ASTC_KIND_EMPTY = 2 };

typedef struct { uint64_t opaque, alpha, empty; } AstcOpaqueScan;

// The endpoint modes whose blocks carry alpha: the modes with an explicit alpha and the two-layer modes.
static inline int astc_cem_has_alpha(uint32_t cem) {
    return cem == 4 || cem == 5 || cem == 10 || cem == 12 || cem == 13 || cem == 14 || cem == 15;
}

// The kind of one 16-byte block.
static inline int astc_block_kind(const uint8_t *b) {
    uint64_t lo, hi;
    memcpy(&lo, b, 8);
    memcpy(&hi, b + 8, 8);
    if (!lo && !hi) return ASTC_KIND_EMPTY;
    const uint32_t mode = (uint32_t)(lo & 0x7FF);
    // void extent: bit 9 HDR, its RGBA as 16-bit values in bits 64-127 (alpha in 112-127)
    if ((mode & 0x1FF) == 0x1FC) {
        if (mode & 0x200) return ASTC_KIND_ALPHA;
        return b[15] == 0xFF ? ASTC_KIND_OPAQUE : ASTC_KIND_ALPHA;
    }
    const uint32_t parts = (uint32_t)(lo >> 11 & 3) + 1;
    uint32_t cem;
    if (parts == 1) {
        cem = (uint32_t)(lo >> 13 & 0xF);
    } else {
        const uint32_t enc = (uint32_t)(lo >> 23 & 0x3F);
        if (enc & 3) return ASTC_KIND_ALPHA;   // partitions of different modes: not read whole here
        cem = enc >> 2 & 0xF;
    }
    return astc_cem_has_alpha(cem) ? ASTC_KIND_ALPHA : ASTC_KIND_OPAQUE;
}

// The 16-byte blocks of [p, p + n) counted by kind (a last partial block is left out). With stop_at_alpha the count
// ends at the first block that may carry alpha, and adds to *s.
static inline void astc_opaque_scan(AstcOpaqueScan *s, const uint8_t *p, uint64_t n, int stop_at_alpha) {
    for (uint64_t i = 0; i + 16 <= n; i += 16) {
        const int k = astc_block_kind(p + i);
        if (k == ASTC_KIND_OPAQUE) s->opaque++;
        else if (k == ASTC_KIND_EMPTY) s->empty++;
        else if (s->alpha++, stop_at_alpha) break;
    }
}

// The choice for a whole upload of a texture not yet decided: BC1 when every block with texels is opaque and they are
// most of the storage (a texture the game has written, not storage still being filled).
static inline int astc_whole_opaque(const AstcOpaqueScan *s) { return !s->alpha && s->opaque && s->opaque >= s->empty; }

// What an upload does with an ASTC image kept as BC3 (astc_bc 1) or BC1 (2) once its blocks are counted: a BC1 image
// whose texture now has alpha goes back to BC3, a BC3 image whose texture is whole and opaque becomes BC1, and
// otherwise it stays. astc_bc_settled is 1 when the texture is to stay BC3 for good and is not counted again: alpha was
// seen in a BC3 image, a BC1 image went back, or a BC3 image could not be made again (remade: whether the change, if
// any, was made).
enum { ASTC_BC_KEEP = -1, ASTC_BC_TO_BC3 = 0, ASTC_BC_TO_BC1 = 1 };

static inline int astc_bc_choice(int astc_bc, const AstcOpaqueScan *s) {
    if (astc_bc == 2) return s->alpha ? ASTC_BC_TO_BC3 : ASTC_BC_KEEP;
    return astc_whole_opaque(s) ? ASTC_BC_TO_BC1 : ASTC_BC_KEEP;
}

static inline int astc_bc_settled(int astc_bc, int choice, int remade, const AstcOpaqueScan *s) {
    if (choice == ASTC_BC_TO_BC1) return !remade;    // no memory for the BC1 image: BC3 for good
    if (choice == ASTC_BC_TO_BC3) return remade;     // back to BC3 for good; not made again: asked again next time
    return astc_bc == 1 && s->alpha != 0;            // alpha in a BC3 image
}
