// The opaque-block classification of runtime/astc_opaque.h, driven with hand-built 16-byte ASTC blocks.
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "astc_opaque.h"

static int fails;
#define CHECK(c, ...)                                                                                                  \
    do {                                                                                                               \
        if (!(c)) { fails++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); }                  \
    } while (0)

static void put_bits(uint8_t b[16], int at, int n, uint32_t v) {
    for (int i = 0; i < n; i++) {
        int bit = at + i;
        if (v >> i & 1) b[bit >> 3] |= (uint8_t)(1u << (bit & 7));
        else b[bit >> 3] &= (uint8_t)~(1u << (bit & 7));
    }
}

// An ordinary block (block mode 0x0 is never the void-extent pattern): parts partitions, endpoint mode cem (one
// partition or all partitions sharing one), with a nonzero weight byte so the block is not counted as empty.
static void normal(uint8_t b[16], int parts, uint32_t cem) {
    memset(b, 0, 16);
    b[15] = 0x55;
    put_bits(b, 11, 2, (uint32_t)(parts - 1));
    if (parts == 1) put_bits(b, 13, 4, cem);
    else put_bits(b, 23, 6, cem << 2);   // the two low bits zero: every partition shares the mode
}

static void void_extent(uint8_t b[16], int hdr, uint8_t alpha_high) {
    memset(b, 0, 16);
    put_bits(b, 0, 9, 0x1FC);
    put_bits(b, 9, 1, (uint32_t)hdr);
    b[8] = 0x12;   // some colour
    b[15] = alpha_high;
}

int main(void) {
    uint8_t b[16];
    // single partition: modes 4, 5, 10, 12, 13, 14 and 15 carry alpha, every other mode is opaque
    for (uint32_t cem = 0; cem < 16; cem++) {
        normal(b, 1, cem);
        int want = (cem == 4 || cem == 5 || cem == 10 || cem == 12 || cem == 13 || cem == 14 || cem == 15)
                       ? ASTC_KIND_ALPHA : ASTC_KIND_OPAQUE;
        CHECK(astc_block_kind(b) == want, "one partition, mode %u: %d, want %d", cem, astc_block_kind(b), want);
    }
    // partitions sharing a mode follow that mode
    for (int parts = 2; parts <= 4; parts++)
        for (uint32_t cem = 0; cem < 16; cem++) {
            normal(b, parts, cem);
            int want = cem == 4 || cem == 5 || cem == 10 || cem == 12 || cem == 13 || cem == 14 || cem == 15
                           ? ASTC_KIND_ALPHA : ASTC_KIND_OPAQUE;
            CHECK(astc_block_kind(b) == want, "%d partitions, shared mode %u: %d, want %d", parts, cem,
                  astc_block_kind(b), want);
        }
    // partitions of different modes are not read whole: maybe alpha
    normal(b, 2, 0);
    put_bits(b, 23, 2, 1);
    CHECK(astc_block_kind(b) == ASTC_KIND_ALPHA, "different modes count as alpha");
    // void extent: LDR with alpha high byte 0xFF is opaque, anything else, and HDR, are not
    void_extent(b, 0, 0xFF);
    CHECK(astc_block_kind(b) == ASTC_KIND_OPAQUE, "LDR void extent, alpha FF");
    void_extent(b, 0, 0xFE);
    CHECK(astc_block_kind(b) == ASTC_KIND_ALPHA, "LDR void extent, alpha FE");
    void_extent(b, 0, 0x00);
    CHECK(astc_block_kind(b) == ASTC_KIND_ALPHA, "LDR void extent, alpha 0");
    void_extent(b, 1, 0xFF);
    CHECK(astc_block_kind(b) == ASTC_KIND_ALPHA, "HDR void extent");
    // zeros are neither
    memset(b, 0, 16);
    CHECK(astc_block_kind(b) == ASTC_KIND_EMPTY, "zero block");

    // scans over several blocks
    uint8_t blocks[16 * 8];
    for (int i = 0; i < 8; i++) normal(blocks + 16 * i, 1, 0);
    AstcOpaqueScan s = {0};
    astc_opaque_scan(&s, blocks, sizeof blocks, 1);
    CHECK(s.opaque == 8 && !s.alpha && !s.empty && astc_whole_opaque(&s), "eight opaque blocks");
    normal(blocks + 16 * 3, 1, 12);
    memset(&s, 0, sizeof s);
    astc_opaque_scan(&s, blocks, sizeof blocks, 1);
    CHECK(s.opaque == 3 && s.alpha == 1, "stops at the first alpha block: opaque %llu alpha %llu",
          (unsigned long long)s.opaque, (unsigned long long)s.alpha);
    CHECK(!astc_whole_opaque(&s), "a texture with alpha is not opaque");
    memset(&s, 0, sizeof s);
    astc_opaque_scan(&s, blocks, sizeof blocks, 0);
    CHECK(s.opaque == 7 && s.alpha == 1, "without stopping it counts them all");
    // a count adds to what it holds, as the per-layer scans of one upload do
    astc_opaque_scan(&s, blocks, 32, 0);
    CHECK(s.opaque == 9, "scans add up");
    // a trailing partial block is left out
    memset(&s, 0, sizeof s);
    astc_opaque_scan(&s, blocks, 16 * 2 + 9, 1);
    CHECK(s.opaque == 2, "the partial block is not counted");
    // storage mostly zeros (not written yet) decides nothing; mostly written does
    uint8_t mix[16 * 10];
    memset(mix, 0, sizeof mix);
    for (int i = 0; i < 3; i++) normal(mix + 16 * i, 1, 0);
    memset(&s, 0, sizeof s);
    astc_opaque_scan(&s, mix, sizeof mix, 1);
    CHECK(s.opaque == 3 && s.empty == 7 && !astc_whole_opaque(&s), "mostly empty storage: no decision");
    for (int i = 3; i < 6; i++) normal(mix + 16 * i, 1, 0);
    memset(&s, 0, sizeof s);
    astc_opaque_scan(&s, mix, sizeof mix, 1);
    CHECK(s.opaque == 6 && s.empty == 4 && astc_whole_opaque(&s), "mostly written: opaque");
    memset(&s, 0, sizeof s);
    astc_opaque_scan(&s, mix + 16 * 9, 16, 1);
    CHECK(s.empty == 1 && !astc_whole_opaque(&s), "all zeros: no decision");

    // the upload's choice and when a texture stays BC3 for good
    AstcOpaqueScan opaque = {.opaque = 6, .empty = 4}, alpha = {.opaque = 3, .alpha = 1}, empty = {.empty = 9};
    CHECK(astc_bc_choice(1, &opaque) == ASTC_BC_TO_BC1, "BC3, whole and opaque: BC1");
    CHECK(astc_bc_choice(1, &alpha) == ASTC_BC_KEEP, "BC3 with alpha: stays");
    CHECK(astc_bc_choice(1, &empty) == ASTC_BC_KEEP, "BC3, storage not written yet: stays");
    CHECK(astc_bc_choice(2, &opaque) == ASTC_BC_KEEP, "BC1, still opaque: stays");
    CHECK(astc_bc_choice(2, &alpha) == ASTC_BC_TO_BC3, "BC1 with alpha now: BC3");
    CHECK(astc_bc_settled(1, ASTC_BC_KEEP, 0, &alpha), "alpha in a BC3 image: BC3 for good");
    CHECK(!astc_bc_settled(1, ASTC_BC_KEEP, 0, &empty), "an unwritten BC3 image is asked again");
    CHECK(!astc_bc_settled(1, ASTC_BC_TO_BC1, 1, &opaque), "made BC1: asked again at the next upload");
    CHECK(astc_bc_settled(1, ASTC_BC_TO_BC1, 0, &opaque), "no memory for BC1: BC3 for good");
    CHECK(astc_bc_settled(2, ASTC_BC_TO_BC3, 1, &alpha), "back to BC3: for good");
    CHECK(!astc_bc_settled(2, ASTC_BC_TO_BC3, 0, &alpha), "could not go back: asked again");
    if (fails) return 1;
    printf("astc_opaque: ok\n");
    return 0;
}
