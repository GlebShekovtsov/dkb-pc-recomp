// Upscaler size note: packing and the staleness rule (runtime/res_upscaler.h). No game files, window or device.
#include <stdio.h>

#include "../runtime/res_upscaler.h"

static unsigned checks, failures;

#define CHECK(c) do { checks++; if (!(c)) { \
    failures++; fprintf(stderr, "line %d: %s\n", __LINE__, #c); \
} } while (0)

int main(void) {
    unsigned in[2], out[2];
    uint64_t p = res_up_pack(1706, 960, 2560, 1440);
    res_up_unpack(p, in, out);
    CHECK(in[0] == 1706 && in[1] == 960 && out[0] == 2560 && out[1] == 1440);
    res_up_unpack(res_up_pack(70000, 1, 2, 3), in, out);
    CHECK(in[0] == 0xffff && in[1] == 1 && out[0] == 2 && out[1] == 3);

    CHECK(!res_up_fresh(0, 100, 100));                          // never noted
    CHECK(res_up_fresh(p, 100, 100));                           // same frame
    CHECK(res_up_fresh(p, 100, 100 + RES_UP_FRESH_FRAMES));     // the last frame it counts
    CHECK(!res_up_fresh(p, 100, 101 + RES_UP_FRESH_FRAMES));    // stale
    CHECK(res_up_fresh(p, 100, 99) == 0);                       // a stamp from the future is not fresh
    if (failures) {
        fprintf(stderr, "%u of %u checks failed\n", failures, checks);
        return 1;
    }
    printf("res_upscaler_test: %u checks passed\n", checks);
    return 0;
}
