// Bananza's virtual display and camera defaults are game state, separate from
// Vulkan's presentation rectangle. The generated dkbwide sites read this page.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"

static unsigned wide_output[2] = {2880, 1620};
static int wide_enabled;

// The game's render-target pool is a fixed 200 MiB sized for 2880x1620 (main+0x458adc in
// f_7000458a88). Grow it with the output pixel count, rounded up to 2 MiB, never below the
// default. The highest accepted output (7680x4320) needs 1424 MiB (about 1.39 GiB), which fits uint32_t.
static uint32_t dkb_wide_pool_bytes(unsigned w, unsigned h) {
    const uint64_t base = 0xc800000, align = 2u << 20;
    uint64_t bytes = (base * w * h + 2880ull * 1620 - 1) / (2880ull * 1620);
    bytes = (bytes + align - 1) / align * align;
    return (uint32_t)(bytes > base ? bytes : base);
}

int dkb_wide_output(unsigned out[2]) {
    if (out) memcpy(out, wide_output, sizeof wide_output);
    return wide_enabled;
}

void dkb_wide_init(void) {
    wide_enabled = 0;
    wide_output[0] = 2880;
    wide_output[1] = 1620;
    int set = rt_patch_set("dkbwide");
    if (set < 0) {
        // The output resolution, ultrawide and frame-rate settings all work through code patches (recomp/patches-dkb100.txt
        // for 1.0.0, recomp/patches-dkb458752.txt for update 3.2.0). A build generated without them has no such set: say
        // so, so a setting that has no effect is not a mystery (#131).
        rt_log("Bananza resolution: this build has no resolution patch; output stays 2880x1620 and the "
               "resolution, ultrawide and real-time settings have no effect");
        return;
    }
    rt_patch_switch(set, 0);
    int allow_wide = cfg_bool("TOTK_ULTRAWIDE");
    unsigned width = 0, height = 0;
    char tail;
    const char *value = cfg_value("TOTK_OUTPUT_RES");
    if (sscanf(value, "%ux%u%c", &width, &height, &tail) != 2 ||
        width < 1280 || width > 7680 || height < 720 || height > 4320 ||
        width % 2 || height % 2 || width * 9 < height * 16 ||
        (!allow_wide && width * 9 != height * 16)) {
        rt_patch_switch(set, 0);
        rt_log("Bananza resolution: use an even 16:9 resolution, or enable ultrawide for a wider resolution; using stock 2880x1620");
        return;
    }
    uint8_t *page = rt_patch_page();
    float aspect = (float)width / height;
    float factor = aspect / (16.0f / 9.0f);
    memcpy(page + 0x100, &width, 4);
    memcpy(page + 0x104, &height, 4);
    memcpy(page + 0x108, &aspect, 4);
    memcpy(page + 0x10c, &factor, 4);
    uint32_t pool = dkb_wide_pool_bytes(width, height);
    memcpy(page + 0x110, &pool, 4);
    wide_output[0] = width;
    wide_output[1] = height;
    wide_enabled = width * 9 > height * 16;
    rt_patch_switch(set, 1);
    rt_log("Bananza resolution: virtual display %ux%u, camera aspect %.6f, ultrawide %s",
           width, height, aspect, wide_enabled ? "on" : "off");
}
