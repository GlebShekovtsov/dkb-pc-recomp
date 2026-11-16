// Bananza's virtual display and camera defaults are game state, separate from
// Vulkan's presentation rectangle. The generated dkbwide sites read this page.
#include <math.h>
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

// TOTK_SCENE_RES: "auto" leaves the game's dynamic resolution alone. "full" (ratio 1.0) or a ratio from 1.0 to 3.0
// draws the 3D scene at the virtual display divided by that ratio: the patched store in the controller at main
// 3c12b4 (patches-dkb100.txt, set dkbscene) loads the ratio from page +0x118. Under emulation the game's own
// control settles at 1.5, so the pool the game sizes for that 3D size (page +0x110) grows by (1.5 / ratio)^2.
// Returns the ratio in effect, 0 for auto. Needs the dkbwide pool word, so it runs after that set is configured.
static float dkb_scene_ratio(void) {
    const char *value = cfg_value("TOTK_SCENE_RES");
    float ratio = 0;
    char tail;
    if (!strcmp(value, "full")) ratio = 1.0f;
    else if (sscanf(value, "%f%c", &ratio, &tail) != 1 || ratio < 1.0f || ratio > 3.0f) ratio = 0;
    return ratio;
}

// The render-target pool is carved from the GSys heap, 518144 KiB (506 MiB) in
// System/HeapSize/Product.Nin_Ounce_NVN2.HeapDef.byml.zs. A pool GSys cannot hold leaves the game with a null
// allocation, and it crashes before it shows anything (access violation reading 0xa0 in main+0x128d40 on 3.2.0): at a
// 3840x2160 output, pools of 474 MiB ran and 514 MiB crashed, and a 5760x3240 output on auto (800 MiB) the same.
// So GSys grows by what the pool adds to the stock 200 MiB (dkb_wide_heap_extra_kib; resolution.c rewrites the table,
// memory.c adds the amount to the application pool, so no other heap shrinks). When the table cannot be rewritten
// (a RomFS without it, a format resolution.c does not know), scene_res keeps the pool within the 474 MiB that ran.
#define DKB_POOL_BASE 0xc800000u
#define DKB_SCENE_POOL_MAX (474u << 20)

// The pool scene_res asks for at that ratio: the game sizes it for its own ratio, 1.5, so it grows by (1.5 / ratio)^2,
// rounded up to 2 MiB.
static uint32_t dkb_scene_pool(uint32_t pool, float ratio) {
    float grow = (1.5f / ratio) * (1.5f / ratio);
    if (grow <= 1.0f) return pool;
    const uint64_t align = 2u << 20;
    uint64_t bytes = (uint64_t)((double)pool * grow + 0.999);
    bytes = (bytes + align - 1) / align * align;
    return (uint32_t)(bytes < 0x80000000u ? bytes : 0x80000000u);   // 2 GiB at most (7680x4320 at ratio 1.0 would ask 3.1)
}

// TOTK_OUTPUT_RES as dkb_wide_init accepts it; 0 for any other value (dkb_wide_init says why).
static int dkb_output_size(unsigned *width, unsigned *height) {
    int allow_wide = cfg_bool("TOTK_ULTRAWIDE");
    char tail;
    return sscanf(cfg_value("TOTK_OUTPUT_RES"), "%ux%u%c", width, height, &tail) == 2 && *width >= 1280 &&
           *width <= 7680 && *height >= 720 && *height <= 4320 && !(*width % 2) && !(*height % 2) &&
           *width * 9 >= *height * 16 && (allow_wide || *width * 9 == *height * 16);
}

// What the render-target pool adds to the stock 200 MiB at the configured output and scene_res, in KiB: GSys grows by
// it (resolution.c). Read once memory.c has read the settings, before dkb_wide_init, from the same settings.
static int heap_extra_done;   // file scope so tests/dkb_wide_test.c can reset the memo
static uint64_t heap_extra;
uint64_t dkb_wide_heap_extra_kib(void) {
    if (heap_extra_done) return heap_extra;
    heap_extra_done = 1;
    unsigned width, height;
    if (rt_patch_set("dkbwide") < 0 || !dkb_output_size(&width, &height)) return heap_extra = 0;
    uint32_t pool = dkb_wide_pool_bytes(width, height);
    float ratio = dkb_scene_ratio();
    if (ratio && rt_patch_set("dkbscene") >= 0) pool = dkb_scene_pool(pool, ratio);
    return heap_extra = pool > DKB_POOL_BASE ? (uint64_t)(pool - DKB_POOL_BASE) >> 10 : 0;
}

static float scene_ratio_now;   // the ratio dkb_wide_scene wrote, 0 for auto

static void dkb_wide_scene(uint8_t *page, float ratio) {
    int set = rt_patch_set("dkbscene");
    if (set < 0) {
        rt_log("Bananza resolution: this build or game version has no 3D scene patch; scene_res has no effect");
        return;
    }
    uint32_t pool;
    memcpy(&pool, page + 0x110, 4);
    if (!res_heapdef_grown()) {
        // GSys keeps its stock size: the pool grows by (1.5 / ratio)^2, so the smallest ratio that keeps it within the
        // cap is 1.5 * sqrt(pool / cap), and not past 1.5, where it does not grow.
        float floor_ratio = (float)(1.5 * sqrt((double)pool / DKB_SCENE_POOL_MAX));
        if (floor_ratio > 1.5f) floor_ratio = 1.5f;
        if (ratio < floor_ratio) {
            rt_log("Bananza resolution: the GSys heap keeps its stock size, so scene_res ratio %.3f (a %u MiB render-target "
                   "pool) is raised to %.3f",
                   ratio, dkb_scene_pool(pool, ratio) >> 20, floor_ratio);
            ratio = floor_ratio;
        }
    }
    uint32_t grown = dkb_scene_pool(pool, ratio);
    // The 2 MiB rounding can pass the cap by a step; the cap only limits growth, never the pool the output itself needs.
    if (!res_heapdef_grown() && grown > pool && grown > DKB_SCENE_POOL_MAX)
        grown = pool > DKB_SCENE_POOL_MAX ? pool : DKB_SCENE_POOL_MAX;
    if (grown != pool) {
        pool = grown;
        memcpy(page + 0x110, &pool, 4);
    }
    memcpy(page + 0x118, &ratio, 4);
    scene_ratio_now = ratio;
    rt_patch_switch(set, 1);
    // Update 3.2.0 makes each display's 3D image a scale from its render parameters times the display size, 2/3, read
    // at main+79024/79028 (patches-dkb458752.txt, set render3d); the controller's ratio divides that image, so the
    // scene is display x scale / ratio. With the controller alone, scene_res = full at a 2880x1620 display drew the
    // scene at 1920x1080 (the upscaler no longer ran, and the image was stretched). The render3d sites read a scale of
    // 1.0 from page +0x120/+0x124, which makes the scene display / ratio, as on 1.0.0, and its sizes those 1.0.0 has
    // at that ratio, which the pool above is grown for.
    int scale_set = rt_patch_set("render3d");
    if (scale_set >= 0) {
        const float one = 1.0f;
        memcpy(page + 0x120, &one, 4);
        memcpy(page + 0x124, &one, 4);
        rt_patch_switch(scale_set, 1);
    }
    rt_log("Bananza resolution: 3D scene ratio %.3f (scene drawn at 1/%.3f of the virtual display%s, render-target pool %u MiB)",
           ratio, ratio, scale_set >= 0 ? ", the 3.2.0 image scale at 1.0" : "", pool >> 20);
}

// Native anti-aliasing (recomp/patches-dkb458752.txt, set dkbnative; docs/dkb-testing.md). With the 3D drawn at the
// output size (a scene ratio of 1.0 in effect) the game skips its upscale step's dispatch, so a temporal upscaler (FSR
// 3.1, FSR 4, DLSS) has nothing to replace. The set makes the game run that dispatch at full size too, from the 3D to
// an image of the same size, and the upscaler takes it as anti-aliasing alone (FSR 4 Native AA, DLAA). The Upscaler
// setting is live, so the choice is checked again whenever the settings change (frame_step_present).
static int native_set = -2, native_on;
static unsigned native_seen;

// The wrapper the upscale step f_7000325a80 calls to issue the dispatch (main f_7000325f30, 3.2.0): the return address
// the dispatch records, 0x7000325fc4, lies inside it. Guest code has the same addresses in every run (layout.py).
#define NATIVE_AA_ISSUER_LO 0x7000325f30ull
#define NATIVE_AA_ISSUER_HI 0x7000325fe0ull

int dkb_native_aa(void) { return __atomic_load_n(&native_on, __ATOMIC_RELAXED); }

// Whether a same-size dispatch issued from the guest address caller is the game's upscale step run at full size by the
// set: only while the set is on, and only from the wrapper above, so no other same-size pass of the same shape (a
// post-process compute pass) is taken for the upscaler (vk_exec_draw.c game_upscaler).
int dkb_native_aa_issued_by(uint64_t caller) {
    return dkb_native_aa() && caller >= NATIVE_AA_ISSUER_LO && caller < NATIVE_AA_ISSUER_HI;
}

void dkb_native_aa_live(void) {
    unsigned v = cfg_version();
    if (native_set != -2 && v == native_seen) return;
    native_seen = v;
    if (native_set == -2) native_set = rt_patch_set("dkbnative");
    if (native_set < 0) return;
    const char *up = cfg_value("TOTK_UPSCALER");
    int want = scene_ratio_now > 0 && scene_ratio_now < 1.0005f &&
               (!strcmp(up, "fsr4") || !strcmp(up, "fsr3") || !strcmp(up, "dlss"));
    if (want == native_on) return;
    __atomic_store_n(&native_on, want, __ATOMIC_RELAXED);
    rt_patch_switch(native_set, want);
    rt_log("Bananza resolution: native anti-aliasing %s (%s with the 3D at the output size: the game's upscale step "
           "runs at full size for it)", want ? "on" : "off", up);
}

void dkb_wide_init(void) {
    wide_enabled = 0;
    scene_ratio_now = 0;
    native_set = -2, native_on = 0;   // the set is off at start (the generated default) and chosen below
    wide_output[0] = 2880;
    wide_output[1] = 1620;
    int scene_set = rt_patch_set("dkbscene"), scale_set = rt_patch_set("render3d");
    if (scene_set >= 0) rt_patch_switch(scene_set, 0);   // on only once dkb_wide_scene has written the ratio
    if (scale_set >= 0) rt_patch_switch(scale_set, 0);   // and the 3.2.0 image scale
    int set = rt_patch_set("dkbwide");
    if (set < 0) {
        // The output resolution, ultrawide and frame-rate settings all work through code patches (recomp/patches-dkb100.txt
        // for 1.0.0, recomp/patches-dkb458752.txt for update 3.2.0). A build generated without them has no such set: say
        // so, so a setting that has no effect is not a mystery (#131).
        rt_log("Bananza resolution: this build has no resolution patch; output stays 2880x1620 and the "
               "resolution, ultrawide and real-time settings have no effect");
        if (dkb_scene_ratio()) rt_log("Bananza resolution: scene_res has no effect either");
        return;
    }
    rt_patch_switch(set, 0);
    float scene = dkb_scene_ratio();
    unsigned width = 0, height = 0;
    if (!dkb_output_size(&width, &height)) {
        rt_patch_switch(set, 0);
        rt_log("Bananza resolution: use an even 16:9 resolution, or enable ultrawide for a wider resolution; using stock 2880x1620");
        if (scene) rt_log("Bananza resolution: scene_res needs a valid output resolution and is not applied");
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
    // RomFS setup runs once and picks the UI overlay from the output set above, so it follows that and precedes the scene:
    // romfs.c serves the grown HeapDef from here on (res_heapdef_served), which dkb_wide_scene asks about.
    if (dkb_wide_heap_extra_kib()) fs_init();
    if (scene) dkb_wide_scene(page, scene);
    dkb_native_aa_live();
}
