// Configuration boundary tests: no game files, window, Vulkan device or save.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../runtime/dkb_wide.c"

static unsigned checks, failures, page_reads, fs_calls, fs_out[2], fs_scene_on;
static int setting, set_index, scene_index, switched, scene_switched, heapdef_grown, fs_wide;
static int native_index = -1, native_switched, native_switches;
static unsigned settings_version;
static const char *resolution, *scene_res, *upscaler = "fsr";
static uint8_t test_page[4096];

#define CHECK(c) do { checks++; if (!(c)) { \
    failures++; fprintf(stderr, "line %d: %s\n", __LINE__, #c); \
} } while (0)

int cfg_bool(const char *key) {
    CHECK(!strcmp(key, "TOTK_ULTRAWIDE"));
    return setting;
}
const char *cfg_value(const char *key) {
    CHECK(!strcmp(key, "TOTK_OUTPUT_RES") || !strcmp(key, "TOTK_SCENE_RES") || !strcmp(key, "TOTK_UPSCALER"));
    return !strcmp(key, "TOTK_UPSCALER") ? upscaler : key[5] == 'O' ? resolution : scene_res;
}
// dkb_native_aa_live reads the settings version to check the Upscaler setting again after a change.
unsigned cfg_version(void) { return settings_version; }
int rt_patch_set(const char *name) {
    if (!strcmp(name, "dkbscene")) return scene_index;
    if (!strcmp(name, "render3d")) return -1;
    if (!strcmp(name, "dkbnative")) return native_index;
    CHECK(!strcmp(name, "dkbwide"));
    return set_index;
}
void rt_patch_switch(int set, int on) {
    CHECK(set == 0 || set == scene_index || set == native_index);
    if (set == 0) switched = on;
    else if (set == native_index) native_switched = on, native_switches++;
    else scene_switched = on;
}
void *rt_patch_page(void) { page_reads++; return test_page; }
void rt_log(const char *fmt, ...) { (void)fmt; }
int res_heapdef_grown(void) { return heapdef_grown; }
// RomFS setup picks the UI overlay from the output at the time it runs (romfs.c overlay_init): record what it sees.
void fs_init(void) {
    fs_calls++;
    fs_wide = dkb_wide_output(fs_out);
    fs_scene_on = scene_switched;
}

static void reset(const char *text) {
    resolution = text;
    scene_res = "auto";
    upscaler = "fsr";
    native_index = -1;
    native_switched = native_switches = 0;
    scene_index = -1;
    heapdef_grown = 0;
    scene_switched = 0;
    fs_calls = 0;
    heap_extra_done = 0;
    set_index = 0;
    setting = 1;
    switched = 1;
    page_reads = 0;
    memset(test_page, 0xa5, sizeof test_page);
}

static void configured(const char *text, unsigned w, unsigned h, int enabled) {
    reset(text);
    setting = enabled;
    dkb_wide_init();
    unsigned got_w, got_h;
    float aspect, factor;
    memcpy(&got_w, test_page + 0x100, 4);
    memcpy(&got_h, test_page + 0x104, 4);
    memcpy(&aspect, test_page + 0x108, 4);
    memcpy(&factor, test_page + 0x10c, 4);
    CHECK(switched == 1 && page_reads == 1);
    CHECK(got_w == w && got_h == h);
    unsigned output[2];
    CHECK(dkb_wide_output(output) == (w * 9 > h * 16));
    CHECK(output[0] == w && output[1] == h);
    CHECK(fabs((double)aspect - (double)w / h) < 0.000002);
    // A projected unit square must have equal pixel dimensions after the x
    // correction: a 16:9 authored canvas followed by this wider output.
    CHECK(fabs((double)w / (1280.0 * factor) - (double)h / 720.0) < 0.000002);
    uint32_t pool;
    memcpy(&pool, test_page + 0x110, 4);
    CHECK(pool == dkb_wide_pool_bytes(w, h) && pool >= 0xc800000);
    for (size_t i = 0; i < sizeof test_page; i++)
        if (i < 0x100 || i >= 0x114) CHECK(test_page[i] == 0xa5);
}

static void valid(const char *text, unsigned w, unsigned h) { configured(text, w, h, 1); }

static void rejected(const char *text) {
    reset(text);
    dkb_wide_init();
    CHECK(switched == 0 && page_reads == 0);
    unsigned output[2];
    CHECK(dkb_wide_output(output) == 0);
    CHECK(output[0] == 2880 && output[1] == 1620);
    for (size_t i = 0; i < sizeof test_page; i++) CHECK(test_page[i] == 0xa5);
}

static uint32_t page_pool(void) {
    uint32_t pool;
    memcpy(&pool, test_page + 0x110, 4);
    return pool;
}

// The pool GSys must hold, in KiB above the stock 200 MiB, at an output and scene_res.
static void extra_case(const char *text, const char *scene, uint64_t kib) {
    reset(text);
    scene_res = scene;
    if (strcmp(scene, "auto")) scene_index = 1;
    CHECK(dkb_wide_heap_extra_kib() == kib);
    CHECK(dkb_wide_heap_extra_kib() == kib);   // read once, same answer after
}

static void heap_extras(void) {
    extra_case("2880x1620", "auto", 0);
    extra_case("1920x1080", "auto", 0);
    extra_case("3840x2160", "auto", (0x16400000u - 0xc800000u) >> 10);
    extra_case("3440x1440", "auto", (0xd600000u - 0xc800000u) >> 10);
    extra_case("3441x1440", "auto", 0);   // rejected output
    // scene_res 1.0 asks for (1.5 / 1.0)^2 times the pool; a build without the scene set ignores it
    extra_case("2880x1620", "full", (uint64_t)(dkb_scene_pool(0xc800000u, 1.0f) - 0xc800000u) >> 10);
    extra_case("2880x1620", "2.0", (uint64_t)(dkb_scene_pool(0xc800000u, 2.0f) - 0xc800000u) >> 10);
    reset("3840x2160");
    scene_res = "full";
    CHECK(dkb_wide_heap_extra_kib() == (0x16400000u - 0xc800000u) >> 10);   // scene_index -1
    reset("3440x1440");
    set_index = -1;
    CHECK(dkb_wide_heap_extra_kib() == 0);
    reset("3440x1440");
    setting = 0;
    CHECK(dkb_wide_heap_extra_kib() == 0);   // ultrawide off
}

// RomFS setup runs once and reads the output then: it comes after the output is set, before the scene, and only
// when GSys grows.
static void ordering(void) {
    reset("3440x1440");
    dkb_wide_init();
    CHECK(fs_calls == 1 && fs_wide == 1 && fs_out[0] == 3440 && fs_out[1] == 1440);
    reset("3840x1600");
    scene_res = "full";
    scene_index = 1;
    dkb_wide_init();
    CHECK(fs_calls == 1 && fs_out[0] == 3840 && fs_out[1] == 1600 && fs_scene_on == 0 && scene_switched == 1);
    reset("2560x1080");   // the stock pool holds it
    dkb_wide_init();
    CHECK(fs_calls == 0);
    reset("1920x1080");
    dkb_wide_init();
    CHECK(fs_calls == 0);
    reset("3440x1441");
    dkb_wide_init();
    CHECK(fs_calls == 0);
}

// dkb_wide_scene with GSys at its stock size (no grown HeapDef) keeps the pool within the cap; with it grown the
// pool is what the ratio asks for.
static void scene_pool(void) {
    reset("3840x2160");
    scene_res = "1.0";
    scene_index = 1;
    dkb_wide_init();
    float ratio;
    memcpy(&ratio, test_page + 0x118, 4);
    CHECK(page_pool() <= DKB_SCENE_POOL_MAX && ratio > 1.0f && scene_switched == 1);
    reset("3840x2160");
    scene_res = "1.0";
    scene_index = 1;
    heapdef_grown = 1;
    dkb_wide_init();
    memcpy(&ratio, test_page + 0x118, 4);
    CHECK(page_pool() == dkb_scene_pool(dkb_wide_pool_bytes(3840, 2160), 1.0f) && ratio == 1.0f);
    reset("7680x4320");   // the output's own pool is above the cap and is never cut
    scene_res = "2.0";
    scene_index = 1;
    dkb_wide_init();
    CHECK(page_pool() >= dkb_wide_pool_bytes(7680, 4320));
}

// Native anti-aliasing: on only with the set present, the 3D at the output size and a temporal upscaler chosen; it
// follows the live Upscaler setting; and only a dispatch issued from the upscale step's wrapper counts as the game's
// upscaler.
static void native_aa(void) {
    reset("3840x2160");
    heapdef_grown = 1;   // else the ratio is raised to what the stock GSys heap holds (dkb_wide_scene)
    scene_res = "full", scene_index = 1, native_index = 2, upscaler = "fsr4";
    dkb_wide_init();
    CHECK(dkb_native_aa() == 1 && native_switched == 1);
    upscaler = "fsr", settings_version++;
    dkb_native_aa_live();
    CHECK(dkb_native_aa() == 0 && native_switched == 0);
    upscaler = "dlss";
    dkb_native_aa_live();   // the settings did not change: not looked at again
    CHECK(dkb_native_aa() == 0);
    settings_version++;
    dkb_native_aa_live();
    CHECK(dkb_native_aa() == 1 && native_switched == 1);
    int n = native_switches;
    settings_version++;
    dkb_native_aa_live();
    CHECK(native_switches == n);   // unchanged: no switch
    CHECK(dkb_native_aa_issued_by(0x7000325fc4ull));
    CHECK(!dkb_native_aa_issued_by(0x7000325f2cull) && !dkb_native_aa_issued_by(0x7000325fe0ull));
    CHECK(!dkb_native_aa_issued_by(0));
    upscaler = "bilinear", settings_version++;
    dkb_native_aa_live();
    CHECK(!dkb_native_aa_issued_by(0x7000325fc4ull));   // set off: nothing is taken
    reset("3840x2160");   // a scaled 3D, auto, and no set
    scene_res = "1.5", scene_index = 1, native_index = 2, upscaler = "fsr4";
    dkb_wide_init();
    CHECK(dkb_native_aa() == 0);
    reset("3840x2160");
    scene_res = "full", scene_index = 1, native_index = -1, upscaler = "fsr4";
    dkb_wide_init();
    CHECK(dkb_native_aa() == 0 && native_switches == 0);
}

int main(void) {
    native_aa();
    heap_extras();
    ordering();
    scene_pool();
    CHECK(dkb_wide_pool_bytes(2880, 1620) == 0xc800000);
    CHECK(dkb_wide_pool_bytes(1280, 720) == 0xc800000);
    CHECK(dkb_wide_pool_bytes(3440, 1440) == 0xd600000);
    CHECK(dkb_wide_pool_bytes(3840, 2160) == 0x16400000);
    CHECK(dkb_wide_pool_bytes(5120, 1440) == 0x13e00000);
    // The largest accepted output needs 1424 MiB (about 1.39 GiB): it fits uint32_t without clamping.
    CHECK(dkb_wide_pool_bytes(7680, 4320) > 0x40000000 && dkb_wide_pool_bytes(7680, 4320) % (2u << 20) == 0);
    valid("3440x1440", 3440, 1440);
    valid("2560x1080", 2560, 1080);
    valid("3840x1600", 3840, 1600);
    valid("5120x2160", 5120, 2160);
    valid("5120x1440", 5120, 1440);
    valid("7680x2160", 7680, 2160);
    valid("1920x1080", 1920, 1080);
    valid("1280x720", 1280, 720);
    configured("1920x1080", 1920, 1080, 0);
    configured("2560x1440", 2560, 1440, 0);
    configured("3840x2160", 3840, 2160, 0);
    const char *bad[] = {"", "3440", "3440:1440", "3440x1440junk", "3440x1440\n",
                         "0x0", "1280x800", "2560x0",
                         "3441x1440", "3440x1441", "1278x720", "7682x1440", "7680x4322",
                         "3440x718", "-3440x1440"};
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) rejected(bad[i]);
    reset("3440x1440");
    setting = 0;
    dkb_wide_init();
    CHECK(switched == 0 && page_reads == 0);
    CHECK(dkb_wide_output(NULL) == 0);
    reset("3440x1440");
    set_index = -1;
    dkb_wide_init();
    CHECK(switched == 1 && page_reads == 0);
    CHECK(dkb_wide_output(NULL) == 0);
    printf("dkb_wide: %u checks, %u failures\n", checks, failures);
    return failures != 0;
}
