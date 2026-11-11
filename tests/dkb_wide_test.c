// Configuration boundary tests: no game files, window, Vulkan device or save.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../runtime/dkb_wide.c"

static unsigned checks, failures, page_reads;
static int setting, set_index, switched;
static const char *resolution;
static uint8_t test_page[4096];

#define CHECK(c) do { checks++; if (!(c)) { \
    failures++; fprintf(stderr, "line %d: %s\n", __LINE__, #c); \
} } while (0)

int cfg_bool(const char *key) {
    CHECK(!strcmp(key, "TOTK_ULTRAWIDE"));
    return setting;
}
const char *cfg_value(const char *key) {
    CHECK(!strcmp(key, "TOTK_OUTPUT_RES"));
    return resolution;
}
int rt_patch_set(const char *name) {
    CHECK(!strcmp(name, "dkbwide"));
    return set_index;
}
void rt_patch_switch(int set, int on) {
    CHECK(set == 0);
    switched = on;
}
void *rt_patch_page(void) { page_reads++; return test_page; }
void rt_log(const char *fmt, ...) { (void)fmt; }

static void reset(const char *text) {
    resolution = text;
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

int main(void) {
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
