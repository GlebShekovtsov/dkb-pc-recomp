#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "host.h"

static GuestFn gamut, tone;
#undef HOST_OVERRIDE
#define HOST_OVERRIDE(sym, fn) \
    static void __attribute__((constructor)) RT_CAT(register_hdr_, __COUNTER__)(void) { \
        if (!strcmp(sym, "_ZN2nn2vi35SetLayerNextHdrContentGamutMetadataEPNS0_5LayerENS0_15HdrContentGamutE")) gamut = fn; \
        if (!strcmp(sym, "_ZN2nn2vi39SetLayerNextHdrToSdrToneMappingMetadataEPNS0_5LayerENS0_14HdrToneMappingE")) tone = fn; \
    }
#include "../runtime/ov_gfx.c"

void nvn_procs_register(void) {}
// Host services ov_gfx.c links against; the test is single-threaded and reads no settings.
int cfg_bool(const char *name) { (void)name; return 0; }
const char rt_game_target[] = "dkb100", rt_game_display[] = "1.0.0";
void rt_log(const char *fmt, ...) { (void)fmt; }
const char *ldr_describe(uint64_t a, char *buf, int n) { (void)a; (void)n; return buf; }
int plat_mem_read(uint64_t addr, void *out, size_t n) { (void)addr; (void)out; (void)n; return 0; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }

int main(void) {
    if (!gamut || !tone) {
        fputs("missing SDR layer metadata binding\n", stderr);
        return 1;
    }
    Cpu c = {0};
    uint64_t layer[16];
    memset(layer, 0xa5, sizeof layer);
    c.x[0] = (uintptr_t)layer;
    c.x[1] = 1;
    gamut(&c);
    assert(c.x[0] == 0);
    c.x[0] = (uintptr_t)layer;
    tone(&c);
    assert(c.x[0] == 0);
    for (size_t i = 0; i < sizeof layer / sizeof *layer; i++) assert(layer[i] == 0xa5a5a5a5a5a5a5a5ull);
    puts("SDR layer metadata ABI passed");
    return 0;
}
