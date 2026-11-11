// Adapted from MacNativeGames games/totk/runtime/frontend.m (nn::vi, lines 345-359) and gfx_bridge.c (nv, lines
// 74-86) @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames contributors). The nvnBootstrapLoader stop takes the place
// of MacNativeGames' NVN entry (third_party/d2r_gfx/gfx/nvn/hle_nvn.c:115-118); no code of it is copied.
// Graphics bring-up:
//   nv::*        record the game's allocator callbacks and graphics work memory (guest addresses) for the NVN layer
//   nn::vi::*    display, layer and native window identities; the vsync event is bound here
//   nvnBootstrapLoader   the end of --boot (rt_boot_event "nvn"); otherwise the synthetic address of an NVN proc (nvn_procs.c)
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "ov.h"
#include "platform/platform.h"

#define VI_R_INVALID 0xf601u   // MacNativeGames' result for a call on an unknown display or layer

static OvNvState nv;
static PlatMutex lock = PLAT_MUTEX_INIT;

const OvNvState *ov_nv_state(void) { return &nv; }

static void set_allocator(Cpu *c, OvNvAllocator *a) {
    plat_mutex_lock(&lock);
    *a = (OvNvAllocator){arg_x(c, 0), arg_x(c, 1), arg_x(c, 2), arg_x(c, 3)};
    plat_mutex_unlock(&lock);
}

// void nv::SetGraphicsAllocator(AllocateFunction, FreeFunction, ReallocateFunction, void *user)
static void ov_graphics_allocator(Cpu *c) { set_allocator(c, &nv.graphics); }
// void nv::SetGraphicsDevtoolsAllocator(AllocateFunction, FreeFunction, ReallocateFunction, void *user)
static void ov_devtools_allocator(Cpu *c) { set_allocator(c, &nv.devtools); }

// void nv::InitializeGraphics(void *memory, size_t size). TotK hands over a work buffer. Donkey Kong Bananza (Switch 2)
// passes (0, 0) and leaves the driver to its allocator callbacks; nothing on the host reads the buffer, so that is fine.
static void ov_initialize_graphics(Cpu *c) {
    uint64_t mem = arg_x(c, 0), size = arg_x(c, 1);
    if (!mem != !size) rt_fatal("nv::InitializeGraphics: work memory %#llx with size %#llx", (unsigned long long)mem,
                                (unsigned long long)size);
    plat_mutex_lock(&lock);
    nv.memory = mem;
    nv.memory_size = size;
    nv.initialized = 1;
    plat_mutex_unlock(&lock);
    rt_log("nv: graphics work memory %#llx+%#llx recorded", (unsigned long long)mem,
           (unsigned long long)size);
}

// void nv::InitCheck(): aborts unless nv::InitializeGraphics ran, and that call is the host's (above), so the SDK's own
// flag is never set. Bananza calls it before nn::codec::video::Initialize when a tutorial guide movie starts.
static void ov_init_check(Cpu *c) { (void)c; }

// ---- nn::vi: the identities are small zeroed blocks in the guest-visible kernel region, in case SDK code reads them
static int vi_ready;
static void *display, *layer, *window;

// void nn::vi::Initialize()
static void ov_vi_initialize(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    if (!display) {
        display = mem_kernel_alloc(64, 16);
        layer = mem_kernel_alloc(64, 16);
        window = mem_kernel_alloc(64, 16);
    }
    vi_ready = 1;
    plat_mutex_unlock(&lock);
}

static int vi_up(void) {
    plat_mutex_lock_shared(&lock);
    int r = vi_ready;
    plat_mutex_unlock_shared(&lock);
    return r;
}

// Result nn::vi::OpenDefaultDisplay(Display **out)
static void ov_open_display(Cpu *c) {
    void **out = arg_p(c, 0);
    if (!out || !vi_up()) { ret_x(c, VI_R_INVALID); return; }
    *out = display;
    ret_x(c, 0);
}

// Result nn::vi::CreateLayer(Layer **out, Display *display)
static void ov_create_layer(Cpu *c) {
    void **out = arg_p(c, 0);
    if (!out || !vi_up() || arg_p(c, 1) != display) { ret_x(c, VI_R_INVALID); return; }
    *out = layer;
    ret_x(c, 0);
}

// Result nn::vi::GetNativeWindow(void **out, Layer *layer): a placeholder the NVN layer recognises
static void ov_native_window(Cpu *c) {
    void **out = arg_p(c, 0);
    if (!out || !vi_up() || arg_p(c, 1) != layer) { ret_x(c, VI_R_INVALID); return; }
    *out = window;
    ret_x(c, 0);
}

// Result nn::vi::GetDisplayVsyncEvent(os::SystemEventType *event, Display *display): auto-clear system event
static void ov_vsync_event(Cpu *c) {
    uint64_t ev = arg_x(c, 0);
    if (!ev || !vi_up() || arg_p(c, 1) != display) { ret_x(c, VI_R_INVALID); return; }
    kern_bind_system_event(c, ev, 1);
    gfx_vsync_bind(ev);
    ret_x(c, 0);
}

// ---- DKB_HDR_TRACE=1 (cfg HDR_TRACE): logs the game's HDR and colour-space calls with their arguments and caller.
// The host answer of each traced call is unchanged; with the key off nothing is counted or printed.
#define HDR_TRACE_NAMES 24
static struct { const char *name; unsigned n; } hdr_tr[HDR_TRACE_NAMES];

int hdr_trace_on(void) {
    static int on = -1;   // racing threads compute the same value
    if (on < 0) on = cfg_bool("TOTK_HDR_TRACE");
    return on;
}

int hdr_trace_hit(const Cpu *c, const char *name) {
    if (!hdr_trace_on()) return 0;
    unsigned n = 0;
    int full = 0;
    plat_mutex_lock(&lock);
    for (int i = 0; i < HDR_TRACE_NAMES; i++) {
        if (!hdr_tr[i].name) hdr_tr[i].name = name;
        if (hdr_tr[i].name == name || !strcmp(hdr_tr[i].name, name)) { n = ++hdr_tr[i].n; break; }
    }
    static int full_logged;   // the table has a row per traced name; a name beyond it is counted nowhere and not logged
    if (!n && !full_logged) full = full_logged = 1;
    plat_mutex_unlock(&lock);
    if (full) rt_log("hdr-trace: more than %d traced names; %s and later new names are not logged", HDR_TRACE_NAMES, name);
    if (!n || (n > 20 && n % 600)) return 0;
    char lr[160];
    rt_log("hdr-trace: %s #%u from %s", name, n, ldr_describe(c->x[30], lr, sizeof lr));
    return 1;
}

void hdr_trace_dump(const char *name, const void *p) {
    if (!p) { rt_log("hdr-trace: %s struct null", name); return; }
    uint8_t b[64];
    if (!plat_mem_read((uint64_t)(uintptr_t)p, b, sizeof b)) {   // a bad guest pointer is logged, not read
        rt_log("hdr-trace: %s struct %p: not readable", name, p);
        return;
    }
    char hex[64 * 3 + 1];
    for (int i = 0; i < 64; i++) snprintf(hex + i * 3, 4, "%02x ", b[i]);
    rt_log("hdr-trace: %s struct %p: %s", name, p, hex);
}

// ---- Switch 2 vi calls (Donkey Kong Bananza): the host presents SDR with a fixed swap interval and no VRR.
// DKB_HDR_NATIVE=1 (cfg HDR_NATIVE, docs/hdr-native-plan.md) reports an HDR display instead, so the game
// switches to its own HDR output: the 3.2.0 display object (main+0x203728) stores the answer, the peak caps its content
// luminance and the paper white scales its HUD and SDR content. vk_present.c present_image then copies the game's
// HDR10 window texture onto the HDR10 swapchain. Only on 3.2.0 (dkb458752) with hdr = 1: 1.0.0 imports the same calls
// by name, but its HDR mode has never been run, so it keeps the SDR answer.
int hdr_native_on(void) {
    static int on = -1;   // racing threads compute the same value
    if (on < 0) {
        int want = cfg_bool("TOTK_HDR_NATIVE");
        int v320 = !strcmp(rt_game_target, "dkb458752");
        if (want && !v320) rt_log("hdr: hdr_native is for game 3.2.0 only, untested on %s; it stays off", rt_game_display);
        else if (want && !cfg_bool("TOTK_HDR")) rt_log("hdr: hdr_native needs hdr = 1; it stays off");
        on = want && v320 && cfg_bool("TOTK_HDR");
    }
    return on;
}
// The answer the game gets: HDR only while the swapchain is HDR10 (gfx_window_hdr10). The game polls it every frame,
// so a fallback to SDR (no HDR10 surface format, Windows HDR switched off while running) switches it back to SDR
// output and an HDR10 swapchain made later switches it to HDR again.
static int hdr_answer(void) { return hdr_native_on() && gfx_window_hdr10(); }
// Result QueryDisplayHdrStatus(bool *enabled, HdrOutputFormat *format, u16 *peak_nits, const Display *): HDR off, or
// on (hdr_answer) with hdr_peak_nits as the display's peak. The game never reads *format; it is left alone.
static void ov_hdr_status(Cpu *c) {
    uint8_t *on = arg_p(c, 0);
    uint16_t *peak = arg_p(c, 2);
    if (hdr_trace_hit(c, "QueryDisplayHdrStatus"))
        rt_log("hdr-trace: QueryDisplayHdrStatus args: enabled=%s format=%s x2=%s display=%s", on ? "set" : "null",
               arg_p(c, 1) ? "set" : "null", peak ? "set" : "null", arg_p(c, 3) ? "set" : "null");
    int hdr = hdr_answer();
    if (on) *on = (uint8_t)hdr;
    if (peak && hdr) *peak = (uint16_t)cfg_double("TOTK_HDR_PEAK_NITS");
    ret_x(c, 0);
}
// Result QueryDisplayHdrPaperWhite(u16 *nits, const Display *): the BT.2408 reference white, 203 nits, or
// hdr_game_nits while the game is told HDR is on (hdr_answer)
static void ov_hdr_paper_white(Cpu *c) {
    uint16_t *nits = arg_p(c, 0);
    if (hdr_trace_hit(c, "QueryDisplayHdrPaperWhite")) rt_log("hdr-trace: QueryDisplayHdrPaperWhite nits=%s", nits ? "set" : "null");
    if (nits) *nits = hdr_answer() ? (uint16_t)cfg_double("TOTK_HDR_GAME_NITS") : 203;
    ret_x(c, 0);
}
// Result QueryDisplayVrrSupported(bool *out, const Display *): no
static void ov_vrr_supported(Cpu *c) {
    uint8_t *out = arg_p(c, 0);
    if (out) *out = 0;
    ret_x(c, 0);
}
// Result SetDisplayVrrCapability(Display *, bool), Set/ClearLayerExtendedSwapInterval(Layer *[, int, int]),
// SetLayerNextHdr{Extended,Standard}ContentLuminanceRangeMetadata(Layer *, u16, u16): accepted, nothing changes
static void ov_vi_accept(Cpu *c) { ret_x(c, 0); }
// The four SetLayerNextHdr* calls: accepted like ov_vi_accept, with their integer arguments traced (x1, x2)
static void hdr_next_trace(Cpu *c, const char *name) {
    if (hdr_trace_hit(c, name)) rt_log("hdr-trace: %s args: x1=%llu x2=%llu", name, (unsigned long long)arg_x(c, 1), (unsigned long long)arg_x(c, 2));
    ret_x(c, 0);
}
// SetLayerNextHdrExtendedContentLuminanceRangeMetadata(Layer *, u16 min, u16 max): max is the peak the game grades
// its HDR picture to, the display's peak (QueryDisplayHdrStatus) capped by the game's own limit for the mode (1400
// nits in 3.2.0 gameplay). vk_hdr.c sends it to the display as the content's light level (hdr_game_content_peak).
static int32_t g_game_peak;   // nits; 0 before the game sent one
int hdr_game_content_peak(void) { return __atomic_load_n(&g_game_peak, __ATOMIC_ACQUIRE); }
static void ov_hdr_ext_lum(Cpu *c) {
    uint64_t max = arg_x(c, 2) & 0xffff;
    if (max) __atomic_store_n(&g_game_peak, (int32_t)max, __ATOMIC_RELEASE);
    hdr_next_trace(c, "SetLayerNextHdrExtendedContentLuminanceRangeMetadata");
}
static void ov_hdr_std_lum(Cpu *c) { hdr_next_trace(c, "SetLayerNextHdrStandardContentLuminanceRangeMetadata"); }
static void ov_hdr_gamut(Cpu *c) { hdr_next_trace(c, "SetLayerNextHdrContentGamutMetadata"); }
static void ov_hdr_tonemap(Cpu *c) { hdr_next_trace(c, "SetLayerNextHdrToSdrToneMappingMetadata"); }

HOST_OVERRIDE("_ZN2nn2vi21QueryDisplayHdrStatusEPbPNS0_15HdrOutputFormatEPtPKNS0_7DisplayE", ov_hdr_status);
HOST_OVERRIDE("_ZN2nn2vi25QueryDisplayHdrPaperWhiteEPtPKNS0_7DisplayE", ov_hdr_paper_white);
HOST_OVERRIDE("_ZN2nn2vi24QueryDisplayVrrSupportedEPbPKNS0_7DisplayE", ov_vrr_supported);
HOST_OVERRIDE("_ZN2nn2vi23SetDisplayVrrCapabilityEPNS0_7DisplayEb", ov_vi_accept);
HOST_OVERRIDE("_ZN2nn2vi28SetLayerExtendedSwapIntervalEPNS0_5LayerEii", ov_vi_accept);
HOST_OVERRIDE("_ZN2nn2vi30ClearLayerExtendedSwapIntervalEPNS0_5LayerE", ov_vi_accept);
HOST_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrExtendedContentLuminanceRangeMetadataEPNS0_5LayerEtt", ov_hdr_ext_lum);
HOST_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrStandardContentLuminanceRangeMetadataEPNS0_5LayerEtt", ov_hdr_std_lum);
HOST_OVERRIDE("_ZN2nn2vi35SetLayerNextHdrContentGamutMetadataEPNS0_5LayerENS0_15HdrContentGamutE", ov_hdr_gamut);
HOST_OVERRIDE("_ZN2nn2vi39SetLayerNextHdrToSdrToneMappingMetadataEPNS0_5LayerENS0_14HdrToneMappingE", ov_hdr_tonemap);

// void *nvnBootstrapLoader(const char *name): the first NVN call
static void ov_nvn_bootstrap(Cpu *c) {
    const char *name = arg_p(c, 0);
    if (!name) name = "(null)";
    static int32_t first = 1;
    if (rt_config()->boot && plat_xchg32(&first, 0)) rt_boot_event("nvn", c, name);   // ends --boot; returns
                                                                                                // when --seconds goes on
    ret_x(c, nvn_proc_address(name));
}

// The NVN procs need their synthetic addresses before hostfn_seal; the loader seals at the end of rt_boot, so the
// registration runs at start-up instead of through boot.c.
__attribute__((constructor)) static void ov_nvn_procs_register(void) { nvn_procs_register(); }

HOST_OVERRIDE("_ZN2nv20SetGraphicsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_", ov_graphics_allocator);
HOST_OVERRIDE("_ZN2nv18InitializeGraphicsEPvm", ov_initialize_graphics);
HOST_OVERRIDE("_ZN2nv9InitCheckEv", ov_init_check);
HOST_OVERRIDE("_ZN2nv28SetGraphicsDevtoolsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_", ov_devtools_allocator);
HOST_OVERRIDE("_ZN2nn2vi10InitializeEv", ov_vi_initialize);
HOST_OVERRIDE("_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE", ov_open_display);
HOST_OVERRIDE("_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE", ov_create_layer);
HOST_OVERRIDE("_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE", ov_native_window);
HOST_OVERRIDE("_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE", ov_vsync_event);
HOST_OVERRIDE("nvnBootstrapLoader", ov_nvn_bootstrap);
