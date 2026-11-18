// How the finished picture is scaled to the window (setting window_filter), as plain functions: the filter the setting
// and the legacy upscaler = bilinear alias give, whether the filter has anything to do at this window size, and which
// passes the present records. No Vulkan and no config: vk_present.c, vk_fsr.c and ui/menu.cpp call them,
// and tests/window_filter_test.c checks them. Modelled on Hookshot's gpu/vk/upscaler_rows.h filter_needed.
#pragma once

#include <string.h>

enum { WF_LINEAR = 0, WF_CATMULL_ROM = 1, WF_LANCZOS = 2, WF_FSR1 = 3, WF_BLIT = 4 };   // WF_BLIT: upscaler = bilinear

// The setting's value as a WF_ constant; anything it does not name gives its default, AMD FSR 1 (today's scaling).
static inline int wf_parse(const char *value) {
    if (!value) return WF_FSR1;
    if (!strcmp(value, "linear")) return WF_LINEAR;
    if (!strcmp(value, "catmull_rom")) return WF_CATMULL_ROM;
    if (!strcmp(value, "lanczos")) return WF_LANCZOS;
    return WF_FSR1;
}

// The filter in effect. filter_set: the player set window_filter (dkb.ini or the environment; cfg_is_set). Before that
// setting existed, upscaler = bilinear chose the plain blit with no sharpening, so a player who has not chosen a filter
// and has that value keeps it (WF_BLIT); every other case takes the setting's value (its default is FSR 1).
static inline int wf_resolve(const char *window_filter, int filter_set, const char *upscaler) {
    if (!filter_set && upscaler && !strcmp(upscaler, "bilinear")) return WF_BLIT;
    return wf_parse(window_filter);
}

// Whether the filter has anything to do: the picture (pw x ph, the crop of the game's image) is not the size the window
// shows it at (ww x wh, the fitted rectangle). While this is 0 the menu's filter row says the filter only matters for
// Sharpness there (wf_plan: Linear, Catmull-Rom and Lanczos still sharpen at the picture's size).
static inline int wf_needed(int pw, int ph, int ww, int wh) { return pw != ww || ph != wh; }

// What the present records for one frame.
typedef struct {
    int fsr1;      // vk_fsr.c: EASU, then RCAS when sharp > 0 (the window is larger than the picture and the filter is FSR 1)
    int pass;      // vk_fsr.c: its pass runs (instead of the plain blit) with mode and sharp below
    int mode;      // the pass's filter, WF_LINEAR, WF_CATMULL_ROM or WF_LANCZOS
    int sharp;     // 0..100: the strength of the sharpen that pass applies, or RCAS's for the FSR 1 path
} WfPlan;

// filter: wf_resolve. sharpness: the setting, 0..100. pw x ph: the picture, ww x wh: the rectangle it is shown in.
// native_hdr: the picture is the game's own HDR one (PQ code values). FSR 1 and the legacy plain blit keep today's
// scaling; the other filters are the player's choice, and only they bring the pass, its sharpen at any size and the
// HDR picture:
// - FSR 1 (the default): EASU + RCAS where the window is larger than an SDR picture, as before; nothing otherwise (the
//   plain blit), and nothing for the HDR picture, which EASU and RCAS do not take.
// - WF_BLIT (upscaler = bilinear, no filter chosen): the plain blit, as before.
// - Catmull-Rom and Lanczos: the pass whenever the size differs; at the size, only to sharpen.
// - Linear: the plain blit, which is the linear filter, unless the pass has a sharpen to apply.
// Nothing extra runs at the picture's size with the slider at 0.
static inline WfPlan wf_plan(int filter, int sharpness, int native_hdr, int pw, int ph, int ww, int wh) {
    WfPlan p = {0, 0, WF_LINEAR, 0};
    int sharp = sharpness < 0 ? 0 : sharpness > 100 ? 100 : sharpness;
    int needed = wf_needed(pw, ph, ww, wh);
    if (filter == WF_BLIT) return p;
    if (filter == WF_FSR1) {
        if (!native_hdr && (ww > pw || wh > ph)) p.fsr1 = 1, p.sharp = sharp;
        return p;
    }
    p.mode = filter == WF_FSR1 ? WF_LINEAR : filter;
    p.sharp = sharp;
    p.pass = sharp > 0 || (needed && p.mode != WF_LINEAR);
    return p;
}

// Whether the Sharpness slider does anything with this filter at these sizes: the plan for a non-zero sharpness runs
// FSR 1 (whose RCAS takes it) or the pass. 0 for FSR 1 where the window is not larger than an SDR picture, and for the
// legacy plain blit; the menu says so under the Sharpness row.
static inline int wf_sharpens(int filter, int native_hdr, int pw, int ph, int ww, int wh) {
    WfPlan p = wf_plan(filter, 100, native_hdr, pw, ph, ww, wh);
    return p.fsr1 || p.pass;
}
