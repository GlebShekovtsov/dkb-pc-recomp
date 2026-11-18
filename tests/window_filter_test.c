// The window scaling filter's rules (runtime/window_filter.h): the legacy upscaler = bilinear alias, when the filter
// has anything to do, which passes the present records and whether the slider sharpens. No Vulkan, no config.
#include <stdio.h>

#include "window_filter.h"

static int failures;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); failures++; } } while (0)

int main(void) {
    // parse: the four values, anything else the default
    CHECK(wf_parse("linear") == WF_LINEAR && wf_parse("catmull_rom") == WF_CATMULL_ROM);
    CHECK(wf_parse("lanczos") == WF_LANCZOS && wf_parse("fsr1") == WF_FSR1);
    CHECK(wf_parse("") == WF_FSR1 && wf_parse("bogus") == WF_FSR1 && wf_parse(NULL) == WF_FSR1);   // the default

    // legacy rule: not set + bilinear = linear; set wins; any other upscaler takes the setting (default Lanczos)
    CHECK(wf_resolve("fsr1", 0, "bilinear") == WF_BLIT);   // the legacy alias keeps its plain blit
    CHECK(wf_resolve("lanczos", 1, "bilinear") == WF_LANCZOS);
    CHECK(wf_resolve("fsr1", 1, "bilinear") == WF_FSR1);
    CHECK(wf_resolve("fsr1", 0, "fsr") == WF_FSR1);
    CHECK(wf_resolve("fsr1", 0, "fsr4") == WF_FSR1 && wf_resolve("catmull_rom", 1, "dlss") == WF_CATMULL_ROM);
    CHECK(wf_resolve("fsr1", 0, NULL) == WF_FSR1);

    // needed: only while the window's rectangle differs from the picture
    CHECK(!wf_needed(1920, 1080, 1920, 1080));
    CHECK(wf_needed(1920, 1080, 2560, 1440) && wf_needed(1920, 1080, 1280, 720) && wf_needed(1920, 1080, 1920, 1081));

    // plan: nothing at the picture's size with the slider at 0, whatever the filter
    for (int f = WF_LINEAR; f <= WF_FSR1; f++) {
        WfPlan p = wf_plan(f, 0, 0, 1920, 1080, 1920, 1080);
        CHECK(!p.fsr1 && !p.pass);
    }
    // at the picture's size the slider alone runs the pass, with FSR 1 as linear
    WfPlan p = wf_plan(WF_FSR1, 80, 0, 1920, 1080, 1920, 1080);   // FSR 1 at the picture's size: nothing, as before
    CHECK(!p.fsr1 && !p.pass);
    p = wf_plan(WF_BLIT, 80, 0, 1920, 1080, 2560, 1440);   // upscaler = bilinear: the plain blit, no sharpen
    CHECK(!p.fsr1 && !p.pass);
    p = wf_plan(WF_LANCZOS, 80, 0, 1920, 1080, 1920, 1080);
    CHECK(!p.fsr1 && p.pass && p.mode == WF_LANCZOS && p.sharp == 80);
    // linear, slider 0, any size: the plain blit; with the slider: the pass
    p = wf_plan(WF_LINEAR, 0, 0, 1920, 1080, 3840, 2160);
    CHECK(!p.fsr1 && !p.pass);
    p = wf_plan(WF_LINEAR, 0, 0, 3840, 2160, 1920, 1080);
    CHECK(!p.fsr1 && !p.pass);
    p = wf_plan(WF_LINEAR, 30, 0, 1920, 1080, 3840, 2160);
    CHECK(p.pass && p.mode == WF_LINEAR && p.sharp == 30);
    // Catmull-Rom and Lanczos run the pass when the size differs, either way
    p = wf_plan(WF_CATMULL_ROM, 0, 0, 1920, 1080, 3840, 2160);
    CHECK(!p.fsr1 && p.pass && p.mode == WF_CATMULL_ROM && p.sharp == 0);
    p = wf_plan(WF_LANCZOS, 0, 0, 3840, 2160, 1920, 1080);
    CHECK(!p.fsr1 && p.pass && p.mode == WF_LANCZOS);
    // FSR 1: EASU + RCAS when the window is larger; linear pass when shrinking
    p = wf_plan(WF_FSR1, 50, 0, 1920, 1080, 2560, 1440);
    CHECK(p.fsr1 && !p.pass && p.sharp == 50);
    p = wf_plan(WF_FSR1, 0, 0, 1920, 1080, 2560, 1440);
    CHECK(p.fsr1 && p.sharp == 0);
    p = wf_plan(WF_FSR1, 0, 0, 1920, 1080, 1280, 720);
    CHECK(!p.fsr1 && p.pass == 0);   // shrinking, slider 0: the plain linear blit
    p = wf_plan(WF_FSR1, 40, 0, 1920, 1080, 1280, 720);   // shrinking with FSR 1: the plain blit, as before
    CHECK(!p.fsr1 && !p.pass);
    // the slider is clamped
    p = wf_plan(WF_LINEAR, 500, 0, 10, 10, 10, 10);
    CHECK(p.sharp == 100);
    p = wf_plan(WF_LINEAR, -5, 0, 10, 10, 10, 10);
    CHECK(!p.pass && p.sharp == 0);

    // native HDR (PQ codes): the same plan as SDR, but AMD FSR 1 enlarging is the pass's Lanczos
    // the game's own HDR picture with FSR 1 (the default) or the legacy blit: blitted as it is, whatever the sizes
    for (int f = WF_FSR1; f <= WF_BLIT; f++) {
        const int sizes[][4] = {{1920, 1080, 2560, 1440}, {1920, 1080, 1920, 2160}, {1920, 1080, 1280, 720},
                                {1920, 1080, 1920, 1080}};
        for (int k = 0; k < 4; k++) {
            p = wf_plan(f, 60, 1, sizes[k][0], sizes[k][1], sizes[k][2], sizes[k][3]);
            CHECK(!p.fsr1 && !p.pass);
        }
    }
    p = wf_plan(WF_CATMULL_ROM, 0, 1, 1920, 1080, 3840, 2160);
    CHECK(!p.fsr1 && p.pass && p.mode == WF_CATMULL_ROM);
    p = wf_plan(WF_LANCZOS, 80, 1, 1920, 1080, 1920, 1080);
    CHECK(p.pass && p.mode == WF_LANCZOS && p.sharp == 80);
    p = wf_plan(WF_LINEAR, 0, 1, 1920, 1080, 3840, 2160);
    CHECK(!p.fsr1 && !p.pass);
    // SDR never reports the stand-in
    p = wf_plan(WF_FSR1, 0, 0, 1920, 1080, 2560, 1440);
    CHECK(p.fsr1);

    // sharpens (the menu's Sharpness note): the three chosen filters at any size, the picture's included, SDR or HDR
    for (int f = WF_LINEAR; f <= WF_LANCZOS; f++)
        for (int hdr = 0; hdr < 2; hdr++)
            CHECK(wf_sharpens(f, hdr, 1920, 1080, 1920, 1080) && wf_sharpens(f, hdr, 1920, 1080, 2560, 1440) &&
                  wf_sharpens(f, hdr, 1920, 1080, 1280, 720));
    // FSR 1 only where the window is larger than an SDR picture; the legacy blit never
    CHECK(wf_sharpens(WF_FSR1, 0, 1920, 1080, 2560, 1440));
    CHECK(!wf_sharpens(WF_FSR1, 0, 1920, 1080, 1920, 1080) && !wf_sharpens(WF_FSR1, 0, 1920, 1080, 1280, 720));
    CHECK(!wf_sharpens(WF_FSR1, 1, 1920, 1080, 2560, 1440));
    CHECK(!wf_sharpens(WF_BLIT, 0, 1920, 1080, 2560, 1440) && !wf_sharpens(WF_BLIT, 0, 1920, 1080, 1920, 1080));

    if (failures) return 1;
    printf("window_filter: ok\n");
    return 0;
}
