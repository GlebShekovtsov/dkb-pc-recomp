// Pacing arithmetic of the frame generation (vk_framegen.c), kept free of Vulkan and of clocks so a test can drive it with
// a synthetic one. Times are nanoseconds. From Hookshot gpu/vk/vk_framegen.cpp (display_main: the interval average and
// the 100 ms rule), https://github.com/NotRexed/Hookshot @ fab8cb37641714bd839ae83c2a7bc582bedc4d70, GPL-3.0.
#pragma once
#include <stdint.h>

#define FG_HITCH_NS 100000000ll   // a gap between two jobs over this is a hitch: no made frame, the history starts again

// Whether the gap since the previous job is a hitch (loading, a pause): nothing to blend across it.
static inline int fg_pace_hitch(int64_t gap) { return gap > FG_HITCH_NS; }

// The game's frame interval, an exponential average of the gaps between jobs: I = (7 I + clamp(gap, 4 ms, 50 ms)) / 8,
// and the first gap as it is. interval 0 means none yet; call it only with a gap that is not a hitch.
static inline int64_t fg_pace_interval(int64_t interval, int64_t gap) {
    int64_t g = gap < 4000000 ? 4000000 : gap > 50000000 ? 50000000 : gap;
    return interval ? (interval * 7 + g) / 8 : g;
}

// When the game's own frame is due once the made frame went out at made_at: half an interval later, so the two are
// evenly spaced and the real frame is shown at the game's pace.
static inline int64_t fg_pace_due(int64_t made_at, int64_t interval) { return made_at + interval / 2; }

// Whether the display is too slow for 2x: the shown rate 2 / interval exceeds the display's refresh by more than 5%, so
// the made frames could not all be shown. refresh_hz <= 0 (unknown) counts as 60 Hz; interval <= 0 (none yet) is not over.
static inline int fg_pace_over_refresh(int64_t interval, double refresh_hz) {
    if (interval <= 0) return 0;
    if (refresh_hz <= 0) refresh_hz = 60.0;
    return 2e9 / (double)interval > refresh_hz * 1.05;
}
