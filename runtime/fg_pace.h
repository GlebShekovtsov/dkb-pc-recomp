// Pacing arithmetic of the frame generation (vk_framegen.c), kept free of Vulkan and of clocks so a test can drive it with
// a synthetic one. Times are nanoseconds. From Hookshot gpu/vk/vk_framegen.cpp (display_main: the interval average and
// the 100 ms rule; show_due, auto_cap, fixed_frames and auto_slots: Auto and the multipliers),
// https://git.notrexed.dev/Hookshot/Hookshot @ 7191250b7b0f8a4becd79ee086229a2620872fc7, GPL-3.0 (the 2x path's
// pacing from https://github.com/NotRexed/Hookshot @ fab8cb37641714bd839ae83c2a7bc582bedc4d70, GPL-3.0).
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

// 2x: when the game's own frame is due once the made frame went out at made_at: half an interval later, so the two are
// evenly spaced and the real frame is shown at the game's pace.
static inline int64_t fg_pace_due(int64_t made_at, int64_t interval) { return made_at + interval / 2; }

// 2x: whether the display is too slow for it: the shown rate 2 / interval exceeds the display's refresh by more than 5%, so
// the made frames could not all be shown. refresh_hz <= 0 (unknown) counts as 60 Hz; interval <= 0 (none yet) is not over.
static inline int fg_pace_over_refresh(int64_t interval, double refresh_hz) {
    if (interval <= 0) return 0;
    if (refresh_hz <= 0) refresh_hz = 60.0;
    return 2e9 / (double)interval > refresh_hz * 1.05;
}

// When the display thread shows frame k of a set (Hookshot vk_framegen.cpp display_main): timed (Auto) at its own
// refresh's time, but never over 100 ms ahead of now (a game frame time gone wrong is not held back); otherwise on the
// even line, next_show (the frame shown before plus the set's step, interval / frames), but never later than one step
// past one game frame after the set was made (job_at + interval + step), so the line stays at most one game frame
// behind what the game handed over and starts again from now when it fell behind; never before now.
static inline int64_t fg_pace_show_due(int timed, int64_t set_due, int64_t next_show, int64_t now, int64_t job_at,
                                       int64_t interval, int64_t step) {
    int64_t due = timed ? set_due : next_show;
    if (due < now) due = now;
    if (!timed && due > job_at + interval + step) due = job_at + interval + step;
    if (timed && due > now + 100000000) due = now;
    return due;
}

#define FG_MAX_MULT 8   // the highest fixed multiplier; a set holds at most this many frames

// Auto's most frames a game frame may be given: the display's refresh over 30 fps, 2 to 4 (Hookshot auto_cap). An
// unknown refresh (<= 0) counts as 60 Hz.
static inline int fg_pace_auto_cap(double refresh_hz) {
    int hz = (int)(refresh_hz + 0.5);
    int n = hz > 0 ? (hz + 29) / 30 : 2;
    return n < 2 ? 2 : n > 4 ? 4 : n;
}

// A fixed multiplier m makes as many frames a game frame as the display can show, m at most, so no frame is made that
// never reaches the screen (Hookshot fixed_frames): the display's refreshes in the game's frame time, 1 (the game's
// frame alone) at the refresh. An unknown refresh or interval: m.
static inline int fg_pace_fixed_frames(int m, double refresh_hz, int64_t interval) {
    if (m <= 1 || refresh_hz <= 0 || interval <= 0) return m > 1 ? m : 1;
    int fit = (int)(refresh_hz * (double)interval / 1e9 + 0.2);   // +0.2: a game a little over 60 fps on 180 Hz keeps its 3
    return fit < 1 ? 1 : fit > m ? m : fit;
}

// Auto (Hookshot auto_slots): the display's refreshes in the game frame's interval (last, now], *next the first one not
// given a frame yet (moved past them). Each one's place between the two game frames goes to t (1: this game frame) and
// its time to at; `most` of them at the most, those past it skipped. A timeline outside the interval (the first frame,
// a hitch) starts again at now. A refresh within a quarter period of the game's frame shows that frame itself, and the
// timeline is pulled an eighth of the way to it, so a game at the display's rate (or a whole fraction of it) stays on
// the line while the two clocks drift. Returns the number of frames; 0 when no refresh fell in the interval.
static inline int fg_pace_auto_slots(int64_t last, int64_t now, int64_t period, int64_t *next, float *t, int64_t *at,
                                     int most) {
    if (now <= last || period <= 0 || most <= 0) return 0;
    if (*next <= last || *next > now + period) *next = now;
    int n = 0;
    for (; *next <= now; *next += period)
        if (n < most) t[n] = (float)(*next - last) / (float)(now - last), at[n++] = *next;
    const int64_t snap = period / 4;
    if (*next - now < snap) {
        if (n < most) n++;
        t[n - 1] = 1.0f, at[n - 1] = now, *next += period + (now - *next) / 8;
    } else if (n > 0 && now - (*next - period) < snap) {
        t[n - 1] = 1.0f, at[n - 1] = now, *next += (now - (*next - period)) / 8;
    }
    return n;
}
