// The frame generation's pacing arithmetic (runtime/fg_pace.h) on a synthetic clock: no Vulkan, no game files.
#include <stdio.h>

#include "fg_pace.h"

static int fails, checks;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

#define MS 1000000ll

// Feeds n jobs gap ns apart and returns the interval reached.
static int64_t run(int64_t interval, int64_t gap, int n) {
    for (int i = 0; i < n; i++) interval = fg_pace_interval(interval, gap);
    return interval;
}

int main(void) {
    // the first gap is taken as it is, and a steady rate stays where it is
    CHECK(fg_pace_interval(0, 16666667) == 16666667);
    for (int fps = 30; fps <= 60; fps += 15) {
        int64_t gap = 1000 * MS / fps, iv = run(0, gap, 200);
        CHECK(iv == gap);
        // the made frame and the game's frame are half an interval apart: shown evenly
        CHECK(fg_pace_due(5 * MS, iv) - 5 * MS == gap / 2);
    }
    // a change of rate is followed: from 60 fps to 30 fps within 1% after 40 jobs, without overshoot
    int64_t iv = run(run(0, 16666667, 50), 33333333, 40);
    CHECK(iv <= 33333333 && iv > 33000000);
    // gaps that alternate around a mean settle between them
    iv = 20 * MS;
    for (int i = 0; i < 200; i++) iv = fg_pace_interval(iv, i & 1 ? 28 * MS : 12 * MS);
    CHECK(iv > 12 * MS && iv < 28 * MS);
    // one gap counts as at least 4 ms and at most 50 ms
    CHECK(fg_pace_interval(0, 1 * MS) == 4 * MS);
    CHECK(fg_pace_interval(0, 90 * MS) == 50 * MS);
    CHECK(fg_pace_interval(16 * MS, 90 * MS) == (16 * MS * 7 + 50 * MS) / 8);
    // a hitch is a gap over 100 ms, not one of exactly 100 ms
    CHECK(!fg_pace_hitch(100 * MS));
    CHECK(fg_pace_hitch(100 * MS + 1));
    CHECK(!fg_pace_hitch(33 * MS));
    // over refresh: 2 / interval above the display's refresh by more than 5%
    CHECK(!fg_pace_over_refresh(0, 60.0));                 // no interval yet
    CHECK(fg_pace_over_refresh(16666667, 60.0));           // 60 fps on 60 Hz: 120 shown
    CHECK(!fg_pace_over_refresh(33333333, 60.0));          // 30 fps on 60 Hz: exactly 60
    CHECK(!fg_pace_over_refresh(31746032, 60.0));          // 63 shown: the 5% margin
    CHECK(fg_pace_over_refresh(31000000, 60.0));           // 64.5 shown
    CHECK(!fg_pace_over_refresh(16666667, 120.0));         // 60 fps on 120 Hz
    CHECK(fg_pace_over_refresh(16666667, 0.0));            // unknown refresh counts as 60 Hz
    CHECK(!fg_pace_over_refresh(33333333, 0.0));
    // a fixed multiplier makes what the display can show, the multiplier at most
    CHECK(fg_pace_fixed_frames(2, 60.0, 16666667) == 1);    // 60 fps on 60 Hz: the game's frames alone
    CHECK(fg_pace_fixed_frames(2, 60.0, 33333333) == 2);    // 30 fps on 60 Hz
    CHECK(fg_pace_fixed_frames(2, 120.0, 16666667) == 2);
    CHECK(fg_pace_fixed_frames(8, 180.0, 16666667) == 3);   // 60 fps on 180 Hz: 3 refreshes a game frame
    CHECK(fg_pace_fixed_frames(8, 180.0, 16400000) == 3);   // a little over 60 fps keeps its 3
    CHECK(fg_pace_fixed_frames(3, 165.0, 33333333) == 3);   // the multiplier is the most
    CHECK(fg_pace_fixed_frames(4, 0.0, 16666667) == 4);     // unknown refresh: the multiplier
    CHECK(fg_pace_fixed_frames(4, 60.0, 0) == 4);           // no interval yet
    CHECK(fg_pace_fixed_frames(0, 60.0, 16666667) == 1);
    // Auto's cap: the refresh over 30 fps, 2 to 4
    CHECK(fg_pace_auto_cap(60.0) == 2);
    CHECK(fg_pace_auto_cap(120.0) == 4);
    CHECK(fg_pace_auto_cap(90.0) == 3);
    CHECK(fg_pace_auto_cap(165.0) == 4);
    CHECK(fg_pace_auto_cap(30.0) == 2);
    CHECK(fg_pace_auto_cap(0.0) == 2);
    // Auto's slots, a steady game on a steady display for 4 s: every refresh gets one frame, no more
    const struct { int fps, hz; } rates[] = {{30, 60}, {45, 60}, {60, 60}, {60, 120}, {40, 165}, {60, 165}, {58, 144}, {90, 60}};
    for (unsigned r = 0; r < sizeof rates / sizeof *rates; r++) {
        const int64_t gap = 1000 * MS / rates[r].fps, period = 1000 * MS / rates[r].hz;
        const int most = fg_pace_auto_cap(rates[r].hz), jobs = rates[r].fps * 4;
        int64_t next = 0, last = 100 * MS, prev_at = 0;
        int frames = 0, game = 0, ordered = 1, in_range = 1, per_job_ok = 1;
        for (int i = 0; i < jobs; i++) {
            const int64_t now = last + gap;
            float t[FG_MAX_MULT];
            int64_t at[FG_MAX_MULT];
            const int n = fg_pace_auto_slots(last, now, period, &next, t, at, most);
            if (n < 0 || n > most) per_job_ok = 0;
            for (int k = 0; k < n; k++) {
                if (t[k] <= 0.0f || t[k] > 1.0f || at[k] <= last || at[k] > now) in_range = 0;
                if (at[k] <= prev_at) ordered = 0;
                prev_at = at[k];
                if (t[k] > 0.999f) {
                    game++;
                    if (k != n - 1) ordered = 0;   // the game's own frame is the set's last
                }
            }
            frames += n;
            last = now;
        }
        const int want = rates[r].hz * 4;   // the display's refreshes in the 4 s
        CHECK(per_job_ok && in_range && ordered);
        // the frames shown follow the display's rate within 3%, or the game's when it is the faster of the two with the cap
        const int shown_most = most * jobs;
        CHECK(frames <= want + want * 3 / 100 + 2);
        CHECK(frames >= (want < shown_most ? want : shown_most) * 97 / 100 - 2 || rates[r].fps >= rates[r].hz);
        CHECK(game <= jobs);
        if (rates[r].fps == rates[r].hz) CHECK(game >= jobs - 2 && frames <= jobs + 1);   // each refresh shows the game's own frame
        if (rates[r].fps > rates[r].hz) CHECK(frames < jobs);                             // some game frames get no refresh
    }
    // a timeline outside the interval (the first frame, after a hitch) starts again at now: the game's own frame
    {
        int64_t next = 0;
        float t[FG_MAX_MULT];
        int64_t at[FG_MAX_MULT];
        CHECK(fg_pace_auto_slots(100 * MS, 116 * MS, 8 * MS, &next, t, at, 4) == 1);
        CHECK(t[0] == 1.0f && at[0] == 116 * MS && next > 116 * MS);
        CHECK(fg_pace_auto_slots(116 * MS, 116 * MS, 8 * MS, &next, t, at, 4) == 0);   // no interval
        CHECK(fg_pace_auto_slots(100 * MS, 116 * MS, 0, &next, t, at, 4) == 0);        // no period
    }
    // the display thread's due times (fg_pace_show_due)
    CHECK(fg_pace_show_due(1, 50 * MS, 0, 40 * MS, 30 * MS, 20 * MS, 10 * MS) == 50 * MS);    // timed: its refresh's time
    CHECK(fg_pace_show_due(1, 30 * MS, 0, 40 * MS, 30 * MS, 20 * MS, 10 * MS) == 40 * MS);    // never before now
    CHECK(fg_pace_show_due(1, 200 * MS, 0, 40 * MS, 30 * MS, 20 * MS, 10 * MS) == 40 * MS);   // over 100 ms ahead: now
    CHECK(fg_pace_show_due(0, 0, 45 * MS, 40 * MS, 30 * MS, 20 * MS, 10 * MS) == 45 * MS);    // the line
    CHECK(fg_pace_show_due(0, 0, 90 * MS, 40 * MS, 30 * MS, 20 * MS, 10 * MS) == 60 * MS);    // a game frame and a step at most
    CHECK(fg_pace_show_due(0, 0, 20 * MS, 40 * MS, 30 * MS, 20 * MS, 10 * MS) == 40 * MS);    // behind: from now
    // The display thread on a synthetic clock: the game hands over a set every gap with up to +-jitter (a fixed pattern),
    // each frame shown at its due time (presents take no time). Fixed 2x and 4x: the frames go out evenly, interval / n
    // apart, the gaps between shown frames within 2 ms of that on at least 95% of them, and no set is shown later than
    // the line allows.
    for (int mult = 2; mult <= 4; mult += 2) {
        const int64_t gap = 22222222;   // 45 fps
        int64_t interval = 16666667, last_at = 0, next_show = 0, shown_at = 0, clock = 0;
        int even = 0, total = 0, behind = 0;
        for (int i = 0; i < 400; i++) {
            const int64_t at = 100 * MS + i * gap + ((i * 7919) % 9 - 4) * MS / 2;   // +-2 ms
            if (clock < at) clock = at;   // the display thread waits for the set
            if (last_at && at - last_at > 0 && !fg_pace_hitch(at - last_at)) interval = fg_pace_interval(interval, at - last_at);
            last_at = at;
            const int64_t step = interval / mult;
            for (int k = 0; k < mult; k++) {
                const int64_t due = fg_pace_show_due(0, 0, next_show, clock, at, interval, step);
                if (due < clock) behind++;
                clock = due;
                if (i > 20 && shown_at) {
                    total++;
                    const int64_t d = clock - shown_at, want = gap / mult;
                    even += d > want - 2 * MS && d < want + 2 * MS;
                }
                shown_at = clock, next_show = clock + step;
            }
        }
        CHECK(!behind);
        CHECK(even * 100 >= total * 95);
    }
    // Auto, 45 fps on 165 Hz: the timed frames go out at the refreshes' times, about 6.06 ms apart
    {
        const int64_t gap = 22222222, period = 6060606;
        int64_t interval = 0, last_at = 0, next = 0, shown_at = 0, clock = 0;
        int even = 0, total = 0;
        for (int i = 0; i < 400; i++) {
            const int64_t at = 100 * MS + i * gap;
            if (last_at) interval = fg_pace_interval(interval, at - last_at);
            float t[FG_MAX_MULT];
            int64_t when[FG_MAX_MULT];
            const int n = last_at ? fg_pace_auto_slots(last_at, at, period, &next, t, when, fg_pace_auto_cap(165)) : 0;
            last_at = at;
            if (clock < at) clock = at;
            for (int k = 0; k < n; k++) {
                const int64_t due = fg_pace_show_due(1, when[k] + interval, 0, clock, at, interval, 0);
                clock = due;
                if (i > 20 && shown_at) {
                    total++;
                    even += clock - shown_at > period - MS && clock - shown_at < period + MS;
                }
                shown_at = clock;
            }
        }
        CHECK(total > 0 && even * 100 >= total * 90);
    }
    printf("fg_pace: %d checks, %d failures\n", checks, fails);
    return fails != 0;
}
