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
    printf("fg_pace: %d checks, %d failures\n", checks, fails);
    return fails != 0;
}
