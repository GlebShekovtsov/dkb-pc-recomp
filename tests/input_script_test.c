// Scripted input parser and scheduler (runtime/input_script.c), driven by synthetic clocks.
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "input_map.h"
#include "input_script.h"

static int parse(const char *text, ScriptEv *e) {
    char line[256], err[64] = "";
    snprintf(line, sizeof line, "%s", text);
    return script_parse_line(line, e, err, sizeof err);
}

// Polls e once per tick from tick 0 to last (untimed clock), returns the tick of the first landing or -1.
static int land_tick(ScriptEv *e, int last) {
    for (int t = 0; t <= last; t++)
        if (script_due(e, (uint64_t)t, -1, 0) == 2) return t;
    return -1;
}

int main(void) {
    ScriptEv e;
    // Comments, blanks and bad lines.
    assert(parse("# only a comment\n", &e) == 0);
    assert(parse("\n", &e) == 0);
    assert(parse("-5 A\n", &e) == 0);
    assert(parse("10 Q\n", &e) == -1);
    assert(parse("10 LSTICK 5\n", &e) == -1);

    // Untimed button with the default hold, as in tools/input/load_save.txt.
    assert(parse("300 A 4\n", &e) == 1);
    assert(!e.timed && e.at == 300 && e.hold == 4 && e.bit == B_A && !e.stick && !strcmp(e.text, "A"));
    assert(land_tick(&e, 400) == 300);
    assert(e.landed == 300);
    assert(script_due(&e, 303, -1, 0) == 1);   // still held
    assert(script_due(&e, 304, -1, 0) == 0);   // released after 4 ticks

    // Untimed stick with a hold and a trailing comment.
    assert(parse("20 RSTICK 32767 -100 90 # turn\n", &e) == 1);
    assert(e.stick == 2 && e.x == 32767 && e.y == -100 && e.hold == 90 && e.at == 20);

    // Timed button: ms to us, default hold 66.667 ms.
    assert(parse("@1500.5 ZR\n", &e) == 1);
    assert(e.timed && e.at == 1500500 && e.hold == 66667 && e.bit == B_ZR);
    assert(script_due(&e, 9999, -1, 1) == 0);         // the world clock has not started
    assert(script_due(&e, 0, 1500000, 1) == 0);
    assert(script_due(&e, 0, 1516667, 1) == 2);       // the first poll at or after the time asked
    assert(e.landed == 1516667);
    assert(script_due(&e, 0, 1533333, 1) == 1);
    assert(script_due(&e, 0, 1600000, 1) == 0);

    // A timed event shorter than a clock step applies once even when the clock jumps past it (a stall).
    assert(parse("@100 A 1\n", &e) == 1);
    assert(e.hold == 1000);
    assert(script_due(&e, 0, 250000, 1) == 2);
    assert(script_due(&e, 0, 266667, 1) == 0);

    // In a mixed script the untimed events end where the world clock starts.
    assert(parse("50 B\n", &e) == 1);
    assert(script_due(&e, 50, 0, 1) == 0);
    assert(script_due(&e, 50, -1, 1) == 2);

    // Summary: pending until every event landed or can no longer land; lateness against the time asked.
    ScriptEv ev[3];
    assert(parse("10 A 2\n", &ev[0]) == 1);
    assert(parse("12 B 2\n", &ev[1]) == 1);
    assert(parse("@40 X\n", &ev[2]) == 1);
    ScriptSummary s = script_summary(ev, 3, 0, -1, 1);
    assert(s.total == 3 && s.landed == 0 && s.pending == 3);
    assert(script_due(&ev[0], 11, -1, 1) == 2);        // one tick late (a frame of interval 2)
    s = script_summary(ev, 3, 11, -1, 1);
    assert(s.landed == 1 && s.pending == 2 && s.max_late_ticks == 1);
    // The world clock starts before B landed: B can no longer land, X can.
    s = script_summary(ev, 3, 12, 0, 1);
    assert(s.pending == 1);
    assert(script_due(&ev[1], 12, 0, 1) == 0);
    assert(script_due(&ev[2], 12, 50000, 1) == 2);
    s = script_summary(ev, 3, 12, 50000, 1);
    assert(s.total == 3 && s.landed == 2 && s.pending == 0 && s.max_late_ticks == 1);
    assert(s.max_late_ms > 9.99 && s.max_late_ms < 10.01);

    // Every button name maps to its own bit.
    for (int i = 0; i < 16; i++) {
        char line[32];
        snprintf(line, sizeof line, "1 %s", script_buttons[i].name);
        assert(parse(line, &e) == 1 && e.bit == script_buttons[i].bit && e.bit == i);
    }
    puts("input_script_test: ok");
    return 0;
}
