// Scripted input (TOTK_INPUT_SCRIPT, docs/controls.md "Scripted input"): the line parser and the scheduler, free of any
// window system or game state so tests/input_script_test.c runs them alone. runtime/input.c reads the file, feeds the
// clocks (frame_rate.c frame_step_ticks, frame_step_world_us), tests the wait conditions and logs.
#pragma once
#include <stddef.h>
#include <stdint.h>

// What a wait line waits for (ScriptEv.wait): the world clock started, or the game opened a file whose path contains
// ScriptEv.arg.
enum { SCRIPT_WAIT_NONE, SCRIPT_WAIT_WORLD, SCRIPT_WAIT_FILE };

// at, hold: 60 Hz ticks of game time, or microseconds of the world clock when timed. landed: the clock reading when the
// event first applied (same unit as at), valid once logged is set.
// A wait line (wait != 0) has its time in at and its timeout in hold (same units); logged is 1 once its condition held,
// 2 once it timed out, and off_t, off_w then hold the shift (ticks, us) of the clocks the lines after it run on.
// seg: the index of the wait line that governs this line (the last one before it), -1 for none (script_link).
typedef struct {
    uint64_t at, hold, landed;
    int bit, stick, timed;
    int32_t x, y;
    int logged;
    int wait, seg;
    int64_t off_t, off_w;
    int64_t armed;   // a bare wait (at == 0): its clock reading when first polled (armed_set), the base of its timeout
    int armed_set;
    char text[48], arg[64];
} ScriptEv;

typedef struct { const char *name; int bit; } ScriptButton;
extern const ScriptButton script_buttons[16];   // the 16 Switch buttons by script name, in HidNpadButton bit order

// Parses one script line (a '#' comment is cut off in place). 1: *e holds the event; 0: blank or comment; -1: a bad
// line, err says why.
int script_parse_line(char *line, ScriptEv *e, char *err, size_t err_n);

// Sets seg of every line: the last wait line before it.
void script_link(ScriptEv *ev, int n);

// The clocks line i runs on: tick and *w shifted by its wait line, so they read that line's time when it held. 0: the
// wait line has not held yet (line i cannot apply).
int script_clock(const ScriptEv *ev, int i, uint64_t *tick, int64_t *w);

// Wait line i at raw clocks tick and w (-1 before the world clock), met: its condition holds now. 1: it holds now, 2: it
// timed out now (its clock reached its time plus its timeout); either way the clocks after it are shifted so they read
// its time from here on. 0: still waiting, or already done, or its own wait line has not held.
int script_wait(ScriptEv *ev, int i, uint64_t tick, int64_t w, int met);

// Whether e is applied in this poll. tick: frame_step_ticks; w: frame_step_world_us (-1 before the world clock), both as
// script_clock gives them for e; mixed: the script has timed lines, so its untimed ones end where the world clock
// starts. On the first poll that applies e, sets logged and landed and returns 2 (log it), later polls inside the hold
// return 1.
int script_due(ScriptEv *e, uint64_t tick, int64_t w, int mixed);

// The landing summary at raw clocks tick and w: how many events applied, how many can still apply (pending; an untimed
// event whose hold passed, or that the world clock ended, never will; an event behind a wait that has not held still
// can), and the latest landing against the time asked, in ticks and in ms of the world clock. Each event lands at the
// first poll at or after its time, so the lateness shows how coarse the clock steps were there (a stall, a load). waits
// and timeouts count the wait lines and those that timed out; a wait still waiting counts as pending.
typedef struct { int total, landed, pending, waits, timeouts; uint64_t max_late_ticks; double max_late_ms; } ScriptSummary;
ScriptSummary script_summary(const ScriptEv *ev, int n, uint64_t tick, int64_t w, int mixed);
