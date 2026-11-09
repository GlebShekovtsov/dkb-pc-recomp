// Scripted input (TOTK_INPUT_SCRIPT, docs/controls.md "Scripted input"): the line parser and the scheduler, free of any
// window system or game state so tests/input_script_test.c runs them alone. runtime/input.c reads the file, feeds the
// clocks (frame_rate.c frame_step_ticks, frame_step_world_us) and logs.
#pragma once
#include <stddef.h>
#include <stdint.h>

// at, hold: 60 Hz ticks of game time, or microseconds of the world clock when timed. landed: the clock reading when the
// event first applied (same unit as at), valid once logged is set.
typedef struct { uint64_t at, hold, landed; int bit, stick, timed; int32_t x, y; int logged; char text[48]; } ScriptEv;

typedef struct { const char *name; int bit; } ScriptButton;
extern const ScriptButton script_buttons[16];   // the 16 Switch buttons by script name, in HidNpadButton bit order

// Parses one script line (a '#' comment is cut off in place). 1: *e holds the event; 0: blank or comment; -1: a bad
// line, err says why.
int script_parse_line(char *line, ScriptEv *e, char *err, size_t err_n);

// Whether e is applied in this poll. tick: frame_step_ticks; w: frame_step_world_us (-1 before the world clock);
// mixed: the script has timed lines, so its untimed ones end where the world clock starts. On the first poll that
// applies e, sets logged and landed and returns 2 (log it), later polls inside the hold return 1.
int script_due(ScriptEv *e, uint64_t tick, int64_t w, int mixed);

// The landing summary at clocks tick and w: how many events applied, how many can still apply (pending; an untimed
// event whose hold passed, or that the world clock ended, never will), and the latest landing against the time asked,
// in ticks and in ms of the world clock. Each event lands at the first poll at or after its time, so the lateness
// shows how coarse the clock steps were there (a stall, a load).
typedef struct { int total, landed, pending; uint64_t max_late_ticks; double max_late_ms; } ScriptSummary;
ScriptSummary script_summary(const ScriptEv *ev, int n, uint64_t tick, int64_t w, int mixed);
