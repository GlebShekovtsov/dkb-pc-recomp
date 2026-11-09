// Scripted input: line parser and scheduler (input_script.h). Format and clocks: docs/controls.md "Scripted input".
#include <stdio.h>
#include <string.h>

#include "input_map.h"
#include "input_script.h"

const ScriptButton script_buttons[16] = {
    {"A", B_A}, {"B", B_B}, {"X", B_X}, {"Y", B_Y}, {"LCLICK", B_STICK_L}, {"RCLICK", B_STICK_R}, {"L", B_L}, {"R", B_R},
    {"ZL", B_ZL}, {"ZR", B_ZR}, {"PLUS", B_PLUS}, {"MINUS", B_MINUS}, {"LEFT", B_LEFT}, {"UP", B_UP}, {"RIGHT", B_RIGHT}, {"DOWN", B_DOWN},
};

int script_parse_line(char *line, ScriptEv *e, char *err, size_t err_n) {
    char *h = strchr(line, '#');
    if (h) *h = 0;
    char name[32];
    double at;
    int off = 0, timed = line[0] == '@';
    if (sscanf(line + timed, "%lf %31s%n", &at, name, &off) < 2 || at < 0) return 0;
    off += timed;
    double unit = timed ? 1000.0 : 1.0;   // ms to us, or ticks
    *e = (ScriptEv){.at = (uint64_t)(at * unit + 0.5), .hold = timed ? 66667 : 4, .bit = -1, .timed = timed};
    long a, b;
    double c;
    if (!strcmp(name, "LSTICK") || !strcmp(name, "RSTICK")) {
        int n = sscanf(line + off, "%ld %ld %lf", &a, &b, &c);
        if (n < 2) { snprintf(err, err_n, "%s needs x y [hold]", name); return -1; }
        e->stick = name[0] == 'L' ? 1 : 2; e->x = (int32_t)a; e->y = (int32_t)b;
        if (n >= 3 && c > 0) e->hold = (uint64_t)(c * unit + 0.5);
        snprintf(e->text, sizeof e->text, "%s %d %d", name, e->x, e->y);
        return 1;
    }
    for (size_t i = 0; i < sizeof script_buttons / sizeof *script_buttons; i++)
        if (!strcmp(name, script_buttons[i].name)) e->bit = script_buttons[i].bit;
    if (e->bit < 0) { snprintf(err, err_n, "unknown button %s", name); return -1; }
    if (sscanf(line + off, "%lf", &c) == 1 && c > 0) e->hold = (uint64_t)(c * unit + 0.5);
    snprintf(e->text, sizeof e->text, "%s", name);
    return 1;
}

int script_due(ScriptEv *e, uint64_t tick, int64_t w, int mixed) {
    if (e->timed ? w < 0 : mixed && w >= 0) return 0;
    uint64_t now = e->timed ? (uint64_t)w : tick;
    // A timed event shorter than a clock step still applies once; an untimed one runs on whole ticks.
    if (now < e->at || (now >= e->at + e->hold && (e->logged || !e->timed))) return 0;
    if (e->logged) return 1;
    e->logged = 1;
    e->landed = now;
    return 2;
}

ScriptSummary script_summary(const ScriptEv *ev, int n, uint64_t tick, int64_t w, int mixed) {
    ScriptSummary s = {.total = n};
    for (int i = 0; i < n; i++) {
        const ScriptEv *e = &ev[i];
        if (!e->logged) {
            s.pending += e->timed || !((mixed && w >= 0) || tick >= e->at + e->hold);
            continue;
        }
        s.landed++;
        uint64_t late = e->landed - e->at;
        if (!e->timed) { if (late > s.max_late_ticks) s.max_late_ticks = late; }
        else if (late / 1000.0 > s.max_late_ms) s.max_late_ms = late / 1000.0;
    }
    return s;
}
