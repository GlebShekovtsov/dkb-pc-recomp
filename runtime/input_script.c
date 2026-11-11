// Scripted input: line parser and scheduler (input_script.h). Format and clocks: docs/controls.md "Scripted input".
#include <stdio.h>
#include <string.h>

#include "input_map.h"
#include "input_script.h"

const ScriptButton script_buttons[16] = {
    {"A", B_A}, {"B", B_B}, {"X", B_X}, {"Y", B_Y}, {"LCLICK", B_STICK_L}, {"RCLICK", B_STICK_R}, {"L", B_L}, {"R", B_R},
    {"ZL", B_ZL}, {"ZR", B_ZR}, {"PLUS", B_PLUS}, {"MINUS", B_MINUS}, {"LEFT", B_LEFT}, {"UP", B_UP}, {"RIGHT", B_RIGHT}, {"DOWN", B_DOWN},
};

enum { WAIT_TIMEOUT_MS = 60000 };

// The rest of a wait line after "wait": "<condition> [file name] [timeout_ms]"; the timeout is in ms whichever clock
// the line runs on.
static int parse_wait(const char *s, ScriptEv *e, char *err, size_t err_n) {
    char cond[16];
    int off = 0;
    double t, ms = e->timed ? 1000.0 : 0.06;   // ms to us, or to ticks
    e->hold = (uint64_t)(WAIT_TIMEOUT_MS * ms + 0.5);
    if (sscanf(s, "%15s%n", cond, &off) < 1) { snprintf(err, err_n, "wait needs a condition"); return -1; }
    s += off;
    if (!strcmp(cond, "world")) e->wait = SCRIPT_WAIT_WORLD;
    else if (!strcmp(cond, "file")) {
        if (sscanf(s, "%63s%n", e->arg, &off) < 1) { snprintf(err, err_n, "wait file needs a name"); return -1; }
        if (strlen(e->arg) >= 63) { snprintf(err, err_n, "wait file name is too long"); return -1; }
        s += off;
        e->wait = SCRIPT_WAIT_FILE;
    } else { snprintf(err, err_n, "unknown wait %s", cond); return -1; }
    if (sscanf(s, "%lf", &t) == 1 && t > 0) e->hold = (uint64_t)(t * ms + 0.5);
    snprintf(e->text, sizeof e->text, "wait %s%s%s", cond, *e->arg ? " " : "", e->arg);
    return 1;
}

int script_parse_line(char *line, ScriptEv *e, char *err, size_t err_n) {
    char *h = strchr(line, '#');
    if (h) *h = 0;
    char name[32];
    double at;
    int off = 0, timed = line[0] == '@';
    if (sscanf(line, " %31s%n", name, &off) == 1 && !strcmp(name, "wait")) {   // a bare wait: tick 0
        *e = (ScriptEv){.bit = -1, .seg = -1};
        return parse_wait(line + off, e, err, err_n);
    }
    off = 0;
    if (sscanf(line + timed, "%lf %31s%n", &at, name, &off) < 2 || at < 0) return 0;
    off += timed;
    double unit = timed ? 1000.0 : 1.0;   // ms to us, or ticks
    *e = (ScriptEv){.at = (uint64_t)(at * unit + 0.5), .hold = timed ? 66667 : 4, .bit = -1, .timed = timed, .seg = -1};
    if (!strcmp(name, "wait")) return parse_wait(line + off, e, err, err_n);
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

void script_link(ScriptEv *ev, int n) {
    int last = -1;
    for (int i = 0; i < n; i++) {
        ev[i].seg = last;
        if (ev[i].wait) last = i;
    }
}

int script_clock(const ScriptEv *ev, int i, uint64_t *tick, int64_t *w) {
    int s = ev[i].seg;
    if (s < 0) return 1;
    if (!ev[s].logged) return 0;
    *tick = (uint64_t)((int64_t)*tick - ev[s].off_t);
    if (*w >= 0) *w -= ev[s].off_w;
    return 1;
}

int script_wait(ScriptEv *ev, int i, uint64_t tick, int64_t w, int met) {
    ScriptEv *e = &ev[i];
    uint64_t t = tick;
    int64_t sw = w;
    if (e->logged || !script_clock(ev, i, &t, &sw)) return 0;
    // A timed wait's timeout runs on the world clock, so it does not run out before that clock starts.
    int64_t now = e->timed ? sw : (int64_t)t;
    // A bare wait (at == 0) times out from the moment it was reached, not from the earlier wait's recorded time.
    if (!e->armed_set && now >= 0) e->armed = now, e->armed_set = 1;
    int64_t base = e->at ? (int64_t)e->at : e->armed;
    int r = met ? 1 : now >= 0 && now >= base + (int64_t)e->hold ? 2 : 0;
    if (!r) return 0;
    // From here on both clocks read the wait's time: the lines after it keep their spacing from it. The world clock
    // is left unshifted when it had not started yet.
    int64_t at_t = e->timed ? (int64_t)(e->at * 60 / 1000000) : (int64_t)e->at;
    int64_t at_w = e->timed ? (int64_t)e->at : (int64_t)(e->at * 1000000 / 60);
    e->off_t = (int64_t)tick - at_t;
    e->off_w = w >= 0 ? w - at_w : 0;
    e->landed = (uint64_t)(now < 0 ? 0 : now);
    e->logged = r;
    return r;
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
        uint64_t t = tick;
        int64_t sw = w;
        if (e->wait) {
            s.total--, s.waits++;
            s.pending += !e->logged;
            s.timeouts += e->logged == 2;
            continue;
        }
        if (!script_clock(ev, i, &t, &sw)) { s.pending++; continue; }
        if (!e->logged) {
            s.pending += e->timed || !((mixed && sw >= 0) || t >= e->at + e->hold);
            continue;
        }
        s.landed++;
        uint64_t late = e->landed - e->at;
        if (!e->timed) { if (late > s.max_late_ticks) s.max_late_ticks = late; }
        else if (late / 1000.0 > s.max_late_ms) s.max_late_ms = late / 1000.0;
    }
    return s;
}
