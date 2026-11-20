// Adapted from MacNativeGames games/totk/runtime/game_patches.c @ ebe621d (MIT, Copyright (c) 2026 MacNativeGames
// contributors): runtime_frame_step_update (2338bb0: no 1/120 s floor) and the speed survey's player chain
// (world_survey.c player_actor).
// Frame rate: the frame limit and the real-time frame step, the host half of the patch sets fps60, realtime and cloth
// (build tables patches-v121.txt, patches-v143.txt).
//
// TotK's adaptive present-interval controller (main+0x8f26c8) turns the main thread's measured work into a number of
// 60 Hz ticks clamped to [minimum, maximum]; shipping data sets the minimum to 2 (30 fps), the set fps60 lowers it to
// 1. The controller still raises the interval when a frame does not fit. Shipping code then steps the game by whole
// vsyncs (VFR step main+0x7ede6c: ratio = interval * 0.5, seconds = interval / 60; global ratio at main+0x8f2394), so
// game speed is right only when frames take exactly their interval. With the set realtime both read the patch page
// instead: +0 ratio = seconds * 30, +4 seconds (1.4.3: +8 interval = seconds * 60, the producer's nominal interval
// before its half-step coefficient), the moving average of the last 5 display intervals, at most 1/15 s
// (slower frames slow the game down, as on the console), no lower bound, so a higher frame-rate cap only changes the
// pacer (frame_step_cap_ns, TOTK_FPS_LIMIT). The set cloth gives hair and cloth the same clock and needs realtime.
//
// The sample is the display time the pacer gives each present (nvnQueuePresentTexture), so the step follows the
// frames as shown, pacing waits included. With the step on, the pacer ignores the game's interval and holds a frame
// only to the frame limit (nvn_api_obj.c present_period): the controller's 2 or 3 vsyncs would only add latency. The
// vsync event stays at 60 Hz whatever the limit: the game does not wait for it each frame (1.4.3 shows its title at 144
// fps with the event at 60 Hz).
// One 64-bit and one 32-bit store per present; the guest reads the floats on its own threads, and each stays finite
// and in range whichever store it sees.
//
// Bananza 1.0.0 opts into experimental dynamic timing with TOTK_REALTIME=1.
// Its controller consumes an integer rate at patch page +12; its game clocks
// consume the precise 60 Hz ratio at +8. The existing FPS limit sets the host cap.
//
// The world clock (frame_step_world_us) sums the game time stepped (the patch page's step, whole vsyncs without the
// real-time step) from the first present that finds the player actor away from the origin, where the title screen
// keeps it: the end of the first save load (Bananza: the first present). Recorded and scripted input
// (input.c) runs on it, so a replay presses at the same game time at any frame rate and after a first load of any
// length; a later load (fast travel) still shifts what follows by its length.
//
// TOTK_TRACE=perf adds a line per 120 presents: fps, the game seconds stepped against wall seconds, and with
// TOTK_SPEED_PROBE=1 the player's position and ground speed (an independent check: Link runs at the same speed at
// any frame rate when game time follows wall time).
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "layout.h"
#include "platform/platform.h"

#define STEP_SAMPLES 5
#define STEP_MAX_S (1.0 / 15.0)    // longest step; a longer frame slows the game down
#define STEP_RESET_S 0.25          // a gap this long (loading, a debugger) restarts the average
#define STEP_FIRST_S (1.0 / 60.0)

static _Atomic uint64_t *step_word;   // patch page +0: ratio (low float), seconds (high float)
static double step_now = STEP_FIRST_S;   // the step the patch page holds: what the game steps each frame
static _Atomic uint32_t *step_interval;   // patch page +8: seconds * 60 (patches-v143.txt realtime)
static _Atomic uint32_t *dkb_rate;   // Bananza controller: integer simulation frames per second
static uint64_t last_ns;
static double samples[STEP_SAMPLES];
static unsigned sample_count, sample_next;

static void step_store(float seconds) {
    step_now = seconds;
    float ratio = seconds * 30.0f, interval = seconds * 60.0f;
    uint32_t lo, hi, iv;
    memcpy(&lo, &ratio, 4);
    memcpy(&hi, &seconds, 4);
    memcpy(&iv, &interval, 4);
    atomic_store_explicit(step_word, (uint64_t)lo | (uint64_t)hi << 32, memory_order_release);
    atomic_store_explicit(step_interval, iv, memory_order_release);
    if (dkb_rate) atomic_store_explicit(dkb_rate, (uint32_t)lroundf(1.0f / seconds), memory_order_release);
}

static void frame_limit_live(void);
static int no_table;   // this game version has no realtime patch table in this build
// TOTK_FIXED_STEP=1 (with the real-time step): every present steps the game 1/60 s whatever its display time, so game
// time is presents / 60 and a scripted run (input.c) steps the same game time per frame in every run. The game slows
// down when frames take longer than 1/60 s and speeds up above 60 fps; meant for A/B and image tests.
static int fixed_step;

// Bananza 3.2.0 radial blur lifetimes (recomp/patches-dkb458752.txt, the site at main+426f54). The blur entries a smash
// starts count down whole 60 Hz frames, subtracting the step truncated to an integer each frame, so a step under 1.0
// subtracted nothing and the blur stayed on screen for good (#158). This stands in for that fcvtzs: it adds the step to
// the entry's leftover fraction and returns the whole frames in w8. Steps of 1.0 and 2.0 give what the game computes.
// An entry is its address (x22, the counter at [x22-4]); a counter the hook did not leave was restarted by the game,
// and its fraction starts again from 0. The table is small, as the game keeps a handful of entries.
#define BLUR_ENTRIES 32
static struct { uint64_t key; int32_t left; float frac; } blur_acc[BLUR_ENTRIES];
static unsigned blur_next;
static PlatMutex blur_lock = PLAT_MUTEX_INIT;

static void blur_frames(Cpu *c) {
    uint32_t bits = (uint32_t)c->v[0];
    float step;
    memcpy(&step, &bits, 4);
    uint64_t key = c->x[22];
    int32_t counter = LD32(key - 4);
    if (!(step > 0) || step > 1000) {   // NaN, negative or absurd: as the game would truncate it
        c->x[8] = (uint32_t)(int32_t)(step > 0 ? 1000 : 0);
        return;
    }
    plat_mutex_lock(&blur_lock);
    unsigned i = 0;
    while (i < BLUR_ENTRIES && blur_acc[i].key != key) i++;
    if (i == BLUR_ENTRIES) {
        i = blur_next++ % BLUR_ENTRIES;
        blur_acc[i].key = key;
        blur_acc[i].left = counter + 1;   // differs from counter: the fraction starts from 0 below
    }
    if (blur_acc[i].left != counter) blur_acc[i].frac = 0;
    float t = blur_acc[i].frac + step;
    int32_t whole = (int32_t)t;
    blur_acc[i].frac = t - (float)whole;
    blur_acc[i].left = counter - whole;
    plat_mutex_unlock(&blur_lock);
    c->x[8] = (uint32_t)whole;
}

// Bananza damped velocity update (the sites at main+13fad88..13fada4 in recomp/patches-dkb458752.txt, main+1241c78..
// 1241c94 in patches-dkb100.txt). main+13fad60 (1.0.0 main+1241c50) steps the player's velocity v toward a target as
// v = r^step * v + sum of r^i * target for i = 1 .. step, adding the target once per whole 60 Hz frame, so a step under
// 1.0 added nothing and the climb velocity decayed to 0 (#201: DK stops on walls and ceilings above 60 fps). This
// stands in for the fmov s12, #1.0 that bounds the loop. On a fractional step it writes the closed form of the sum,
// r * (1 - r^step) / (1 - r) * target, which equals the loop's sum at every whole step and keeps the steady velocity
// r / (1 - r) * target at any step, then sets s12 to +inf so the loop does not run. On a whole step, or a rate outside
// (0, 1), it leaves s12 = 1.0 and s9..s11 = 0, the game's own state: the loop runs as on the console. At the BL x0 is
// the mover ([actor+0x60], step at [[x0+0x18]]), x1 the target vector and s8 the rate r.
static float reg_f32(const Cpu *c, int n) {
    uint32_t bits = (uint32_t)c->v[n];
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void set_f32(Cpu *c, int n, float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    c->v[n] = bits;   // a scalar write clears the rest of the vector register, as fmov and movi do
}

static float ld_f32(uint64_t a) {
    uint32_t bits = LD32(a);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void damped_sum(Cpu *c) {
    float step = ld_f32(LD64(c->x[0] + 0x18)), r = reg_f32(c, 8);
    set_f32(c, 9, 0), set_f32(c, 10, 0), set_f32(c, 11, 0), set_f32(c, 12, 1.0f);
    if (!(step > 0 && step < 1000) || step == floorf(step) || !(r > 0 && r < 1)) return;
    float g = r * (1.0f - powf(r, step)) / (1.0f - r);
    uint64_t t = c->x[1];
    set_f32(c, 11, g * ld_f32(t)), set_f32(c, 10, g * ld_f32(t + 4)), set_f32(c, 9, g * ld_f32(t + 8));
    set_f32(c, 12, INFINITY);
}

// ---- TOTK_FP_DIAG=1 (Bananza 3.2.0, set fpdiag in recomp/patches-dkb458752.txt): what the present-interval controller
// measures. The controller (main+0x204a18) turns the frame's +0x128 ticks into floor(seconds * 60 + bias) + 1 vsyncs
// (bias: the controller's +8 float), clamped to its +0 and +4 and raised only after +0xc frames in a row. +0x128 is
// the time from the frame's start (+0xf0) to the end of the main thread's three frame calls (main+0x20383c, slot
// +0x280), plus two callback calls through the frame's +0x158 timed around them (main+0x204240 with argument 0, slot
// +0x290; main+0x2042e0 with argument 1, slot +0x2a0), which run in the next call (vtable +0x110) before the next
// start. Each slot runs after the original instruction, so it sees the value the game keeps. Every 120 frames a line
// gives the three parts and their sum (avg and max ms), the frames whose sum passed 1/60 s and the controller's
// intervals; each change of interval gets a line with the parts of the three frames before it (the first 200).
#define FPD_HIST 4
static struct { uint64_t main, cb0, cb1; } fpd_frame[FPD_HIST];   // [0] the frame in progress
static uint64_t fpd_obj, fpd_n, fpd_sum[4], fpd_max[4], fpd_over, fpd_iv[4], fpd_changes;
static uint32_t fpd_last_iv;

static double fpd_ms(uint64_t ticks) { return rt_ticks_to_ns(ticks) / 1e6; }
static double fpd_f32(uint32_t bits) {
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void fpd_window(void) {
    if (!fpd_n) return;
    double n = (double)fpd_n;
    rt_log("fpdiag: %llu frames: main %.2f/%.2f ms, callback(0) %.2f/%.2f, callback(1) %.2f/%.2f, measured %.2f/%.2f "
           "(avg/max); over 1/60 s %llu; interval 1/2/3+ %llu/%llu/%llu",
           (unsigned long long)fpd_n, fpd_ms(fpd_sum[0]) / n, fpd_ms(fpd_max[0]), fpd_ms(fpd_sum[1]) / n,
           fpd_ms(fpd_max[1]), fpd_ms(fpd_sum[2]) / n, fpd_ms(fpd_max[2]), fpd_ms(fpd_sum[3]) / n, fpd_ms(fpd_max[3]),
           (unsigned long long)fpd_over, (unsigned long long)fpd_iv[1], (unsigned long long)fpd_iv[2],
           (unsigned long long)fpd_iv[3]);
    memset(fpd_sum, 0, sizeof fpd_sum);
    memset(fpd_max, 0, sizeof fpd_max);
    memset(fpd_iv, 0, sizeof fpd_iv);
    fpd_n = fpd_over = 0;
}

// main+0x20383c: str x8, [x19, #0x128]; x8 is the main thread's part of this frame, x19 the frame object. The frame
// before it is complete: its two callbacks ran after its own store.
static void fpd_main(Cpu *c) {
    uint64_t obj = c->x[19], ctrl = LD64(obj + 0x218);
    if (fpd_obj != obj) {
        fpd_obj = obj;
        rt_log("fpdiag: frame object %#llx, controller %#llx: interval %u..%u, bias %.3f, %u frames to raise; "
               "callback %#llx", (unsigned long long)obj, (unsigned long long)ctrl, ctrl ? LD32(ctrl) : 0,
               ctrl ? LD32(ctrl + 4) : 0, fpd_f32(ctrl ? LD32(ctrl + 8) : 0), ctrl ? LD32(ctrl + 0xc) : 0,
               (unsigned long long)LD64(obj + 0x158));
    }
    if (fpd_frame[0].main) {
        uint64_t parts[4] = {fpd_frame[0].main, fpd_frame[0].cb0, fpd_frame[0].cb1, 0};
        parts[3] = parts[0] + parts[1] + parts[2];
        for (int k = 0; k < 4; k++) {
            fpd_sum[k] += parts[k];
            if (parts[k] > fpd_max[k]) fpd_max[k] = parts[k];
        }
        if (parts[3] * 60 > rt_tick_hz()) fpd_over++;
        fpd_n++;
    }
    memmove(fpd_frame + 1, fpd_frame, sizeof fpd_frame[0] * (FPD_HIST - 1));
    fpd_frame[0].main = c->x[8], fpd_frame[0].cb0 = fpd_frame[0].cb1 = 0;
    uint32_t iv = ctrl ? LD32(ctrl + 0x40) : 0;   // the interval the controller chose at this frame's start
    fpd_iv[iv < 1 ? 1 : iv > 3 ? 3 : iv]++;
    if (fpd_last_iv && iv != fpd_last_iv && fpd_changes++ < 200)
        rt_log("fpdiag: interval %u -> %u after frames of main/callback(0)/callback(1) ms: %.2f/%.2f/%.2f, "
               "%.2f/%.2f/%.2f, %.2f/%.2f/%.2f (newest first)", fpd_last_iv, iv,
               fpd_ms(fpd_frame[1].main), fpd_ms(fpd_frame[1].cb0), fpd_ms(fpd_frame[1].cb1),
               fpd_ms(fpd_frame[2].main), fpd_ms(fpd_frame[2].cb0), fpd_ms(fpd_frame[2].cb1),
               fpd_ms(fpd_frame[3].main), fpd_ms(fpd_frame[3].cb0), fpd_ms(fpd_frame[3].cb1));
    fpd_last_iv = iv;
    if (fpd_n >= 120) fpd_window();
}

// main+0x204240: add x8, x8, x9 after the timed callback(0): x0 and x21 the ticks after and before it.
static void fpd_cb0(Cpu *c) { fpd_frame[0].cb0 += c->x[0] - c->x[21]; }
// main+0x2042e0: add x8, x21, x8 after the timed callback(1): x21 its ticks.
static void fpd_cb1(Cpu *c) { fpd_frame[0].cb1 += c->x[21]; }

void frame_step_init(void) {
    int fps60 = rt_patch_set("fps60"), realtime = rt_patch_set("realtime"), cloth = rt_patch_set("cloth");
    int fpdiag = rt_patch_set("fpdiag");
    if (rt_patch_enabled(fpdiag)) {   // the main thread alone runs these sites, one frame after another
        rt_register_host((uint64_t)(uintptr_t)rt_patch_page() + 0x280, fpd_main);
        rt_register_host((uint64_t)(uintptr_t)rt_patch_page() + 0x290, fpd_cb0);
        rt_register_host((uint64_t)(uintptr_t)rt_patch_page() + 0x2a0, fpd_cb1);
        rt_log("fpdiag: logging the present-interval controller's measure (TOTK_FP_DIAG=1)");
    }
    if (realtime < 0) {
        no_table = 1;
        rt_log("frame rate: no timing patch table; using the game's original frame timing");
        return;
    }
    if (!rt_patch_enabled(realtime) && rt_patch_enabled(cloth)) {
        rt_patch_disable(cloth);
        rt_log("frame rate: cloth clock off, it needs the real-time step (TOTK_REALTIME=0)");
    }
    if (rt_patch_enabled(realtime)) {
        step_word = (_Atomic uint64_t *)rt_patch_page();
        step_interval = (_Atomic uint32_t *)((uint8_t *)rt_patch_page() + 8);
        if (!rt_game_version) {
            dkb_rate = (_Atomic uint32_t *)((uint8_t *)rt_patch_page() + 12);
            rt_register_host((uint64_t)(uintptr_t)rt_patch_page() + 0x250, blur_frames);   // radial blur lifetimes, both versions
            rt_register_host((uint64_t)(uintptr_t)rt_patch_page() + 0x260, damped_sum);    // damped velocity, both versions
        }
        step_store((float)STEP_FIRST_S);
    }
    const char *fx = cfg_get("TOTK_FIXED_STEP");
    if (fx && !strcmp(fx, "1")) {
        fixed_step = step_word != NULL;
        rt_log("frame rate: %s", fixed_step ? "fixed step, 1/60 s of game time per present (TOTK_FIXED_STEP=1)"
               : "TOTK_FIXED_STEP=1 needs the real-time step (TOTK_REALTIME=1); the game keeps its own timing");
    }
    frame_limit_live();   // TOTK_FPS_LIMIT decides the set fps60 from the start
    if (!rt_game_version) {
        rt_log("frame rate: Bananza %s", step_word ? "experimental dynamic step; host-paced frames" :
               "original game timing (TOTK_REALTIME=1 opts into experimental dynamic timing)");
        return;
    }
    rt_log("frame rate: present interval %s, %s", rt_patch_enabled(fps60) ? "1..3 (above 30 fps)" : "2..3 (shipping 30 fps)",
           step_word ? "real-time step (5-frame average of display intervals, at most 1/15 s)" : "whole-vsync steps");
}

int frame_step_realtime(void) { return step_word != NULL; }

// The frame limit (menu, TOTK_FPS_LIMIT, live): 30, 60, 90, 120, 144, any whole rate from 30 to 360, display (the
// refresh rate of the window's monitor) or unlimited (no pacer wait: nvn_api_obj.c pace() treats the period 0 as
// "present at once"; with TOTK_VK_PRESENT=fifo the display still caps the rate). Unset, the older TOTK_FPS60=0 still means 30. A limit of 30 switches the set
// fps60 off (the shipping minimum interval 2), every other limit switches it on; the controller reads its minimum
// interval from the patched sites each frame. With the real-time step the pacer holds frames to the limit; above 60
// only the pacer changes, as the step has no lower bound. Without it the game steps whole 60 Hz vsyncs and the pacer
// follows the game's interval, so the limit stops at 60.
#define LIMIT_MIN_HZ 30
#define LIMIT_MAX_HZ 360
#define LIMIT_NONE (-2)   // limit_hz for unlimited; 0 is display and -1 is "not read yet"
static int fps60_set = -2;
static unsigned limit_seen;
static int limit_read;
static int limit_hz = -1;   // 0: display; LIMIT_NONE: unlimited; -1 before the first read
// Bananza with the real-time step: the highest frame rate the pacer allows, whatever the limit (TOTK_REALTIME_MAX_FPS,
// default 120, 0 for none). Above 120 fps the step falls under half a 60 Hz frame, and the 3.2.0 Emerald Rush challenge
// levels then crash in the draw-record merge job (main+0xa68f80, #411: a record block freed while still linked); the
// same levels play through at 120 fps and with a fixed 1/60 s step. Which game code breaks below 1/120 s is not found
// yet, so the ceiling keeps every step at 1/120 s or longer instead of handing the game a shorter one.
static int rt_max_hz;

static int limit_parse(void) {
    const char *e = cfg_get("TOTK_FPS_LIMIT"), *old = cfg_get("TOTK_FPS60");
    if (!e || !*e) return old && !strcmp(old, "0") ? 30 : 60;
    if (!strcmp(e, "display")) return 0;
    if (!strcmp(e, "unlimited")) return LIMIT_NONE;
    char *end;
    long n = strtol(e, &end, 10);
    if (!*end && n >= LIMIT_MIN_HZ && n <= LIMIT_MAX_HZ) return (int)n;
    rt_log("frame rate: TOTK_FPS_LIMIT=%s is not 30..360, display or unlimited; 60", e);
    return 60;
}

static void frame_limit_live(void) {
    unsigned v = cfg_version();
    if (limit_read && v == limit_seen) return;
    limit_read = 1;
    limit_seen = v;
    const char *mx = cfg_get("TOTK_REALTIME_MAX_FPS");
    int max_hz = !step_word || rt_game_version ? 0 : mx && *mx ? atoi(mx) : 120;
    if (max_hz && max_hz < LIMIT_MIN_HZ) max_hz = LIMIT_MIN_HZ;
    int hz = limit_parse();
    if (hz == limit_hz && max_hz == rt_max_hz) return;
    if (max_hz != rt_max_hz && step_word && !rt_game_version)
        rt_log("frame rate: real-time step held to %d fps at most%s", max_hz,
               max_hz ? " (TOTK_REALTIME_MAX_FPS; 0 lifts it, #411)" : ", no ceiling (TOTK_REALTIME_MAX_FPS=0)");
    rt_max_hz = max_hz;
    limit_hz = hz;
    if (fps60_set == -2) fps60_set = rt_patch_set("fps60");
    if (fps60_set >= 0) rt_patch_switch(fps60_set, hz != 30);
    if (hz == LIMIT_NONE) {
        if (!step_word) rt_log("frame rate: unlimited needs the real-time step (TOTK_REALTIME=1); 60 fps");
        else rt_log("frame rate: no limit (unlimited)%s", strcmp(cfg_value("TOTK_VK_PRESENT"), "fifo") ? "" :
                    "; TOTK_VK_PRESENT=fifo still caps it at the monitor's rate, use mailbox or immediate");
    } else if ((hz > 60 || hz == 0) && !step_word)
        rt_log("frame rate: limits above 60 need the real-time step (TOTK_REALTIME=1); 60 fps");
    else if (hz && hz < 60 && no_table && !rt_game_version)
        rt_log("frame rate: this game version has no real-time frame-rate support in this build; limit %d fps cannot be held", hz);
    else if (hz && hz < 60 && !step_word && !rt_game_version)
        // Bananza's own controller picks the present interval and the pacer follows it, so a lower limit is not held
        // without the real-time step (issue #118: a 30 fps limit still ran at 60).
        rt_log("frame rate: limit %d fps needs the real-time step (DKB_REALTIME=1, 'Allow higher frame rates'); the game keeps its own timing", hz);
    else if (hz) rt_log("frame rate: limit %d fps", hz);
    else if (gfx_window_refresh_hz() > 0) rt_log("frame rate: limit at the monitor's refresh rate (%.3f Hz now)", gfx_window_refresh_hz());
    else rt_log("frame rate: limit at the monitor's refresh rate (60 until the window measures it)");
}

uint64_t frame_step_cap_ns(void) {
    frame_limit_live();
    if (limit_hz == LIMIT_NONE && step_word && !rt_max_hz) return 0;   // pace() presents at once
    double hz = limit_hz > 0 ? limit_hz : limit_hz == LIMIT_NONE ? LIMIT_MAX_HZ : gfx_window_refresh_hz();
    if (hz < LIMIT_MIN_HZ) hz = 60;   // display rate unknown (no window yet)
    if (hz > LIMIT_MAX_HZ) hz = LIMIT_MAX_HZ;
    if (rt_max_hz && hz > rt_max_hz) hz = rt_max_hz;
    return (uint64_t)(1e9 / hz + 0.5);
}

// The frame rate the limit gives, for the performance overlay: the setting, the monitor's rate for "display", 0 for unlimited, 60 at
// most without the real-time step; 0 before the first frame read the setting.
double frame_step_limit_hz(void) {
    int hz = limit_hz;
    if (hz == LIMIT_NONE && rt_max_hz) return rt_max_hz;
    if (hz < 0) return 0;   // not read yet, or unlimited: the overlay shows no limit
    double r = hz ? hz : gfx_window_refresh_hz();
    if (r < LIMIT_MIN_HZ) r = 60;
    if (rt_max_hz && r > rt_max_hz) r = rt_max_hz;
    return !step_word && r > 60 ? 60 : r > LIMIT_MAX_HZ ? LIMIT_MAX_HZ : r;
}

// ---- TOTK_TRACE=perf speed lines
static uint64_t win_t0, win_frames, win_hist[4];   // win_hist: frames by present interval 1, 2, 3+ ([0] unused)
static double win_game;
static int probe = -1;
static float last_pos[3];
static uint64_t last_pos_ns;

static int read_guest(uint64_t a, void *out, size_t n) { return a && plat_mem_read(a, out, n); }
static uint64_t read_ptr(uint64_t a) {
    uint64_t p = 0;
    return read_guest(a, &p, 8) ? p : 0;
}
// Where main sits in this process, from rt_layout and not from layout.h: the layout is the one modules.c generated
// for this install (tools/build_game.py), while the runtime library is compiled ahead of it.
static uint64_t main_base(void) {
    for (int i = 0; i < rt_layout_count; i++)
        if (!strcmp(rt_layout[i].name, "main")) return rt_layout[i].base;
    return 0;
}
// 1.2.1: [main+0x4728538] (1.1.1: [main+0x4630a68], 1.4.3: [main+0x3add238]) +0x1e8 scene, its component 13 lists the
// resident actors; the one named "Player" has its position at +0x2b4 (MacNativeGames world_survey.c, game_version.h
// TOTK_GAME_V121/V143). The 1.1.1 slot was measured from its own dereference chain: 181 scene accesses in that build
// read it, and one of them reproduces this walk (the 14-component check, component 13, the 0x70 stride and the name
// compare) instruction for instruction.
static int player_pos(float xyz[3]) {
    uint64_t module = read_ptr(main_base() + (RT_V143 ? 0x3add238 : RT_V111 ? 0x4630a68 : 0x4728538)),
             scene = module ? read_ptr(module + 0x1e8) : 0;
    int32_t count = 0;
    if (!scene || !read_guest(scene + 0x50, &count, 4) || count < 14 || count > 256) return 0;
    uint64_t comps = read_ptr(scene + 0x58), resident = comps ? read_ptr(comps + 13 * 8) : 0;
    if (!resident || !read_guest(resident + 0x20, &count, 4) || count < 1 || count > 64) return 0;
    uint64_t links = read_ptr(resident + 0x28);
    for (int i = 0; links && i < count; i++) {
        uint64_t data = read_ptr(links + (uint64_t)i * 0x70 + 8), actor = data ? read_ptr(data + 0x40) : 0;
        uint64_t name = actor ? read_ptr(actor + 0x218) : 0;
        char text[7];
        if (name && read_guest(name, text, 7) && !memcmp(text, "Player", 7))
            return read_guest(actor + 0x2b4, xyz, 12) && isfinite(xyz[0]) && isfinite(xyz[2]);
    }
    return 0;
}

static void speed_report(uint64_t now, double step, int interval) {
    if (!win_t0) {
        win_t0 = now;
        return;
    }
    win_frames++;
    win_game += step;
    win_hist[interval < 1 ? 1 : interval > 3 ? 3 : interval]++;
    if (win_frames < 120) return;
    if (probe < 0) {
        const char *e = cfg_get("TOTK_SPEED_PROBE");
        probe = e && !strcmp(e, "1");
    }
    double wall = (double)(now - win_t0) * 1e-9;
    char pos[160] = "";
    float p[3];
    if (probe && player_pos(p)) {
        double d = last_pos_ns ? hypot(p[0] - last_pos[0], p[2] - last_pos[2]) : 0,
               s = last_pos_ns ? (double)(now - last_pos_ns) * 1e-9 : 0;
        snprintf(pos, sizeof pos, "; player %.1f %.1f %.1f, %.2f m in %.3f s = %.2f m/s", p[0], p[1], p[2], d, s,
                 s > 0 ? d / s : 0);
        memcpy(last_pos, p, sizeof p);
        last_pos_ns = now;
    }
    if (step_word)
        rt_log("fps: %llu frames in %.3f s (%.1f fps), %.3f s game stepped, speed %.3f; interval 1/2/3+ %llu/%llu/%llu%s",
               (unsigned long long)win_frames, wall, (double)win_frames / wall, win_game, win_game / wall,
               (unsigned long long)win_hist[1], (unsigned long long)win_hist[2], (unsigned long long)win_hist[3], pos);
    else
        rt_log("fps: %llu frames in %.3f s (%.1f fps), whole-vsync steps; interval 1/2/3+ %llu/%llu/%llu%s",
               (unsigned long long)win_frames, wall, (double)win_frames / wall, (unsigned long long)win_hist[1],
               (unsigned long long)win_hist[2], (unsigned long long)win_hist[3], pos);
    win_t0 = now;
    win_frames = 0;
    win_game = 0;
    memset(win_hist, 0, sizeof win_hist);
}

static _Atomic int64_t world_us = -1;
static _Atomic uint64_t game_us;   // game time stepped since the first present (frame_step_ticks)
static uint64_t presents;

int64_t frame_step_world_us(void) { return atomic_load_explicit(&world_us, memory_order_relaxed); }
uint64_t frame_step_ticks(void) { return atomic_load_explicit(&game_us, memory_order_relaxed) * 60 / 1000000; }

// game: the game time this present stepped, the same amount frame_step_ticks adds. The world clock once summed the
// display time without the real-time step and skipped the frames of a step reset (first frame, a long gap) with it,
// so a load or a shader-compile stall moved every later timed script event by a few frames between runs.
static void world_clock(double game) {
    presents++;
    int64_t w = atomic_load_explicit(&world_us, memory_order_relaxed);
    if (w < 0) {
        float p[3];
        // TotK: a few guest reads, until the load ends. Another game (Bananza) has no known player position: the clock
        // starts at the first present.
        p[0] = p[1] = p[2] = 0;
        if (rt_game_version && (!player_pos(p) || fabsf(p[0]) + fabsf(p[2]) < 1.0f)) return;
        rt_log("frame step: world clock starts at present %llu, %.2f s after the guest started (the player at %.1f %.1f %.1f)",
               (unsigned long long)presents, rt_guest_start_ns ? (kern_now_ns() - rt_guest_start_ns) / 1e9 : 0.0, p[0],
               p[1], p[2]);
        atomic_store_explicit(&world_us, 0, memory_order_relaxed);
        return;
    }
    atomic_store_explicit(&world_us, w + (int64_t)(game * 1e6 + 0.5), memory_order_relaxed);
}

void frame_step_present(uint64_t display_ns, int interval) {
    frame_limit_live();
    if (!rt_game_version) dkb_native_aa_live();   // Bananza: native anti-aliasing follows the live Upscaler setting
    double step = 0, dt = last_ns ? (double)(display_ns - last_ns) * 1e-9 : 0;
    if (step_word && fixed_step) {
        step = STEP_FIRST_S;   // the patch page keeps the 1/60 s frame_step_init stored
    } else if (step_word && last_ns) {
        if (dt > STEP_RESET_S || dt < 0.001) {
            sample_count = sample_next = 0;
        } else {
            if (dt > STEP_MAX_S) dt = STEP_MAX_S;
            samples[sample_next++ % STEP_SAMPLES] = dt;
            if (sample_count < STEP_SAMPLES) sample_count++;
            double sum = 0;
            for (unsigned i = 0; i < sample_count; i++) sum += samples[i];
            step = sum / sample_count;
            step_store((float)step);
        }
    }
    last_ns = display_ns;
    // The game clock of untimed script lines: what the game stepped this frame, the patch page's step with the
    // real-time step (kept through a reset), whole vsyncs without it.
    double game = step_word ? step_now : (interval > 0 ? interval : 1) / 60.0;
    atomic_store_explicit(&game_us, atomic_load_explicit(&game_us, memory_order_relaxed) + (uint64_t)(game * 1e6 + 0.5),
                          memory_order_relaxed);
    world_clock(game);
    if (rt_trace & RT_TRACE_PERF) speed_report(display_ns, step, interval);
}
