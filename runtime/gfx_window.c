// The window's host side, shared by the graphics back ends: the close and exit sequence, focus and its messages, the
// window settings, the 60 Hz vsync thread and the aspect fit. The device side (swapchain or drawable, present,
// capture, in-game menu) belongs to the back end: gfx_window_start, gfx_window_resize and gfx_window_present_mode call
// GfxBackend.window_start / window_resize / present_mode (gfx.h), and vk_present.c or mtl_*.m own the rest.
//
// Threads: the window thread (the back end's) owns the window and never runs guest code. The vsync thread signals the
// bound system events. Presents come from any host thread (the back end's present worker); the swapchain is rebuilt
// lazily: a resize only marks it dirty (gfx_window_mark_dirty), the next present (or gfx_window_resize) rebuilds it.
//
// The in-game menu (ui/menu.h) draws over the game image in the back end's present, after its blit. The window's size
// and fullscreen state are the settings TOTK_WINDOW_SIZE and TOTK_FULLSCREEN, which the back end applies at start and
// whenever a cfg_set changes them (window_settings_live -> gfx_window_apply_settings).
//
// Focus and vsync behaviour from MacNativeGames, games/totk/runtime/frontend.m:88-97, 175-176, 350-352
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// MacNativeGames contributors).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "fs.h"
#include "gfx.h"
#include "gfx_wsys.h"
#include "nvn_api.h"
#include "platform/platform.h"

#define VSYNC_NS 1000000000ull
#define VSYNC_HZ 60

// ---- window state shared with the back end
static volatile int32_t g_dirty;               // the window was resized; the back end's swapchain or drawable is stale
static volatile int32_t g_closed;              // gfx_window_closed or an exit: present and start drop out
static volatile int32_t g_focused = 1;
static volatile int32_t g_focus_msgs;
static PlatRMutex g_start_lock;       // gfx_window_start and gfx_exit_release: one back end start or release at a time

void gfx_window_mark_dirty(void) { plat_xchg32(&g_dirty, 1); }
int gfx_window_exiting(void) { return plat_cas32(&g_closed, 0, 0); }
int gfx_window_closing(void) { return gfx_window_exiting() != 0; }
// Reads the mark without clearing it: the present path asks with it, and only the swapchain build clears it (with
// gfx_window_take_dirty), so a resize seen while the window is minimised still rebuilds after the restore.
int gfx_window_dirty(void) { return plat_cas32(&g_dirty, 0, 0) != 0; }
int gfx_window_take_dirty(void) { return plat_xchg32(&g_dirty, 0); }

// The window gained or lost the focus: the game hears it as nn::oe's focus state and a notification message.
void gfx_window_focus_changed(int in) {
    plat_xchg32(&g_focused, in);
    oe_set_focus(in);                          // pinned to 1 by TOTK_ASSUME_FOCUS inside
    oe_push_message(OE_MSG_FOCUS_STATE_CHANGED);
    plat_inc32(&g_focus_msgs);
    rt_log("gfx: window %s the focus", in ? "gained" : "lost");
}

// The window was closed: the process ends with it, as in MNG, once a save Commit still running in the background is
// done.
//
// Every exit path (this one, the guest's ExitProcess, the --seconds end) calls gfx_exit_release after its saves: the
// queued present is dropped once the present thread is idle, then the device is released (idle, swapchain and surface
// destroyed) before the process ends. plat_exit while a present or a swapchain rebuild is running hung some AMD
// mobile drivers. g_closed keeps the present thread from rebuilding the surface in between. The first caller does the
// work; a later one parks, because returning would run guest code while the device goes away, and the first caller
// ends the process.
static void exit_watchdog(void *arg) {
    (void)arg;
    plat_sleep_ms(3000);
    plat_exit(0);
}

void gfx_exit_release(void) {
    if (plat_cas32(&g_closed, 1, 0) != 0)
        for (;;) plat_sleep_ms(1000);
    // A hung GPU or driver present must not keep the process alive: end it after a grace period regardless.
    plat_thread_spawn(exit_watchdog, NULL, "exitwd", 0);
    // The queued present goes out first, without the start lock: the thread presenting it may be waiting for that lock
    // in gfx_window_start (or for the window thread, which a resize holds there).
    gfx_backend()->exit_wait();
    plat_rmutex_lock(&g_start_lock);   // the back end's window_start and this release are serialised
    gfx_backend()->exit_release();
    plat_rmutex_unlock(&g_start_lock);
}

// The window thread must keep pumping messages: on the SDL back end the present thread asks it for the window size,
// and vk_present_drain waits for that thread. So the exit runs on its own thread and gfx_window_closed returns at once.
static void exit_main(void *arg) {
    (void)arg;
    fs_save_flush();
    gfx_backend()->save();
    prof_write();   // plat_exit runs no atexit handlers
    gfx_exit_release();
    plat_exit(0);
}

void gfx_window_closed(void) {
    static volatile int32_t requested;
    if (plat_cas32(&requested, 1, 0) != 0) return;
    rt_log("gfx: window closed; exiting");
    if (!plat_thread_spawn(exit_main, NULL, "exit", 0)) {
        // No thread to run the full exit on: save and end the process inline, as before the release step existed.
        fs_save_flush();
        gfx_backend()->save();
        prof_write();
        plat_exit(0);
    }
}

// TOTK_WINDOW_MODE: windowed, borderless (desktop fullscreen), fullscreen (the display's own mode where the back end can
// ask for it). Unset, TOTK_FULLSCREEN=1 (the older setting, the Steam Deck profile) means borderless.
// TOTK_WINDOW_SIZE=<w>x<h>: client area of the window (the swapchain follows the window; the game image is scaled to it).
int gfx_cfg_window_mode(void) {
    const char *m = cfg_get("TOTK_WINDOW_MODE");
    if (cfg_from_env("TOTK_FULLSCREEN") && !cfg_from_env("TOTK_WINDOW_MODE")) m = NULL;   // the older variable wins over the ini
    if (m && *m) {
        if (!strcmp(m, "borderless")) return 1;
        if (!strcmp(m, "fullscreen")) return 2;
        if (!strcmp(m, "windowed")) return 0;
    }
    return cfg_bool("TOTK_FULLSCREEN") ? 1 : 0;
}

int gfx_cfg_fullscreen(void) { return gfx_cfg_window_mode() != 0; }

void gfx_cfg_window_size(int *w, int *h) {
    int x = 0, y = 0;
    if (sscanf(cfg_value("TOTK_WINDOW_SIZE"), "%dx%d", &x, &y) != 2 || x < 640 || y < 360 || x > 7680 || y > 4320)
        sscanf(cfg_meta("WINDOW_SIZE")->def, "%dx%d", &x, &y);   // unparsable or out of range: the default
    *w = x;
    *h = y;
}

// The refresh rate of the window's monitor (TOTK_FPS_LIMIT=display), measured by the back end's window thread when the
// window opens, lands on another monitor or the display mode changes; readers take the stored value.
static volatile int32_t g_rate_mhz;   // millihertz, 0 unknown

void gfx_window_set_refresh(double hz) {
    int32_t mhz = (int32_t)(hz * 1000 + 0.5);
    if (plat_xchg32(&g_rate_mhz, mhz) != mhz) rt_log("gfx: monitor refresh rate %.3f Hz", hz);
}

double gfx_window_refresh_hz(void) { return plat_cas32(&g_rate_mhz, 0, 0) / 1000.0; }

// ---- vsync
static PlatMutex g_vs_lock = PLAT_MUTEX_INIT;
static uint64_t g_vs_ev[16];
static int g_vs_nev;
static volatile int32_t *g_vs_ctr[4];
static int g_vs_nctr;
static volatile int64_t g_ticks;
static volatile int64_t g_vs_start_ns;
static int g_vs_started;

static void vsync_main(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_HIGHEST);
    uint64_t n = 0;
    for (;;) {
        uint64_t start = (uint64_t)g_vs_start_ns;
        uint64_t due = start + (n + 1) * VSYNC_NS / VSYNC_HZ;
        uint64_t now = kern_now_ns();
        if (now > due + 100000000ull) {                  // the process stalled: restart the schedule, no catch-up burst
            g_vs_start_ns = (int64_t)(now - (n + 1) * VSYNC_NS / VSYNC_HZ);
        } else if (due > now) {
            plat_wait_ns(due - now);
        }
        n++;
        plat_xchg64(&g_ticks, (int64_t)n);
        uint64_t ev[16];
        volatile int32_t *ctr[4];
        int ne, nc;
        plat_mutex_lock_shared(&g_vs_lock);
        ne = g_vs_nev;
        nc = g_vs_nctr;
        memcpy(ev, g_vs_ev, sizeof ev);
        memcpy(ctr, g_vs_ctr, sizeof ctr);
        plat_mutex_unlock_shared(&g_vs_lock);
        for (int i = 0; i < nc; i++) plat_inc32(ctr[i]);
        for (int i = 0; i < ne; i++) kern_signal_system_event(ev[i]);
    }
    return;
}

static void vsync_ensure(void) {
    plat_rmutex_lock(&g_start_lock);
    if (!g_vs_started) {
        g_vs_started = 1;
        g_vs_start_ns = (int64_t)kern_now_ns();
        if (!plat_thread_spawn(vsync_main, NULL, "vsync", 0)) rt_fatal("gfx: cannot start the vsync thread");
    }
    plat_rmutex_unlock(&g_start_lock);
}

void gfx_vsync_bind(uint64_t system_event) {
    plat_mutex_lock(&g_vs_lock);
    int have = 0;
    for (int i = 0; i < g_vs_nev; i++) have |= g_vs_ev[i] == system_event;
    if (!have) {
        if (g_vs_nev == 16) rt_fatal("gfx: more than 16 vsync events");
        g_vs_ev[g_vs_nev++] = system_event;
    }
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

void gfx_vsync_bind_counter(volatile int32_t *counter) {
    plat_mutex_lock(&g_vs_lock);
    if (g_vs_nctr == 4) rt_fatal("gfx: more than 4 vsync counters");
    g_vs_ctr[g_vs_nctr++] = counter;
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

uint64_t gfx_vsync_ticks(void) { return (uint64_t)plat_cas64(&g_ticks, 0, 0); }
uint64_t gfx_vsync_time_ns(uint64_t tick) { return (uint64_t)g_vs_start_ns + tick * VSYNC_NS / VSYNC_HZ; }

// ---- focus
int gfx_window_focused(void) { return g_focused; }
unsigned gfx_window_focus_messages(void) { return (unsigned)g_focus_msgs; }

// The rectangle {x, y, w, h} of a dw x dh swapchain image the cw x ch picture goes into: the whole image with
// TOTK_ASPECT=stretch, else (fit, the default) the largest one of the picture's own aspect ratio, centred (a 16:9 picture
// on the Steam Deck's 1280x800 screen gets bars of 40 rows above and below; a 16:9 picture in a 21:9 window, bars at
// the sides).
void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]) {
    out[0] = out[1] = 0, out[2] = dw, out[3] = dh;
    if (cw <= 0 || ch <= 0 || !strcmp(cfg_value("TOTK_ASPECT"), "stretch")) return;
    if ((int64_t)dw * ch > (int64_t)dh * cw) out[2] = (int)(((int64_t)dh * cw + ch / 2) / ch);   // window wider: pillarbox
    else out[3] = (int)(((int64_t)dw * ch + cw / 2) / cw);                                        // taller: letterbox
    // A side of the window's parity keeps the two bars equal; a picture one pixel short of the window fills it
    out[2] += (dw - out[2]) & 1, out[3] += (dh - out[3]) & 1;
    out[0] = (dw - out[2]) / 2, out[1] = (dh - out[3]) / 2;
}

// The window thread, the back end's device and its swapchain or drawable; idempotent. The start lock keeps a present,
// a resize and the exit from entering the back end's start or release at the same time.
void gfx_window_start(void) {
    if (gfx_window_exiting()) return;   // the surface is gone for good
    plat_rmutex_lock(&g_start_lock);
    if (!gfx_window_exiting()) gfx_backend()->window_start();
    plat_rmutex_unlock(&g_start_lock);
    vsync_ensure();
}

void gfx_window_resize(int client_w, int client_h) {
    if (gfx_window_exiting()) return;
    gfx_window_start();
    wsys_resize(client_w, client_h);
    gfx_backend()->window_resize();
}

// The present mode the back end's swapchain or layer got (TOTK_VK_PRESENT); the menu shows it. "" before one, "none"
// when the back end has no swapchain.
const char *gfx_window_present_mode(void) { return gfx_backend()->present_mode(); }

// Whether the swapchain is HDR10 at the moment (any thread): ov_gfx.c tells the game HDR is on only then (native HDR).
// 0 before the first swapchain, on SDR and on a back end without one.
int gfx_window_hdr10(void) { return gfx_backend()->window_hdr10(); }
