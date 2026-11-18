// Frame timing (TOTK_TRACE=perf, the frame-time summary, the performance overlay's counts), the A/B switches (rt_ab,
// rt_knob) and TOTK_ONE_PER_CORE. The sampling profiler (TOTK_PROFILE) is the OS's (prof_win.c); without one the weak
// prof_sampler_* below do nothing.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "flight.h"
#include "gfx.h"
#include "host.h"
#include "nvn_mode_trace.h"
#include "perfmon.h"
#include "prof.h"
#include "platform/platform.h"

volatile uint32_t prof_render_tid, prof_main_tid, prof_replay_tid;

__attribute__((weak)) int prof_sampler_start(void) {
    if (cfg_get("TOTK_PROFILE")) rt_log("prof: TOTK_PROFILE: no sampling profiler on this OS");
    return 0;
}
__attribute__((weak)) void prof_sampler_write(void) {}
__attribute__((weak)) void prof_sampler_frame(uint64_t frame, uint64_t now, uint64_t d) { (void)frame, (void)now, (void)d; }

// TOTK_TRACE=perf, TOTK_FLIGHT=1 or the full performance overlay: event counts are taken. Wait times are always taken
// (frame-time summary). perf_base: the part fixed at start.
static int perf_on, perf_base;
static int overlay;   // performance overlay mode (perf_set_overlay): 0 off, 1 fps, 2 full, 3 breakdown

void perf_set_overlay(int mode) {
    __atomic_store_n(&overlay, mode, __ATOMIC_RELAXED);
    __atomic_store_n(&perf_on, perf_base || mode >= 2, __ATOMIC_RELAXED);
}

int perf_gpu_timing(void) { return (rt_trace & RT_TRACE_PERF) || __atomic_load_n(&overlay, __ATOMIC_RELAXED) >= 2; }

static volatile int64_t gpu_busy_total;

void perf_gpu_busy(uint64_t ns) { plat_add64(&gpu_busy_total, (int64_t)ns); }

uint64_t perf_gpu_busy_total(void) { return (uint64_t)__atomic_load_n(&gpu_busy_total, __ATOMIC_RELAXED); }

static volatile int64_t present_gpu_ns, present_timed;

void perf_present_gpu(uint64_t ns) { plat_add64(&present_gpu_ns, (int64_t)ns), plat_inc64(&present_timed); }

void prof_start(void) {
    flight_init();
    atexit(perf_summary_flush);   // boot.c's _exit path goes through prof_write instead
    perf_base = (rt_trace & RT_TRACE_PERF) || flight_on;
    if (rt_trace & RT_TRACE_PERF) plat_op_trace();   // operations in flight (platform.h), for ops_frame_log and vk_memory.c
    perf_on = perf_base || overlay >= 2;
    if (prof_sampler_start()) atexit(prof_write);   // a --seconds run ends in _exit: boot.c calls prof_write itself first
}

void prof_write(void) {
    nvn_mode_trace_finish(); // current callers are process exit paths (boot, window close, ExitProcess)
    perf_summary_flush();
    prof_sampler_write();
}

// ---- TOTK_TRACE=perf: frame interval and the time the render thread spends waiting, logged every 60 presents. The
// flight recorder (TOTK_FLIGHT=1, flight.c) takes the same numbers per frame.
static uint64_t wait_ns[PERF_N], frame_ns[PERF_N], worst_ns[PERF_N];   // window totals, this frame, the slowest frame

// ---- TOTK_TRACE=perf: the replay of a slow frame by part (host.h PERF_PH_). The recording thread keeps a stack of the
// parts open on it: a part's time is its interval less the intervals of the parts that ended inside it, so the parts
// add up to no more than the replay and what is left is the replay's own walk. Only the thread inside submit_record
// pushes (ph_on), and the slice workers while they record a slice (perf_ph_worker_begin), so the other threads'
// perf_begin costs one thread-local load more. A perf_begin whose perf_end never comes (an early return) stays on the
// stack until the part below it ends, and the time of the parts inside it goes to that part's children all the same
// (ph_close). Two banks of sums: the recording thread's (0) and the slice workers' together (1), whose work the
// recording thread sees only as its waits for slices. The sums are per present, cleared by the render thread there.
#define PH_DEPTH 32
typedef struct { uint64_t t0, child, cpu0; } PhFrame;
static _Thread_local int ph_on, ph_n, ph_bank;
static _Thread_local PhFrame ph_st[PH_DEPTH];
static volatile int64_t ph_ns[2][PERF_PH_N], ph_cnt[2][PERF_PH_N], ph_max[2][PERF_PH_N], ph_cpu[2][PERF_PH_N];
static volatile int64_t ph_all_ns[2], ph_all_cpu[2];   // the time inside the brackets and the thread CPU time in them

static void ph_push(uint64_t t, uint64_t cpu0) {
    if (ph_n < PH_DEPTH) ph_st[ph_n++] = (PhFrame){t, 0, cpu0};
}

static void ph_close(int part, uint64_t t0, uint64_t now) {
    int i = ph_n - 1;
    while (i >= 0 && ph_st[i].t0 != t0) i--;
    uint64_t d = now - t0, child = 0, cpu0 = 0;
    if (i >= 0) {
        for (int j = ph_n - 1; j > i; j--) ph_st[j - 1].child += ph_st[j].child;   // perf_begins without their end
        child = ph_st[i].child, cpu0 = ph_st[i].cpu0;
        ph_n = i;
    }
    if (ph_n) ph_st[ph_n - 1].child += d;
    if (part < 0 || part >= PERF_PH_N) return;
    int b = ph_bank;
    plat_add64(&ph_ns[b][part], (int64_t)(d > child ? d - child : 0));
    plat_inc64(&ph_cnt[b][part]);
    int64_t m = __atomic_load_n(&ph_max[b][part], __ATOMIC_RELAXED);
    while ((int64_t)d > m && !__atomic_compare_exchange_n(&ph_max[b][part], &m, (int64_t)d, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    if (cpu0) plat_add64(&ph_cpu[b][part], (int64_t)(plat_self_cpu_ns() - cpu0));
}

static _Thread_local uint64_t ph_t_start, ph_cpu_start;

static int ph_open(int bank) {
    int prev = ph_on | ph_bank << 1;
    ph_on = (rt_trace & RT_TRACE_PERF) != 0;
    ph_n = 0, ph_bank = bank;
    if (ph_on) ph_cpu_start = plat_self_cpu_ns(), ph_t_start = kern_now_ns();
    return prev;
}

static void ph_shut(int prev) {
    if (ph_on) {
        plat_add64(&ph_all_ns[ph_bank], (int64_t)(kern_now_ns() - ph_t_start));
        plat_add64(&ph_all_cpu[ph_bank], (int64_t)(plat_self_cpu_ns() - ph_cpu_start));
    }
    ph_on = prev & 1, ph_bank = prev >> 1, ph_n = 0;
}

int perf_ph_replay_begin(void) { return ph_open(0); }
void perf_ph_replay_end(int prev) { ph_shut(prev); }
int perf_ph_worker_begin(void) { return ph_open(1); }
void perf_ph_worker_end(int prev) { ph_shut(prev); }

uint64_t perf_ph_begin(void) {
    if (!ph_on) return 0;
    uint64_t t = kern_now_ns();
    ph_push(t, 0);
    return t;
}

uint64_t perf_ph_begin_cpu(void) {
    if (!ph_on) return 0;
    uint64_t c = plat_self_cpu_ns(), t = kern_now_ns();
    ph_push(t, c ? c : 1);
    return t;
}

void perf_ph_end(int part, uint64_t t0) {
    if (t0 && ph_on) ph_close(part, t0, kern_now_ns());
}

static volatile int64_t ph_new_ns, ph_new_draws, ph_new_passes;

void perf_ph_unsliced(uint64_t ns, uint32_t draws) {
    plat_add64(&ph_new_ns, (int64_t)ns), plat_add64(&ph_new_draws, draws), plat_inc64(&ph_new_passes);
}

uint64_t perf_begin(void) {
    uint64_t t = kern_now_ns();
    if (ph_on) ph_push(t, 0);
    return t;
}

void perf_replay_thread(void) {
    prof_replay_tid = plat_thread_id();
    plat_op_role(PLAT_OP_REPLAY);
}
void perf_main_thread(void) { prof_main_tid = plat_thread_id(); }

void perf_end(int kind, uint64_t t0) {
    if (!t0) return;
    if (ph_on) ph_close(kind, t0, kern_now_ns());
    uint32_t tid = plat_thread_id();
    if (tid != prof_render_tid && tid != prof_replay_tid) return;
    uint64_t d = kern_now_ns() - t0;
    if (tid == prof_render_tid && !prof_replay_tid) {   // one thread adds: no atomics on the synchronous path
        wait_ns[kind] += d;
        frame_ns[kind] += d;
        return;
    }
    __atomic_fetch_add(&wait_ns[kind], d, __ATOMIC_RELAXED);
    __atomic_fetch_add(&frame_ns[kind], d, __ATOMIC_RELAXED);
}

static volatile int64_t other_wait_ns;     // perf_gpu_wait on threads other than the render thread, over the window
static volatile int64_t render_wait_ns;    // the same on the render thread
void (*perf_gpu_stats)(PerfGpu *out);

void perf_gpu_wait(uint64_t t0) {
    if (!t0 || !__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) return;
    int64_t d = (int64_t)(kern_now_ns() - t0);
    plat_add64(plat_thread_id() == prof_render_tid ? &render_wait_ns : &other_wait_ns, d);
}

static volatile int64_t replay_ns;         // perf_replay: summed over the window, any thread
static volatile int64_t frame_replay_ns;   // the same since the previous present (frame-time summary)

void perf_replay(uint64_t t0) {
    if (!t0) return;
    int64_t d = (int64_t)(kern_now_ns() - t0);
    plat_add64(&frame_replay_ns, d);
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) plat_add64(&replay_ns, d);
}

static volatile uint64_t replay_tot[3];   // perf_replay_totals: draws, pools taken, pools created (running totals)

void perf_replay_totals(uint64_t draws, uint64_t pools_taken, uint64_t pools_created) {
    __atomic_store_n(&replay_tot[0], draws, __ATOMIC_RELAXED);
    __atomic_store_n(&replay_tot[1], pools_taken, __ATOMIC_RELAXED);
    __atomic_store_n(&replay_tot[2], pools_created, __ATOMIC_RELAXED);
}

static volatile int64_t counts[PERF_CNT_N];   // running totals: the perf log and the flight recorder take differences

void perf_count(int kind) {
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) plat_inc64(&counts[kind]);
}

// ---- Frame-time summary, always on: every 60 s of play and at exit one line of frame count, mean fps, 1% low and the
// frames over 25/50/100 ms, then the worst frames of the period with the render thread's waits in them. Per frame: a
// histogram increment, the compare against the worst list, a memset of the wait array. No allocation.
#define FT_BINS 1024              // 0.25 ms bins up to 256 ms; slower frames land in the last bin
#define FT_WORST 5
#define FT_PERIOD_NS 60000000000ull
typedef struct { uint64_t frame, ns, wait[PERF_N], replay; } FtWorst;
static struct {
    uint64_t t0, frames, sum, over25, over50, over100;
    uint32_t hist[FT_BINS];
    FtWorst worst[FT_WORST];   // slowest first
    int nworst;
} ft;
static uint64_t summary_t0;   // start of the current period (0: not started)

static void ft_report(uint64_t now, const char *tag) {
    if (ft.frames && ft.sum) {
        uint64_t cut = (ft.frames + 99) / 100, acc = 0, low = 0;   // the slowest 1% of frames
        for (int b = FT_BINS - 1; b >= 0; b--) {
            acc += ft.hist[b];
            if (acc >= cut) { low = (uint64_t)(b + 1) * 250000ull; break; }   // upper edge of the bin
        }
        rt_log("frametime: %s: %llu frames in %.1f s, mean %.1f fps, 1%% low %.1f fps, over 25 ms %llu, 50 ms %llu, 100 ms %llu",
               tag, (unsigned long long)ft.frames, (now - ft.t0) / 1e9, ft.frames * 1e9 / ft.sum, low ? 1e9 / low : 0.0,
               (unsigned long long)ft.over25, (unsigned long long)ft.over50, (unsigned long long)ft.over100);
        for (int i = 0; i < ft.nworst; i++) {
            const FtWorst *w = &ft.worst[i];
            rt_log("frametime:   worst frame %llu: %.1f ms; shader translation %.1f, pipeline build %.1f, texture upload %.1f (astc %.1f), replay %.1f, fence/present wait %.1f, pacing %.1f",
                   (unsigned long long)w->frame, w->ns / 1e6, w->wait[PERF_XLATE] / 1e6, w->wait[PERF_PIPE] / 1e6,
                   w->wait[PERF_TEX] / 1e6, w->wait[PERF_ASTC] / 1e6, w->replay / 1e6,
                   (w->wait[PERF_FENCE] + w->wait[PERF_DRAIN]) / 1e6, w->wait[PERF_PACE] / 1e6);
        }
    }
    memset(&ft, 0, sizeof ft);
    ft.t0 = now;
}

__attribute__((weak)) void gfx_exit_log(void) {}

void perf_summary_flush(void) {   // at exit (prof_write)
    if (summary_t0) {
        ft_report(kern_now_ns(), "exit");
        gfx_exit_log();
        mem_commit_report("exit");
    }
    summary_t0 = 0;
}

// Returns the replay time since the previous present.
static uint64_t ft_frame(uint64_t frame, uint64_t now, uint64_t d) {
    int64_t rep = plat_xchg64(&frame_replay_ns, 0);
    if (!summary_t0) {   // the first present starts the period; there is no interval yet
        summary_t0 = now;
        ft.t0 = now;
        return (uint64_t)rep;
    }
    ft.frames++;
    ft.sum += d;
    uint64_t b = d / 250000ull;
    ft.hist[b < FT_BINS ? b : FT_BINS - 1]++;
    if (d > 25000000ull) ft.over25++;
    if (d > 50000000ull) ft.over50++;
    if (d > 100000000ull) ft.over100++;
    if (d > 25000000ull && (ft.nworst < FT_WORST || d > ft.worst[FT_WORST - 1].ns)) {
        int i = ft.nworst < FT_WORST ? ft.nworst++ : FT_WORST - 1;
        while (i > 0 && ft.worst[i - 1].ns < d) ft.worst[i] = ft.worst[i - 1], i--;
        ft.worst[i].frame = frame;
        ft.worst[i].ns = d;
        memcpy(ft.worst[i].wait, frame_ns, sizeof frame_ns);
        ft.worst[i].replay = (uint64_t)rep;
    }
    if (now - summary_t0 >= FT_PERIOD_NS) {
        ft_report(now, "60 s");
        summary_t0 = now;
    }
    return (uint64_t)rep;
}

// The second line of the 60-frame window: GPU busy time (timestamps), the time threads waited for GPU fences, VRAM and
// process memory, and which side holds the frame back. pace: the frame waited for pacing (vsync, the frame limiter) at
// least 15% of the window; gpu: the GPU was busy for 80% of the frame interval or more (without timestamps: threads
// waited for fences for half the interval or more); cpu otherwise.
static void perf_gpu_line(uint64_t frame, double interval_ms, double pace_share, uint64_t n) {
    PerfGpu g;
    memset(&g, 0, sizeof g);
    if (perf_gpu_stats) perf_gpu_stats(&g);
    double rw = plat_xchg64(&render_wait_ns, 0) / 1e6 / n, ow = plat_xchg64(&other_wait_ns, 0) / 1e6 / n;
    double gpu_ms = g.have_gpu ? g.gpu_ns / 1e6 / n : 0;
    char gs[96], vs[96];
    if (g.have_gpu) snprintf(gs, sizeof gs, "gpu avg %.2f ms (%llu parts timed, %llu not)", gpu_ms,
                             (unsigned long long)g.gpu_parts, (unsigned long long)g.gpu_missed);
    else snprintf(gs, sizeof gs, "gpu avg n/a");
    if (g.have_vram) snprintf(vs, sizeof vs, "vram %.0f of %.0f MiB%s", g.vram_used / 1048576.0, g.vram_budget / 1048576.0,
                              g.vram_exact ? "" : " (own allocations of the heap size)");
    else snprintf(vs, sizeof vs, "vram n/a");
    char us[224];   // upload batches and presents: outside the timed submission parts, so not in "gpu avg"
    int64_t pg = plat_xchg64(&present_gpu_ns, 0), pt = plat_xchg64(&present_timed, 0);
    snprintf(us, sizeof us, "uploads %llu batches, %.1f MiB staged, gpu %.2f ms per frame (%llu timed), waited %.2f ms per frame; "
             "present gpu %.2f ms per frame (%lld timed)",
             (unsigned long long)g.up_batches, g.up_bytes / 1048576.0, g.up_gpu_ns / 1e6 / n, (unsigned long long)g.up_timed,
             g.up_wait_ns / 1e6 / n, pg / 1e6 / n, (long long)pt);
    uint64_t res, prv;
    plat_process_memory(&res, &prv);
    double ws = res / 1048576.0, priv = prv / 1048576.0;
    const char *bound = pace_share >= 0.15 ? "pace"
                      : g.have_gpu ? (gpu_ms >= 0.8 * interval_ms ? "gpu" : "cpu")
                      : (rw + ow >= 0.5 * interval_ms ? "gpu" : "cpu");
    rt_log("perf: frame %llu: %s; %s; fence wait per frame: render thread %.2f ms, other threads %.2f ms; %s; ram working "
           "set %.0f MiB, private %.0f MiB; bound: %s",
           (unsigned long long)frame, gs, us, rw, ow, vs, ws, priv, bound);
}

int rt_ab(void) {
    static int64_t period = -1;   // us of world clock; 0: off
    if (period < 0) {
        const char *e = cfg_get("TOTK_AB");
        period = e && *e ? (int64_t)(atof(e) * 1e6) : 0;
    }
    int64_t w = period ? frame_step_world_us() : -1;
    if (w < 0) return 0;   // before the world clock: A
    int64_t k = w / period % 4;
    return k == 1 || k == 2;
}

int rt_knob(const char *name, const char *def) {
    char key[64];
    snprintf(key, sizeof key, "TOTK_%s", name);
    const char *e = cfg_get(key);
    if (!e || !*e) e = def;
    return !strcmp(e, "ab") ? 2 : strcmp(e, "0") != 0;
}

// TOTK_ONE_PER_CORE=1: the process keeps the first logical processor of each physical core, so busy threads (the
// render thread, the guest cores, the job workers) never share a core's execution units with another; "ab" switches
// it with the A/B blocks. Off by default until measured here; Savage Wind Recomp
// reported +4-10% from the same rule on an 8-core SMT CPU.
static void cpu_placement(void) {
    static int mode = -1, applied = -1;
    static uint64_t all, one;
    if (mode < 0) {
        mode = rt_knob("ONE_PER_CORE", "0");
        int cores = plat_cpu_cores(&all, &one);
        if (mode && one && one != all)
            rt_log("cpu: %d cores, %d logical processors: one per core%s", cores, __builtin_popcountll(all),
                   mode == 2 ? " in the B blocks" : "");
        else
            mode = 0;
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    applied = on;
    plat_process_affinity(on ? one : all);
}

// TOTK_AB: every frame's interval and replay time summed per side (A, B) once the world clock passes 10 s, logged
// every 600 frames. Frame by frame, so blocks of a few seconds count whole (the 60-frame perf windows mostly straddle
// them); a 2-3 s period keeps both sides on nearly the same scenery and the same CPU temperature.
static void ab_frame(uint64_t frame, uint64_t d, uint64_t rep, int side) {
    static int on = -1;
    static double sum[2], rsum[2], n[2], sq[2];
    if (on < 0) {
        const char *e = cfg_get("TOTK_AB");
        on = e && *e && atof(e) > 0;
    }
    if (!on || !d) return;
    if (frame_step_world_us() >= 10000000 && d < 250000000ull) {
        double ms = d / 1e6;
        sum[side] += ms, sq[side] += ms * ms, rsum[side] += rep / 1e6, n[side]++;
    }
    if (frame % 600 || !n[0] || !n[1]) return;
    double a = sum[0] / n[0], b = sum[1] / n[1], ra = rsum[0] / n[0], rb = rsum[1] / n[1];
    double va = sq[0] / n[0] - a * a, vb = sq[1] / n[1] - b * b;
    double se = sqrt((va > 0 ? va : 0) / n[0] + (vb > 0 ? vb : 0) / n[1]);
    rt_log("perf: ab frames: A %.0f frames %.2f ms (%.1f fps) replay %.2f ms; B %.0f frames %.2f ms (%.1f fps) replay %.2f "
           "ms; B-A interval %+.1f%% (+-%.1f%%), fps %+.1f%%, replay %+.1f%%",
           n[0], a, 1000 / a, ra, n[1], b, 1000 / b, rb, 100 * (b - a) / a, 200 * se / a, 100 * (a / b - 1),
           ra > 0 ? 100 * (rb - ra) / ra : 0);
}

static volatile int64_t lock_waits[PERF_LOCK_N][3];   // waits over 100 us: count, total ns, longest ns

void perf_lock_wait(int lock, uint64_t ns) {
    if (ns < 100000 || lock < 0 || lock >= PERF_LOCK_N) return;
    plat_add64(&lock_waits[lock][0], 1), plat_add64(&lock_waits[lock][1], (int64_t)ns);
    int64_t m = __atomic_load_n(&lock_waits[lock][2], __ATOMIC_RELAXED);
    while ((int64_t)ns > m && !__atomic_compare_exchange_n(&lock_waits[lock][2], &m, (int64_t)ns, 0, __ATOMIC_RELAXED,
                                                            __ATOMIC_RELAXED)) {}
}

static void lock_waits_log(uint64_t frame) {
    static const char *const name[PERF_LOCK_N] = {"texture", "memory pools", "mirror", "generations"};
    char line[300];
    size_t o = 0;
    for (int i = 0; i < PERF_LOCK_N; i++) {
        int64_t n = plat_xchg64(&lock_waits[i][0], 0), t = plat_xchg64(&lock_waits[i][1], 0), m = plat_xchg64(&lock_waits[i][2], 0);
        o += (size_t)snprintf(line + o, sizeof line - o, "%s%s %lld (%.1f ms, longest %.0f us)", i ? ", " : "", name[i],
                              (long long)n, t / 1e6, m / 1e3);
    }
    rt_log("perf: frame %llu: lock waits over 100 us: %s", (unsigned long long)frame, line);
}

// A slow frame's replay by part (TOTK_TRACE=perf): every part that took 0.05 ms or more, longest first, with its count,
// its longest single interval (with the parts inside it) and, for the parts timed with perf_ph_begin_cpu, the thread's
// CPU time in them. "rest of the walk" is the replay less the parts: the walk's own work and everything it does that no
// part names. "outside the replay" is the frame interval less the replay; with TOTK_REPLAY_ASYNC the replay runs on its
// own thread beside the render thread, so the two need not add up to the interval. A second line gives the slice
// workers' slices the same way, summed over the workers (they run side by side, so it can exceed the frame). Every
// present clears the sums, so the lines hold what was recorded since the previous one.
static void ph_frame(uint64_t frame, uint64_t d, uint64_t rep, int slow) {
    static const char *const name[PERF_PH_N] = {
        [PERF_FENCE] = "fence waits", [PERF_DRAIN] = "recorder drains", [PERF_PIPE] = "pipeline waits",
        [PERF_XLATE] = "shader translation", [PERF_TEX] = "texture uploads", [PERF_ASTC] = "ASTC decode",
        [PERF_PH_IMPORT] = "pool imports on first use", [PERF_PH_MIRROR] = "mirrors made on first use",
        [PERF_PH_MIRROR_BEGIN] = "mirror pre-snapshot", [PERF_PH_MSYNC] = "mirror syncs",
        [PERF_PH_WW] = "write-watch queries", [PERF_PH_SHADER] = "shader lookups (new program stages)",
        [PERF_PH_IMAGE] = "texture images made", [PERF_PH_DLSS_CREATE] = "DLSS feature creation",
        [PERF_PH_DLSS_EVAL] = "DLSS evaluation", [PERF_PH_TEMPORAL] = "temporal inputs (motion pass)",
        [PERF_PH_SPIRV_HASH] = "SPIR-V hashes (upscaler match)", [PERF_PH_SLICE_START] = "slice worker start",
        [PERF_PH_SLICE_WAIT] = "waits for slices", [PERF_PH_SUBMIT] = "part submits",
        [PERF_PH_USNAP] = "uniform snapshot", [PERF_PH_BUDGET] = "VRAM budget queries"};
    int64_t ns[2][PERF_PH_N], cnt[2][PERF_PH_N], mx[2][PERF_PH_N], cpu[2][PERF_PH_N], all[2], allc[2];
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < PERF_PH_N; i++)
            ns[b][i] = plat_xchg64(&ph_ns[b][i], 0), cnt[b][i] = plat_xchg64(&ph_cnt[b][i], 0),
            mx[b][i] = plat_xchg64(&ph_max[b][i], 0), cpu[b][i] = plat_xchg64(&ph_cpu[b][i], 0);
        all[b] = plat_xchg64(&ph_all_ns[b], 0), allc[b] = plat_xchg64(&ph_all_cpu[b], 0);
    }
    int64_t nns = plat_xchg64(&ph_new_ns, 0), ndraws = plat_xchg64(&ph_new_draws, 0), npass = plat_xchg64(&ph_new_passes, 0);
    if (!slow) return;
    for (int b = 0; b < 2; b++) {
        if (b && !all[1]) break;
        int order[PERF_PH_N], k = 0;
        int64_t parts = 0;
        for (int i = 0; i < PERF_PH_N; i++) {
            parts += ns[b][i];
            if (name[i] && ns[b][i] >= 50000) order[k++] = i;
        }
        for (int a = 1; a < k; a++)
            for (int c = a; c > 0 && ns[b][order[c]] > ns[b][order[c - 1]]; c--) {
                int t = order[c];
                order[c] = order[c - 1], order[c - 1] = t;
            }
        char line[2048];
        size_t o = 0;
        line[0] = 0;
        for (int j = 0; j < k && o < sizeof line - 160; j++) {
            int i = order[j];
            o += (size_t)snprintf(line + o, sizeof line - o, "; %s %.1f (%lld, longest %.1f", name[i], ns[b][i] / 1e6,
                                  (long long)cnt[b][i], mx[b][i] / 1e6);
            if (cpu[b][i]) o += (size_t)snprintf(line + o, sizeof line - o, ", cpu %.1f", cpu[b][i] / 1e6);
            o += (size_t)snprintf(line + o, sizeof line - o, ")");
        }
        if (!b)
            rt_log("perf: slow frame %llu replay by part: %.1f ms on the recording thread (cpu %.1f)%s; rest of the walk "
                   "%.1f; outside the replay %.1f of the %.1f ms frame; passes left unsliced for new targets %lld (%lld "
                   "draws, %.1f ms with the parts inside them)", (unsigned long long)frame, rep / 1e6, allc[0] / 1e6,
                   line, ((int64_t)rep - parts) / 1e6, ((int64_t)d - (int64_t)rep) / 1e6, d / 1e6, (long long)npass,
                   (long long)ndraws, nns / 1e6);
        else
            rt_log("perf: slow frame %llu slice workers by part: %.1f ms recording slices, summed over the workers (cpu "
                   "%.1f)%s; rest of their slices %.1f", (unsigned long long)frame, all[1] / 1e6, allc[1] / 1e6, line,
                   (all[1] - parts) / 1e6);
    }
}

// TOTK_TRACE=perf: the host operations (platform.h plat_op_begin) that ended in a slow frame, by thread role and kind,
// so that a long frame outside the replay (a layer's teardown and load) splits into pool finalizes, thaws, imports,
// releases, file reads and the system calls inside them. Taken every present, logged for the slow ones.
static void ops_frame_log(uint64_t frame, int slow) {
    static PlatOpTotal t[PLAT_OP_ROLES][PLAT_OP_N];
    plat_op_totals(t);
    if (!slow) return;
    static const int order[PLAT_OP_ROLES] = {PLAT_OP_GAME, PLAT_OP_REPLAY, PLAT_OP_RENDER, PLAT_OP_OTHER};
    char line[3000];
    size_t o = 0;
    line[0] = 0;
    for (int i = 0; i < PLAT_OP_ROLES; i++) {
        int r = order[i], first = 1;
        for (int k = 0; k < PLAT_OP_N && o < sizeof line - 160; k++) {
            if (!t[r][k].calls) continue;
            if (first) o += (size_t)snprintf(line + o, sizeof line - o, "%s%s: ", o ? "; " : "", plat_op_role_name(r));
            o += (size_t)snprintf(line + o, sizeof line - o, "%s%s %llu (%.2f ms, longest %.2f)", first ? "" : ", ",
                                  plat_op_name(k), (unsigned long long)t[r][k].calls, t[r][k].ns / 1e6,
                                  t[r][k].max_ns / 1e6);
            first = 0;
        }
    }
    rt_log("perf: slow frame %llu host operations that ended in it (nested ones also count in their outer one): %s",
           (unsigned long long)frame, o ? line : "none");
}

void perf_present(uint64_t frame) {
    static uint64_t last, sum, max, n, win_start, ab_b;
    static int64_t win_prev[PERF_CNT_N];
    static uint32_t op_render;   // the thread plat_op_role was told is the render thread
    prof_render_tid = plat_thread_id();
    if (op_render != prof_render_tid && prof_render_tid != prof_replay_tid)
        op_render = prof_render_tid, plat_op_role(PLAT_OP_RENDER);
    if (rt_trace & RT_TRACE_SCHED) kern_sched_report(frame);
    uint64_t now = kern_now_ns(), d = last ? now - last : 0;
    prof_sampler_frame(frame, now, last ? d : 0);
    uint64_t rep = ft_frame(frame, now, d);   // d is 0 on the first present: it only starts the period
    int ab_side = rt_ab();
    ab_b += ab_side;
    ab_frame(frame, d, rep, ab_side);
    cpu_placement();
    if (__atomic_load_n(&overlay, __ATOMIC_RELAXED))   // the overlay's only cost while it is off: this branch
        perfmon_frame(now, d, frame_ns, rep, __atomic_load_n(&counts[PERF_CNT_TEX_UP], __ATOMIC_RELAXED));
    kern_core_diag_frame(frame, d);
    if (flight_on) flight_frame(frame, now, d, frame_ns, counts);
    if (!(rt_trace & RT_TRACE_PERF)) {
        memset(frame_ns, 0, sizeof frame_ns);
        last = now;
        return;
    }
    static uint64_t slow_ns;   // TOTK_PERF_SLOW_MS (default 50): frames at least this long get a line of their own
    if (!slow_ns) {
        const char *e = cfg_get("TOTK_PERF_SLOW_MS");
        slow_ns = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 50) * 1000000ull;
    }
    if (last) {
        sum += d;
        n++;
        if (d > max) max = d, memcpy(worst_ns, frame_ns, sizeof worst_ns);
        if (d >= slow_ns)
            rt_log("perf: slow frame %llu: %.1f ms; blocked pipeline %.1f xlate %.1f texture %.1f astc %.1f fence %.1f pacing %.1f; "
                   "replay %.1f", (unsigned long long)frame, d / 1e6, frame_ns[PERF_PIPE] / 1e6, frame_ns[PERF_XLATE] / 1e6,
                   frame_ns[PERF_TEX] / 1e6, frame_ns[PERF_ASTC] / 1e6, frame_ns[PERF_FENCE] / 1e6,
                   frame_ns[PERF_PACE] / 1e6, rep / 1e6);
    }
    ph_frame(frame, d, rep, last && d >= slow_ns);
    ops_frame_log(frame, last && d >= slow_ns);
    memset(frame_ns, 0, sizeof frame_ns);
    last = now;
    if (!win_start) win_start = now;
    if (frame % 60) return;
    lock_waits_log(frame);
    uint64_t span = now - win_start;
    int64_t win[PERF_CNT_N];
    for (int i = 0; i < PERF_CNT_N; i++) {
        int64_t v = __atomic_load_n(&counts[i], __ATOMIC_RELAXED);
        win[i] = v - win_prev[i];
        win_prev[i] = v;
    }
    int64_t async = win[PERF_CNT_PIPE_ASYNC], skips = win[PERF_CNT_PIPE_SKIP], up = win[PERF_CNT_TEX_UP];
    int64_t reup = win[PERF_CNT_TEX_REUP], sskip = win[PERF_CNT_SHADER_SKIP], waited = win[PERF_CNT_PIPE_WAIT];
    int64_t oneshot = win[PERF_CNT_ONESHOT_WAIT];
    uint64_t shader_keys, shader_xlated;
    gfx_shader_counts(&shader_keys, &shader_xlated);
    static uint64_t rt_prev[3];   // replay_tot at the previous line
    double rd[3];
    for (int i = 0; i < 3; i++) {
        uint64_t v = __atomic_load_n(&replay_tot[i], __ATOMIC_RELAXED);
        rd[i] = (double)(v - rt_prev[i]), rt_prev[i] = v;
    }
    int64_t rep_ns = plat_xchg64(&replay_ns, 0);
    if (n && span)
        rt_log("perf: frame %llu: interval avg %.2f ms max %.2f ms (%.1f fps); render thread blocked: pacing %.0f%% fence %.0f%% drain %.0f%% pipeline build %.0f%% shader xlate %.0f%% texture %.0f%% (astc %.0f%%); pipelines built in background %lld, draws skipped pending %lld, waited %lld, waiting for a shader %lld, one-shot waited %lld; textures uploaded %lld re-uploaded %lld; slowest frame: xlate %.1f ms texture %.1f ms (astc %.1f) pipeline %.1f ms fence %.1f ms; shaders: %llu keys, %llu translated; render busy avg %.2f ms; replay avg %.2f ms, %.0f draws and dispatches per frame, %.2f us each, descriptor pools taken %.1f created %.1f per frame; ab %llu/%llu; world %.1f s",
               (unsigned long long)frame, sum / 1e6 / n, max / 1e6, n * 1e9 / sum, 100.0 * wait_ns[PERF_PACE] / span,
               100.0 * wait_ns[PERF_FENCE] / span, 100.0 * wait_ns[PERF_DRAIN] / span,
               100.0 * wait_ns[PERF_PIPE] / span, 100.0 * wait_ns[PERF_XLATE] / span,
               100.0 * wait_ns[PERF_TEX] / span, 100.0 * wait_ns[PERF_ASTC] / span, (long long)async, (long long)skips, (long long)waited, (long long)sskip, (long long)oneshot,
               (long long)up, (long long)reup, worst_ns[PERF_XLATE] / 1e6, worst_ns[PERF_TEX] / 1e6,
               worst_ns[PERF_ASTC] / 1e6, worst_ns[PERF_PIPE] / 1e6, worst_ns[PERF_FENCE] / 1e6,
               (unsigned long long)shader_keys, (unsigned long long)shader_xlated,
               (span > wait_ns[PERF_PACE] ? span - wait_ns[PERF_PACE] : 0) / 1e6 / n,
               rep_ns / 1e6 / n, rd[0] / n, rd[0] > 0 ? rep_ns / 1e3 / rd[0] : 0.0, rd[1] / n, rd[2] / n,
               (unsigned long long)ab_b, (unsigned long long)n, frame_step_world_us() / 1e6);
    if (n && span) perf_gpu_line(frame, sum / 1e6 / n, (double)wait_ns[PERF_PACE] / span, n);
    sum = max = n = ab_b = 0;
    memset(wait_ns, 0, sizeof wait_ns);
    win_start = now;
}
