"""Find slowdowns in game logs taken with TOTK_TRACE=perf, say what held each one back, and compare groups of runs.

    python tools/perf_check.py build/dkb100/logs/totk.log
    python tools/perf_check.py run1.out run2.out --min-fps 50 --hitch-ms 100
    python tools/perf_check.py --group off='runs/gs0-*.out' --group on='runs/gs1-*.out'
    python tools/perf_check.py scenario.out --fail            exit 1 when a run has a slowdown (scripted scenarios)

It reads the lines runtime/perf.c writes every 60 presents ("perf: frame N: interval avg ..." and its "gpu avg" line), the
kernel's preemption statistics ("kernel: frame N: asynchronous preemption ...") and the 60-second frametime summaries.
A window is slow when its mean interval is longer than 1000 / --min-fps ms. Consecutive slow windows form an episode, and
each episode gets the cause most of its windows show, checked in this order:

  compile      the render thread waited for shader translation or pipeline builds (expected on a cold shader cache)
  texture      it waited for texture uploads or decodes
  gpu          the GPU was busy for most of the interval, or the render thread waited for fences or a drain
  render-cpu   the render thread was busy itself (replaying commands, host-side work) for most of the interval
  game-cpu     none of the above: the game's own threads were slow (guest code, host calls, waits between threads)

An episode is major when its mean interval passes 1.5 times the budget, it lasts --major-windows windows or more, or one
of its frames takes a second or longer; otherwise it is minor (a short dip such as a streaming load at the same point of
every run). Verdicts: SLOW with a major episode, HITCHY with minor ones only, OK without slow windows, "no data" when the
log has no perf windows (the run did not get that far, or TOTK_TRACE=perf was not set).

Windows before --skip-world seconds of game time are left out (title screen and loading). A hitch is a window whose
longest frame passed --hitch-ms; hitches are counted but do not change the verdict. Each cause comes with the tool that
looks deeper (crit_path.py, prof_report.py, TOTK_PERF_SLOW_GPU_MS). Standard library only.
"""
import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

WINDOW = re.compile(
    r'perf: frame (?P<frame>\d+): interval avg (?P<avg>[\d.]+) ms max (?P<max>[\d.]+) ms \([\d.]+ fps\); '
    r'render thread blocked: pacing (?P<pacing>\d+)% fence (?P<fence>\d+)% drain (?P<drain>\d+)% '
    r'pipeline build (?P<pipeline>\d+)% shader xlate (?P<xlate>\d+)% texture (?P<texture>\d+)%'
    r'(?:.*?shaders: \d+ keys, (?P<translated>\d+) translated)?'
    r'(?:.*?render busy avg (?P<busy>[\d.]+) ms)?(?:.*?replay avg (?P<replay>[\d.]+) ms)?'
    r'(?:.*?world (?P<world>[\d.]+) s)?')
GPU = re.compile(r'perf: frame (?P<frame>\d+): gpu avg (?P<gpu>[\d.]+|n/a)(?: ms)?.*?bound: (?P<bound>\w+)')
KERNEL = re.compile(r'kernel: frame (?P<frame>\d+): asynchronous preemption: (?P<interrupted>\d+) interrupted.*?'
                    r'refused: (?P<host>\d+) in a host call, (?P<outside>\d+) outside game code')
SUMMARY = re.compile(r'frametime: (?P<what>60 s|exit): (?P<frames>\d+) frames in (?P<secs>[\d.]+) s, mean (?P<fps>[\d.]+) '
                     r'fps, 1% low (?P<low>[\d.]+) fps')

CAUSES = ('compile', 'texture', 'gpu', 'render-cpu', 'game-cpu')
HINTS = {
    'compile': 'shader translation or pipeline builds: expected on a first run with a new translator; on a warm cache, '
               'look for shaders translated again (gfx_shader.c keys) or pipelines missing from pipeline-cache.bin',
    'texture': 'texture uploads: rerun with TOTK_TRACE=perf,gfx, or TOTK_FLIGHT=1 and tools/flight_view.py to see the '
               'frames and their uploads',
    'gpu': 'GPU work: rerun with TOTK_PERF_SLOW_GPU_MS=50 to log the slow replay parts, their stages and indirect arguments',
    'render-cpu': 'the render thread itself: rerun with TOTK_PROFILE=1 TOTK_PROFILE_RENDER=1, then tools/prof_report.py',
    'game-cpu': "the game's threads: rerun with TOTK_CRIT=1, then tools/crit_path.py for each frame's critical path "
                '(running, waiting for a core, in a host call, or waiting on another thread)',
}


def parse(path):
    """Windows of one log: [{frame, avg, max, shares..., gpu, bound, host_calls}], plus its frametime summaries."""
    windows, gpu, kernel, summaries = {}, {}, [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'perf: frame' in line:
                m = WINDOW.search(line)
                if m:
                    d = m.groupdict()
                    w = {k: float(d[k]) for k in ('avg', 'max', 'pacing', 'fence', 'drain', 'pipeline', 'xlate', 'texture')}
                    w['frame'] = int(d['frame'])
                    for k in ('translated', 'busy', 'replay', 'world'):
                        w[k] = float(d[k]) if d[k] is not None else None
                    windows[w['frame']] = w
                    continue
                m = GPU.search(line)
                if m:
                    gpu[int(m['frame'])] = (None if m['gpu'] == 'n/a' else float(m['gpu']), m['bound'])
            elif 'kernel: frame' in line:
                m = KERNEL.search(line)
                if m:
                    kernel.append((int(m['frame']), int(m['host'])))
            elif 'frametime:' in line:
                m = SUMMARY.search(line)
                if m:
                    summaries.append(dict(what=m['what'], frames=int(m['frames']), secs=float(m['secs']),
                                          fps=float(m['fps']), low=float(m['low'])))
    out, prev_total = [], 0.0
    for frame in sorted(windows):
        w = windows[frame]
        # The perf line's "N translated" is the run's total so far (gfx_shader_counts): keep the window's own count.
        if w['translated'] is not None:
            w['translated'], prev_total = max(0.0, w['translated'] - prev_total), w['translated']
        w['gpu'], w['bound'] = gpu.get(frame, (None, None))
        # Host-call refusals per present since the previous kernel line: how often a waiting thread found the core's
        # owner inside a host call (kernel.c ap_try). A jump means game threads sit in host code.
        before = [(f, h) for f, h in kernel if f <= frame]
        w['host_calls'] = None
        if len(before) >= 2:
            (f0, h0), (f1, h1) = before[-2], before[-1]
            w['host_calls'] = (h1 - h0) / max(1, f1 - f0)
        elif len(before) == 1:
            w['host_calls'] = before[0][1] / max(1, before[0][0])
        out.append(w)
    return out, summaries


def cause(w):
    """The side that held window w back, from the runtime's own measurements (runtime/perf.c)."""
    if w['xlate'] + w['pipeline'] >= 20 or (w['translated'] or 0) > 0 and w['xlate'] + w['pipeline'] >= 5:
        return 'compile'
    if w['texture'] >= 20:
        return 'texture'
    if w['bound'] == 'gpu' or w['fence'] + w['drain'] >= 20 or w['gpu'] is not None and w['gpu'] >= 0.8 * w['avg']:
        return 'gpu'
    if w['busy'] is not None and w['busy'] >= 0.8 * w['avg']:
        return 'render-cpu'
    return 'game-cpu'


def analyse(path, min_fps, hitch_ms, skip_world, major_windows=3):
    windows, summaries = parse(path)
    budget = 1000.0 / min_fps
    played = [w for w in windows if w['world'] is None or w['world'] >= skip_world]
    for w in played:
        w['slow'] = w['avg'] > budget
        w['cause'] = cause(w) if w['slow'] else None
    episodes, cur = [], None
    for w in played:
        if w['slow']:
            if cur is None:
                cur = dict(windows=[])
                episodes.append(cur)
            cur['windows'].append(w)
        else:
            cur = None
    for e in episodes:
        ws = e.pop('windows')
        counts = {c: sum(w['cause'] == c for w in ws) for c in CAUSES}
        e.update(first_frame=ws[0]['frame'] - 59, last_frame=ws[-1]['frame'], windows=len(ws),
                 world=(ws[0]['world'], ws[-1]['world']), mean_ms=statistics.mean(w['avg'] for w in ws),
                 worst_ms=max(w['max'] for w in ws), cause=max(CAUSES, key=lambda c: counts[c]),
                 causes={c: n for c, n in counts.items() if n},
                 render_busy_ms=_mean(w['busy'] for w in ws), replay_ms=_mean(w['replay'] for w in ws),
                 gpu_ms=_mean(w['gpu'] for w in ws), host_calls=_mean(w['host_calls'] for w in ws))
        e['major'] = e['mean_ms'] > 1.5 * budget or e['windows'] >= major_windows or e['worst_ms'] >= 1000
    steady = [w for w in played if not w['slow']]
    hitches = [w for w in played if w['max'] > hitch_ms]
    slow = sum(w['slow'] for w in played)
    verdict = ('no data' if not played else 'SLOW' if any(e['major'] for e in episodes)
               else 'HITCHY' if episodes else 'OK')
    return dict(
        log=str(path), windows=len(played), skipped=len(windows) - len(played), verdict=verdict,
        fps=_fps(played), slow_windows=slow, slow_share=slow / len(played) if played else 0.0,
        steady_host_calls=_mean(w['host_calls'] for w in steady),
        hitches=len(hitches), worst_ms=max((w['max'] for w in played), default=None),
        episodes=episodes, summaries=summaries)


def _mean(values):
    v = [x for x in values if x is not None]
    return statistics.mean(v) if v else None


def _fps(windows):
    """Frames per second over the windows: 60 presents per window over their summed time."""
    t = sum(w['avg'] for w in windows) * 60
    return 60 * len(windows) * 1000.0 / t if t else None


def _f(v, fmt='{:.1f}', none='-'):
    return none if v is None else fmt.format(v)


def report(r, budget_ms, out):
    out(f'{r["log"]}: {r["verdict"]}, {_f(r["fps"])} fps over {r["windows"]} windows'
        f' ({r["skipped"]} before the skip), {r["slow_windows"]} slow ({r["slow_share"] * 100:.0f}%), '
        f'{r["hitches"]} with a frame over the hitch limit, worst frame {_f(r["worst_ms"], "{:.0f}")} ms')
    for e in r['episodes']:
        world = '' if e['world'][0] is None else f', world {e["world"][0]:.1f}-{e["world"][1]:.1f} s'
        mix = ', '.join(f'{c} {n}' for c, n in e['causes'].items())
        out(f'  {"major" if e["major"] else "minor"} frames {e["first_frame"]}-{e["last_frame"]}{world}: '
            f'{e["windows"]} windows at {e["mean_ms"]:.1f} ms '
            f'(budget {budget_ms:.1f}), worst frame {e["worst_ms"]:.0f} ms; cause {e["cause"]} ({mix})')
        detail = [f'render thread busy {_f(e["render_busy_ms"])} ms', f'replay {_f(e["replay_ms"])} ms',
                  f'gpu {_f(e["gpu_ms"], "{:.2f}")} ms']
        if e['host_calls'] is not None:
            steady = r['steady_host_calls']
            detail.append(f'host-call preemption refusals {e["host_calls"]:.2f} per frame'
                          + (f' (steady windows {steady:.2f})' if steady is not None else ''))
        out('    per frame: ' + ', '.join(detail))
    causes = sorted({e['cause'] for e in r['episodes']}, key=CAUSES.index)
    for c in causes:
        out(f'  next ({c}): {HINTS[c]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='*', type=Path)
    ap.add_argument('--group', action='append', default=[], metavar='NAME=GLOB',
                    help='a named set of logs to compare with the other groups (repeatable)')
    ap.add_argument('--min-fps', type=float, default=50.0, help='a window below this is slow (default 50)')
    ap.add_argument('--hitch-ms', type=float, default=100.0, help='a frame longer than this is a hitch (default 100)')
    ap.add_argument('--skip-world', type=float, default=5.0,
                    help='leave out windows before this much game time, in seconds (default 5)')
    ap.add_argument('--major-windows', type=int, default=3,
                    help='an episode this many windows long is major whatever its mean (default 3)')
    ap.add_argument('--json', type=Path, help='also write the analysis here')
    ap.add_argument('--fail', action='store_true', help='exit 1 when a run is SLOW or has no perf data')
    ap.add_argument('--fail-hitchy', action='store_true', help='with --fail, a HITCHY run fails too')
    a = ap.parse_args(argv)

    groups = []
    for g in a.group:
        name, sep, pattern = g.partition('=')
        if not sep:
            ap.error(f'--group needs NAME=GLOB, got {g!r}')
        files = sorted(Path(p) for p in glob.glob(pattern))
        if not files:
            ap.error(f'--group {name}: no file matches {pattern}')
        groups.append((name, files))
    if a.logs:
        groups.append(('', a.logs))
    if not groups:
        ap.error('give logs or --group')

    budget = 1000.0 / a.min_fps
    results = {}
    for name, files in groups:
        if name:
            print(f'== {name}')
        results[name] = [analyse(f, a.min_fps, a.hitch_ms, a.skip_world, a.major_windows) for f in files]
        for r in results[name]:
            report(r, budget, print)
    named = [(n, rs) for n, rs in results.items() if n]
    if len(named) > 1:
        print('\ngroup       runs  SLOW runs  median fps  slow windows  major episodes by cause')
        for n, rs in named:
            fps = [r['fps'] for r in rs if r['fps'] is not None]
            by = {}
            for r in rs:
                for e in r['episodes']:
                    if e['major']:
                        by[e['cause']] = by.get(e['cause'], 0) + 1
            share = statistics.mean(r['slow_share'] for r in rs) * 100
            print(f'{n:10} {len(rs):5} {sum(r["verdict"] == "SLOW" for r in rs):6}/{len(rs):<3} '
                  f'{_f(statistics.median(fps) if fps else None):>10} {share:12.0f}%  '
                  + (', '.join(f'{c} {k}' for c, k in sorted(by.items(), key=lambda kv: CAUSES.index(kv[0]))) or '-'))
    if a.json:
        a.json.write_text(json.dumps(dict(min_fps=a.min_fps, hitch_ms=a.hitch_ms, skip_world=a.skip_world,
                                          groups=results), indent=2) + '\n', encoding='utf-8')
    passing = {'OK', 'HITCHY'} if not a.fail_hitchy else {'OK'}
    bad = [r for rs in results.values() for r in rs if r['verdict'] not in passing]
    return 1 if a.fail and bad else 0


if __name__ == '__main__':
    sys.exit(main())
