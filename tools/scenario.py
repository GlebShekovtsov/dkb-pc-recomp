"""Run the game on a saved scenario without anyone playing, and say whether it worked.

    python tools/scenario.py --list                         the scenarios and whether their private save is present
    python tools/scenario.py tutorial-cave                  one run with the default build (build/dkb100)
    python tools/scenario.py tutorial-cave --repeat 3       three runs
    python tools/scenario.py tutorial-cave --ab native_gs   variants of the scenario, alternating: off, on, off, on, ...
    python tools/scenario.py tutorial-cave --build build/dkb100-pr67 --env TOTK_VK_VALIDATION=1
    python tools/scenario.py tutorial-cave --warp HillLevelStage    start in another stage (tools/save_warp.py)
    python tools/scenario.py tutorial-cave --sweep 'LevelStage$'    once per matching stage (the 20 layers)
    python tools/scenario.py tutorial-cave --sweep 'LevelStage$' --jobs 3   three games at once (timings unreliable)
    python tools/scenario.py tutorial-cave --repeat 2 --env TOTK_DUMP=4300 --env TOTK_DUMP_DIR={run}/dump
                                                         {run} in a value: that run's own folder

A scenario is tests/scenarios/<name>.toml: which private save to start from, the scripted input that gets from the title
screen into it, how long to run, and what must hold afterwards. The save itself is never in the repository: it is the
folder build/saves/<save>/ (account.uid and save/ of a game profile from your own console or an earlier run). The input
scripts are timings only (tools/input/), not game data.

Each run gets its own folder build/scenarios/<stamp>-<name>/<variant>-<n>/ with the game's whole output (game.log), the
last frame (last.png), the perf analysis (perf.json, tools/perf_check.py) and result.json. The profile is a fresh copy of
the save and is deleted after the run (--keep-profile keeps it), so a run never changes the save.

How a run ended is one of:
  completed        the game reached --seconds of game time and exited 0
  crashed          the crash reporter wrote FATAL (exit 71): the log has the host address, the lifted function, the
                   guest registers and frames; crash.txt has that part
  crashed, no report   the process died with no FATAL line (stack overflow, a fault in a driver or a validation layer,
                   or the reporter itself failing); the exit code says how (0xC0000005 access violation, 0xC00000FD stack
                   overflow, 0xC0000409 fast fail)
  device lost      Vulkan lost the device; the log has the VK_EXT_device_fault report when the driver gives one
  hung             no exit within the time limit (seconds x 6 + 180 s), or, while the run was not over, STALL_S without
                   a frame (the stall watchdog's "stall: no frame" lines, TOTK_STALL_DUMP) or without any other log
                   line; before the kill, hang-stacks.txt gets every thread's stack (gdb attached, when installed)
  warp failed      tools/save_warp.py could not move the profile copy to the stage (unknown save layout, a stage name
                   too long for its field); no game was started and the batch goes on
  did not start    the exe could not be started (missing build, wrong path)
Then the checks: a perf verdict (OK, HITCHY or SLOW, tools/perf_check.py); that the last frame is in the scene (its
HUD band has bright pixels; a loading screen or fade has none, and then nothing about the picture is judged); and
the last frame's share of near-black pixels below the HUD band (the black destructible terrain of #45 shows as a
high share).

Before each run the machine is checked: another game process running stops the run (two games share the GPU); other
programs using more than 1 GiB of memory or less than 4 GiB of free commit memory mark the run as on a busy machine, and
its timings as unreliable; a SLOW verdict then does not fail the run (parallel runs, --jobs, likewise). The exit code is 0 when every run completed and every check held, 1 otherwise, 2 for a usage
or setup error, 77 when the scenario's private save is missing (tools/test.py reports that as a skip).
"""
import argparse
import bisect
import concurrent.futures
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / 'tests' / 'scenarios'
SAVES = ROOT / 'build' / 'saves'
RUNS = ROOT / 'build' / 'scenarios'
WIN = os.name == 'nt'
GAME_EXES = ('dkb.exe', 'totk.exe') if WIN else ('dkb', 'totk')
sys.path.insert(0, str(ROOT / 'tools'))
import perf_check  # noqa: E402
import save_warp  # noqa: E402

SKIP = 77
STAGGER_S = 20             # seconds between the starts of parallel runs
MAX_JOBS = 3               # games at once; more exhausts an 8 GiB GPU (tools/run_game.py found the same)
BUSY_PRIVATE_MIB = 1024      # another program with this much private memory makes the machine busy
MIN_FREE_COMMIT_GIB = 4.0    # less free commit memory than this makes it busy


# ---------------------------------------------------------------- scenario files

def load(name):
    path = SCENARIOS / f'{name}.toml'
    if not path.is_file():
        known = ', '.join(sorted(p.stem for p in SCENARIOS.glob('*.toml'))) or 'none'
        raise SystemExit(f'scenario: no {_rel(path)} (known: {known})')
    s = tomllib.loads(path.read_text(encoding='utf-8'))
    for key in ('description', 'save', 'input', 'seconds'):
        if key not in s:
            raise SystemExit(f'scenario: {path.name} lacks "{key}"')
    s['name'] = name
    s.setdefault('env', {})
    s.setdefault('checks', {})
    s.setdefault('variants', {})
    return s


def save_dir(s):
    return SAVES / s['save']


def has_save(s):
    d = save_dir(s)
    return (d / 'save').is_dir() and (d / 'account.uid').is_file()


# ---------------------------------------------------------------- machine state

def machine_state(exe_names=GAME_EXES):
    """Other game processes, programs with over BUSY_PRIVATE_MIB of private memory, and free commit memory (Windows)."""
    if not WIN:
        return dict(games=[], heavy=[], free_commit_gib=None, checked=False)
    ps = ('$o = Get-CimInstance Win32_OperatingSystem; '
          '$p = Get-Process | Where-Object { $_.PrivateMemorySize64 -gt %d } | '
          'ForEach-Object { @{name = $_.ProcessName; id = $_.Id; mib = [int]($_.PrivateMemorySize64 / 1MB)} }; '
          '@{free_commit_kib = $o.FreeVirtualMemory; heavy = @($p)} | ConvertTo-Json -Depth 3 -Compress'
          % (BUSY_PRIVATE_MIB << 20))
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return dict(games=[], heavy=[], free_commit_gib=None, checked=False)
    heavy = [h for h in data.get('heavy') or [] if h]
    games = [h for h in heavy if f'{h["name"].lower()}.exe' in exe_names]
    try:   # a game process with little memory still holds the GPU
        tl = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True, timeout=30).stdout
        for line in tl.splitlines():
            parts = line.strip('"').split('","')
            if parts and parts[0].lower() in exe_names and not any(g['id'] == int(parts[1]) for g in games):
                games.append(dict(name=parts[0], id=int(parts[1]), mib=None))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    others = [h for h in heavy if f'{h["name"].lower()}.exe' not in exe_names]
    free = data.get('free_commit_kib')
    return dict(games=games, heavy=others, free_commit_gib=free / 1048576 if free else None, checked=True)


def busy_reasons(m):
    why = [f'{h["name"]} ({h["mib"]} MiB)' for h in m['heavy']]
    if m['free_commit_gib'] is not None and m['free_commit_gib'] < MIN_FREE_COMMIT_GIB:
        why.append(f'only {m["free_commit_gib"]:.1f} GiB of commit memory free')
    return why


# ---------------------------------------------------------------- one run

FATAL = re.compile(r'^\[?\w*\]?\s*FATAL: (.*)$', re.M)
NTSTATUS = {0xC0000005: 'access violation', 0xC00000FD: 'stack overflow', 0xC0000409: 'fast fail (stack buffer overrun '
            'or __fastfail)', 0xC000001D: 'illegal instruction', 0xC0000094: 'integer divide by zero',
            0xC0000374: 'heap corruption', 0xC0000017: 'out of memory', 0x40010004: 'killed (debugger or Task Manager)'}


EXIT_GRACE_S = 120   # a run that saved its last frame but has not exited this long after is killed and counted done
STALL_S = 120   # the game logs perf lines every second while it runs; this long without a log write is a hang


def hang_stacks(pid, out):
    """Every thread's stack of the hung process into out (gdb attached in batch mode; host addresses, no symbols:
    tools/hang_stacks.py names them from totk.map). Best effort: no gdb, or gdb failing, leaves a note."""
    gdb = shutil.which('gdb')
    if not gdb:
        out.write_text('no gdb on PATH: no stacks\n', encoding='utf-8')
        return
    # Two snapshots 5 s apart: threads whose stacks stay put are blocked, ones that move are running (a livelock).
    text = ''
    for i in range(2):
        if i:
            time.sleep(5)
        try:
            r = subprocess.run([gdb, '-batch', '-nx', '-p', str(pid), '-ex', 'thread apply all bt 40'],
                               capture_output=True, text=True, errors='replace', timeout=300, stdin=subprocess.DEVNULL)
            text += f'==== snapshot {i + 1}\n' + r.stdout + r.stderr
        except (OSError, subprocess.TimeoutExpired) as e:
            text += f'==== snapshot {i + 1}\ngdb failed: {e}\n'
    out.write_text(text, encoding='utf-8')


GPUAV_SETTINGS = ROOT / 'tools' / 'vk_gpuav_safe.txt'
VK_REPORT = re.compile(r'^\[host\] vk validation (error|warning): (.*)$', re.M)
VK_SHADER = re.compile(r'Shader Module \(0x[0-9a-f]+\) \(internal ID (\d+)\)')
VK_STAGE = re.compile(r'Stage = (\w+)')


def safe_gpu_env():
    """The environment for --safe-gpu: the Khronos validation layer from MSYS2 with tools/vk_gpuav_safe.txt."""
    layer = Path(os.environ.get('MSYS2_ROOT', 'C:/msys64')) / 'ucrt64' / 'bin'
    if not (layer / 'VkLayer_khronos_validation.json').is_file():
        raise SystemExit(f'scenario: --safe-gpu needs the Vulkan validation layer in {layer} '
                         '(pacman -S mingw-w64-ucrt-x86_64-vulkan-validation-layers)')
    return ['TOTK_VK_VALIDATION=1', f'VK_LAYER_PATH={layer}', f'VK_LAYER_SETTINGS_PATH={GPUAV_SETTINGS}']


def gpuav_reports(text):
    """Sort the validation layer's reports in a --safe-gpu log. The shader sanitizer's undefined results (a NaN into
    min/max and the like) are counted per shader and stage but do not fail a run: the game has some on real hardware
    too. Every other error is GPU-AV catching an access that safe mode then skipped, which fails the run."""
    nan, errors, first = {}, 0, None
    clock = [(w.start(), float(w.group(1))) for w in re.finditer(r'\bworld ([0-9.]+) s\b', text)]
    for m in VK_REPORT.finditer(text):
        kind, msg = m.groups()
        if kind != 'error' or 'duplicate_message_limit' in msg:
            continue
        follow = text[m.end():m.end() + 600].split('\n[', 1)[0]   # the report's own lines, up to the next log line
        if 'has an undefined result' in msg:
            sh, st = VK_SHADER.search(follow), VK_STAGE.search(follow)
            key = f'{st.group(1) if st else "?"} shader {sh.group(1) if sh else "?"}'
            if key not in nan:   # the game time of the last perf line before the first report: when it started
                i = bisect.bisect(clock, (m.start(), float('inf')))
                nan[key] = dict(count=0, from_world_s=clock[i - 1][1] if i else None)
            nan[key]['count'] += 1
        else:
            errors += 1
            first = first or msg.strip()[:200]
    return dict(errors=errors, first_error=first, undefined_results=nan)


def ending(code, text, timed_out):
    """(kind, detail) for how a run ended, from its exit code and output."""
    if code is None and not timed_out:
        return 'did not start', text.strip()[-300:]
    # A lost device is checked before FATAL: the runtime stops on it with "FATAL: vk: upload submit failed (-4)"
    # (-4 is VK_ERROR_DEVICE_LOST), which is a GPU fault, not a crash of the game's code.
    lost = re.search(r'.*(device lost|VK_ERROR_DEVICE_LOST|failed \(-4\)).*', text)
    m = FATAL.search(text)
    if timed_out is True:
        if 'scenario: hung (log silent)' in text:
            return 'hung', f'no frame or log line for {STALL_S} s; stacks in hang-stacks.txt; killed'
        return 'hung', 'no exit within the time limit; killed'
    if lost:
        fault = re.search(r'.*device fault address \d+: .*', text)
        return 'device lost', (fault or lost).group(0).strip()
    if m:
        return 'crashed', m.group(1).strip()
    if timed_out == 'exit':
        return 'completed', f'finished the run, then did not exit within {EXIT_GRACE_S} s; killed'
    if code == 0:
        if 'frametime: exit' not in text and 'perf: frame' not in text:
            return 'completed', 'exited 0 without perf output (TOTK_TRACE=perf missing?)'
        return 'completed', ''
    u = code & 0xFFFFFFFF
    name = NTSTATUS.get(u)
    return 'crashed, no report', f'exit {u:#x}' + (f' ({name})' if name else '') + '; no FATAL line in the log'


def vram(text):
    """(peak MiB used, its budget) of device-local memory over a run's log: the perf lines' "vram U of B MiB" and the
    "heap 0 usage U MiB of budget B MiB" a device loss logs. (None, None) without either."""
    pairs = [(int(u), int(b)) for u, b in re.findall(r'vram (\d+) of (\d+) MiB', text)]
    pairs += [(int(u), int(b)) for u, b in re.findall(r'heap 0 usage (\d+) MiB of budget (\d+) MiB', text)]
    return max(pairs) if pairs else (None, None)


def crash_excerpt(text):
    """The crash reporter's block: the FATAL line and the indented lines after it."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'FATAL:' in line:
            block = [line]
            for nxt in lines[i + 1:i + 80]:
                if not nxt.startswith(' '):
                    break
                block.append(nxt)
            return '\n'.join(block) + '\n'
    return ''


def frame_stats(png, hud_band, threshold=8, bright=96):
    """(near-black share below the top hud_band, bright share inside it) of a frame; (None, None) without Pillow or
    the image. In gameplay the HUD (hearts, counters) puts a few percent of bright pixels in the band, also when the
    terrain is black; a loading screen or a fade has none (measured: 0.00% against 2.4-3.6% in the tutorial cave)."""
    try:
        from PIL import Image
    except ImportError:
        return None, None
    if not png.is_file():
        return None, None
    with Image.open(png) as im:
        g = im.convert('L')
        w, h = g.size
        below = g.crop((0, int(h * hud_band), w, h)).histogram()
        top = g.crop((0, 0, w, max(1, int(h * hud_band)))).histogram()
    nb, nt = sum(below), sum(top)
    return (sum(below[:threshold]) / nb if nb else None, sum(top[bright:]) / nt if nt else None)


def black_share(png, hud_band, threshold=8):
    """Share of pixels with luma below threshold in the frame below its top hud_band (None without Pillow or image)."""
    return frame_stats(png, hud_band, threshold)[0]


def reference_check(png, spec):
    """Compare a normalized frame region to a private reference; lower RGB error is closer."""
    from PIL import Image, ImageChops, ImageStat

    path = (ROOT / spec['reference']).resolve()
    if not path.is_relative_to((ROOT / 'build').resolve()):
        raise ValueError('reference image must be under build/')
    box = spec['region']
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError('reference region must be [left, top, right, bottom] in 0..1')
    limit = spec['max_mean_error']
    if not 0 <= limit <= 255:
        raise ValueError('max_mean_error must be in 0..255')

    def region(image):
        with Image.open(image) as im:
            w, h = im.size
            rect = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
            if rect[0] >= rect[2] or rect[1] >= rect[3]:
                raise ValueError('reference region is empty')
            return im.convert('RGB').crop(rect).resize((180, 100))

    difference = ImageChops.difference(region(path), region(png))
    score = sum(ImageStat.Stat(difference).mean) / 3
    return {'mean_error': round(score, 3), 'max_mean_error': limit, 'passed': score <= limit}


def run_env(s, env_extra, out, base):
    """The environment of one run: base, the scenario's [env], then env_extra (--env and the variant). {run} in a value
    is the run's folder, so repeated runs do not overwrite each other's outputs (TOTK_DUMP_DIR={run}/dump). perf is
    always traced, because the perf verdict needs it; other TOTK_TRACE categories asked for are added to it. Runs are
    muted (TOTK_VOLUME=0) unless the caller's environment, the scenario or --env sets a volume."""
    env = dict(base)
    env.setdefault('TOTK_VOLUME', '0')
    env['TOTK_INPUT_SCRIPT'] = str(ROOT / s['input'])
    env.update({k: str(v) for k, v in s['env'].items()})
    env.update({k: str(v) for k, v in env_extra.items()})
    for k in set(s['env']) | set(env_extra):
        env[k] = env[k].replace('{run}', str(out))
    env['TOTK_TRACE'] = ','.join(sorted(set(filter(None, env.get('TOTK_TRACE', '').split(',') + ['perf']))))
    return env


def perf_fails(checks, verdict):
    """Speed is judged only when the scenario asks: perf = "SLOW" fails a SLOW run, "HITCHY" a SLOW or HITCHY one.
    Without it the verdict is reported and nothing fails on it."""
    want = checks.get('perf')
    return want == 'SLOW' and verdict == 'SLOW' or want == 'HITCHY' and verdict in ('SLOW', 'HITCHY')


def dump_env(checks):
    """The frame dump a scenario's checks ask for: dump_frame = N dumps that present into the run's dump/ folder,
    --env overrides it."""
    if 'dump_frame' not in checks:
        return {}
    env = {'TOTK_DUMP': str(checks['dump_frame']), 'TOTK_DUMP_DIR': '{run}/dump'}
    return env


def dump_nan(folder):
    """NaN pixels over every image a frame dump copied (the "nan N" fields of its images.txt files); None without one."""
    files = sorted(Path(folder).rglob('images.txt')) if Path(folder).is_dir() else []
    if not files:
        return None
    total = 0
    for f in files:
        total += sum(int(n) for n in re.findall(r'\bnan (\d+)', f.read_text(encoding='utf-8', errors='replace')))
    return total


STALL_LINE = re.compile(rb'stall: no frame for ([\d.]+) s')


class LogTail:
    """Reads only what the game appended to its log since the last call. shot: the end-of-run screenshot was logged;
    progress: when a line other than the stall watchdog's last arrived; stall: (seconds without a frame, when read) of
    the latest "stall: no frame" line, until a perf line shows frames again."""

    def __init__(self, path):
        self.path, self.pos, self.rest = path, 0, b''
        self.shot, self.progress, self.stall = False, None, None

    def read(self, now):
        try:
            with open(self.path, 'rb') as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return
        self.pos += len(data)
        lines = (self.rest + data).split(b'\n')
        self.rest = lines.pop()
        for line in lines:
            if b'boot: shot ' in line:
                self.shot = True
            m = STALL_LINE.search(line)
            if m:
                self.stall = (float(m[1]), now)
                continue
            if b'stall:' in line:   # the rest of the watchdog's dump
                continue
            self.progress = now
            if b'perf: frame' in line:
                self.stall = None

    def stalled_for(self, now):
        return 0.0 if self.stall is None else self.stall[0] + now - self.stall[1]


def warp_failed(s, out, prof, warp, env_extra, exe, args, e):
    """The result of a run whose profile could not be warped: no game started, the batch goes on."""
    shutil.rmtree(prof, ignore_errors=True)
    detail = f'{type(e).__name__}: {e}'
    r = dict(scenario=s['name'], variant=out.name, warp=warp, env=env_extra, exe=str(exe),
             seconds=args.seconds or s['seconds'], wall_s=0.0, exit_code=None, ending='warp failed', ending_detail=detail,
             perf='no data', fps=None, slow_windows=None, worst_frame_ms=None, major_episode_causes=[],
             failed=[f'warp failed: {detail}'], ok=False, perf_judged=False, busy_machine=[], timings_reliable=False,
             vram_peak_mib=None, vram_budget_mib=None, vram_over_budget=False, gpuav=None, black_share=None,
             hud_bright=None, in_scene=None, nan_pixels=None, reference_frame=None, machine={})
    (out / 'result.json').write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8')
    return r


def run_once(s, exe, out, env_extra, args, warp=None):
    out.mkdir(parents=True)
    prof = out / 'profile'
    shutil.copytree(save_dir(s), prof)
    warp = warp or args.warp or s.get('warp')
    if warp:   # this run's profile copy starts in another stage (tools/save_warp.py); the private save is unchanged
        try:
            save_warp.warp_profile(prof, warp)
        except (save_warp.SaveError, UnicodeEncodeError, OSError) as e:
            return warp_failed(s, out, prof, warp, env_extra, exe, args, e)
    env = run_env(s, {**dump_env(s['checks']), **env_extra}, out, os.environ)
    seconds = args.seconds or s['seconds']
    cmd = [str(exe), '--profile', str(prof), '--seconds', str(seconds), '--shot', str(out / 'last.png')]
    limit = seconds * 6 + 180
    if args.safe_gpu:   # every shader is instrumented: the game runs at a fifth of its speed or less
        limit = seconds * 30 + 600
    m = machine_state()
    start = time.monotonic()
    code, timed_out, text = None, False, ''
    with open(out / 'game.log', 'w', encoding='utf-8', errors='replace') as log:
        log.write('$ ' + ' '.join(f'{k}={v}' for k, v in sorted(env_extra.items())) + ' ' + subprocess.list2cmdline(cmd)
                  + '\n')
        log.flush()
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            finished = None   # when the game logged its end-of-run screenshot (runtime/boot.c)
            tail = LogTail(out / 'game.log')
            last_write = time.monotonic()
            while code is None:
                try:
                    code = p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    tail.read(now)
                    if finished is None and tail.shot:
                        finished = now
                    if tail.progress is not None:
                        last_write = max(last_write, tail.progress)
                    stall_limit = STALL_S * (5 if args.safe_gpu else 1)
                    # The stall watchdog (TOTK_STALL_DUMP, runtime/kernel.c stall_main) keeps logging while no frame
                    # comes, so its lines are not progress: a hang is either a log silent apart from them, or a
                    # "stall: no frame" that lasted the limit.
                    stalled = finished is None and (now - last_write > stall_limit or
                                                    tail.stalled_for(now) > stall_limit)
                    if now - start > limit or stalled or finished is not None and now - finished > EXIT_GRACE_S:
                        # Under the validation layer the process can hang in its teardown after the run is over.
                        timed_out = 'exit' if finished is not None else True
                        if timed_out is True:
                            log.write(f'\nscenario: hung ({"log silent" if stalled else "time limit"}); '
                                      'stacks in hang-stacks.txt\n')
                            log.flush()
                            hang_stacks(p.pid, out / 'hang-stacks.txt')
                        p.kill()
                        p.wait()
                        break
        except OSError as e:
            log.write(f'\nscenario: could not start: {e}\n')
    wall = time.monotonic() - start
    text = (out / 'game.log').read_text(encoding='utf-8', errors='replace')
    kind, detail = ending(code, text, timed_out)
    vram_used, vram_budget = vram(text)
    if kind == 'crashed':
        (out / 'crash.txt').write_text(crash_excerpt(text), encoding='utf-8')
    checks = s['checks']
    perf = perf_check.analyse(out / 'game.log', checks.get('min_fps', 50), checks.get('hitch_ms', 100),
                              checks.get('skip_world', 5))
    (out / 'perf.json').write_text(json.dumps(perf, indent=2) + '\n', encoding='utf-8')
    black, hud = frame_stats(out / 'last.png', checks.get('hud_band', 0.15))
    in_scene = None if hud is None else hud >= checks.get('min_hud_bright', 0.005)
    failed = []
    if kind != 'completed':
        failed.append(kind)
    busy = busy_reasons(m)
    # Speed counts only when the timings do: on a busy machine a SLOW verdict is reported, not failed.
    perf_judged = not busy and not args.safe_gpu
    gpuav = gpuav_reports(text) if args.safe_gpu else None
    if gpuav and gpuav['errors']:   # safe mode skipped an access that would have faulted or read garbage
        failed.append(f'GPU-AV: {gpuav["errors"]} error report(s), first: {gpuav["first_error"]}')
    if perf_judged and perf_fails(checks, perf['verdict']):
        failed.append(f'perf {perf["verdict"]}')
    if kind == 'completed' and perf['verdict'] == 'no data':
        failed.append('no perf data')
    if kind == 'completed' and in_scene is False:
        # Judging a loading screen's black would report black terrain that is not there: say what happened instead.
        failed.append(f'not in the scene at the end (no HUD in the last frame: a loading screen, menu or fade?); '
                      f'give it more time with --seconds')
    elif 'max_black' in checks and kind == 'completed':
        if black is None:
            failed.append('no last frame to check')
        elif black > checks['max_black']:
            failed.append(f'near-black {black * 100:.1f}% > {checks["max_black"] * 100:.0f}%')
    nan = dump_nan(out / 'dump') if 'dump_frame' in checks else None
    if 'max_nan' in checks and kind == 'completed':
        if nan is None:
            failed.append(f'no frame dump (the run ended before frame {checks.get("dump_frame")}, or the dump failed)')
        elif nan > checks['max_nan']:
            failed.append(f'{nan} NaN pixels in the dumped frame > {checks["max_nan"]}')
    reference = None
    if 'reference_frame' in checks and kind == 'completed':
        try:
            reference = reference_check(out / 'last.png', checks['reference_frame'])
            if not reference['passed']:
                failed.append(f'reference frame mismatch: mean RGB error {reference["mean_error"]:.2f} '
                              f'> {reference["max_mean_error"]:.2f}')
        except (ImportError, OSError, ValueError, KeyError, TypeError) as e:
            failed.append(f'reference frame check unavailable: {e}')
    if not args.keep_profile:
        shutil.rmtree(prof, ignore_errors=True)
    r = dict(perf_judged=perf_judged, vram_peak_mib=vram_used, vram_budget_mib=vram_budget,
             vram_over_budget=bool(vram_used and vram_budget and vram_used > vram_budget), gpuav=gpuav,
             scenario=s['name'], variant=out.name, warp=warp, env=env_extra, exe=str(exe), seconds=seconds, wall_s=round(wall, 1),
             exit_code=code, ending=kind, ending_detail=detail, perf=perf['verdict'], fps=perf['fps'],
             slow_windows=perf['slow_windows'], worst_frame_ms=perf['worst_ms'],
             major_episode_causes=sorted({e['cause'] for e in perf['episodes'] if e['major']}),
             black_share=black, hud_bright=hud, in_scene=in_scene, nan_pixels=nan, reference_frame=reference,
             failed=failed, ok=not failed,
             busy_machine=busy,
             timings_reliable=not busy, machine=m)
    (out / 'result.json').write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8')
    return r


# ---------------------------------------------------------------- command line

def _pct(v):
    return '-' if v is None else f'{v * 100:.1f}%'


def _rel(p):
    """p relative to the repository when it is inside it, else as it is."""
    p = Path(p)
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenario', nargs='?')
    ap.add_argument('--list', action='store_true', help='list the scenarios and whether their save is present')
    ap.add_argument('--build', type=Path, default=ROOT / 'build' / 'dkb100', help='build folder with dkb.exe')
    ap.add_argument('--repeat', type=int, default=1, help='runs per variant (default 1)')
    ap.add_argument('--jobs', '-j', type=int, default=1,
                    help=f'games at once (default 1, at most {MAX_JOBS}); for crash, NaN and picture checks: the runs are marked '
                         'as having unreliable timings. Big stages need about 3 GiB of video memory each: on an 8 GiB card a '
                         'device loss with 3 at once may be the memory; confirm it with --jobs 1')
    ap.add_argument('--ab', metavar='VARIANTS', help='the scenario\'s [variants.<name>] table to alternate')
    ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE', help='extra environment for every run')
    ap.add_argument('--seconds', type=float, help="override the scenario's game time")
    ap.add_argument('--keep-profile', action='store_true', help='keep the run\'s profile copy (for a save to inspect)')
    ap.add_argument('--force', action='store_true', help='run even while another game process is running')
    ap.add_argument('--sweep', metavar='REGEX', help='run once per stage whose name matches, e.g. LevelStage$ for the '
                    '20 layers (combines with --ab and --repeat)')
    ap.add_argument('--warp', metavar='STAGE', help="start each run in this stage instead of the save's "
                    '(tools/save_warp.py; `python tools/save_warp.py --stages` lists them)')
    ap.add_argument('--safe-gpu', action='store_true',
                    help='run under GPU-assisted validation in safe mode (tools/vk_gpuav_safe.txt): an access that '
                    'would fault the GPU is skipped and reported in game.log, so the driver is never reset. About '
                    'ten times slower; speed is not judged')
    a = ap.parse_args(argv)

    if a.list:
        for p in sorted(SCENARIOS.glob('*.toml')):
            s = load(p.stem)
            state = 'save present' if has_save(s) else f'missing save build/saves/{s["save"]}/'
            print(f'{p.stem:22} {state:40} {s["description"]}')
            for v, envs in s['variants'].items():
                print(f'{"":22}   variants {v}: ' + ', '.join(f'{k}: {e}' for k, e in envs.items()))
        return 0
    if not a.scenario:
        ap.error('name a scenario (see --list)')
    s = load(a.scenario)
    if not has_save(s):
        print(f'scenario {s["name"]}: skipped, no private save at {_rel(save_dir(s))}/ (account.uid and '
              f'save/ of a profile; tests/scenarios/README.md says how to make one)')
        return SKIP
    exe = next((a.build / n for n in GAME_EXES if (a.build / n).is_file()), None)
    if exe is None:
        print(f'scenario: no dkb.exe in {a.build} (build it with tools/build_game.py)')
        return 2
    extra = {}
    if a.safe_gpu:
        a.env = safe_gpu_env() + a.env
    for item in a.env:
        k, sep, v = item.partition('=')
        if not sep:
            ap.error(f'--env needs KEY=VALUE, got {item!r}')
        extra[k] = v
    if not 1 <= a.jobs <= MAX_JOBS:
        ap.error(f'--jobs takes 1 to {MAX_JOBS}')
    faulting = {e.partition('=')[0]: e.partition('=')[2] for e in a.env}.get('TOTK_FIX_VS_FAULT') == '0'
    if a.jobs > 1 and faulting and not a.force:
        # TOTK_FIX_VS_FAULT=0 makes known stages lose the GPU; several of those at once crashed Windows (bugcheck 0x119).
        ap.error('--jobs > 1 with TOTK_FIX_VS_FAULT=0 runs GPU faults in parallel, which crashed Windows on an RTX 5060; '
                 'use --jobs 1 (or --force if you accept the risk)')
    if a.ab:
        table = s['variants'].get(a.ab)
        if not table:
            ap.error(f'scenario {s["name"]} has no variants "{a.ab}" (has: {", ".join(s["variants"]) or "none"})')
        variants = [(name, {**extra, **{k: str(v) for k, v in env.items()}}, None) for name, env in table.items()]
    else:
        variants = [('run', extra, None)]
    warp = a.warp or s.get('warp')
    known = save_warp.stages() if warp or a.sweep else None
    if warp and known is not None and warp not in known:
        close = [k for k in known if warp.lower() in k.lower()][:8]
        print(f'scenario: no stage "{warp}" in the RomFS' + (f' (did you mean {", ".join(close)}?)' if close else ''))
        return 2
    if a.sweep:   # one variant per matching stage, each with every --ab variant
        if known is None:
            print('scenario: --sweep needs the RomFS index (build/extracted/dkb100/romfs/index.json) for the stage list')
            return 2
        try:
            pick = [k for k in known if re.search(a.sweep, k)]
        except re.error as e:
            ap.error(f'--sweep: {e}')
        if not pick:
            print(f'scenario: no stage matches {a.sweep!r} (python tools/save_warp.py --stages lists them)')
            return 2
        variants = [(stage if name == 'run' else f'{stage}-{name}', env, stage)
                    for stage in pick for name, env, _ in variants]
    m = machine_state()
    if m['games'] and not a.force:
        print('scenario: another game process is running (' + ', '.join(f'{g["name"]} {g["id"]}' for g in m['games'])
              + '); close it or pass --force')
        return 2
    busy = busy_reasons(m)
    if busy:
        print('scenario: busy machine, timings will be marked unreliable: ' + '; '.join(busy))

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    base = RUNS / f'{stamp}-{s["name"]}'
    print(f'scenario {s["name"]}: {s["description"]}\n  {len(variants) * a.repeat} runs of {a.seconds or s["seconds"]} s '
          f'game time with {_rel(exe)}; results in '
          f'{_rel(base)}')
    width = max(12, max(len(v[0]) for v in variants) + 2)
    plan = [(name, env, stage, n) for n in range(1, a.repeat + 1) for name, env, stage in variants]
    lock = threading.Lock()

    def one(job, slot):
        name, env, stage, n = job
        if slot:   # stagger the start of parallel runs, so their loads and shader translation do not all collide
            time.sleep(min(slot, a.jobs - 1) * STAGGER_S)
        r = run_once(s, exe, base / f'{name}-{n}', env, a, stage)
        if a.jobs > 1:   # runs that share the GPU and the CPU: their frame rates say nothing about speed
            r['timings_reliable'] = False
            r['perf_judged'] = False
            r['busy_machine'] = r['busy_machine'] + [f'{a.jobs} runs in parallel']
            r['failed'] = [f for f in r['failed'] if not f.startswith('perf ')]
            if r['ending'] == 'device lost' and r['vram_over_budget']:
                # The games share one GPU's memory; a stage that needs about 3 GiB, three times, passes an 8 GiB card.
                r['ending_detail'] += (f' [video memory {r["vram_peak_mib"]} of {r["vram_budget_mib"]} MiB with '
                                       f'{a.jobs} games at once: may come from running in parallel; rerun with --jobs 1]')
            r['ok'] = not r['failed']
            (base / f'{name}-{n}' / 'result.json').write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8')
        causes = f' ({", ".join(r["major_episode_causes"])})' if r['major_episode_causes'] else ''
        fps = '-' if r['fps'] is None else f'{r["fps"]:.1f}'
        with lock:
            print(f'  {r["variant"]:{width}} {"PASS" if r["ok"] else "FAIL":4}  {r["ending"]:18} perf {r["perf"]:7}{causes:16}'
                  f' {fps:>5} fps  near-black {_pct(r["black_share"]):>6}  {r["wall_s"]:.0f} s'
                  + (f'  [{r["ending_detail"]}]' if r['ending_detail'] else '')
                  + ('' if r['timings_reliable'] else '  [busy machine]'), flush=True)
            if r['failed']:
                print(f'    failed: {", ".join(r["failed"])}', flush=True)
        return r

    if a.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures = [pool.submit(one, job, i) for i, job in enumerate(plan)]
            results = [f.result() for f in futures]   # in plan order, whatever order they finished in
    else:
        results = [one(job, 0) for job in plan]
    summary = dict(scenario=s['name'], build=str(a.build), runs=results,
                   ok=all(r['ok'] for r in results), timings_reliable=all(r['timings_reliable'] for r in results))
    (base / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if len(variants) > 1:
        print(f'\n  {"variant":{width - 2}} runs  pass  median fps  median near-black')
        for name, _, _ in variants:
            rs = [r for r in results if r['variant'].rsplit('-', 1)[0] == name]
            fps = sorted(r['fps'] for r in rs if r['fps'] is not None)
            blk = sorted(r['black_share'] for r in rs if r['black_share'] is not None)
            print(f'  {name:{width - 2}} {len(rs):4} {sum(r["ok"] for r in rs):5} '
                  f'{fps[len(fps) // 2] if fps else float("nan"):11.1f} {_pct(blk[len(blk) // 2] if blk else None):>18}')
    if not summary['timings_reliable']:
        print('  timings unreliable: other programs were using the machine (result.json "busy_machine")')
    print(f'{"PASS" if summary["ok"] else "FAIL"}: {sum(r["ok"] for r in results)} of {len(results)} runs passed')
    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
