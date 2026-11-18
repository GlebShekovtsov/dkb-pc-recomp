"""Run the project's tests from one place: tests/manifest.toml lists them, this finds the tools and runs them.

    python tools/test.py                    tiers 0-2: no game files needed (GPU tests skip without a Vulkan device)
    python tools/test.py --tier 0-3         also the checks that need your dump or your build of the game
    python tools/test.py --area nvn,geometry
    python tools/test.py nvn2-compute-c0    names or globs (fnmatch) of tests
    python tools/test.py --list             what would run, and why anything would be skipped
    python tools/test.py --set cc=C:/llvm/bin/clang.exe --set xlate=build/xlate/totk_xlate.exe

Tiers: 0 = Python only, 1 = host toolchain (Clang, spirv-val, the shader translator), 2 = a Vulkan device,
3 = your dump or your build of the game, 4 = scripted game runs. A test whose tools or device are missing is reported as
SKIP with the reason; it never passes silently. A test that exits 77 is skipped too, with its last
output line as the reason. A test with `known_failure` in the manifest is XFAIL while it fails and
XPASS (reported, not an error) once it passes, so the entry can be removed.

Tools are found in this order: --set NAME=PATH, the environment variable DKB_TEST_<NAME>, the [tools] table of
test.local.toml at the repository root (ignored by git), then the defaults below (MSYS2 UCRT64 on Windows, PATH).
Paths in game-related tools are local and never leave the machine; the report records them only in build/test/runner.

Every test runs from the repository root with its own output folder build/test/runner/<name>/ ({out} in the
manifest; removed before the run, the test creates it) and its full log in build/test/runner/<name>.log. The summary is written to build/test/runner/report.json, and
--junit FILE writes JUnit XML for CI. The exit code is 1 when a test fails, errors or times out, else 0.
"""
import argparse
import concurrent.futures
import ctypes
import fnmatch
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'tests' / 'manifest.toml'
LOCAL = ROOT / 'test.local.toml'
RUNS = ROOT / 'build' / 'test' / 'runner'
WIN = os.name == 'nt'
EXE = '.exe' if WIN else ''
MSYS = Path(os.environ.get('MSYS2_ROOT', 'C:/msys64')) / 'ucrt64'
TIER_NAMES = {0: 'python', 1: 'host', 2: 'gpu', 3: 'game', 4: 'scenario'}
SKIP_EXIT = 77   # a test that exits 77 is skipped, with its last output line as the reason (the automake convention)


# ---------------------------------------------------------------- tools

def _first(*paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _which(name):
    w = shutil.which(name)
    return Path(w) if w else None


def _msys(name):
    return MSYS / 'bin' / (name + EXE) if WIN else None


def _deps():
    """A root holding include/vulkan/vulkan.h and include/vma/vk_mem_alloc.h (MSYS2's ucrt64, or /usr)."""
    for root in ([MSYS] if WIN else []) + [Path('/usr'), ROOT / 'toolchain' / 'deps-mingw']:
        if (root / 'include' / 'vulkan' / 'vulkan.h').is_file() and (root / 'include' / 'vma' / 'vk_mem_alloc.h').is_file():
            return root
    return None


def _layer_dir():
    for d in ([MSYS / 'bin', MSYS / 'share' / 'vulkan' / 'explicit_layer.d'] if WIN else []) + \
             [Path('/usr/share/vulkan/explicit_layer.d')]:
        if (d / 'VkLayer_khronos_validation.json').is_file():
            return d
    sdk = os.environ.get('VULKAN_SDK')
    if sdk:
        return _first(Path(sdk) / 'Bin', Path(sdk) / 'share' / 'vulkan' / 'explicit_layer.d')
    return None


def _vulkan_library():
    if WIN:
        return _first(MSYS / 'lib' / 'libvulkan-1.dll.a',
                      os.environ.get('VULKAN_SDK') and Path(os.environ['VULKAN_SDK']) / 'Lib' / 'vulkan-1.lib')
    return _first('/usr/lib/x86_64-linux-gnu/libvulkan.so', '/usr/lib/libvulkan.so')


def _exefs():
    for d in (ROOT / 'build' / 'extracted' / 'dkb100-dump' / 'exefs', ROOT / 'build' / 'extracted' / 'dkb100' / 'exefs'):
        if (d / 'main').is_file():
            return d
    return None


# name -> (what it is, default finder). The manifest refers to them as {name}.
TOOLS = {
    'python': ('this Python (the .venv the runner runs in)', lambda: Path(sys.executable)),
    'python314': ('Python 3.14 environment for the installer GUI tests',
                  lambda: _first(ROOT / '.venv-py314' / ('Scripts/python.exe' if WIN else 'bin/python'))),
    'cc': ('C compiler (Clang)', lambda: _first(os.environ.get('CC') and _which(os.environ['CC']), _msys('clang'),
                                                _which('clang'))),
    'cxx': ('C++ compiler (Clang)', lambda: _first(os.environ.get('CXX') and _which(os.environ['CXX']), _msys('clang++'),
                                                   _which('clang++'))),
    'glslc': ('glslc (shaderc)', lambda: _first(_msys('glslc'), _which('glslc'))),
    'spirv_val': ('spirv-val (SPIRV-Tools)', lambda: _first(_msys('spirv-val'), _which('spirv-val'))),
    'deps': ('dependency root with include/vulkan and include/vma', _deps),
    'vulkan_include': ('include root with vulkan/ and vma/', lambda: _deps() and _deps() / 'include'),
    'vulkan_library': ('Vulkan import library', _vulkan_library),
    'layer_path': ('folder with VkLayer_khronos_validation.json', _layer_dir),
    'dotnet': ('dotnet with the .NET 10 SDK', lambda: _first(ROOT / 'toolchain' / 'dotnet' / ('dotnet' + EXE),
                                                            _which('dotnet'))),
    'xlate': ('shader translator (python tools/build_xlate.py)',
              lambda: _first(ROOT / 'build' / 'xlate' / ('totk_xlate' + EXE))),
    'xlate_defined_minmax': ('translator built with defined float min/max (python tools/build_xlate.py '
                             '--defined-float-minmax --out build/xlate-defined-minmax)',
                             lambda: _first(ROOT / 'build' / 'xlate-defined-minmax' / ('totk_xlate' + EXE))),
    'xlate_baseline':('an unmodified translator build to compare against (set it explicitly)', lambda: None),
    'ryu_src': ('the patched translator sources build_xlate.py prepared',
                lambda: _first(ROOT / 'build' / 'xlate' / 'ryujinx')),
    'exefs': ('your dump\'s ExeFS folder (main, sdk, main.npdm)', _exefs),
    'game_exe': ('your build of the game (tools/build_game.py)', lambda: _first(ROOT / 'build' / 'dkb100' / ('dkb' + EXE))),
}
# Placeholders that are not tools.
BUILTIN = {'root', 'out', 'gpu_name'}


def resolve_tools(overrides):
    local = {}
    if LOCAL.is_file():
        local = tomllib.loads(LOCAL.read_text(encoding='utf-8')).get('tools', {})
    found, origin = {}, {}
    for name, (_, finder) in TOOLS.items():
        for source, value in (('--set', overrides.get(name)), ('env', os.environ.get('DKB_TEST_' + name.upper())),
                              (LOCAL.name, local.get(name))):
            if value:
                p = Path(value)
                found[name] = p if p.is_absolute() else ROOT / p
                origin[name] = source
                break
        else:
            p = finder()
            if p:
                found[name], origin[name] = Path(p), 'default'
    unknown = set(overrides) - set(TOOLS)
    if unknown:
        sys.exit(f'test.py: unknown tool(s) {", ".join(sorted(unknown))}; known: {", ".join(TOOLS)}')
    # A tool set explicitly to a missing path is a configuration error, not a skip.
    for name, p in found.items():
        if origin[name] != 'default' and not p.exists():
            sys.exit(f'test.py: {name} = {p} ({origin[name]}) does not exist')
    return found, origin


# ---------------------------------------------------------------- Vulkan device

class _AppInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('pApplicationName', ctypes.c_char_p),
                ('applicationVersion', ctypes.c_uint32), ('pEngineName', ctypes.c_char_p),
                ('engineVersion', ctypes.c_uint32), ('apiVersion', ctypes.c_uint32)]


class _InstanceInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('flags', ctypes.c_uint32),
                ('pApplicationInfo', ctypes.POINTER(_AppInfo)), ('enabledLayerCount', ctypes.c_uint32),
                ('ppEnabledLayerNames', ctypes.c_void_p), ('enabledExtensionCount', ctypes.c_uint32),
                ('ppEnabledExtensionNames', ctypes.c_void_p)]


def _driver_version(vendor, v):
    if vendor == 0x10DE:   # NVIDIA packs 10.8.8.6 bits
        return f'{(v >> 22) & 0x3ff}.{(v >> 14) & 0xff:02d}'
    if WIN and vendor == 0x8086:   # Intel on Windows packs 18.14 bits
        return f'{v >> 14}.{v & 0x3fff}'
    return f'{v >> 22}.{(v >> 12) & 0x3ff}.{v & 0xfff}'


def probe_vulkan():
    """The Vulkan 1.3 devices the loader reports, as [{name, vendor, driver, api}], or (None, reason).
    Only creates and destroys an instance; no device, memory or queue."""
    try:
        lib = ctypes.WinDLL('vulkan-1.dll') if WIN else ctypes.CDLL('libvulkan.so.1')
    except OSError as e:
        return None, f'no Vulkan loader ({e.__class__.__name__})'
    app = _AppInfo(0, None, b'dkb-test', 0, None, 0, (1 << 22) | (3 << 12))   # VK_STRUCTURE_TYPE_APPLICATION_INFO
    info = _InstanceInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)    # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    inst = ctypes.c_void_p()
    if lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(inst)) != 0:
        return None, 'vkCreateInstance failed (no Vulkan 1.3 driver?)'
    try:
        n = ctypes.c_uint32(0)
        lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
        handles = (ctypes.c_void_p * n.value)()
        lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), handles)
        devices = []
        for h in handles[:n.value]:
            props = ctypes.create_string_buffer(4096)   # VkPhysicalDeviceProperties is under 1 KiB
            lib.vkGetPhysicalDeviceProperties(ctypes.c_void_p(h), props)
            api, drv, vendor, _, _ = (int.from_bytes(props.raw[i:i + 4], 'little') for i in range(0, 20, 4))
            name = props.raw[20:276].split(b'\0', 1)[0].decode('utf-8', 'replace')
            if api >= (1 << 22) | (3 << 12):
                devices.append(dict(name=name, vendor=f'0x{vendor:04x}', driver=_driver_version(vendor, drv),
                                    api=f'{api >> 22}.{(api >> 12) & 0x3ff}.{api & 0xfff}'))
        return (devices, None) if devices else (None, 'no Vulkan 1.3 device')
    finally:
        lib.vkDestroyInstance(inst, None)


# ---------------------------------------------------------------- manifest

PLACEHOLDER = re.compile(r'\{(\w+)\}')


def tier_set(text):
    out = set()
    for part in text.split(','):
        a, _, b = part.partition('-')
        out |= set(range(int(a), int(b or a) + 1))
    return out


def load_manifest():
    data = tomllib.loads(MANIFEST.read_text(encoding='utf-8'))
    tests, names = [], set()
    for t in data.get('test', []):
        for key in ('name', 'tier', 'area', 'cmd'):
            if key not in t:
                sys.exit(f'test.py: {MANIFEST.name}: an entry lacks "{key}": {t}')
        if t['name'] in names:
            sys.exit(f'test.py: {MANIFEST.name}: duplicate name {t["name"]}')
        names.add(t['name'])
        used = {m for part in t['cmd'] for m in PLACEHOLDER.findall(part)}
        bad = used - set(TOOLS) - BUILTIN
        if bad:
            sys.exit(f'test.py: {t["name"]}: unknown placeholder(s) {", ".join(sorted(bad))}')
        t.setdefault('needs', [])
        t['tools'] = sorted((used & set(TOOLS)) | (set(t['needs']) & set(TOOLS)))
        t['gpu'] = 'gpu' in t['needs'] or 'gpu_name' in used
        t.setdefault('timeout', 600)
        t.setdefault('exclusive', t['gpu'])
        tests.append(t)
    return tests


def skip_reason(t, tools, gpu_reason):
    if t.get('platform') and t['platform'] != ('windows' if WIN else sys.platform):
        return f'{t["platform"]} only'
    missing = [n for n in t['tools'] if n not in tools]
    if missing:
        return 'no ' + ', no '.join(f'{n} ({TOOLS[n][0]}; set DKB_TEST_{n.upper()})' for n in missing)
    if t['gpu'] and gpu_reason:
        return gpu_reason
    return None


# ---------------------------------------------------------------- running

UNITTEST_RAN = re.compile(r'^Ran (\d+) tests? in', re.M)
UNITTEST_SKIPPED = re.compile(r'^(?:OK|FAILED) \(.*?skipped=(\d+)', re.M)


def run_one(t, tools, gpu, verbose):
    # {out} does not exist when the test starts: several tests create it themselves and refuse an existing folder, so
    # their outputs never mix with an earlier run's.
    out = RUNS / t['name']
    if out.exists():
        shutil.rmtree(out)
    values = {n: str(p) for n, p in tools.items()}
    values.update(root=str(ROOT), out=str(out), gpu_name=gpu[0]['name'] if gpu else '')
    cmd = [PLACEHOLDER.sub(lambda m: values[m.group(1)], part) for part in t['cmd']]
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('TOTK_VOLUME', '0')   # games the tests start are silent unless the caller sets a volume
    # The tests that pick their own compiler read CC/CXX; MSYS2's bin goes first so its DLLs and tools are found.
    if 'cc' in tools:
        env['CC'] = str(tools['cc'])
        env['PATH'] = str(tools['cc'].parent) + os.pathsep + env.get('PATH', '')
    if 'cxx' in tools:
        env['CXX'] = str(tools['cxx'])
    if t.get('env'):
        env.update({k: PLACEHOLDER.sub(lambda m: values[m.group(1)], v) for k, v in t['env'].items()})
    log = RUNS / (t['name'] + '.log')
    start = time.monotonic()
    with open(log, 'w', encoding='utf-8', errors='replace') as f:
        f.write('$ ' + subprocess.list2cmdline(cmd) + '\n\n')
        f.flush()
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               timeout=t['timeout'])
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = None
        except OSError as e:
            f.write(f'\ntest.py: could not start: {e}\n')
            code = -1
    seconds = time.monotonic() - start
    text = log.read_text(encoding='utf-8', errors='replace')
    note = ''
    ran = UNITTEST_RAN.findall(text)
    if ran:
        cases = sum(map(int, ran))
        skipped = sum(map(int, UNITTEST_SKIPPED.findall(text)))
        note = f'{cases} cases' + (f', {skipped} skipped' if skipped else '')
    if code is None:
        status = 'TIMEOUT'
        note = f'over {t["timeout"]} s'
    elif code == SKIP_EXIT:
        # The test found it cannot run here (tools/scenario.py without its private save): its last line says why.
        status = 'SKIP'
        note = ([ln.strip() for ln in text.splitlines() if ln.strip()][-1:] or ['skipped'])[0][:200]
    elif code == 0:
        status = 'XPASS' if t.get('known_failure') else 'PASS'
    else:
        status = 'XFAIL' if t.get('known_failure') else 'FAIL'
        if status == 'XFAIL':
            note = t['known_failure']
        else:
            tail = [ln for ln in text.splitlines() if ln.strip()][-1:] or ['']
            note = (f'exit {code}: ' + tail[0].strip())[:160]
    if verbose and status in ('FAIL', 'TIMEOUT'):
        print(f'---- {t["name"]} ({log})\n' + '\n'.join(text.splitlines()[-40:]) + '\n----', flush=True)
    return dict(status=status, seconds=round(seconds, 1), note=note, log=str(log.relative_to(ROOT)),
                cmd=subprocess.list2cmdline(cmd))


def git_revision():
    try:
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return rev + ('-dirty' if dirty else '')
    except OSError:
        return 'unknown'


def write_junit(path, results, tests):
    by = {t['name']: t for t in tests}
    counts = {s: sum(r['status'] == s for r in results.values()) for s in ('FAIL', 'TIMEOUT', 'SKIP')}
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<testsuite name="dkb" tests="{len(results)}" failures="{counts["FAIL"]}" errors="{counts["TIMEOUT"]}" '
             f'skipped="{counts["SKIP"]}">']
    for name, r in results.items():
        t = by[name]
        lines.append(f'  <testcase classname={quoteattr("tier%d.%s" % (t["tier"], t["area"]))} name={quoteattr(name)} '
                     f'time="{r.get("seconds", 0)}">')
        if r['status'] == 'SKIP':
            lines.append(f'    <skipped message={quoteattr(r["note"])}/>')
        elif r['status'] in ('FAIL', 'TIMEOUT'):
            tag = 'failure' if r['status'] == 'FAIL' else 'error'
            body = ''
            if r.get('log'):
                body = '\n'.join((ROOT / r['log']).read_text(encoding='utf-8', errors='replace').splitlines()[-200:])
            lines.append(f'    <{tag} message={quoteattr(r["note"])}>{escape(body)}</{tag}>')
        lines.append('  </testcase>')
    lines.append('</testsuite>')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def stale_translator(exe):
    """Why the translator exe is older than its sources (tools/xlate, the vendored Ryujinx, build_xlate.py), or None.
    A test run against a stale translator tests the old shader code: after a merge or a patch, translator tests then
    fail (or pass) for the wrong reason."""
    if not exe or not Path(exe).is_file():
        return None
    built = Path(exe).stat().st_mtime
    newest, which = 0.0, None
    for rel in ('tools/xlate', 'tools/vendor/ryujinx', 'tools/build_xlate.py'):
        p = ROOT / rel
        for f in ([p] if p.is_file() else p.rglob('*') if p.is_dir() else []):
            if f.is_file() and '__pycache__' not in f.parts and f.stat().st_mtime > newest:
                newest, which = f.stat().st_mtime, f
    if which is not None and newest > built + 2:
        return (f'the translator {Path(exe)} is older than {which.relative_to(ROOT)}: its tests check old shader code. '
                f'Rebuild it (python tools/build_xlate.py) or point at a newer one (--set xlate=build/<name>/totk_xlate.exe)')
    return None


# ---------------------------------------------------------------- --changed: the tests a change can affect

# The source paths each area's tests cover (fnmatch on repository-relative paths with /). A test also covers the files
# its command names and the fixtures beside its script (tests/<stem>*), and the ones its manifest entry lists in
# `paths`. Keep these broad: a test run too often costs seconds, one skipped wrongly costs a regression.
AREA_PATHS = {
    'cpu': ['recomp/*', 'runtime/cpu.h', 'runtime/simd.h', 'runtime/excl.c', 'runtime/dispatch*', 'tests/diff/*',
            'tools/spin_waits.py'],
    'os': ['runtime/kernel*', 'runtime/ipc*', 'runtime/svc*', 'runtime/ov_*', 'runtime/fs*', 'runtime/romfs*',
           'runtime/cfg*', 'runtime/main.c', 'runtime/loader*', 'runtime/hostfn*', 'runtime/host.h', 'runtime/platform/*',
           'runtime/input*', 'runtime/boot*'],
    'audio': ['runtime/audio*', 'runtime/audren*', 'runtime/ov_codec*'],
    'nvn': ['runtime/nvn*', 'runtime/gfx*', 'runtime/vk_exec*'],
    'geometry': ['runtime/vk_exec_draw.c', 'runtime/gfx_shader.c', 'runtime/nvn_geometry*', 'runtime/nvn2_lower*',
                 'runtime/third_party/eglantine/*', 'tools/xlate/*', 'tools/vendor/ryujinx/*', 'tools/build_xlate.py'],
    'shaders': ['tools/xlate/*', 'tools/vendor/ryujinx/*', 'tools/build_xlate.py', 'runtime/gfx_shader.c',
                'runtime/nvn2_lower*', 'runtime/third_party/eglantine/*', 'tools/*shader*', 'tools/inspect_spirv*',
                'tools/fragment_output*'],
    'descriptors': ['runtime/vk_*', 'runtime/gfx*'],
    'gfx': ['runtime/vk_*', 'runtime/gfx*', 'runtime/tex*', 'runtime/astc*', 'runtime/stencil*'],
    'tools': ['tools/*.py', 'tools/input/*', 'tests/scenarios/*'],
    'installer': ['installer/*', 'tools/build_game.py', 'tools/make_release*.py'],
    'scenario': ['runtime/*', 'tools/xlate/*', 'tools/vendor/ryujinx/*', 'tools/scenario.py', 'tools/save_warp.py',
                 'tools/perf_check.py', 'tests/scenarios/*', 'tools/input/*'],
}
# Changes that no test checks: documents and CI files. Anything else that matches no area or test runs everything.
UNTESTED = ['*.md', 'docs/*', '.github/*', 'LICENSE*', '.gitignore', '.gitattributes', 'test.local.toml']
# A change to the runner itself runs everything.
RUNNER = ['tools/test.py', 'tests/manifest.toml']


def changed_files(rev):
    """Repository-relative paths changed since rev (HEAD by default): committed, staged, unstaged and untracked."""
    def git(*a):
        p = subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True)
        if p.returncode:
            sys.exit(f'test.py: git {" ".join(a)} failed: {p.stderr.strip()}')
        return [line.strip() for line in p.stdout.splitlines() if line.strip()]
    return sorted(set(git('diff', '--name-only', rev) + git('ls-files', '--others', '--exclude-standard')))


def covers(t):
    """The path patterns test t covers: its area's, its own files and fixtures, and its manifest `paths`."""
    pats = list(AREA_PATHS.get(t['area'], [])) + list(t.get('paths', []))
    for part in t['cmd']:
        if PLACEHOLDER.search(part) or not ('/' in part and part.endswith(('.py', '.c', '.cpp', '.h'))):
            continue
        pats.append(part)
        stem = Path(part).stem.removesuffix('_test').removesuffix('_check')
        if part.startswith('tests/'):
            pats.append(f'tests/{stem}*')
    return pats


def select_changed(tests, files):
    """(tests to run, reason) for a change touching files. All tests when the runner or an unmapped file changed."""
    hit = lambda f, pats: any(fnmatch.fnmatchcase(f, p) for p in pats)
    if not files:
        return [], 'nothing changed'
    if any(hit(f, RUNNER) for f in files):
        return list(tests), 'the test runner or the manifest changed: running everything'
    relevant = [f for f in files if not hit(f, UNTESTED)]
    if not relevant:
        return [], 'only documents or CI files changed'
    cov = {t['name']: covers(t) for t in tests}
    unmapped = [f for f in relevant if not any(hit(f, c) for c in cov.values())]
    if unmapped:
        shown = ', '.join(unmapped[:5]) + (f' and {len(unmapped) - 5} more' if len(unmapped) > 5 else '')
        return list(tests), f'no test is mapped to {shown}: running everything (map it in tools/test.py AREA_PATHS)'
    picked = [t for t in tests if any(hit(f, cov[t['name']]) for f in relevant)]
    return picked, f'{len(relevant)} changed file(s) select {len(picked)} test(s)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help='test names or fnmatch globs (default: all)')
    ap.add_argument('--tier', default='0-2', help='tiers to include, e.g. 0-1 or 0,2 (default 0-2)')
    ap.add_argument('--area', type=lambda s: [a.strip() for a in s.split(',') if a.strip()],
                    help='only these areas, comma-separated (cpu, os, audio, nvn, geometry, descriptors, shaders, gfx, '
                         'tools, installer)')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=PATH', help='tool path override')
    ap.add_argument('--jobs', '-j', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument('--changed', nargs='?', const='HEAD', metavar='REV',
                    help='only the tests the changes since REV (default HEAD, with uncommitted and untracked files) can '
                         'affect; see AREA_PATHS')
    ap.add_argument('--list', action='store_true', help='list the selected tests with their state, run nothing')
    ap.add_argument('--tools', action='store_true', help='show where each tool was found, run nothing')
    ap.add_argument('--junit', type=Path, help='also write JUnit XML here')
    ap.add_argument('--verbose', '-v', action='store_true', help='print the end of the log of each failing test')
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        name, sep, value = item.partition('=')
        if not sep:
            sys.exit(f'test.py: --set needs NAME=PATH, got {item!r}')
        overrides[name.strip()] = value.strip()
    tools, origin = resolve_tools(overrides)

    if args.tools:
        for name, (what, _) in TOOLS.items():
            where = f'{tools[name]}  ({origin[name]})' if name in tools else '-  not found'
            print(f'{name:15} {where}\n{"":15} {what}')
        return 0

    tests = load_manifest()
    tiers = tier_set(args.tier)
    selected = [t for t in tests if t['tier'] in tiers and (not args.area or t['area'] in args.area)
                and (not args.names or any(fnmatch.fnmatchcase(t['name'], n) for n in args.names))]
    if args.names:
        for n in args.names:
            if not any(fnmatch.fnmatchcase(t['name'], n) for t in tests):
                sys.exit(f'test.py: no test matches {n!r} (see --list)')
    if args.changed:
        selected, why = select_changed(selected, changed_files(args.changed))
        print(f'--changed {args.changed}: {why}')
        if not selected:
            return 0
    if not selected:
        sys.exit('test.py: no tests selected')

    gpu, gpu_reason = (None, None)
    if any(t['gpu'] for t in selected):
        gpu, gpu_reason = probe_vulkan()

    results, todo = {}, []
    for t in selected:
        why = skip_reason(t, tools, gpu_reason)
        if why:
            results[t['name']] = dict(status='SKIP', note=why)
        else:
            todo.append(t)

    if args.list:
        for t in selected:
            state = 'skip: ' + results[t['name']]['note'] if t['name'] in results else 'run'
            if t.get('known_failure') and t['name'] not in results:
                state = 'run (known failure: ' + t['known_failure'] + ')'
            print(f'{t["tier"]} {t["area"]:11} {t["name"]:36} {state}')
        return 0

    RUNS.mkdir(parents=True, exist_ok=True)
    gpu_desc = ', '.join(f'{d["name"]} (driver {d["driver"]})' for d in gpu) if gpu else (gpu_reason or 'not needed')
    print(f'DKB tests at {git_revision()}: {len(todo)} to run, {len(results)} skipped; tiers {args.tier}; '
          f'GPU: {gpu_desc}', flush=True)
    old = stale_translator(tools.get('xlate')) if any('xlate' in t['tools'] for t in todo) else None
    if old:
        print(f'warning: {old}', flush=True)

    lock = threading.Lock()
    width = max(len(t['name']) for t in selected)

    def report(t, r):
        with lock:
            results[t['name']] = r
            extra = f'  {r["note"]}' if r['note'] else ''
            print(f'{r["status"]:7} {t["name"]:{width}} {r["seconds"]:7.1f}s{extra}', flush=True)

    # Exclusive tests (GPU work, tests that use every core) run one at a time after the parallel ones.
    shared = [t for t in todo if not t['exclusive']]
    exclusive = [t for t in todo if t['exclusive']]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_one, t, tools, gpu, args.verbose): t for t in shared}
        for f in concurrent.futures.as_completed(futures):
            report(futures[f], f.result())
    for t in exclusive:
        report(t, run_one(t, tools, gpu, args.verbose))

    order = {t['name']: i for i, t in enumerate(selected)}
    results = dict(sorted(results.items(), key=lambda kv: order[kv[0]]))
    counts = {}
    for r in results.values():
        counts[r['status']] = counts.get(r['status'], 0) + 1
    skipped = [(n, r['note']) for n, r in results.items() if r['status'] == 'SKIP']
    if skipped:
        print('\nskipped:')
        for n, why in skipped:
            print(f'  {n:{width}}  {why}')
    bad = [n for n, r in results.items() if r['status'] in ('FAIL', 'TIMEOUT')]
    if bad:
        print('\nfailed (logs in build/test/runner/):')
        for n in bad:
            print(f'  {n:{width}}  {results[n]["note"]}')
    xpass = [n for n, r in results.items() if r['status'] == 'XPASS']
    if xpass:
        print('\nnow passing; remove known_failure from tests/manifest.toml: ' + ', '.join(xpass))
    summary = ', '.join(f'{v} {k.lower()}' for k, v in sorted(counts.items()))
    print(f'\n{summary} in {time.monotonic() - started:.0f} s')

    by = {t['name']: t for t in selected}
    report_data = dict(
        revision=git_revision(), tiers=args.tier, os=platform.platform(), python=platform.python_version(),
        gpu=gpu, gpu_reason=gpu_reason, tools={n: str(p) for n, p in tools.items()},
        tests=[dict(name=n, tier=by[n]['tier'], area=by[n]['area'], **r) for n, r in results.items()])
    (RUNS / 'report.json').write_text(json.dumps(report_data, indent=2) + '\n', encoding='utf-8')
    if args.junit:
        write_junit(args.junit, results, selected)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
