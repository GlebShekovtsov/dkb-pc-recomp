"""Actual-source submitted-tick lifetime regression; no Vulkan loader or GPU.

--source-root or --revision selects older source for RED comparisons. Vulkan submission and command decoding are adapters;
the frontend, replay submission branch, recorder context, report registration and completion callback are production.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(text, declaration):
    start = text.index(declaration)
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument('--source-root', type=Path, default=ROOT)
    selection.add_argument('--revision', help='read production source and headers from this Git revision')
    ap.add_argument('--vulkan-include', type=Path, required=True)
    args = ap.parse_args()
    if not args.cc:
        ap.error('pass --cc')
    source = args.source_root.resolve()
    if args.revision:
        revision = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options',
                                           args.revision + '^{commit}'], cwd=ROOT, text=True).strip()
        source = ROOT / 'build/tests/nvn-submit-ticks' / revision[:12] / 'source'
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, '--', 'runtime'],
                                        cwd=ROOT, text=True).splitlines()
        needed = {'nvn_api_core.c', 'vk_backend.c', 'vk_exec.c', 'vk_exec_xfer.c'}
        for name in names:
            rel = Path(name)
            if rel.suffix != '.h' and rel.name not in needed:
                continue
            target = source / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT))
    texts = {name: (source / 'runtime' / name).read_text() for name in
             ('nvn_api_core.c', 'vk_backend.c', 'vk_exec.c', 'vk_exec_xfer.c')}
    core, backend, execute, xfer = (texts[x] for x in texts)
    api = 'uint64_t submitted_ticks' in function(backend, 'static uint64_t vk_be_submit(')
    globals_ = core[core.index('#define LOD_RING'):core.index('NVN_API(nvnQueueSubmitCommands)')]
    types = backend[backend.index('enum { RI_SUBMIT'):backend.index('#define RQ_CAP')]
    callback = xfer[xfer.index('static void write_counter_at('):xfer.index('// Reports queued by the replay')]
    branch = function(backend, 'if (it.kind == RI_SUBMIT)')
    # The handle entry now forwards into a shared recorder. Keep executing its actual lifecycle, not a stand-in.
    record = function(execute, 'int vk_exec_record(')
    if 'static int record_impl(' in execute:
        record = function(execute, 'static int record_impl(') + '\n' + record
    if 'static void views_to_fence(' in execute:   # the view list of a command buffer, shared with the part split
        record = function(execute, 'static void views_to_fence(') + '\n' + record
    if 'static _Thread_local const op_hdr *walk_at;' in execute:   # the lookahead's view of the walk, cleared after it
        record = 'static _Thread_local const op_hdr *walk_at __attribute__((unused));\nstatic const uint64_t *la_handles;\nstatic int la_n, la_i;\n' + record
    replacements = {'@API@': str(int(api)), '@GLOBALS@': globals_, '@REPLAY_TYPES@': types,
                    '@COUNTERS@': callback + function(xfer, 'static void report_counter('),
                    '@RECORD@': record,
                    '@BACKEND@': function(backend, 'static uint64_t submit_sync(') + '\n' +
                                 function(backend, 'static uint64_t vk_be_submit('),
                    '@FRONTEND@': function(core, 'NVN_API(nvnQueueSubmitCommands)'), '@SUBMIT_BRANCH@': branch}
    fixture = (ROOT / 'tests/nvn_submit_ticks.c').read_text()
    for key, value in replacements.items():
        fixture = fixture.replace(key, value)
    out = ROOT / 'build/tests/nvn-submit-ticks' / ('working' if source == ROOT else 'baseline')
    out.mkdir(parents=True, exist_ok=True)
    generated = out / 'fixture.c'
    generated.write_text(fixture)
    results = []
    for version in (0, 143):
        for opt in ('-O0', '-O2'):
            exe = out / f'ticks-{version}-{opt[1:]}.exe'
            cmd = [args.cc, '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                   '-DTEST_GAME_VERSION=' + str(version), '-I', str(source / 'runtime'),
                   '-I', str(args.vulkan_include), '-I', str(args.vulkan_include / 'vma'),
                   str(generated), *([] if os.name == 'nt' else ['-pthread']), '-o', str(exe)]
            subprocess.run(cmd, check=True, timeout=30)
            p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
            row = {'version': version, 'optimization': opt, 'exit': p.returncode, 'output': p.stdout + p.stderr,
                   'command': cmd}
            print(json.dumps(row), flush=True)
            results.append(row)
    (out / 'results.json').write_text(json.dumps({'source': str(source), 'by_value_api': api,
        'sources': {p: hashlib.sha256(t.encode()).hexdigest() for p, t in texts.items()},
        'extracted': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in replacements.items()},
        'fixture_sha256': hashlib.sha256(fixture.encode()).hexdigest(), 'results': results}, indent=2) + '\n')
    return int(any(x['exit'] for x in results))


if __name__ == '__main__':
    raise SystemExit(main())
