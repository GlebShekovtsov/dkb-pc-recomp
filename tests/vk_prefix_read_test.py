"""CPU provider checks of actual prefix split/readback functions, without linking a Vulkan loader.

Usage: python tests/vk_prefix_read_test.py --cc clang --deps /path/to/runtime/deps
Providers check submission/recorder boundaries and data lifetime, not GPU or pool-mapping correctness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from nvn2_compute_c0_test import function

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--deps', type=Path, required=True, help='existing dependency root or include directory')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    include = args.deps / 'include' if (args.deps / 'include').is_dir() else args.deps
    out = ROOT / 'build' / 'tests' / 'vk-prefix-read'
    out.mkdir(parents=True, exist_ok=True)
    sync = (ROOT / 'runtime/vk_sync.c').read_text()
    execute = (ROOT / 'runtime/vk_exec.c').read_text()
    code = '\n'.join(function(sync, name) for name in (
        'static VkCommandBuffer split_impl(', 'VkCommandBuffer vk_sync_split(', 'VkCommandBuffer vk_sync_split_wait('))
    if 'VkResult vk_sync_queue_submit(' in sync:   # the parts' vkQueueSubmit2 goes through it
        code = function(sync, 'VkResult vk_sync_queue_submit(') + '\n' + code
    code += '\n' + function(execute, 'int vk_exec_read_buffer(')
    code += '\n' + function(execute, 'static void views_to_fence(')
    if 'static _Thread_local const op_hdr *walk_at;' in execute:   # the lookahead's view of the walk, cleared after it
        code += '\nstatic _Thread_local const op_hdr *walk_at __attribute__((unused));\nstatic const uint64_t *la_handles;\nstatic int la_n, la_i;'
    code += '\n' + function(execute, 'static int record_impl(')
    code += '\n' + function(execute, 'int vk_exec_record(')
    code += '\n' + function(execute, 'int vk_exec_record_work(')
    if 'static uint64_t submit_record(' in sync:
        code += '\n' + function(sync, 'static uint64_t submit_record(')
    code += '\n' + function(sync, 'uint64_t vk_sync_submit_record(')
    replay_start = sync.index('typedef struct { const uint64_t *handles; int n; } ReplayHandles;')
    code += '\n' + sync[replay_start:sync.index('} ReplayHandles;', replay_start) + len('} ReplayHandles;')]
    code += '\n' + function(sync, 'static int record_handles(')
    code += '\n' + function(sync, 'uint64_t vk_sync_submit(')
    view_ent = execute[execute.index('typedef struct {'):execute.index('} ViewEnt;') + len('} ViewEnt;')]
    view_list_start = execute.index('typedef struct { int n; VkImageView v[]; } ViewList;')
    types = view_ent + '\n' + execute[view_list_start:execute.index('} ViewList;', view_list_start) + len('} ViewList;')]
    fixture = out / 'fixture.c'
    fixture.write_text((ROOT / 'tests/vk_prefix_read.c').read_text().replace('/* PRODUCTION_FUNCTIONS */', code)
                       .replace('/* PRODUCTION_TYPES */', types))
    results = []
    for opt in ('O0', 'O2'):
        exe = out / (f'prefix-{opt}' + ('.exe' if os.name == 'nt' else ''))
        command = [args.cc, '-std=gnu11', '-' + opt, '-Wall', '-Wextra', '-Werror',
                   '-I', str(ROOT / 'runtime'), '-I', str(include), '-I', str(include / 'vma'),
                   str(fixture), '-o', str(exe)]
        compiled = subprocess.run(command, capture_output=True, text=True)
        (out / f'{opt}-compile.log').write_text(compiled.stdout + compiled.stderr)
        if compiled.returncode:
            print(compiled.stderr)
            return compiled.returncode
        for mode in ('', 'submit-fail', 'wait-fail'):
            run = subprocess.run([str(exe), *([mode] if mode else [])], capture_output=True, text=True, timeout=15)
            log = run.stdout + run.stderr
            (out / f'{opt}-{mode or "checks"}.log').write_text(log)
            expected = 70 if mode else 0
            failure = 'vk: wait for a prefix of value 99 failed' if mode == 'wait-fail' else 'vk: vkQueueSubmit2 of a part of value 99 failed'
            state = f'frees=0 waits={1 if mode == "wait-fail" else 0} begins=0 signals=0'
            if run.returncode != expected or (mode and (failure not in log or state not in log)):
                print(log)
                raise AssertionError(f'{opt} {mode}: exit {run.returncode}, expected {expected}, no in-flight free')
            print(log.strip())
            results.append({'opt': opt, 'mode': mode or 'checks', 'exit': run.returncode})
    (out / 'manifest.json').write_text(json.dumps({
        'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(), 'results': results,
        'limits': 'CPU provider checks only; no Vulkan loader, GPU execution, mapping implementation or concurrency coverage',
    }, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
