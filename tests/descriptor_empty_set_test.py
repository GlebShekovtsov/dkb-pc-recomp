"""CPU regression for bind_resources descriptor packing and recorder copies.

The actual binder, set writer, recorder push producers and two push consumer
cases run against inert descriptor/packet providers. No Vulkan loader, driver,
guest memory, ring thread or shader translation is used. --baseline REV uses
that revision's unmodified bind_resources, including its set-boundary logic.
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


def source(draw):
    recorder = (ROOT / 'runtime/vk_rec.c').read_text()
    context_start = draw.index('static struct {', draw.index('// State of the command buffer being recorded'))
    context_end = draw.index('// The bound-slot bytes of VkExec', context_start)
    context = draw[context_start:context_end]
    declarations = '\n'.join(function(recorder, prefix) + suffix for prefix, suffix in (
        ('typedef struct { PFN_vkCmdPushDescriptorSetWithTemplateKHR fn;', ' A_PushTpl;'),
        ('typedef struct { PFN_vkCmdPushDescriptorSetKHR fn;', ' A_Push;')))
    producers = '\n'.join(function(recorder, prefix) for prefix in (
        'static void *tail_copy(', 'void rec_push_tpl(', 'void rec_push('))
    consumers = '\n'.join(line for line in recorder.splitlines()
                          if line.strip().startswith(('case R_PushTpl:', 'case R_Push:')))
    if len(consumers.splitlines()) != 2:
        raise RuntimeError('the recorder push consumer seam changed')
    production = '\n'.join(function(draw, prefix) for prefix in (
        'static int nvn_stage_of(', 'static void write_set(', 'static void bind_resources('))
    fixture = (ROOT / 'tests/descriptor_empty_set.c').read_text()
    return fixture.replace('@CONTEXT@', context).replace('@RECORDER@', declarations + '\n' + producers).replace(
        '@CONSUMERS@', consumers).replace('@BINDER@', production)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc'))
    parser.add_argument('--deps', type=Path, required=True, help='runtime dependency root with include/vulkan and include/vma')
    parser.add_argument('--baseline', metavar='REV', help='use an earlier unmodified production binder')
    parser.add_argument('--out', type=Path, default=ROOT / 'build/test/descriptor-empty-set',
                        help='output root; current/baseline results use separate subdirectories')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    if args.baseline:
        revision = subprocess.check_output(['git', 'rev-parse', '--verify', args.baseline + '^{commit}'], cwd=ROOT, text=True).strip()
        draw = subprocess.check_output(['git', 'show', revision + ':runtime/vk_exec_draw.c'], cwd=ROOT, text=True)
        tag = 'baseline-' + revision[:12]
    else:
        draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
        tag = 'current'
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'fixture.c'
    src.write_text(source(draw))
    results = []
    for opt in ('-O0', '-O2'):
        exe = out / ('fixture' + opt + ('.exe' if os.name == 'nt' else ''))
        command = [args.cc, '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror',
                   '-Wno-missing-field-initializers',  # unchanged production BS/RS initializers
                   '-Wno-unused-variable', '-Wno-unused-const-variable',  # context statics the binder does not read
                   '-I', str(ROOT / 'runtime'), '-I', str(args.deps / 'include'),
                   '-I', str(args.deps / 'include/vma'), str(src), '-o', str(exe)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (out / (opt + '-compile.log')).write_text(compiled.stdout + compiled.stderr)
        if compiled.returncode:
            print(compiled.stdout + compiled.stderr)
            return compiled.returncode
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        (out / (opt + '-run.log')).write_text(run.stdout + run.stderr)
        print(opt, run.stdout + run.stderr, end='')
        results.append({'optimization': opt, 'exit': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr})
    (out / 'results.json').write_text(json.dumps({'results': results,
        'binder_sha256': hashlib.sha256(function(draw, 'static void bind_resources(').encode()).hexdigest(),
        'fixture_sha256': hashlib.sha256(src.read_bytes()).hexdigest()}, indent=2) + '\n')
    return int(any(row['exit'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
