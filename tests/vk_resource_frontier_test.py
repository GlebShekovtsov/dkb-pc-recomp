"""CPU-only retirement regression through production submit, split, alias and shutdown code.

No Vulkan library is linked. The C fixture names every synthetic driver adapter explicitly.
Use --source-root with an unmodified checkout to reproduce the original failure.
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


def function(text, name):
    match = re.search(r'^(?:static )?(?:inline )?[^\n;{}]+\b' + re.escape(name) + r'\([^\n]*\)\s*\{', text, re.M)
    if not match:
        raise ValueError(f'function not found: {name}')
    depth = 0
    for at in range(text.index('{', match.start()), len(text)):
        depth += (text[at] == '{') - (text[at] == '}')
        if not depth:
            return text[match.start():at + 1]
    raise ValueError(f'unterminated function: {name}')


def between(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--cc', default=os.environ.get('CC'))
    parser.add_argument('--vulkan-include', type=Path, help='include root containing vulkan/ and vma/')
    parser.add_argument('--out', type=Path, help='generated CPU fixture and results directory')
    args = parser.parse_args()
    source = args.source_root.resolve()
    package = Path(os.environ['TOTK_TOOLCHAIN']) if os.environ.get('TOTK_TOOLCHAIN') else None
    cc = args.cc or shutil.which('clang')
    if not cc and package:
        cc = str(package / 'llvm/bin/clang.exe')
    inc = args.vulkan_include or (package / 'runtime/deps/include' if package else ROOT / 'runtime/deps/include')
    if not cc or not (inc / 'vulkan/vulkan.h').is_file() or not (inc / 'vma/vk_mem_alloc.h').is_file():
        parser.error('provide --cc and --vulkan-include, or set TOTK_TOOLCHAIN to the dependency package')
    work = (args.out or ROOT / 'build/test/vk-resource-frontier').resolve()
    work.mkdir(parents=True, exist_ok=True)
    texts = {name: (source / 'runtime' / name).read_text(encoding='utf-8')
             for name in ('vk_memory.c', 'vk_texture.c', 'vk_exec_draw.c', 'vk_sync.c')}
    memory, texture, draw, sync = (texts[name] for name in texts)
    frontier = 'vk_mem_protect_until' if 'void vk_mem_protect_until(' in memory else 'vk_mem_note_submit'
    memory_names = ('cmd_init', 'grow', 'free_staging',
                    *(('batch_status',) if 'static VkResult batch_status(' in memory else ()), 'retire',
                    'collect_locked', 'defer', 'vk_defer_image', 'vk_defer_view', frontier, 'vk_mem_collect',
                    'vk_mem_shutdown')
    texture_names = ('make_key', 'same_but_swizzle', 'rel_unlink', 'destroy_tex', 'entry_of', 'slot_resolve')
    sync_names = ('claim_slot', 'take_slot', 'vk_sync_split', 'vk_sync_submit')
    extracts = {}

    def functions(filename, names):
        result = []
        for name in names:
            body = function(texts[filename], name)
            extracts[f'{filename}:{name}'] = hashlib.sha256(body.encode()).hexdigest()
            result.append(body)
        return '\n\n'.join(result)

    if 'static VkCommandBuffer split_impl(' in sync:
        # Follow the real shared split/submit lifecycle while retaining older-source RED comparisons.
        sync_functions = functions('vk_sync.c', (*(('vk_sync_queue_submit',) if 'VkResult vk_sync_queue_submit(' in sync
                                                   else ()), 'claim_slot', 'take_slot', 'split_impl', 'vk_sync_split',
                                                *(('submit_record',) if 'static uint64_t submit_record(' in sync else ()),
                                                'vk_sync_submit_record'))
        sync_functions += '\n' + between(sync, 'typedef struct { const uint64_t *handles; int n; } ReplayHandles;',
                                          'static int record_handles(')
        sync_functions += '\n' + functions('vk_sync.c', ('record_handles', 'vk_sync_submit'))
    else:
        sync_functions = functions('vk_sync.c', sync_names)

    replacements = {
        '@TEXTURE_TYPES@': between(texture, 'typedef struct {\n    uint64_t addr;', 'typedef struct { samp_par p;'),
        '@MEMORY_TYPES@': between(memory, 'typedef struct { VkBuffer buf; VmaAllocation alloc; } Staging;',
                                 'static void cmd_init('),
        '@VIEW_TYPES@': between(draw, 'typedef struct {\n    VkImage image;', '_Static_assert(sizeof(ViewKey)') +
                       '\n' + re.search(r'typedef struct ViewNode[^\n]+', draw).group(),
        '@MEMORY_FUNCTIONS@': functions('vk_memory.c', memory_names),
        '@TEXTURE_FUNCTIONS@': functions('vk_texture.c', texture_names),
        '@VIEW_FUNCTION@': functions('vk_exec_draw.c', ('vk_exec_image_gone',)),
        '@SYNC_FUNCTIONS@': sync_functions,
        '@FRONTIER@': frontier,
    }
    fixture = (ROOT / 'tests/vk_resource_frontier.c').read_text(encoding='utf-8')
    if 'uint64_t submitted_ticks' not in function(sync, 'vk_sync_submit'):
        # Preserve retirement-baseline comparisons made before the explicit submitted-tick API existed.
        fixture = fixture.replace(', uint64_t submitted_ticks)', ')')
        fixture = fixture.replace('    check(submitted_ticks == 123, "actual submit forwards immutable entry tick to recorder");\n', '')
        fixture = fixture.replace('vk_sync_submit(&handle, scenario == EMPTY ? 0 : 1, 123)',
                                  'vk_sync_submit(&handle, scenario == EMPTY ? 0 : 1)')
    for marker, replacement in replacements.items():
        fixture = fixture.replace(marker, replacement)
    generated = work / 'fixture.c'
    generated.write_text(fixture, encoding='utf-8')
    results = []
    for opt in ('-O0', '-O2'):
        exe = work / f'frontier-{opt[1:]}.exe'
        command = [cc, '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                   '-Wno-unused-variable', '-I', str(source / 'runtime'), '-I', str(inc), '-I', str(inc / 'vma'),
                   str(generated), '-o', str(exe)]
        subprocess.run(command, check=True, timeout=30)
        for scenario in ('alias', 'split', 'empty', 'skipped', 'shutdown'):
            run = subprocess.run([str(exe), scenario], capture_output=True, text=True, timeout=15)
            row = {'optimization': opt, 'scenario': scenario, 'exit': run.returncode,
                   'stdout': run.stdout.strip(), 'stderr': run.stderr.strip()}
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {'source': str(source), 'functions_sha256': extracts, 'driver': 'synthetic; no Vulkan library linked',
              'results': results, 'pass': all(row['exit'] == 0 for row in results)}
    (work / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
