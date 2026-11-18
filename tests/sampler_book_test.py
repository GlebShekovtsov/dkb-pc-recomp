"""Builds and runs tests/sampler_book_test.c: the sampler book of runtime/sampler_book.h against a modelled driver.

Pure logic: no Vulkan, no game data. Also checks that vk_texture.c keys the book on the whole create info and no
longer ends the run when vkCreateSampler fails.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cc', default='clang')
    ap.add_argument('--opt', default='1')
    args = ap.parse_args()
    source = (ROOT / 'runtime/vk_texture.c').read_text(encoding='utf-8')
    for needle in ('sampler_admit(c, &key)', 'maxSamplerAllocationCount', 'vk_sync_fence_done()', 'sb_pin(&book',
                   'sb_admit(&book'):
        if needle not in source:
            raise SystemExit(f'vk_texture.c: missing {needle!r}')
    if 'rt_fatal("vk: vkCreateSampler failed (' in source:
        raise SystemExit('vk_texture.c: vkCreateSampler failure is fatal again')
    with tempfile.TemporaryDirectory(prefix='sampler-book-') as tmp:
        exe = Path(tmp) / 'sampler_book_test'
        subprocess.run([args.cc, f'-O{args.opt}', '-Wall', '-Werror', '-I', str(ROOT / 'runtime'),
                        str(ROOT / 'tests/sampler_book_test.c'), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
