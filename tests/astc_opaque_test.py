"""Builds and runs tests/astc_opaque_test.c: the opaque-block classification of runtime/astc_opaque.h.

Pure logic, no Vulkan and no game data. Also checks that the pieces that use it are in place: the BC1 shader, its
header's source hash, the encoder's BC1 mode, and that the choice, the count and the remaking of the image are all
defined in the GPL module vk_astc_bc1.c, which is what the calls in vk_texture.c reach.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cc', default='clang')
    ap.add_argument('--opt', default='1')
    args = ap.parse_args()
    shader = ROOT / 'runtime/shaders/bc1_encode.comp'
    header = (ROOT / 'runtime/bc1_encode_spv.h').read_text(encoding='utf-8')
    digest = hashlib.sha256(shader.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if f'// source sha256 {digest}' not in header:
        raise SystemExit('runtime/bc1_encode_spv.h does not match runtime/shaders/bc1_encode.comp')
    texture = (ROOT / 'runtime/vk_texture.c').read_text(encoding='utf-8')
    astc = (ROOT / 'runtime/vk_astc.c').read_text(encoding='utf-8')
    bc1 = (ROOT / 'runtime/vk_astc_bc1.c').read_text(encoding='utf-8')
    # What vk_texture.c holds for the ASTC formats: the calls, and the bookkeeping of its own after one.
    for needle in ('vk_astc_bc_kinds(agpu, e->f.astc_bc, &e->pub, e->alloc, e->bound_to, e->bc_settled)',
                   'vk_astc_bc_scan(&sc, tmp, raw)', 'vk_astc_bc_choice(&e->pub,', 'vk_astc_bc3(&e->pub,',
                   'astc_reformat_done(e, cb)', 'vk_texture_astc_bc3(&e->pub)'):
        if needle not in texture:
            raise SystemExit(f'vk_texture.c: missing {needle!r}')
    # The rule, the count, the choice, the image made again and the old one retired: defined in the GPL module,
    # not in vk_texture.c (a Hookshot method). They are there, and they are not in the MIT file as well.
    for needle in ('astc_opaque_scan(s, p, n, 1)', 'astc_bc_choice(was, s)',
                   'astc_bc_settled(was, choice, remade, s)', 'vk_astc_bc_remake(', 'vk_astc_bc1()',
                   'vk_exec_image_gone(img->image)', 'vk_defer_image(img->image, *io_alloc)',
                   'img->image = new_img, img->format = f'):
        if needle not in bc1:
            raise SystemExit(f'vk_astc_bc1.c: missing {needle!r}')
        if needle in texture:
            raise SystemExit(f'vk_texture.c: {needle!r} is defined in vk_astc_bc1.c, not here')
    if 'bc == 2 ? bc1_pipe : bc_pipe' not in astc:
        raise SystemExit('vk_astc.c: the BC1 pipeline is not bound for bc 2')
    with tempfile.TemporaryDirectory(prefix='astc-opaque-') as tmp:
        exe = Path(tmp) / 'astc_opaque_test'
        subprocess.run([args.cc, f'-O{args.opt}', '-Wall', '-Werror', '-I', str(ROOT / 'runtime'),
                        str(ROOT / 'tests/astc_opaque_test.c'), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
