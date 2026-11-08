"""Standalone check of runtime/aoc_romfs.c on synthetic add-on content; no game data, no GPU."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', required=True)
    ap.add_argument('--opt', choices=('0', '2'), default='2')
    ap.add_argument('--out', type=Path, default=ROOT / 'build/tests/aoc-romfs')
    args = ap.parse_args()
    out = args.out.resolve()
    out.relative_to((ROOT / 'build').resolve())
    out.mkdir(parents=True, exist_ok=True)
    exe = out / 'aoc_romfs_test.exe'   # the reader looks for aoc-<index>/ next to the executable, so the test writes them here
    rt = ROOT / 'runtime'
    command = [args.cc, '-std=gnu11', '-O' + args.opt, '-Wall', '-Wextra', '-Werror', '-Wno-missing-field-initializers', '-Wno-sign-compare',
               '-Wno-unused-parameter', '-I', rt,
               ROOT / 'tests/aoc_romfs_test.c', rt / 'aoc_romfs.c', rt / 'platform/win32.c', rt / 'platform/common.c',
               '-o', exe, '-lwinmm', '-lbcrypt', '-lpdh', '-lpsapi', '-lshell32', '-lrstrtmgr']   # win32.c's imports
    compiled = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=120)
    if compiled.returncode:
        print(compiled.stdout + compiled.stderr)
        return compiled.returncode
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    print(result.stdout + result.stderr, end='')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
