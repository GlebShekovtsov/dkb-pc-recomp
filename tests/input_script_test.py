"""Scripted input parser and scheduler (runtime/input_script.c), without game files or a GPU.

Usage: python tests/input_script_test.py --cc <clang>
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', required=True)
    args = ap.parse_args()
    out = ROOT / 'build/tests/input-script'
    out.mkdir(parents=True, exist_ok=True)
    for opt in ('-O0', '-O2'):
        exe = out / f'input-script-{opt[1:]}.exe'
        subprocess.run([args.cc, '-std=gnu11', opt, '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                        str(ROOT / 'tests/input_script_test.c'), str(ROOT / 'runtime/input_script.c'),
                        '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
