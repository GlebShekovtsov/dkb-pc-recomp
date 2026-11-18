"""Compile the Switch 2 controller HD rumble checks (runtime/switch2_proto.h) at O0/O2 and run them. No SDL, Bluetooth
or game files.

Usage: python tests/switch2_rumble_test.py --cc <clang>
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', required=True)
    args = ap.parse_args()
    out = ROOT / 'build/tests/switch2-rumble'
    out.mkdir(parents=True, exist_ok=True)
    for opt in ('-O0', '-O2'):
        exe = out / f'switch2-rumble-{opt[1:]}.exe'
        subprocess.run([args.cc, '-std=gnu11', opt, '-Wall', '-Werror', '-I', str(ROOT / 'runtime'),
                        str(ROOT / 'tests/switch2_rumble_test.c'), '-lm', '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
