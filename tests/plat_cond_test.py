"""Stress test of the Linux condition variables (runtime/platform/linux.c, plat_cond_*), with no game or GPU.

Usage: python tests/plat_cond_test.py --cc /path/to/clang

tests/plat_cond.c includes linux.c and drives signals, broadcasts and timed waits from several threads. The subprocess
timeout catches a wake-up that never comes (a thread left asleep); the fixture's own checks catch a waiter count left
over after the waits. The lost-wake-up race the waiter count must not open is a few nanoseconds wide, too narrow for a
test to hit reliably: the ordering argument in linux.c's comment covers it.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with the project Clang executable')
    out = ROOT / 'build' / 'tests' / 'plat-cond'
    out.mkdir(parents=True, exist_ok=True)
    exe = out / 'plat_cond'
    # Only plat_cond_* and the futex locks are used: the rest of linux.c is dropped at link time.
    subprocess.run([args.cc, '-std=gnu11', '-O2', '-Wall', '-fno-strict-aliasing', '-ffunction-sections',
                    '-fdata-sections', '-I', str(ROOT / 'runtime'), str(ROOT / 'tests' / 'plat_cond.c'),
                    '-Wl,--gc-sections', '-pthread', '-o', str(exe)], check=True)
    return subprocess.run([str(exe)], timeout=60).returncode


if __name__ == '__main__':
    raise SystemExit(main())
