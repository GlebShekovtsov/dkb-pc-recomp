"""Add-on content in a .nup (tools/extract_nup.py aoc_packs, write_aoc, --aoc) on synthetic containers; no game data.

Run: python tests/nup_aoc_test.py
"""
import contextlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
import extract_nup as nup  # noqa: E402
from nup_extract_check import build  # noqa: E402

APP = 0x0100AAAA00BBB000
FIRST = APP + 0x1000
BASE = (0, APP, 1, {'exefs': {'main': b'x'}, 'romfs': {'a': b'a'}})


def dlc(index, data=b'dlc', version=1, tree=None):
    return (2, FIRST + index, version, {'romfs': tree if tree is not None else {'d': {'f.bin': data}}})


class AocTest(unittest.TestCase):
    def open(self, packages):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'x.nup'
        path.write_bytes(build(packages))
        container = nup.Nup(path)
        self.addCleanup(container.close)
        return container, Path(tmp.name)

    def test_index_and_filter(self):
        update = (1, APP + 0x800, 2, {'exefs': {'main': b'y'}})
        n, _ = self.open([BASE, update, dlc(0), dlc(3), dlc(5, version=1), dlc(5, version=2),
                          (2, FIRST + 32, 1, {'romfs': {'z': b'z'}}),       # index 32: out of range
                          (2, APP + 0x2000, 1, {'romfs': {'z': b'z'}}),     # another application's: negative
                          (2, FIRST + 7, 1, {'exefs': {'main': b'm'}})])    # no RomFS
        got = nup.aoc_packs(n)
        self.assertEqual([i for i, _ in got], [0, 3, 5])
        self.assertEqual(got[2][1].version, 2)

    def test_zero_title_id_base(self):
        n, _ = self.open([(0, 0, 1, {'exefs': {'main': b'x'}}), (1, APP + 0x800, 2, {'exefs': {'main': b'y'}}),
                          dlc(1)])
        self.assertEqual([i for i, _ in nup.aoc_packs(n)], [1])

    def test_no_program_package(self):
        n, _ = self.open([dlc(0)])
        self.assertEqual(nup.aoc_packs(n), [])

    def test_write_aoc(self):
        n, tmp = self.open([BASE, dlc(1, b'one'), dlc(2, b'two')])
        out = tmp / 'aoc'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(nup.write_aoc(n, nup.Zstd(), out, True, True), 2)
        for i, data in ((1, b'one'), (2, b'two')):
            image = (out / f'aoc-{i}' / 'romfs.bin').read_bytes()
            self.assertEqual(image, data)
            index = (out / f'aoc-{i}' / 'aoc-index.bin').read_bytes()
            self.assertEqual(index[:8], b'TOTKRFS1')
            self.assertEqual(struct.unpack_from('<Q', index, 8)[0], 1)

    def test_bad_package_is_skipped(self):
        n, tmp = self.open([BASE, dlc(1), dlc(2)])
        f = n.files[-1]   # the last package's file: its SHA-256 record no longer matches
        n.files[-1] = f[:4] + (bytes(32),) + f[5:]
        out = tmp / 'aoc'
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(nup.write_aoc(n, nup.Zstd(), out, True, True), 1)
        self.assertIn('skipped', err.getvalue())
        self.assertTrue((out / 'aoc-1' / 'aoc-index.bin').is_file())
        self.assertFalse((out / 'aoc-2').exists())


if __name__ == '__main__':
    unittest.main()
