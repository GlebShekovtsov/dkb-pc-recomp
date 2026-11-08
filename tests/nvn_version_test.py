"""Native NVN version queries for both supported Bananza targets; no GPU."""
import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_game  # noqa: E402  # type: ignore


class TargetGlue(unittest.TestCase):
    PATHS = {name: Path(name) for name in ('exefs', 'romfs', 'profile')}

    def glue(self, target, expected=None):
        return build_game.module_glue([], [], {}, self.PATHS, target, expected)

    def test_target_identity_is_not_the_legacy_totk_version(self):
        for target in ('dkb100', 'dkb458752', 'v121'):
            self.assertIn(f'const char rt_game_target[] = "{target}";', self.glue(target))

    def test_display_version_names_the_build_instead_of_zero(self):
        self.assertIn('const char rt_game_display[] = "1.2.1";', self.glue('v121'))
        self.assertIn('const char rt_game_display[] = "1.4.3";', self.glue('v143'))
        self.assertIn('const char rt_game_display[] = "1.0.0";', self.glue('dkb100', {'display_version': '1.0.0'}))
        self.assertIn('const char rt_game_display[] = "3.2.0";',
                      self.glue('dkb458752', {'display_version': '3.2.0', 'title_version': 458752}))
        # No metadata: the target name, never a decimal 0.0.0 that reads like a release.
        self.assertIn('const char rt_game_display[] = "dkb100";', self.glue('dkb100'))

    def test_shipped_update_metadata_names_the_update(self):
        expected = json.loads((ROOT / 'tools/data/expected-dkb458752.json').read_text(encoding='utf-8'))
        self.assertEqual((expected['display_version'], expected['title_version']), ('3.2.0', 458752))
        self.assertIn('const char rt_game_display[] = "3.2.0";', self.glue('dkb458752', expected))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', required=True)
    ap.add_argument('--deps', type=Path, required=True)
    args = ap.parse_args()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TargetGlue))
    if not result.wasSuccessful():
        return 1
    out = ROOT / 'build/tests/nvn-version'
    out.mkdir(parents=True, exist_ok=True)
    for target, minor in (('dkb100', 16), ('dkb458752', 30)):
        for opt in ('-O0', '-O2'):
            exe = out / f'{target}-{opt[1:]}.exe'
            subprocess.run([args.cc, '-std=gnu11', opt, '-ffunction-sections', '-fdata-sections',
                            '-Wl,--gc-sections', '-I', str(ROOT / 'runtime'),
                            '-I', str(args.deps / 'include'), f'-DTEST_TARGET="{target}"',
                            f'-DTEST_MINOR={minor}', str(ROOT / 'tests/nvn_version.c'),
                            '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
