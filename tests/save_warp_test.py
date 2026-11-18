"""tools/save_warp.py on synthetic saves laid out like Bananza 1.0.0's (three 64-byte fields: checkpoint, location path,
stage name); no game files. Run: python tests/save_warp_test.py
"""
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('save_warp', ROOT / 'tools' / 'save_warp.py')
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)

KNOWN = ['AlphaLevelStage', 'BetaLevelStage']


def field(text):
    b = text.encode()
    return b + bytes(64 - len(b))


def progress(stage='AlphaLevelStage', checkpoint='0000aaaa-1111-2222-3333-444455556666_CheckPointTmp'):
    return (bytes(0x200) + field(checkpoint) + field(f'Develop/{stage}/Location') + field(stage) + bytes(0x100))


def caption(stage='AlphaLevelStage'):
    return bytes(0x38) + field(f'Develop/{stage}/Location') + bytes(0x40)


class SaveWarp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.profile = Path(self.tmp.name) / 'profile'
        self.user = self.profile / 'save' / 'abcdef'
        (self.user / '0').mkdir(parents=True)
        (self.profile / 'account.uid').write_bytes(b'synthetic')

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, prog=None, cap=None):
        (self.user / '0' / 'progress.sav').write_bytes(prog if prog is not None else progress())
        if cap is not False:
            (self.user / 'caption.sav').write_bytes(cap if cap is not None else caption())

    def test_warp_rewrites_location_stage_and_caption(self):
        self.write()
        old = sw.warp_profile(self.profile, 'BetaLevelStage', known=KNOWN)
        self.assertEqual(old, 'AlphaLevelStage')
        p = (self.user / '0' / 'progress.sav').read_bytes()
        self.assertEqual(p, progress('BetaLevelStage', checkpoint=''))
        self.assertEqual((self.user / 'caption.sav').read_bytes(), caption('BetaLevelStage'))
        self.assertEqual(sw.location(p)[1:], ('BetaLevelStage', ''))

    def test_keep_checkpoint(self):
        self.write()
        sw.warp_profile(self.profile, 'BetaLevelStage', known=KNOWN, keep_checkpoint=True)
        self.assertEqual((self.user / '0' / 'progress.sav').read_bytes(), progress('BetaLevelStage'))

    def test_unknown_stage_is_refused_with_suggestions(self):
        self.write()
        with self.assertRaisesRegex(sw.SaveError, 'did you mean BetaLevelStage'):
            sw.warp_profile(self.profile, 'Beta', known=KNOWN)
        self.assertEqual((self.user / '0' / 'progress.sav').read_bytes(), progress())   # untouched

    def test_unrecognised_layouts_are_refused(self):
        bad = {
            'no record': bytes(0x400),
            'two records': progress() + progress(),
            'stage field differs': progress().replace(field('AlphaLevelStage'), field('OtherStage')),
            'not a checkpoint before': progress(checkpoint='something else'),
        }
        for name, data in bad.items():
            with self.subTest(name):
                self.write(prog=data)
                with self.assertRaises(sw.SaveError):
                    sw.warp_profile(self.profile, 'BetaLevelStage', known=KNOWN)
                self.assertEqual((self.user / '0' / 'progress.sav').read_bytes(), data)

    def test_caption_mismatch_leaves_both_files(self):
        self.write(cap=caption('SomewhereElse'))
        with self.assertRaisesRegex(sw.SaveError, 'caption.sav'):
            sw.warp_profile(self.profile, 'BetaLevelStage', known=KNOWN)
        self.assertEqual((self.user / '0' / 'progress.sav').read_bytes(), progress())

    def test_missing_caption_is_fine(self):
        self.write(cap=False)
        sw.warp_profile(self.profile, 'BetaLevelStage', known=KNOWN)
        self.assertEqual(sw.location((self.user / '0' / 'progress.sav').read_bytes())[1], 'BetaLevelStage')

    def test_name_too_long(self):
        self.write()
        long_name = 'X' * 60
        with self.assertRaisesRegex(sw.SaveError, 'does not fit'):
            sw.warp_profile(self.profile, long_name, known=KNOWN + [long_name])

    def test_save_names_stay_under_saves(self):
        for name in ('..', '../escape', 'a/b', '.'):
            with self.subTest(name), redirect_stdout(io.StringIO()) as out:
                self.assertEqual(sw.main(['some-save', name, 'BetaLevelStage']), 1)
                self.assertIn('is not a save name', out.getvalue())


if __name__ == '__main__':
    unittest.main()
