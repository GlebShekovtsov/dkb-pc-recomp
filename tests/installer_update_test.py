"""Update notice tests: the release lookup is injected, no network."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from installer import update


def release(tag, **extra):
    return lambda: {'tag_name': tag, 'html_url': f'https://example.invalid/{tag}', **extra}


def broken():
    raise OSError('offline')


class UpdateTest(unittest.TestCase):
    def test_newer(self):
        self.assertEqual(update.newer_release('0.2.0+33f8bbc', release('v0.3.0')), ('v0.3.0', 'https://example.invalid/v0.3.0'))
        self.assertIsNotNone(update.newer_release('0.2.0-dev+abc', release('v0.2.1')))
        self.assertIsNotNone(update.newer_release('0.9.0', release('v0.10.0')))   # numeric, not text, order

    def test_equal_and_older(self):
        self.assertIsNone(update.newer_release('0.2.0+33f8bbc', release('v0.2.0')))
        self.assertIsNone(update.newer_release('0.2.0', release('v0.1.9')))

    def test_failures_return_none(self):
        self.assertIsNone(update.newer_release('0.2.0', broken))
        self.assertIsNone(update.newer_release('0.2.0', lambda: None))
        self.assertIsNone(update.newer_release('0.2.0', lambda: {'oops': 1}))
        self.assertIsNone(update.newer_release('unknown', release('v0.3.0')))
        with patch('urllib.request.urlopen', side_effect=OSError):
            self.assertIsNone(update.fetch_latest())

    def test_prerelease_ignored(self):
        self.assertIsNone(update.newer_release('0.2.0', release('v0.3.0-rc1')))
        self.assertIsNone(update.newer_release('0.2.0', release('v0.3.0', prerelease=True)))

    def test_switch(self):
        with patch.dict(os.environ, {'DKB_NO_UPDATE_CHECK': '1'}):
            self.assertTrue(update.disabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('DKB_NO_UPDATE_CHECK', None)
            self.assertFalse(update.disabled())
            self.assertTrue(update.disabled(True))

    def test_text_in_both_languages(self):
        for lang in ('en', 'ru'):
            text = json.loads((ROOT / 'installer' / 'text' / f'{lang}.json').read_text(encoding='utf-8'))
            self.assertIn('{tag}', text['ui.update_available'])
            self.assertIn('{url}', text['warn.update_available'])
        en = json.loads((ROOT / 'installer' / 'text' / 'en.json').read_text(encoding='utf-8'))
        ru = json.loads((ROOT / 'installer' / 'text' / 'ru.json').read_text(encoding='utf-8'))
        self.assertNotEqual(en['ui.update_available'], ru['ui.update_available'])


if __name__ == '__main__':
    unittest.main()
