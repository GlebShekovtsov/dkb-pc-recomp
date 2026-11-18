"""tools/perf_check.py on synthetic TOTK_TRACE=perf logs in runtime/perf.c's line format; no game files.
Run: python tests/perf_check_test.py
"""
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('perf_check', ROOT / 'tools' / 'perf_check.py')
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def window(frame, avg, world, busy=8.0, replay=3.0, gpu=1.0, pacing=40, fence=0, drain=0, pipeline=0, xlate=0,
           texture=0, translated=0, bound='pace'):
    """The two lines runtime/perf.c writes for one 60-present window."""
    return (f'[host] perf: frame {frame}: interval avg {avg:.2f} ms max {avg * 3:.2f} ms ({1000 / avg:.1f} fps); render '
            f'thread blocked: pacing {pacing}% fence {fence}% drain {drain}% pipeline build {pipeline}% shader xlate '
            f'{xlate}% texture {texture}% (astc 0%); pipelines built in background 0, draws skipped pending 0, waited 0, '
            f'waiting for a shader 0, one-shot waited 0; textures uploaded 0 re-uploaded 0; slowest frame: xlate 0.0 ms '
            f'texture 0.0 ms (astc 0.0) pipeline 0.0 ms fence 0.0 ms; shaders: 77 keys, {translated} translated; render '
            f'busy avg {busy:.2f} ms; replay avg {replay:.2f} ms; ab 0/60; world {world:.1f} s\n'
            f'[host] perf: frame {frame}: gpu avg {gpu:.2f} ms (100 parts timed, 0 not); uploads 1 batches, 1.0 MiB '
            f'staged, gpu 0.02 ms per frame (1 timed), waited 0.00 ms per frame; present gpu 0.02 ms per frame (60 timed); '
            f'fence wait per frame: render thread 0.00 ms, other threads 0.00 ms; vram 900 of 7000 MiB; ram working set '
            f'2500 MiB, private 11000 MiB; bound: {bound}\n')


def log(*windows):
    return ''.join(window(60 * (i + 1), world=1.0 * (i + 1), **w) for i, w in enumerate(windows))


STEADY = dict(avg=16.7)


class PerfCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        p = self.dir / name
        p.write_text(text, encoding='utf-8')
        return p

    def analyse(self, text, **kw):
        return pc.analyse(self.write('run.log', text), kw.pop('min_fps', 50), kw.pop('hitch_ms', 100),
                          kw.pop('skip_world', 0), **kw)

    def test_steady_run_is_ok(self):
        r = self.analyse(log(*[STEADY] * 10))
        self.assertEqual((r['verdict'], r['slow_windows'], r['episodes']), ('OK', 0, []))
        self.assertAlmostEqual(r['fps'], 1000 / 16.7, places=3)

    def test_the_instrumentation_lines_do_not_change_the_verdict(self):
        # lines added by TOTK_TRACE=perf for slow-window diagnosis (runtime/perf.c, vk_memory.c): the submit lock, the process, GetWriteWatch timing
        extra = ('[host] perf: frame 60: submit lock: 3.00 submits per frame, wait 4.100 ms per frame (max 30.00 ms), held 11.000 ms per frame (max 50.00 ms)\n'
                 '[host] perf: frame 60: process: page faults +123456 (2057 per frame), working set 4054 MiB (peak 4100), private 11516 MiB; system: '
                 'available 11000 of 32000 MiB, commit 20000 of 60000 MiB\n'
                 '[host] mem: write-watch timing frame 60: 800 calls, 9.400 ms total (0.0118 ms per call), slowest 3.100 ms (2.00 MiB queried, '
                 '5 pages found, reset), 12 calls over 0.2 ms\n'
                 '[host] mem: write-watch call 3.100 ms: 2.00 MiB queried, 5 pages found\n')
        plain = log(*[STEADY] * 3)
        with_extra = ''.join(line + (extra if i % 2 else '') for i, line in enumerate(plain.splitlines(True)))   # after each window's second line
        a, b = self.analyse(plain), self.analyse(with_extra)
        self.assertEqual((a['verdict'], a['slow_windows'], a['episodes']), (b['verdict'], b['slow_windows'], b['episodes']))
        self.assertAlmostEqual(a['fps'], b['fps'], places=6)

    def test_short_dip_is_minor(self):
        r = self.analyse(log(STEADY, STEADY, dict(avg=24.0), STEADY, STEADY))
        self.assertEqual(r['verdict'], 'HITCHY')
        self.assertEqual(len(r['episodes']), 1)
        self.assertFalse(r['episodes'][0]['major'])
        self.assertEqual((r['episodes'][0]['first_frame'], r['episodes'][0]['last_frame']), (121, 180))

    def test_long_or_deep_slowdown_is_major(self):
        long = self.analyse(log(STEADY, *[dict(avg=24.0)] * 3, STEADY))
        deep = self.analyse(log(STEADY, dict(avg=60.0), STEADY))
        self.assertEqual((long['verdict'], deep['verdict']), ('SLOW', 'SLOW'))
        self.assertEqual(long['episodes'][0]['windows'], 3)

    def test_causes(self):
        cases = {
            'compile': dict(avg=80.0, xlate=30, pacing=0, translated=12),
            'texture': dict(avg=80.0, texture=40, pacing=0),
            'gpu': dict(avg=80.0, gpu=70.0, pacing=0, bound='gpu'),
            'render-cpu': dict(avg=80.0, busy=75.0, replay=20.0, pacing=0, bound='cpu'),
            'game-cpu': dict(avg=80.0, busy=10.0, pacing=0, bound='cpu'),
        }
        for want, w in cases.items():
            with self.subTest(want):
                r = self.analyse(log(STEADY, w, STEADY))
                self.assertEqual(r['episodes'][0]['cause'], want)

    def test_translated_is_a_run_total(self):
        # The perf line's count is the run's total: a slow window with a little compile time after boot, when nothing
        # new was translated, is not a compile window; one where the total grew is.
        boot = dict(avg=16.7, translated=500)
        same = dict(avg=80.0, busy=10.0, pacing=0, bound='cpu', xlate=3, pipeline=3, translated=500)
        grew = dict(same, translated=512)
        self.assertEqual(self.analyse(log(boot, same, boot))['episodes'][0]['cause'], 'game-cpu')
        self.assertEqual(self.analyse(log(boot, grew, dict(boot, translated=512)))['episodes'][0]['cause'], 'compile')

    def test_skip_world_leaves_out_loading(self):
        text = log(dict(avg=300.0), dict(avg=300.0), STEADY, STEADY)   # worlds 1 and 2 s: the load
        self.assertEqual(self.analyse(text, skip_world=2.5)['verdict'], 'OK')
        self.assertEqual(self.analyse(text, skip_world=0)['verdict'], 'SLOW')

    def test_no_perf_lines(self):
        r = self.analyse('[host] something else\n')
        self.assertEqual((r['verdict'], r['windows'], r['fps']), ('no data', 0, None))

    def test_host_call_refusals_per_frame(self):
        text = log(STEADY, STEADY) + ('[host] kernel: frame 60: asynchronous preemption: 1 interrupted, 1 cores given '
                                      'up there; refused: 60 in a host call, 0 outside game code, 0 preemption disabled\n'
                                      '[host] kernel: frame 120: asynchronous preemption: 2 interrupted, 2 cores given '
                                      'up there; refused: 660 in a host call, 0 outside game code, 0 preemption disabled\n')
        r = self.analyse(text)
        self.assertAlmostEqual(r['steady_host_calls'], (1.0 + 10.0) / 2)   # 60 over 60 frames, then 600 over 60

    def test_fail_and_groups(self):
        ok = self.write('ok.log', log(*[STEADY] * 6))
        dip = self.write('dip.log', log(STEADY, dict(avg=24.0), STEADY))
        slow = self.write('slow.log', log(STEADY, *[dict(avg=90.0, busy=85.0, pacing=0, bound='cpu')] * 4))
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(pc.main([str(ok), str(dip), '--fail', '--skip-world', '0']), 0)
            self.assertEqual(pc.main([str(dip), '--fail', '--fail-hitchy', '--skip-world', '0']), 1)
            self.assertEqual(pc.main([str(slow), '--fail', '--skip-world', '0']), 1)
            code = pc.main(['--group', f'a={self.dir / "ok.log"}', '--group', f'b={self.dir / "slow.log"}',
                            '--skip-world', '0', '--json', str(self.dir / 'r.json')])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn('render-cpu', text)
        self.assertIn('TOTK_PROFILE_RENDER=1', text)   # the hint for the cause
        self.assertTrue((self.dir / 'r.json').is_file())


if __name__ == '__main__':
    unittest.main()
