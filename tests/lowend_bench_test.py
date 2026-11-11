"""Check tools/lowend_bench.py parse_log on a synthetic log built from the real line formats.

No game and no GPU: it only guards the regexes and the statistics.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lowend_bench import parse_log  # noqa: E402


def frame(n, avg, mx, replay=None):
    tail = "; replay avg %.2f ms, 65 draws" % replay if replay else ""
    return ("[host] perf: frame %d: interval avg %.2f ms max %.2f ms (60.0 fps); render thread blocked: pacing 80%% fence 1%% "
            "drain 2%% pipeline build 3%% shader xlate 4%% texture 5%% (astc 0%%); shaders: 1 keys%s; world 1.0 s\n" % (n, avg, mx, tail))


LOG = (
    frame(60, 20.0, 300.0, 3.0) + "[host] perf: frame 60: gpu avg 1.40 ms (127 parts timed, 0 not); uploads 1\n"
    + frame(120, 16.0, 30.0, 1.0) + "[host] perf: frame 120: gpu avg 0.60 ms (121 parts timed, 0 not)\n"
    + frame(180, 18.0, 40.0)
    + "svc: 0x18 blocked for 12.0 ms at 1.0 s tid 5, called from x\n"
    + "romfs: slow read a/b: 64 KiB in 9.0 ms\nromfs: slow read c: 1 KiB in 8.0 ms\n"
    + "vk_exec: draw profile over 100 draws (5 other ops), ns per draw: a 10, b 20; total 30\n"
    + "vk_exec: draw profile over 100 draws (5 other ops), ns per draw: a 11, b 22; total 33\n"
    + "boot: frames 180 presented (nvn calls 1, distinct 2, unimplemented 0, validation errors 1, shader failures 2, draws skipped 3)\n"
)


class ParseLog(unittest.TestCase):
    def test_numbers(self):
        r = parse_log(LOG)
        self.assertEqual((r["frames"], r["errors"], r["shader_failures"], r["draws_skipped"]), (180, 1, 2, 3))
        self.assertEqual(r["windows"], 3)
        self.assertAlmostEqual(r["steady_fps"], 1000 / 18.0)
        self.assertEqual(r["worst_ms"], 300.0)
        self.assertEqual(r["low1_ms"], 300.0)
        self.assertEqual(r["blocked"], [80, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(r["gpu_ms"], 1.0)
        self.assertAlmostEqual(r["replay_ms"], 2.0)
        self.assertEqual(r["profile"], "a 11, b 22")
        self.assertEqual((r["svc_blocked"], r["romfs_slow"]), (1, 2))

    def test_empty(self):
        r = parse_log("nothing here\n")
        self.assertIsNone(r["frames"])
        self.assertIsNone(r["steady_fps"])
        self.assertIsNone(r["profile"])


if __name__ == "__main__":
    unittest.main()
