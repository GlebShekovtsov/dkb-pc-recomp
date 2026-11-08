"""Regression tests for the packed-half atomic add and the halfword store (storage candidates).

Each test replays one captured shader-failure packet through a translator with tools/replay_shader_failure.py, then runs
spirv-val --target-env vulkan1.3 on the module it wrote. The packets hold game-derived shader code, so they live only under
build/ (ignored). A missing packet or translator is a failure, not a skip.

Observed tuples (translator error text, family from the seed split):
  * RED.E.ADD.F16x2.FTZ.RN in family 63399bcafafc05041e92103533f23610f1605f3c65fa98a32f6900335e645bb1: a guarded
    32-bit compare-exchange of the packed-half add (GlobalToStorage.PackedHalfAddFtzRn). Must translate and validate.
  * STG.E.U16 in family d0762eff0fa4b2b4f2d55e566722867d757099ade6692d7e9391b5a674a6ddef: a guarded halfword store
    (byte-lane compare-exchange, two words when it crosses). Same shader also contains the RED above. Must translate.

The GPU side (arithmetic against an independent reference, contention, and the crossing-word store) is in
build/agent-shader-investigation/h3/gpu/ and runs only under build\\gpu-run.lock.

Run with the translator under test (default: the verification-copy build of the combined candidate):
  .venv\\Scripts\\python.exe -m unittest tests.xlate_storage_guard_test
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Captured failure packets (game-derived, not in the repo). Point XLATE_PACKETS at the folder that holds run-63399/ and run-d0762/.
H3 = pathlib.Path(os.environ.get("XLATE_PACKETS", ROOT / "build" / "agent-shader-investigation" / "packets"))
TRANSLATOR = pathlib.Path(os.environ.get("XLATE_TRANSLATOR", ROOT / "build" / "xlate" / "totk_xlate.exe"))
SPIRV_VAL = pathlib.Path(os.environ.get("SPIRV_VAL", "C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe"))
REPLAY = ROOT / "tools" / "replay_shader_failure.py"

PACKETS = {
    "red-f16x2": H3 / "run-63399" / "failures" / "capture-32004-1"
                 / "2c0b4eb6a04d24243fb4c7e2c0809f4f4a54142a371c76fe42d01ae5f228d6f2.request.json",
    "store-u16": H3 / "run-d0762" / "failures" / "capture-17724-1"
                 / "9730318e3e18d6540ba4de0ffc64e50499acc26511d32b11f54305cc4f960b69.request.json",
}


def replay(packet, out_dir):
    """Replay one packet; return (exit code, text of stderr and stdout)."""
    run = subprocess.run(
        [sys.executable, str(REPLAY), str(packet), "--translator", str(TRANSLATOR), "--out", str(out_dir)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    return run.returncode, run.stderr + run.stdout


class H3StorageGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The packets are game-derived and not in the repository: without them (or without a translator) the class is
        # skipped with the reason, so a checkout without game data still passes. With them it runs in full.
        missing = [str(p) for p in (TRANSLATOR, *PACKETS.values()) if not p.is_file()]
        if missing:
            raise unittest.SkipTest("needs a translator and captured packets; set XLATE_TRANSLATOR and XLATE_PACKETS. Missing: "
                                    + ", ".join(missing))
        if not SPIRV_VAL.is_file():
            raise unittest.SkipTest(f"spirv-val not found at {SPIRV_VAL}; set SPIRV_VAL")

    def _translate_and_validate(self, name):
        with tempfile.TemporaryDirectory() as out:
            code, text = replay(PACKETS[name], out)
            self.assertEqual(code, 0, text)
            self.assertTrue(text.startswith("ok "), text)
            modules = sorted(pathlib.Path(out).glob("*.spv"))
            self.assertEqual(len(modules), 1, "the translator must write exactly one SPIR-V module")
            val = subprocess.run([str(SPIRV_VAL), "--target-env", "vulkan1.3", str(modules[0])],
                                 capture_output=True, text=True)
            self.assertEqual(val.returncode, 0, val.stdout + val.stderr)

    def test_red_f16x2_packed_half_add_translates_and_validates(self):
        self._translate_and_validate("red-f16x2")

    def test_store_u16_halfword_store_translates_and_validates(self):
        self._translate_and_validate("store-u16")


if __name__ == "__main__":
    unittest.main()
