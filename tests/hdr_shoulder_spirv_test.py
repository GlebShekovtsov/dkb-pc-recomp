"""Fragment modules through the real spirv_hdr_shoulder (runtime/spirv_hdr_shoulder.c), then spirv_nan_guard
(runtime/spirv_nan_guard.c), each output checked with spirv-val. No Vulkan, GPU or game runs.

The modules are synthetic and laid out the way the translator lays out Bananza's HDR composite: a knee comparison
Load Ys >= Load U[0][4].y, then three colour components stored as FMin(FMax(0, FMul(x, Load Y)), Load U[0][4].x).
The test checks that such a module is rewritten (one new OpSelect, whose result all three FMul then read) and still
validates and takes the NaN guard, and that near misses are refused: a store to Ys between the comparison and the
FMul, a store to Y among the FMul, only two components, a missing comparison and a vertex stage. It does not execute
the shaders.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
OP_EXT_INST, OP_LOAD, OP_STORE, OP_FMUL, OP_SELECT = 12, 61, 62, 133, 169
GLSL_FMIN, GLSL_FMAX = 37, 40


def instr(op, *words):
    return [(len(words) + 1) << 16 | op, *words]


def f32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def string(s):
    b = s.encode() + b'\0'
    b += b'\0' * (-len(b) % 4)
    return list(struct.unpack(f'<{len(b) // 4}I', b))


def composite(stage=4, components=3, store_ys_after_compare=False, store_y_among_muls=False, compare=True):
    """The composite's shape: c = U[0] is a vec4[8] uniform array, the colour output a vec4 written per component."""
    ids = iter(range(1, 1000))
    (GLSL, MAIN, OUT, U, VOID, FN, F32, BOOL, INT, VEC4, ARR, BLOCK, PTR_U, PTR_U_F32, PTR_O, PTR_O_F32, PTR_F_F32,
     ZERO, I0, I1, I2, I4, I8, ENTRY, THEN, MERGE, YV, YSV) = (next(ids) for _ in range(28))
    w = []
    w += instr(17, 1)                                            # OpCapability Shader
    w += instr(11, GLSL, *string('GLSL.std.450'))                # OpExtInstImport
    w += instr(14, 0, 1)                                         # OpMemoryModel Logical GLSL450
    w += instr(15, stage, MAIN, *string('main'), OUT)            # OpEntryPoint
    if stage == 4:
        w += instr(16, MAIN, 7)                                  # OpExecutionMode OriginUpperLeft
    w += instr(71, OUT, 30, 0)                                   # Location 0
    w += instr(71, U, 34, 0) + instr(71, U, 33, 0)               # DescriptorSet 0, Binding 0
    w += instr(71, BLOCK, 2)                                     # Block
    w += instr(72, BLOCK, 0, 35, 0)                              # OpMemberDecorate Offset 0
    w += instr(71, ARR, 6, 16)                                   # ArrayStride 16
    w += instr(19, VOID) + instr(33, FN, VOID)
    w += instr(22, F32, 32) + instr(20, BOOL) + instr(21, INT, 32, 1)
    w += instr(23, VEC4, F32, 4)
    w += instr(43, INT, I0, 0) + instr(43, INT, I1, 1) + instr(43, INT, I2, 2) + instr(43, INT, I4, 4)
    w += instr(43, INT, I8, 8)
    w += instr(28, ARR, VEC4, I8) + instr(30, BLOCK, ARR)        # OpTypeArray, OpTypeStruct
    w += instr(32, PTR_U, 2, BLOCK) + instr(32, PTR_U_F32, 2, F32)
    w += instr(32, PTR_O, 3, VEC4) + instr(32, PTR_O_F32, 3, F32) + instr(32, PTR_F_F32, 7, F32)
    w += instr(43, F32, ZERO, f32(0.0))
    w += instr(59, PTR_O, OUT, 3) + instr(59, PTR_U, U, 2)       # OpVariable Output, Uniform

    def load_u(row, comp):
        ac, ld = next(ids), next(ids)
        nonlocal w
        w += instr(65, PTR_U_F32, ac, U, I0, row, comp) + instr(OP_LOAD, F32, ld, ac)
        return ld

    w += instr(54, VOID, MAIN, 0, FN) + instr(248, ENTRY)        # OpFunction, OpLabel
    w += instr(59, PTR_F_F32, YV, 7) + instr(59, PTR_F_F32, YSV, 7)
    x = load_u(I0, I0)                                           # the graded colour stand-in, c[0].x
    w += instr(OP_STORE, YSV, x) + instr(OP_STORE, YV, x)
    ys = next(ids)
    w += instr(OP_LOAD, F32, ys, YSV)
    cmp = next(ids)
    if compare:                                                  # OpFOrdGreaterThanEqual Ys, c[4].y: the knee
        k = load_u(I4, I1)
        w += instr(190, BOOL, cmp, ys, k)
    else:                                                        # OpFOrdGreaterThan Ys, x: not the knee
        w += instr(186, BOOL, cmp, ys, x)
    w += instr(247, MERGE, 0) + instr(250, cmp, THEN, MERGE)     # OpSelectionMerge, OpBranchConditional
    w += instr(248, THEN)
    w += instr(OP_STORE, YV, load_u(I4, I0))                     # the shoulder's Y stand-in
    w += instr(249, MERGE) + instr(248, MERGE)                   # OpBranch, OpLabel
    if store_ys_after_compare:
        w += instr(OP_STORE, YSV, x)
    for c, idx in enumerate((I0, I1, I2)[:components]):
        if store_y_among_muls and c == 1:
            w += instr(OP_STORE, YV, x)
        y, mul, mx = next(ids), next(ids), next(ids)
        w += instr(OP_LOAD, F32, y, YV) + instr(OP_FMUL, F32, mul, x, y)
        w += instr(OP_EXT_INST, F32, mx, GLSL, GLSL_FMAX, ZERO, mul)
        m = load_u(I4, I0)
        mn, ac = next(ids), next(ids)
        w += instr(OP_EXT_INST, F32, mn, GLSL, GLSL_FMIN, mx, m)
        w += instr(65, PTR_O_F32, ac, OUT, idx) + instr(OP_STORE, ac, mn)
    w += instr(253) + instr(56)                                  # OpReturn, OpFunctionEnd
    return [0x07230203, 0x00010300, 0, next(ids), 0] + w


def ops(words):
    i, out = 5, []
    while i < len(words):
        n = words[i] >> 16
        out.append((words[i] & 0xffff, words[i + 1:i + n]))
        i += n
    return out


class HdrShoulder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args.out.mkdir(parents=True, exist_ok=True)
        harness = args.out / 'harness.c'
        harness.write_text(r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_hdr_shoulder.h"
#include "spirv_nan_guard.h"
// harness shoulder|guard in.spv out.spv: exit 3 when the rewrite refuses the module
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    static uint32_t w[1 << 20];
    FILE *f = fopen(argv[2], "rb");
    if (!f) return 2;
    size_t n = fread(w, 4, 1 << 20, f);
    fclose(f);
    uint64_t out_n = 0;
    uint32_t *o;
    if (!strcmp(argv[1], "shoulder")) {
        o = spirv_hdr_shoulder(w, n, &out_n);
    } else {
        const float marker[4] = {0, 0, 0, 0};
        unsigned stores = 0;
        o = spirv_nan_guard(w, n, marker, &out_n, &stores);
    }
    if (!o) return 3;
    f = fopen(argv[3], "wb");
    fwrite(o, 4, out_n, f);
    fclose(f);
    free(o);
    return 0;
}
''', encoding='utf-8')
        cls.exe = args.out / 'harness.exe'
        build = subprocess.run([str(args.cc), '-O1', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                                str(harness), str(ROOT / 'runtime/spirv_hdr_shoulder.c'),
                                str(ROOT / 'runtime/spirv_nan_guard.c'), '-o', str(cls.exe)],
                               capture_output=True, text=True, timeout=120)
        assert build.returncode == 0, build.stdout + build.stderr

    def validate(self, name, path):
        check = subprocess.run([str(args.spirv_val), '--target-env', 'vulkan1.3', str(path)], capture_output=True,
                               text=True, timeout=30)
        self.assertEqual(check.returncode, 0, name + ': ' + check.stdout + check.stderr)

    def run_pass(self, mode, name, words):
        src = args.out / f'{name}.{mode}.in.spv'
        dst = args.out / f'{name}.{mode}.out.spv'
        src.write_bytes(struct.pack(f'<{len(words)}I', *words))
        run = subprocess.run([str(self.exe), mode, str(src), str(dst)], capture_output=True, text=True, timeout=30)
        if run.returncode == 3:
            return None
        self.assertEqual(run.returncode, 0, run.stderr)
        self.validate(f'{name} after {mode}', dst)
        data = dst.read_bytes()
        return list(struct.unpack(f'<{len(data) // 4}I', data))

    def test_the_synthetic_module_is_valid_before_the_rewrite(self):
        for name, words in (('plain', composite()), ('vertex', composite(stage=0))):
            path = args.out / f'{name}.spv'
            path.write_bytes(struct.pack(f'<{len(words)}I', *words))
            self.validate(name, path)

    def test_the_composite_is_rewritten(self):
        before = composite()
        out = self.run_pass('shoulder', 'composite', before)
        self.assertIsNotNone(out)
        found = ops(out)
        selects = [a[1] for op, a in found if op == OP_SELECT]
        self.assertEqual(len(selects), 1)
        muls = [a for op, a in found if op == OP_FMUL]
        colour_muls = [a for a in muls if a[3] == selects[0]]
        self.assertEqual(len(colour_muls), 3, 'all three colour FMul read the new Y')
        self.assertGreater(out[3], before[3], 'the id bound grows')
        self.assertEqual(sum(op == OP_STORE for op, _ in found), sum(op == OP_STORE for op, _ in ops(before)),
                         'no store is added')

    def test_the_rewritten_composite_takes_the_nan_guard(self):
        out = self.run_pass('shoulder', 'guarded', composite())
        self.assertIsNotNone(out)
        self.assertIsNotNone(self.run_pass('guard', 'guarded', out))

    def test_near_misses_are_refused(self):
        cases = {
            'ys_stored_after_compare': composite(store_ys_after_compare=True),
            'y_stored_among_muls': composite(store_y_among_muls=True),
            'two_components': composite(components=2),
            'no_knee_compare': composite(compare=False),
            'vertex': composite(stage=0),
        }
        for name, words in cases.items():
            with self.subTest(name):
                self.assertIsNone(self.run_pass('shoulder', name, words))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', type=Path, required=True)
    parser.add_argument('--spirv-val', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, rest = parser.parse_known_args()
    raise SystemExit(not unittest.main(argv=[__file__] + rest, exit=False).result.wasSuccessful())
