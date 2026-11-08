"""Fragment modules through the real spirv_nan_guard (runtime/spirv_nan_guard.c), checked with spirv-val. No Vulkan, GPU
or game runs.

It checks that a whole-vector store and per-component stores through access chains each get one OpIsNan and one
OpSelect per float, that the selects pick the marker constant of their component, and that integer outputs, built-in
outputs and vertex stages are refused. It does not execute the shaders.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
OP_TYPE_BOOL, OP_CONSTANT, OP_STORE, OP_COMPOSITE_CONSTRUCT, OP_IS_NAN, OP_SELECT = 20, 43, 62, 80, 156, 169
MARKER = (16.0, 0.0, 4.0, 1.0)


def instr(op, *words):
    return [(len(words) + 1) << 16 | op, *words]


def f32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def module(stage=4, per_component=False, int_output=False, builtin=False, with_bool=True):
    """A shader that stores a vec4 to one output, whole or one component at a time. Ids are fixed; bound is 40."""
    (MAIN, OUT, VOID, FN, SCALAR, VEC, PTR, ZERO, VALUE, ENTRY, BOOL, INT, PTR1, *IDX) = range(1, 18)
    w = [0x07230203, 0x00010300, 0, 40, 0]
    w += instr(17, 1)                                    # OpCapability Shader
    w += instr(14, 0, 1)                                 # OpMemoryModel Logical GLSL450
    w += instr(15, stage, MAIN, 0x6e69616d, 0, OUT)      # OpEntryPoint "main"
    if stage == 4:
        w += instr(16, MAIN, 7)                          # OpExecutionMode OriginUpperLeft
    w += instr(71, OUT, 11, 22) if builtin else instr(71, OUT, 30, 0)   # BuiltIn FragDepth / Location 0
    if int_output:
        w += instr(71, OUT, 14)                          # Flat
    w += instr(19, VOID) + instr(33, FN, VOID)
    if with_bool:
        w += instr(OP_TYPE_BOOL, BOOL)
    w += instr(21, SCALAR, 32, 1) if int_output else instr(22, SCALAR, 32)
    w += instr(21, INT, 32, 1) if not int_output else []
    index_type = SCALAR if int_output else INT
    if builtin:
        w += instr(32, PTR, 3, SCALAR) + instr(59, PTR, OUT, 3)
        w += instr(OP_CONSTANT, SCALAR, ZERO, 0)
    else:
        w += instr(23, VEC, SCALAR, 4) + instr(32, PTR, 3, VEC) + instr(32, PTR1, 3, SCALAR) + instr(59, PTR, OUT, 3)
        w += instr(OP_CONSTANT, SCALAR, ZERO, 0) + instr(44, VEC, VALUE, ZERO, ZERO, ZERO, ZERO)
        for c, i in enumerate(IDX):
            w += instr(OP_CONSTANT, index_type, i, c)
    w += instr(54, VOID, MAIN, 0, FN) + instr(248, ENTRY)
    if builtin:
        w += instr(OP_STORE, OUT, ZERO)
    elif per_component:
        for c, i in enumerate(IDX):
            w += instr(65, PTR1, 30 + c, OUT, i)         # OpAccessChain
            w += instr(OP_STORE, 30 + c, ZERO)
    else:
        w += instr(OP_STORE, OUT, VALUE)
    w += instr(253) + instr(56)                          # OpReturn, OpFunctionEnd
    return w


def ops(words):
    i, out = 5, []
    while i < len(words):
        n = words[i] >> 16
        out.append((words[i] & 0xffff, words[i + 1:i + n]))
        i += n
    return out


class NanGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args.out.mkdir(parents=True, exist_ok=False)
        harness = args.out / 'harness.c'
        harness.write_text(r'''#include <stdio.h>
#include <stdlib.h>
#include "spirv_nan_guard.h"
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    static uint32_t w[1 << 20];
    FILE *f = fopen(argv[1], "rb");
    size_t n = fread(w, 4, 1 << 20, f);
    fclose(f);
    const float marker[4] = {16.0f, 0.0f, 4.0f, 1.0f};
    uint64_t out_n = 0;
    unsigned stores = 0;
    uint32_t *o = spirv_nan_guard(w, n, marker, &out_n, &stores);
    if (!o) return 3;
    f = fopen(argv[2], "wb");
    fwrite(o, 4, out_n, f);
    fclose(f);
    free(o);
    printf("%u\n", stores);
    return 0;
}
''', encoding='utf-8')
        cls.exe = args.out / 'harness.exe'
        subprocess.run([str(args.cc), '-O1', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'), str(harness),
                        str(ROOT / 'runtime/spirv_nan_guard.c'), '-o', str(cls.exe)], check=True, capture_output=True,
                       text=True, timeout=120)

    def run_guard(self, name, words):
        src = args.out / (name + '.in.spv')
        dst = args.out / (name + '.out.spv')
        src.write_bytes(struct.pack(f'<{len(words)}I', *words))
        run = subprocess.run([str(self.exe), str(src), str(dst)], capture_output=True, text=True, timeout=30)
        if run.returncode == 3:
            return None
        self.assertEqual(run.returncode, 0, run.stderr)
        data = dst.read_bytes()
        check = subprocess.run([str(args.spirv_val), '--target-env', 'vulkan1.3', str(dst)], capture_output=True, text=True, timeout=30)
        self.assertEqual(check.returncode, 0, name + ': ' + check.stdout + check.stderr)
        return int(run.stdout), ops(list(struct.unpack(f'<{len(data) // 4}I', data)))

    def markers_selected(self, found):
        consts = {a[1]: a[2] for op, a in found if op == OP_CONSTANT}
        return [consts[a[3]] for op, a in found if op == OP_SELECT]

    def test_a_whole_vector_store_is_guarded_per_component(self):
        stores, found = self.run_guard('whole', module())
        self.assertEqual(stores, 1)
        self.assertEqual(sum(op == OP_IS_NAN for op, _ in found), 4)
        self.assertEqual(sum(op == OP_COMPOSITE_CONSTRUCT for op, _ in found), 1)
        self.assertEqual(self.markers_selected(found), [f32(x) for x in MARKER])

    def test_component_stores_pick_their_component_marker(self):
        stores, found = self.run_guard('components', module(per_component=True))
        self.assertEqual(stores, 4)
        self.assertEqual(sum(op == OP_IS_NAN for op, _ in found), 4)
        self.assertEqual(self.markers_selected(found), [f32(x) for x in MARKER])

    def test_a_missing_bool_type_is_added(self):
        stores, found = self.run_guard('no_bool', module(with_bool=False))
        self.assertEqual(stores, 1)
        self.assertEqual(sum(op == OP_TYPE_BOOL for op, _ in found), 1)

    def test_other_modules_are_refused(self):
        self.assertIsNone(self.run_guard('int_output', module(int_output=True)))
        self.assertIsNone(self.run_guard('frag_depth', module(builtin=True)))
        self.assertIsNone(self.run_guard('vertex', module(stage=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', type=Path, required=True)
    parser.add_argument('--spirv-val', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, rest = parser.parse_known_args()
    raise SystemExit(not unittest.main(argv=[__file__] + rest, exit=False).result.wasSuccessful())
