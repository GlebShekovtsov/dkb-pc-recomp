"""Original fragment modules through the real spirv_alpha_test, checked with spirv-val. No Vulkan, GPU or game runs.

The function is cut out of runtime/spirv_rewrite.c and compiled alone, so the test follows the source. It checks that each
NVN alpha function gives a valid module with one OpKill per return of the entry point, that a helper's return is left
alone, and that a module without a float vec4 output at Location 0 is refused. It does not execute the shaders.
"""
import argparse
from pathlib import Path
import re
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
OP_TYPE_BOOL, OP_CONSTANT, OP_FUNCTION, OP_RETURN, OP_KILL = 20, 43, 54, 253, 252
COMPARE = {1: 184, 2: 184, 3: 180, 4: 188, 5: 186, 6: 182, 7: 190}   # by NVN_ALPHA_FUNC_*


def instr(op, *words):
    return [(len(words) + 1) << 16 | op, *words]


def module(location=0, scalar_int=False, returns=1, helper=False):
    """A fragment shader that stores a constant vec4 to one output. Ids are fixed; bound is 40."""
    MAIN, OUT, VOID, FN, SCALAR, VEC, PTR, ZERO, VALUE, ENTRY, BOOL, TRUE, THEN, ELSE, MERGE, HELPER, HENTRY, CALL = range(1, 19)
    w = [0x07230203, 0x00010300, 0, 40, 0]
    w += instr(17, 1)                                    # OpCapability Shader
    w += instr(14, 0, 1)                                 # OpMemoryModel Logical GLSL450
    w += instr(15, 4, MAIN, 0x6e69616d, 0, OUT)          # OpEntryPoint Fragment "main"
    w += instr(16, MAIN, 7)                              # OpExecutionMode OriginUpperLeft
    w += instr(71, OUT, 30, location)                    # OpDecorate Location
    if scalar_int:
        w += instr(71, OUT, 14)                          # OpDecorate Flat
    w += instr(19, VOID) + instr(33, FN, VOID)
    w += instr(21, SCALAR, 32, 1) if scalar_int else instr(22, SCALAR, 32)
    w += instr(23, VEC, SCALAR, 4) + instr(32, PTR, 3, VEC) + instr(59, PTR, OUT, 3)
    w += instr(OP_CONSTANT, SCALAR, ZERO, 0) + instr(44, VEC, VALUE, ZERO, ZERO, ZERO, ZERO)
    if returns == 2:
        w += instr(OP_TYPE_BOOL, BOOL) + instr(41, BOOL, TRUE)
    if helper:
        w += instr(OP_FUNCTION, VOID, HELPER, 0, FN) + instr(248, HENTRY) + instr(OP_RETURN) + instr(56)
    w += instr(OP_FUNCTION, VOID, MAIN, 0, FN) + instr(248, ENTRY)
    w += instr(62, OUT, VALUE)                           # OpStore
    if helper:
        w += instr(57, VOID, CALL, HELPER)               # OpFunctionCall
    if returns == 2:
        w += instr(247, MERGE, 0) + instr(250, TRUE, THEN, ELSE)
        w += instr(248, THEN) + instr(OP_RETURN) + instr(248, ELSE) + instr(OP_RETURN) + instr(248, MERGE) + instr(255)
    else:
        w += instr(OP_RETURN)
    w += instr(56)                                       # OpFunctionEnd
    return w


def ops(words):
    i, out = 5, []
    while i < len(words):
        n = words[i] >> 16
        out.append((words[i] & 0xffff, words[i + 1:i + n]))
        i += n
    return out


class AlphaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args.out.mkdir(parents=True, exist_ok=False)
        source = (ROOT / 'runtime/spirv_rewrite.c').read_text(encoding='utf-8')
        header = (ROOT / 'runtime/spirv_rewrite.h').read_text(encoding='utf-8')
        enum = re.search(r'enum \{ SPV_OP_ENTRY_POINT.*?\};', header, re.S).group(0)
        start = source.index('uint32_t *spirv_alpha_test(')
        function = source[start:source.index('\n}\n', start) + 3]
        harness = args.out / 'harness.c'
        harness.write_text('#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n' + enum + '\n' +
                           function + r'''
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    static uint32_t w[1 << 20];
    FILE *f = fopen(argv[1], "rb");
    size_t n = fread(w, 4, 1 << 20, f);
    fclose(f);
    uint64_t out_n = 0;
    uint32_t *o = spirv_alpha_test(w, n, (unsigned)atoi(argv[3]), (float)atof(argv[4]), &out_n);
    if (!o) return 3;
    f = fopen(argv[2], "wb");
    fwrite(o, 4, out_n, f);
    fclose(f);
    free(o);
    return 0;
}
''', encoding='utf-8')
        cls.exe = args.out / 'harness.exe'
        subprocess.run([str(args.cc), '-O1', '-Wall', '-Wextra', '-Werror', str(harness), '-o', str(cls.exe)], check=True,
                       capture_output=True, text=True, timeout=120)

    def run_function(self, name, words, func, ref):
        src = args.out / (name + '.in.spv')
        dst = args.out / (name + '.out.spv')
        src.write_bytes(struct.pack(f'<{len(words)}I', *words))
        run = subprocess.run([str(self.exe), str(src), str(dst), str(func), str(ref)], capture_output=True, timeout=30)
        if run.returncode == 3:
            return None
        self.assertEqual(run.returncode, 0, run.stderr)
        data = dst.read_bytes()
        check = subprocess.run([str(args.spirv_val), '--target-env', 'vulkan1.3', str(dst)], capture_output=True, text=True, timeout=30)
        self.assertEqual(check.returncode, 0, name + ': ' + check.stdout + check.stderr)
        return list(struct.unpack(f'<{len(data) // 4}I', data))

    def test_every_function_gives_a_valid_module_with_its_compare(self):
        for func, opcode in COMPARE.items():
            out = self.run_function(f'func{func}', module(), func, 0.5)
            found = ops(out)
            self.assertEqual(sum(op == OP_KILL for op, _ in found), 1)
            self.assertEqual(sum(op == OP_RETURN for op, _ in found), 1)
            self.assertEqual(sum(op == opcode for op, _ in found), 1)
            self.assertIn(struct.unpack('<I', struct.pack('<f', 0.5))[0], [a[-1] for op, a in found if op == OP_CONSTANT])

    def test_each_return_of_the_entry_point_gets_the_test(self):
        found = ops(self.run_function('two_returns', module(returns=2), 7, 1.0))
        self.assertEqual(sum(op == OP_KILL for op, _ in found), 2)
        self.assertEqual(sum(op == OP_TYPE_BOOL for op, _ in found), 1)   # the module's own bool type is reused

    def test_a_helper_function_is_left_alone(self):
        found = ops(self.run_function('helper', module(helper=True), 7, 1.0))
        self.assertEqual(sum(op == OP_KILL for op, _ in found), 1)
        self.assertEqual(sum(op == OP_RETURN for op, _ in found), 2)

    def test_modules_without_the_output_are_refused(self):
        self.assertIsNone(self.run_function('location1', module(location=1), 7, 1.0))
        self.assertIsNone(self.run_function('int_output', module(scalar_int=True), 7, 1.0))

    def test_always_is_not_a_test(self):
        self.assertIsNone(self.run_function('always', module(), 8, 1.0))
        self.assertIsNone(self.run_function('unset', module(), 0, 1.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', type=Path, required=True)
    parser.add_argument('--spirv-val', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, rest = parser.parse_known_args()
    raise SystemExit(not unittest.main(argv=[__file__] + rest, exit=False).result.wasSuccessful())
