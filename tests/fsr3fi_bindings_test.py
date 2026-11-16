"""The descriptors each of AMD's frame-interpolation and optical-flow passes declares against the binding lists of
runtime/vk_fsr3fi.c, without game files or a GPU.

Each pass pushes its descriptors at the binding numbers its GLSL fixes, and the lists in vk_fsr3fi.c are written by
hand: a number or a kind out of place is a wrong picture or a validation error at run time, not a compile error. This
reads the committed SPIR-V headers (runtime/fsr3_fi_*_spv.h, runtime/fsr3_of_*_spv.h) and checks, for every pass, that
each (binding, kind) the module declares is in its list, the sampler at binding 1000 included. The headers are built by
glslc with its optimizer (tools/build_shaders.py), which drops a descriptor a pass declares and never reads, such as a
constant block the SDK declares for every pass; the list still names it, which a pipeline layout allows, so the module
may declare less than the list but never something the list does not have or has as another kind.

Usage: python tests/fsr3fi_bindings_test.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / 'runtime'

# SPIR-V opcodes and decorations read here
OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_STRUCT, OP_TYPE_POINTER, OP_VARIABLE, OP_DECORATE = 25, 26, 30, 32, 59, 71
OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY = 28, 29
DEC_BUFFER_BLOCK, DEC_BINDING = 3, 33
SC_STORAGE_BUFFER = 12


def words_of(header):
    text = header.read_text(encoding='utf-8')
    body = text[text.index('{') + 1:text.rindex('}')]
    return [int(w, 16) for w in body.replace('\n', ' ').split(',') if w.strip()]


def declared(words):
    """The sorted (binding, kind) of a module: T sampled texture, I storage image, U uniform buffer, B storage buffer,
    S sampler."""
    types, pointers, variables, bindings, buffer_blocks = {}, {}, {}, {}, set()
    i = 5
    while i < len(words):
        op, n = words[i] & 0xffff, words[i] >> 16
        a = words[i + 1:i + n]
        if op == OP_DECORATE and a[1] == DEC_BINDING:
            bindings[a[0]] = a[2]
        elif op == OP_DECORATE and a[1] == DEC_BUFFER_BLOCK:
            buffer_blocks.add(a[0])
        elif op == OP_TYPE_IMAGE:
            types[a[0]] = 'T' if a[6] == 1 else 'I'   # Sampled: 1 with a sampler, 2 a storage image
        elif op == OP_TYPE_SAMPLER:
            types[a[0]] = 'S'
        elif op == OP_TYPE_STRUCT:
            types[a[0]] = 'struct'
        elif op in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY):
            types[a[0]] = types.get(a[1], '?')
        elif op == OP_TYPE_POINTER:
            pointers[a[0]] = a[2]
        elif op == OP_VARIABLE:
            variables[a[1]] = (a[0], a[2])
        i += n
    out = []
    for var, (pointer, storage_class) in variables.items():
        if var not in bindings:
            continue
        pointee = pointers[pointer]
        kind = types.get(pointee, '?')
        if kind == 'struct':
            kind = 'B' if storage_class == SC_STORAGE_BUFFER or pointee in buffer_blocks else 'U'
        out.append((bindings[var], kind))
    return sorted(out)


def main():
    src = (RUNTIME / 'vk_fsr3fi.c').read_text(encoding='utf-8')
    tables = {m.group(1): [(int(b), k) for b, k in re.findall(r'\{(\d+), ([TIUB])\}', m.group(2))]
              for m in re.finditer(r'static const Bind (\w+)\[\] = \{(.*?)\};', src, re.S)}
    passes = re.findall(r'PASS\("([^"]+)", (\w+), (\w+), (\d)\)', src)
    bad = 0
    if len(passes) != 17:
        print(f'FAIL: {len(passes)} passes in vk_fsr3fi.c, 17 expected')
        bad += 1
    for name, spv, table, sampler in passes:
        got = declared(words_of(RUNTIME / f'{spv}.h'))
        want = sorted(tables[table] + ([(1000, 'S')] if sampler == '1' else []))
        extra = [d for d in got if d not in want]
        if extra or not got:
            bad += 1
            print(f'FAIL {name}:\n  the module declares {got}\n  {table} lists       {want}\n  not in the list: {extra}')
    widest = max(len(tables[t]) + int(s) for _, _, t, s in passes)
    m = re.search(r'#define MAX_BIND (\d+)', src)
    if not m or int(m.group(1)) + 1 < widest:
        bad += 1
        print(f'FAIL: MAX_BIND + 1 is below the widest pass ({widest} descriptors)')
    print(f'fsr3fi bindings: {len(passes)} passes, {bad} failures')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
