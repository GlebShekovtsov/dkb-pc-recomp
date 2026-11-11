"""CPU-only GS pipeline contract: real functions and reflected original modules.

Vulkan module/pipeline creation, device capabilities and timing are bounded spies;
no Vulkan loader is linked. Shader structure/accessors and interface reflection
are extracted verbatim from gfx_shader.c. Original probe programs are authored,
translated and spirv-val checked before the C checks run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import spirv_interface_test as interface
import vk_geometry_probe_test as probe
import vk_geometry_passthrough_probe_test as portable_probe
import geometry_shader_fixture as original

ROOT = Path(__file__).resolve().parents[1]


def run(command, log, timeout=120, input=None):
    result = subprocess.run([str(x) for x in command], input=input, capture_output=True, text=True, timeout=timeout)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{command[0]} exited {result.returncode}; see {log}\n{result.stdout}{result.stderr}')
    return result.stdout


def shader_adapter():
    source = (ROOT / 'runtime/gfx_shader.c').read_text()
    struct = source[source.index('struct GfxShader {'):source.index('\n};', source.index('struct GfxShader {')) + 3]
    start = source.index('enum { OP_ENTRY_POINT')
    enum = source[start:source.index('};', start) + 2]
    names = ['static void reflect_spirv(', 'const uint32_t *gfx_shader_spirv(',
             'int gfx_shader_stage(', 'const GfxShaderGeometry *gfx_shader_geometry(',
             'int gfx_shader_resources(', 'uint32_t gfx_shader_inputs(',
             'uint32_t gfx_shader_outputs(', 'const uint8_t *gfx_shader_input_id(']
    return '\n'.join([struct, enum] + [interface.function(source, name) for name in names])


def resources(out, names):
    text = ['static void fixture_resources(GfxShader *s, const char *name) {']
    for name in names:
        report = json.loads((out / f'{name}.res.json').read_text())
        text.append(f'if (!strcmp(name,"{name}")) {{')
        if report['stage'] == 'gs':
            g = report['geometry']
            input = ('points', 'lines', 'lines_adjacency', 'triangles', 'triangles_adjacency').index(report['gs_input']) + 1
            output = ('points', 'line_strip', 'triangle_strip').index(g['output_topology']) + 1
            text.append(f's->geometry=(GfxShaderGeometry){{{report["geometry_version"]},{input},{g["input_vertices"]},{output},'
                        f'{g["invocations"]},{g["max_output_vertices"]},{int(g["passthrough"])}}};')
        for kind, key in enumerate(('cbufs', 'sbufs', 'textures', 'images')):
            entries = report[key]
            if not entries:
                continue
            text.append(f'static GfxShaderRes r{kind}[] = {{')
            for entry in entries:
                fields = {'set': 0, 'binding': entry['binding'], 'count': entry.get('count', 1),
                          'slot': entry.get('slot', -1), 'flags': entry.get('flags', 0),
                          'cbuf_slot': entry.get('cb_slot', -1), 'handle': entry.get('cb_offset', 0)}
                text.append('{' + ','.join(f'.{k}={v}' for k, v in fields.items()) + '},')
            text.append(f'}}; s->res[{kind}]=r{kind}; s->nres[{kind}]={len(entries)};')
        text.append('}')
    text.append('}')
    return '\n'.join(text)


def pipeline_source():
    source = (ROOT / 'runtime/vk_pipeline.c').read_text()
    prefix = source[:source.index('static Layout *layouts')]
    prefix += '\n#define FNV0 0xcbf29ce484222325ull\n'
    # Unexecuted legacy varying/vertex-input branches fail loudly. This fixture
    # intentionally verifies that ordinary GS pipelines do not take those paths.
    start = source.index('typedef struct {\n    uint32_t n_attr, n_bind;')
    prefix += source[start:source.index('} VtxInput;', start) + len('} VtxInput;')]
    prefix += '''
static void vertex_input(const VkPipeState *s,VtxInput *v) { (void)s;(void)v;abort(); }
static uint32_t vs_outs(const GfxShader *v,const GfxShader *f,GfxShaderIo io[32]) {(void)v;(void)f;(void)io;abort();}
// gfx_spirv_final's optional rewrites (screen UV, FMZ, HDR shoulder, NaN guard, dump, override) are off here: this fixture
// checks the geometry-chain module path only.
VkPipelineCreateFlags vk_pipeline_capture_flags(void) {return 0;}
static int screen_uv_stage(const GfxShader *src) {(void)src;return 0;}
uint32_t *spirv_screen_uv(const uint32_t *w,uint64_t n,uint64_t *out_n);
uint32_t *spirv_fmz(const uint32_t *w,uint64_t n,int vertex_only,uint64_t *out_n,unsigned *nm,unsigned *nf) {(void)w;(void)n;(void)vertex_only;(void)out_n;(void)nm;(void)nf;return NULL;}
static uint32_t *hdr_shoulder(const GfxShader *src,const uint32_t *w,uint64_t words,uint64_t *out_n) {(void)src;(void)w;(void)words;(void)out_n;return NULL;}
static uint32_t *nan_guard(const GfxShader *src,const uint32_t *w,uint64_t words,uint64_t *out_n) {(void)src;(void)w;(void)words;(void)out_n;return NULL;}
static void spv_dump(const uint32_t *w,uint64_t words,const GfxShader *src) {(void)w;(void)words;(void)src;}
static uint32_t *spv_override(const uint32_t *w,uint64_t words,uint64_t *out_n) {(void)w;(void)words;(void)out_n;return NULL;}
'''
    names = ['static uint64_t fnv(', 'static VkBlendFactor blend_factor(', 'static VkBlendOp blend_op(',
             'static VkPolygonMode polygon_mode(', 'VkPrimitiveTopology vk_pipeline_topology(',
             'uint8_t vk_geometry_input(', 'static int topology_class(', 'static int format_type(',
             'static VkFormatFeatureFlags format_features(', 'static VkFormat target_format(',
             'static uint32_t clip_cull_limit(',
             'static void pin_subgroup32(', 'static VkShaderModule make_module_linked(', 'static VkShaderModule make_module(']
    # The rewrites are spirv_rewrite.c's and the module chain is gfx_spirv.c's; both are cut out like the rest.
    rewrite = (ROOT / 'runtime/spirv_rewrite.c').read_text()
    rewrites = ['uint32_t *spirv_rebind(', 'uint32_t *spirv_set(', 'uint32_t *spirv_dual_source(',
                'uint32_t *spirv_alpha_test(', 'uint32_t *spirv_add_outputs(', 'uint32_t *spirv_fixup_linked(',
                'uint32_t *spirv_fixup(', 'int spirv_uses_subgroup_ops(']
    prefix += '\n'.join(interface.function(rewrite, name) for name in rewrites) + '\n'
    prefix += interface.function((ROOT / 'runtime/gfx_spirv.c').read_text(), 'uint32_t *gfx_spirv_final(') + '\n'
    prefix += '\n'.join(interface.function(source, name) for name in names)
    # Contiguous original definitions: geometry IDs/validation and descriptor merge.
    start = source.index('typedef struct {', source.index('static VkShaderModule make_module('))
    prefix += source[start:source.index('static VkDescriptorUpdateTemplate make_tpl(', start)]
    start = source.index('typedef struct {                       // normalized:')
    prefix += source[start:source.index('// ---- graphics pipeline library', start)]
    start = source.index('typedef struct Job {')
    prefix += source[start:source.index('// Appends the pending records', start)]
    return prefix


def cull_module(use):
    """Original complete point GS with an arrayed Clip[8]/Cull[1] input block."""
    w = [0x07230203, 0x00010300, 0, 100, 0]
    def op(code, *args):
        w.extend([(len(args) + 1) << 16 | code, *args])
    for cap in (1, 2, 32, 33):
        op(17, cap)
    if use == 'ext_literal':
        op(11, 50, *struct.unpack('<4I', b'GLSL.std.450\0\0\0\0'))
    op(14, 0, 1)
    op(15, 3, 30, 0x6e69616d, 0, 14, 17)
    for mode, args in ((19, ()), (0, (1,)), (26, (1,)), (27, ())):
        op(16, 30, mode, *args)
    op(71, 11, 2)
    for member, builtin in ((0, 0), (1, 3), (2, 4)):
        op(72, 11, member, 11, builtin)
    op(71, 17, 11, 0)
    if use.endswith('dynamic'):
        op(71, 44, 11, 8)  # InvocationId, bounded to zero by Invocations1.
        # Append this interface variable to the original entry point.
        entry = next(i for i in range(5, len(w)) if w[i] & 65535 == 15)
        end = entry + (w[entry] >> 16)
        w[entry] += 1 << 16
        w.insert(end, 44)
    op(19, 1); op(33, 2, 1); op(22, 3, 32); op(23, 4, 3, 4); op(21, 5, 32, 0)
    for id, value in ((6, 0), (7, 1), (8, 8), (21, 2)):
        op(43, 5, id, value)
    op(28, 9, 3, 8); op(28, 10, 3, 7); op(30, 11, 4, 9, 10); op(28, 12, 11, 7)
    op(32, 13, 1, 12); op(59, 13, 14, 1); op(32, 15, 1, 3)
    op(32, 16, 3, 4); op(59, 16, 17, 3); op(32, 22, 1, 11)
    op(43, 3, 18, 0); op(43, 3, 19, 0x3f800000); op(44, 4, 20, 18, 18, 18, 19)
    if use.endswith('dynamic'):
        op(32, 45, 1, 5); op(59, 45, 44, 1)
    op(54, 1, 30, 0, 2); op(248, 31)
    if use.endswith('dynamic'):
        op(61, 5, 43, 44)
        op(65, 15, 41, 14, 6, 7 if use.startswith('clip') else 21, 43); op(61, 3, 42, 41)
    elif use == 'ext_literal':
        # GLSL.std.450 Cos is instruction 14, the input block's id: a literal, not a use.
        op(12, 3, 46, 50, 14, 19)
    elif use == 'clip':
        op(65, 15, 41, 14, 6, 7, 6); op(61, 3, 42, 41)
    elif use == 'clip_alias':
        op(83, 13, 40, 14); op(65, 15, 41, 40, 6, 7, 6); op(61, 3, 42, 41)
    elif use == 'clip_derived':
        op(65, 22, 40, 14, 6); op(65, 15, 41, 40, 7, 6); op(61, 3, 42, 41)
    elif use == 'direct':
        op(65, 15, 41, 14, 6, 21, 6); op(61, 3, 42, 41)
    elif use == 'alias':
        op(83, 13, 40, 14); op(65, 15, 41, 40, 6, 21, 6); op(61, 3, 42, 41)
    elif use == 'whole':
        op(65, 22, 40, 14, 6); op(61, 11, 42, 40)
    elif use == 'derived':
        op(65, 22, 40, 14, 6); op(65, 15, 41, 40, 21, 6); op(61, 3, 42, 41)
    op(62, 17, 20); op(218); op(219); op(253); op(56)
    return struct.pack('<' + 'I' * len(w), *w)


def primitive_id_fragment(data):
    """Add a real PrimitiveId read to our original FS; no captured instructions."""
    old = list(struct.unpack('<' + 'I' * (len(data) // 4), data))
    bound = old[3]
    int_type = next(o[0] for op, o in original.instructions(data) if op == 21 and o[1:] == (32, 1))
    pointer, variable, loaded = range(bound, bound + 3)
    result = old[:5] + [2 << 16 | 17, 2]  # FS PrimitiveId requires Geometry capability.
    result[3] += 3
    decorated = defined = read = label = False
    i = 5
    while i < len(old):
        size, code = old[i] >> 16, old[i] & 65535
        instruction = old[i:i + size]
        if not decorated and code in (19, 20, 21, 22):
            result += [4 << 16 | 71, variable, 11, 7, 3 << 16 | 71, variable, 14]
            decorated = True
        if not defined and code == 54:
            result += [4 << 16 | 32, pointer, 1, int_type,
                       4 << 16 | 59, pointer, variable, 1]
            defined = True
        if code == 15:
            instruction[0] += 1 << 16
            instruction.append(variable)
        if label and not read and code not in (59, 8, 317):
            result += [4 << 16 | 61, int_type, loaded, variable]
            read = True
        result += instruction
        if code == 248:
            label = True
        i += size
    assert decorated and defined and read
    return struct.pack('<' + 'I' * len(result), *result)


def input_block_variant(data, flat=False):
    """Original valid GS controls: reordered members or a Flat builtin member."""
    types, pointers, arrays, variables, builtins = {}, {}, {}, {}, {}
    for op, a in original.instructions(data):
        if op == 30: types[a[0]] = a[1:]
        elif op == 32: pointers[a[0]] = a[2]
        elif op == 28: arrays[a[0]] = a[1]
        elif op == 59 and a[2] == 1: variables[a[1]] = a[0]
        elif op == 72 and a[2] == 11: builtins[a[0], a[1]] = a[3]
    blocks = [(var, arrays[pointers[pointer]]) for var, pointer in variables.items()
              if pointers[pointer] in arrays and arrays[pointers[pointer]] in types]
    assert len(blocks) == 1
    variable, block = blocks[0]
    assert (builtins[block, 0], builtins[block, 1]) == (0, 1)
    assert not any(op in (65, 66) and a[2] == variable for op, a in original.instructions(data))
    result = list(struct.unpack('<5I', data[:20]))
    flat_added = False
    for op, a in original.instructions(data):
        a = list(a)
        if flat and not flat_added and 19 <= op <= 39:
            result += [4 << 16 | 72, block, 0, 14]
            flat_added = True
        if not flat and op == 30 and a[0] == block:
            a[1], a[2] = a[2], a[1]
        elif not flat and op in (6, 72) and a[0] == block and a[1] in (0, 1):
            a[1] ^= 1
        result += [(len(a) + 1) << 16 | op] + a
    return struct.pack('<' + 'I' * len(result), *result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', required=True)
    parser.add_argument('--deps', type=Path, required=True)
    parser.add_argument('--translator', type=Path, required=True)
    parser.add_argument('--spirv-val', required=True)
    parser.add_argument('--opt', choices=('0', '2'), default='2')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    author = args.out / 'author.c'
    author.write_text(probe.cpu_source())
    author_exe = args.out / 'author.exe'
    run([args.cc, '-std=c11', '-O2', author, '-o', author_exe], args.out / 'author-compile.log')
    run([author_exe.resolve(), args.out.resolve()], args.out / 'author.log')
    portable_author = args.out / 'portable-author.c'
    portable_author.write_text(portable_probe.cpu_source())
    portable_exe = args.out / 'portable-author.exe'
    run([args.cc, '-std=c11', '-O2', '-I', ROOT / 'runtime', portable_author, '-o', portable_exe],
        args.out / 'portable-author-compile.log')
    run([portable_exe.resolve(), args.out.resolve(), '1'], args.out / 'portable-author.log')
    jobs = []
    for stage, name in ((0, 'vs'), (1, 'ps'), (2, 'gs')):
        job = {'key': name, 'stage': name, 'code': str((args.out / f'stage{stage}.code').resolve()), 'c1': ''}
        if stage == 0:
            job['attr'] = {'0': 'float', '1': 'float'}
        if stage == 2:
            job.update(geometry_version=1, gs_input='points')
        jobs.append(job)
    for stage, name in ((0, 'vs_ubo'), (1, 'ps_ubo')):
        code = (args.out / f'stage{stage}.code').read_bytes()
        words = struct.unpack('<' + 'Q' * ((len(code) - 80) // 8), code[80:])
        ops = [word for i, word in enumerate(words) if i % 4]
        ops = ops[:ops.index(original.EXIT)]
        ops += ([original.movc(0, 3, 0), original.ast(0, 0x70, 1)] if stage == 0 else
                [original.movc(3, 3, 0)]) + [original.EXIT]
        path = args.out / f'{name}.code'
        path.write_bytes(code[:80] + original.framed(ops))
        job = {'key': name, 'stage': 'vs' if stage == 0 else 'ps', 'code': str(path.resolve()), 'c1': ''}
        if stage == 0:
            job['attr'] = {'0': 'float', '1': 'float'}
        jobs.append(job)
    for input in ('lines', 'lines_adjacency', 'triangles', 'triangles_adjacency'):
        jobs.append(dict(jobs[2], key='gs_' + input, gs_input=input))
    code = (args.out / 'stage2.code').read_bytes()
    words = struct.unpack('<' + 'Q' * ((len(code) - 80) // 8), code[80:])
    ops = [word for i, word in enumerate(words) if i % 4]
    ops = ops[:ops.index(original.EXIT)]
    live = [original.movi(12, 0)]
    for word in ops:
        if word >> 51 == original.out_vertex() >> 51 and (word >> 39) & 1:
            # Carry all eight actually used input clip distances into every
            # emitted vertex. They must never be deleted as boilerplate.
            for address in (0x2c0, 0x2d0):
                live += [original.ald(8, address, 12), original.ast(8, address)]
        live.append(word)
    path = args.out / 'gs_live_clip.code'
    path.write_bytes(code[:80] + original.framed(live + [original.EXIT]))
    jobs.append(dict(jobs[2], key='gs_live_clip', code=str(path.resolve())))
    code = (args.out / 'stage0.code').read_bytes()
    words = struct.unpack('<' + 'Q' * ((len(code) - 80) // 8), code[80:])
    ops = [word for i, word in enumerate(words) if i % 4]
    ops = ops[:ops.index(original.EXIT)]
    ops += [original.movi(register, 0) for register in range(8, 12)]
    ops += [original.ast(8, address) for address in (0x2c0, 0x2d0)] + [original.EXIT]
    path = args.out / 'vs_live_clip.code'
    path.write_bytes(code[:80] + original.framed(ops))
    jobs.append(dict(jobs[0], key='vs_live_clip', code=str(path.resolve())))
    jobs.extend(portable_probe.jobs(args.out))
    request = ''.join(json.dumps({'job': job, 'out': str(args.out.resolve())}) + '\n' for job in jobs)
    (args.out / 'request.jsonl').write_text(request)
    result = run([args.translator.resolve(), '--serve'], args.out / 'translate.log', input=request)
    if result.splitlines() != ['ok ' + job['key'] for job in jobs]:
        raise RuntimeError(f'translator rejected original programs: {result}')
    names = [job['key'] for job in jobs]
    # A valid GS producer at Location1 with FS consuming Location1. VS still
    # writes Location0: the legacy direct VS→FS repair would corrupt this path.
    for name, source, storage in (('gs_output1', 'gs', 3), ('ps_input1', 'ps', 1)):
        data = (args.out / f'{source}.spv').read_bytes()
        words = list(struct.unpack('<' + 'I' * (len(data) // 4), data))
        variables = {o[1]: o[2] for op, o in original.instructions(data) if op == 59}
        i = 5
        while i < len(words):
            size, op = words[i] >> 16, words[i] & 65535
            if op == 71 and size == 4 and words[i + 2] == 30 and variables.get(words[i + 1]) == storage:
                assert words[i + 3] == 0
                words[i + 3] = 1
            i += size
        (args.out / f'{name}.spv').write_bytes(struct.pack('<' + 'I' * len(words), *words))
        (args.out / f'{name}.res.json').write_bytes((args.out / f'{source}.res.json').read_bytes())
        names.append(name)
    (args.out / 'ps_primitive.spv').write_bytes(primitive_id_fragment((args.out / 'ps.spv').read_bytes()))
    (args.out / 'ps_primitive.res.json').write_bytes((args.out / 'ps.res.json').read_bytes())
    names.append('ps_primitive')
    (args.out / 'gs_reordered.spv').write_bytes(input_block_variant((args.out / 'gs.spv').read_bytes()))
    (args.out / 'gs_reordered.res.json').write_bytes((args.out / 'gs.res.json').read_bytes())
    names.append('gs_reordered')
    (args.out / 'gs_flat_builtin.spv').write_bytes(input_block_variant((args.out / 'gs.spv').read_bytes(), flat=True))
    (args.out / 'gs_flat_builtin.res.json').write_bytes((args.out / 'gs.res.json').read_bytes())
    names.append('gs_flat_builtin')
    for name in names:
        run([args.spirv_val, '--target-env', 'vulkan1.3', args.out / f'{name}.spv'], args.out / f'{name}-val.log')
    for use in ('unused', 'direct', 'alias', 'whole', 'derived', 'clip', 'clip_alias', 'clip_derived', 'clip_dynamic', 'cull_dynamic', 'ext_literal'):
        path = args.out / f'cull-{use}.spv'
        path.write_bytes(cull_module(use))
        run([args.spirv_val, '--target-env', 'vulkan1.3', path], args.out / f'cull-{use}-val.log')
    generated = args.out / 'pipeline.c'
    generated.write_text(pipeline_source() + '\n' + shader_adapter() + '\n' + resources(args.out, names) + '\n' +
                         (ROOT / 'tests/geometry_pipeline.c').read_text())
    exe = args.out / 'pipeline.exe'
    run([args.cc, '-std=gnu11', f'-O{args.opt}', '-ffunction-sections', '-fdata-sections',
         '-I', ROOT / 'runtime', '-I', args.deps / 'include', '-I', args.deps / 'include/vma',
         generated, '-Wl,--gc-sections,--exclude-all-symbols', '-o', exe], args.out / 'compile.log')
    output = run([exe.resolve(), args.out.resolve()], args.out / 'test.log')
    legacy = (args.out / 'legacy-key.bin').read_bytes()
    # Frozen from d9fe9f7's actual VkPipeState/KeyRec/rec_fill on the 64-bit ABI,
    # seeded with 0x5a, VS id=01*32, FS id=02*32. No new implementation mirror.
    if len(legacy) != 424 or hashlib.sha256(legacy).hexdigest() != '5c6280aa0184a67b236cc57d4ce5bdd8f554244cbdfc9dfd77f720be67077c2a':
        raise AssertionError('legacy version-3 key bytes changed')
    for index in (1, 2, 3):
        run([args.spirv_val, '--target-env', 'vulkan1.3', args.out / f'module{index}.spv'], args.out / f'module{index}-val.log')
        run([args.spirv_val, '--target-env', 'vulkan1.3', args.out / f'portable-module{index}.spv'],
            args.out / f'portable-module{index}-val.log')
    run([args.spirv_val, '--target-env', 'vulkan1.3', args.out / 'plain-vs.spv'], args.out / 'plain-vs-val.log')
    for use in ('unused', 'clip', 'ext_literal'):
        run([args.spirv_val, '--target-env', 'vulkan1.3', args.out / f'fixed-cull-{use}.spv'], args.out / f'fixed-cull-{use}-val.log')
    manifest = {'optimization': args.opt, 'result': output.strip(),
        'complete_spirv_validations': len(list(args.out.glob('*-val.log'))), 'providers': [
        'Vulkan module/pipeline creation spies; no loader or device',
        'owned translated modules plus exact production SPIR-V interface reflector',
        'resource reflection metadata populated from original shader JSON',
        'fixed generous device limits and disabled timing',
        'legacy vertex input/VS-FS union/format branches abort if reached'], 'source_sha256': {}}
    for name in ('runtime/vk.h', 'runtime/vk_pipeline.c', 'runtime/gfx.h', 'runtime/gfx_shader.c',
                 'runtime/vk_geometry_probe.c', 'runtime/nvn_geometry_control.h',
                 'tests/vk_geometry_passthrough_probe_test.py', 'tests/geometry_pipeline.c', 'tests/geometry_pipeline_test.py'):
        manifest['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (args.out / 'handoff.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output, end='')


if __name__ == '__main__':
    main()
