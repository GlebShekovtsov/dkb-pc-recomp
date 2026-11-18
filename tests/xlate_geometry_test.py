"""CPU-only GS protocol/emission tests with original Maxwell fixtures.

No Vulkan loader or GPU runs. The optional baseline compares existing VS/PS/CS
outputs against an unchanged translator; it is not used to change expectations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import unittest

import geometry_shader_fixture as fixture

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGIES = {'points': (19, 1), 'lines': (20, 2), 'lines_adjacency': (21, 4),
              'triangles': (22, 3), 'triangles_adjacency': (23, 6)}


class GeometrySpirv:
    """Inspect/evaluate only these original straight-line scalar-copy fixtures.

    Values are raw 32-bit words, so this is not a floating-point implementation.
    No branches, calls, arithmetic, storage buffers or other shader behavior are
    supported: an unexpected body opcode fails. Output memory is cleared after
    every EmitVertex, matching its undefined-after-emission lifetime.
    """
    def __init__(self, data):
        self.ops = list(fixture.instructions(data))
        self.constants = {o[1]: o[2] for op, o in self.ops if op == 43 and len(o) == 3}
        self.types = {o[0]: (op, o[1:]) for op, o in self.ops if 19 <= op <= 33}
        self.variables = {o[1]: (o[2], self.types[o[0]][1][1]) for op, o in self.ops if op == 59}
        self.locations = {o[0]: o[2] for op, o in self.ops if op == 71 and o[1] == 30}
        self.members = {(o[0], o[1]): o[3] for op, o in self.ops if op == 72 and o[2] == 11}
        self.entries = [o[1] for op, o in self.ops if op == 15 and o[0] == 3]
        assert len(self.entries) == 1
        functions = [i for i, (op, _) in enumerate(self.ops) if op == 54]
        assert len(functions) == 1 and self.ops[functions[0]][1][1] == self.entries[0]
        assert self.ops[-1][0] == 56
        self.body = self.ops[functions[0] + 1:-1]
        assert self.body[0][0] == 248 and self.body[-1][0] == 253
        assert sum(op == 248 for op, _ in self.body) == sum(op == 253 for op, _ in self.body) == 1

    def location(self, storage, location):
        matches = [var for var, (sc, _) in self.variables.items()
                   if sc == storage and self.locations.get(var) == location]
        assert len(matches) == 1
        return matches[0]

    def position(self, storage):
        matches = []
        for var, (sc, type_id) in self.variables.items():
            if sc != storage:
                continue
            if self.types[type_id][0] == 28:
                type_id = self.types[type_id][1][0]
            if self.types[type_id][0] == 30:
                matches += [(var, member) for (struct_id, member), builtin in self.members.items()
                            if struct_id == type_id and builtin == 0]
        assert len(matches) == 1
        return matches[0]

    def input_array_length(self, variable):
        storage, type_id = self.variables[variable]
        assert storage == 1 and self.types[type_id][0] == 28
        return self.constants[self.types[type_id][1][1]]

    def vec4_float(self, type_id):
        op, operands = self.types[type_id]
        return op == 23 and operands[1] == 4 and self.types[operands[0]] == (22, (32,))

    def run(self, positions, colors, output_location):
        values, memory = dict(self.constants), {}
        pointers = {var: (var,) for var in self.variables}
        input_position, input_member = self.position(1)
        output_position, output_member = self.position(3)
        output_color = self.location(3, output_location)
        point_members = [member for (type_id, member), builtin in self.members.items()
                         if type_id == self.variables[output_position][1] and builtin == 1]
        points = any(op == 16 and o[1:] == (27,) for op, o in self.ops)
        if points:
            assert len(point_members) == 1
        for vertex, position in enumerate(positions):
            for component, value in enumerate(position):
                memory[(input_position, vertex, input_member, component)] = value
        if colors:
            input_color = self.location(1, 6)
            for vertex, color in enumerate(colors):
                for component, value in enumerate(color):
                    memory[(input_color, vertex, component)] = value
        emitted, cuts, reads = [], [], []
        for op, o in self.body:
            if op in (248, 253):  # the single label and return
                continue
            if op == 65:  # OpAccessChain
                pointers[o[1]] = pointers[o[2]] + tuple(values[index] for index in o[3:])
            elif op == 61:  # OpLoad: an undefined or out-of-range leaf fails
                pointer = pointers[o[2]]
                values[o[1]] = memory[pointer]
                reads.append(pointer)
            elif op == 62:  # OpStore
                memory[pointers[o[0]]] = values[o[1]]
            elif op in (83, 124):  # scalar OpCopyObject / OpBitcast preserve bits
                values[o[1]] = values[o[2]]
            elif op == 218:
                emitted.append(dict(position=tuple(memory[(output_position, output_member, c)] for c in range(4)),
                                    color=tuple(memory[(output_color, c)] for c in range(4))))
                if points:
                    emitted[-1]['point_size'] = memory[(output_position, point_members[0])]
                memory = {p: value for p, value in memory.items() if self.variables[p[0]][0] != 3}
            elif op == 219:
                cuts.append(len(emitted))
            else:
                raise AssertionError('unsupported fixture body opcode ' + str(op))
        return emitted, cuts, reads


def command(path):
    return [str(args.dotnet), str(path)] if path.suffix == '.dll' else [str(path)]


def packet(key, code=None, **changes):
    path = args.out / (key + '.bin')
    path.write_bytes(fixture.program() if code is None else code)
    job = dict(key=key, stage='gs', code=str(path.resolve()), c1='', debug=False,
               geometry_version=1, gs_input='points', sbuf_align=256)
    job.update(changes)
    return job


class GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args.out.mkdir(parents=True, exist_ok=True)
        cls.env = {k: v for k, v in os.environ.items() if k.upper() in {
            'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA',
            'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMDATA', 'SYSTEMDRIVE'}}
        cls.env.update(DOTNET_ROOT=str(args.dotnet.parent), DOTNET_PROCESSOR_COUNT='2',
                       DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_GENERATE_ASPNET_CERTIFICATE='false',
                       DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1', DOTNET_NOLOGO='1', DOTNET_ADD_GLOBAL_TOOLS_TO_PATH='false')

    def translate(self, jobs, valid=True, translator=None):
        run = subprocess.run(command(translator or args.xlate) + ['--serve'], env=self.env,
                             input=''.join(json.dumps(dict(job=j, out=str(args.out.resolve()))) + '\n' for j in jobs),
                             capture_output=True, text=True, timeout=60)
        (args.out / (jobs[0]['key'] + '.service.log')).write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stderr)
        replies = run.stdout.splitlines()
        self.assertEqual(len(replies), len(jobs), run.stdout + run.stderr)
        for job, reply in zip(jobs, replies):
            if valid:
                self.assertEqual(reply, 'ok ' + job['key'], reply)
                spv = args.out / (job['key'] + '.spv')
                validation = subprocess.run([str(args.spirv_val), '--target-env', 'vulkan1.3', str(spv)],
                                            capture_output=True, text=True, timeout=15)
                self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
            else:
                self.assertTrue(reply.startswith('fail ' + job['key'] + ' '), reply)
                self.assertFalse((args.out / (job['key'] + '.spv')).exists())
                self.assertFalse((args.out / (job['key'] + '.res.json')).exists())
        return replies

    def reflection(self, key):
        return json.loads((args.out / (key + '.res.json')).read_text())

    def test_topologies_emission_and_store(self):
        jobs = [packet(topology + ('_store' if store else '_emit'), fixture.program(store=store), gs_input=topology)
                for topology in TOPOLOGIES for store in (False, True)]
        self.translate(jobs)
        for job in jobs:
            data = (args.out / (job['key'] + '.spv')).read_bytes()
            instructions = list(fixture.instructions(data))
            self.assertTrue(any(op == 15 and o[0] == 3 for op, o in instructions))
            modes = [o[1:] for op, o in instructions if op == 16]
            self.assertIn((TOPOLOGIES[job['gs_input']][0],), modes)
            reflection = self.reflection(job['key'])
            self.assertEqual(reflection['stage'], 'gs')
            self.assertEqual(reflection['geometry_version'], 1)
            self.assertEqual(reflection['gs_input'], job['gs_input'])
            self.assertEqual(reflection['geometry'], dict(input_vertices=TOPOLOGIES[job['gs_input']][1],
                             output_topology='points', invocations=1, max_output_vertices=1, passthrough=False))
            for mode in ((0, 1), (26, 1), (27,)):
                self.assertIn(mode, modes)
            self.assertEqual(sum(op == 218 for op, _ in instructions), 1)
            self.assertEqual(sum(op == 219 for op, _ in instructions), 1)
            self.assertNotIn(b'SPV_NV_geometry_shader_passthrough', data)
            variables = {o[1]: o[2] for op, o in instructions if op == 59}
            self.assertTrue(any(op == 71 and o[1:] == (30, 0) and variables.get(o[0]) == 3
                                for op, o in instructions), 'missing red color output Location0')
            self.assertEqual(reflection['notes'], [])
            if job['key'].endswith('_store'):
                self.assertEqual(len(reflection['sbufs']), 1)
                buffer = reflection['sbufs'][0]
                self.assertEqual((buffer['cb_slot'], buffer['cb_offset'], buffer['flags']), (3, 0, 1))
                bindings = {o[0]: o[2] for op, o in instructions if op == 71 and o[1] == 33}
                buffers = {var for var, binding in bindings.items() if binding == buffer['binding']}
                chains = {o[1]: o[2] for op, o in instructions if op == 65}
                self.assertEqual(sum(op == 62 and chains.get(o[0]) in buffers for op, o in instructions), 1)
                self.assertTrue(any(op == 43 and o[-1] == 0x1234abcd for op, o in instructions))
            else:
                self.assertEqual(reflection['sbufs'], [])

    def test_strip_outputs_invocations_and_vertex_limits(self):
        jobs = [packet(output, fixture.strip_program(output)) for output in ('line_strip', 'triangle_strip')]
        self.translate(jobs)
        for job in jobs:
            with self.subTest(output=job['key']):
                triangle = job['key'] == 'triangle_strip'
                count, invocations, mode = (3, 3, 29) if triangle else (2, 2, 28)
                module = GeometrySpirv((args.out / (job['key'] + '.spv')).read_bytes())
                modes = [o[1:] for op, o in module.ops if op == 16]
                for expected in ((19,), (mode,), (0, invocations), (26, count * 2)):
                    self.assertIn(expected, modes)
                self.assertEqual(self.reflection(job['key'])['geometry'],
                                 dict(input_vertices=1, output_topology=job['key'], invocations=invocations,
                                      max_output_vertices=count * 2, passthrough=False))
                self.assertEqual(sum(op == 218 for op, _ in module.ops), count * 2)
                self.assertEqual(sum(op == 219 for op, _ in module.ops), 2)
                emitted, cuts, reads = module.run([], [], 0)
                expected = []
                base_positions = ((-0.75, -0.5, 0, 1), (0.75, -0.5, 0, 1), (0, 0.5, 0, 1))
                for primitive in range(2):
                    for vertex, p in enumerate(base_positions[:count]):
                        expected.append(dict(position=tuple(fixture.bits(v) for v in (p[0], p[1] + primitive * 0.25, 0, 1)),
                            color=tuple(fixture.bits(v) for v in (0.25 * (vertex + 1), 0.25 * (primitive + 1), 0.5, 1))))
                self.assertEqual(emitted, expected)
                self.assertEqual(cuts, [count, count * 2])
                self.assertEqual(reads, [])
                self.assertEqual(self.reflection(job['key'])['notes'], [])

    def test_per_vertex_input_arrays_and_copy_semantics(self):
        selections = {'points': ('points', [0]), 'lines': ('line_strip', [1, 0]),
                      'lines_adjacency': ('line_strip', [2, 1]), 'triangles': ('triangle_strip', [2, 0, 1]),
                      'triangles_adjacency': ('triangle_strip', [4, 0, 2])}
        jobs = [packet('input_' + name, fixture.input_program(output, vertices), gs_input=name)
                for name, (output, vertices) in selections.items()]
        self.translate(jobs)
        for job in jobs:
            with self.subTest(topology=job['gs_input']):
                topology = job['gs_input']
                output, vertices = selections[topology]
                input_count = TOPOLOGIES[topology][1]
                module = GeometrySpirv((args.out / (job['key'] + '.spv')).read_bytes())
                input_color = module.location(1, 6)
                output_color = module.location(3, 2)
                input_position, input_member = module.position(1)
                self.assertEqual(module.input_array_length(input_color), input_count)
                self.assertEqual(module.input_array_length(input_position), input_count)
                array_type = module.variables[input_color][1]
                self.assertTrue(module.vec4_float(module.types[array_type][1][0]))
                self.assertTrue(module.vec4_float(module.variables[output_color][1]))
                modes = [o[1:] for op, o in module.ops if op == 16]
                for expected in ((TOPOLOGIES[topology][0],), ({'points': 27, 'line_strip': 28, 'triangle_strip': 29}[output],),
                                 (0, 2), (26, len(vertices))):
                    self.assertIn(expected, modes)
                self.assertEqual(self.reflection(job['key'])['geometry'],
                                 dict(input_vertices=input_count, output_topology=output, invocations=2,
                                      max_output_vertices=len(vertices), passthrough=False))
                positions = [tuple(fixture.bits(v) for v in (-0.75 + i * 0.125, -0.5 + i * 0.0625, i * 0.03125, 1))
                             for i in range(input_count)]
                colors = [tuple(fixture.bits((i * 4 + component + 1) / 32) for component in range(4))
                          for i in range(input_count)]
                emitted, cuts, reads = module.run(positions, colors, 2)
                expected_vertices = [dict(position=positions[v], color=colors[v]) for v in vertices]
                if output == 'points':
                    for vertex in expected_vertices:
                        vertex['point_size'] = fixture.bits(1)
                self.assertEqual(emitted, expected_vertices)
                self.assertEqual(cuts, [len(vertices)])
                self.assertEqual(sum(op == 218 for op, _ in module.ops), len(vertices))
                self.assertEqual(sum(op == 219 for op, _ in module.ops), 1)
                for root, suffix in ((input_position, (input_member,)), (input_color, ())):
                    self.assertEqual([p for p in reads if p[0] == root],
                                     [(root, vertex, *suffix, component) for vertex in vertices for component in range(4)])
                self.assertEqual(self.reflection(job['key'])['notes'], [])
                self.assertEqual(self.reflection(job['key'])['sbufs'], [])

    def test_version_is_required_exact_integer_one(self):
        jobs = []
        for index, version in enumerate((None, 0, 2, -1, 1.0, True, '1', {}, [])):
            jobs.append(packet('bad_version_' + str(index), geometry_version=version))
        missing = packet('missing_version')
        del missing['geometry_version']
        self.translate([missing, *jobs], valid=False)

    def test_topology_is_required_exact_supported_string(self):
        jobs = [packet('bad_topology_' + str(i), gs_input=value)
                for i, value in enumerate((None, '', 'POINTS', 'triangle_strip', 'quads', 1, True, {}, []))]
        missing = packet('missing_topology')
        del missing['gs_input']
        self.translate([missing, *jobs], valid=False)

    def test_passthrough_is_explicitly_unsupported(self):
        jobs = [packet('passthrough_' + str(store), fixture.program(store=store, passthrough=True)) for store in (False, True)]
        replies = self.translate(jobs, valid=False)
        self.assertTrue(all('passthrough' in reply.lower() and 'linkage' in reply.lower() for reply in replies))

    def test_header_stage_size_and_geometry_metadata(self):
        jobs = [packet('header_' + stage, fixture.program(stage=stage)) for stage in ('vs', 'ps')]
        jobs += [packet('short_' + str(size), fixture.program()[:size]) for size in (0, 4, 79, 80, 88)]
        for name, offset, value in (('zero_invocations', 8, 0), ('zero_vertices', 16, 0), ('bad_output', 12, 2 << 24)):
            code = bytearray(fixture.program())
            struct.pack_into('<I', code, offset, value)
            jobs.append(packet(name, bytes(code)))
        self.translate(jobs, valid=False)

    def test_geometry_fields_and_sph_cannot_hide_under_other_stages(self):
        jobs = []
        for stage in ('vs', 'ps', 'cs'):
            code = fixture.framed([fixture.EXIT]) if stage == 'cs' else fixture.program(stage=stage)
            for field in ('geometry_version', 'gs_input'):
                job = packet('wrong_stage_' + stage + '_' + field, code, stage=stage)
                del job['geometry_version' if field == 'gs_input' else 'gs_input']
                jobs.append(job)
        disguised = packet('disguised_geometry', stage='vs')
        del disguised['geometry_version'], disguised['gs_input']
        self.translate([*jobs, disguised], valid=False)

    def test_existing_non_geometry_requests(self):
        jobs = []
        for stage in ('vs', 'ps', 'cs'):
            code = fixture.framed([fixture.EXIT]) if stage == 'cs' else fixture.program(stage=stage)
            job = packet('legacy_' + stage, code, stage=stage, local=[1, 1, 1], shared=0)
            del job['geometry_version'], job['gs_input']
            jobs.append(job)
        self.translate(jobs)
        for job in jobs:
            res = self.reflection(job['key'])
            self.assertNotIn('geometry_version', res)
            self.assertNotIn('gs_input', res)
            self.assertNotIn('geometry', res)
        if args.baseline:
            controls = [dict(job, key=job['key'] + '_baseline') for job in jobs]
            self.translate(controls, translator=args.baseline)
            for job, baseline in zip(jobs, controls):
                self.assertEqual((args.out / (job['key'] + '.spv')).read_bytes(),
                                 (args.out / (baseline['key'] + '.spv')).read_bytes())
                old, new = self.reflection(baseline['key']), self.reflection(job['key'])
                old.pop('key'); new.pop('key')
                self.assertEqual(old, new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xlate', required=True, type=Path)
    parser.add_argument('--dotnet', type=Path, default=Path('dotnet'))
    parser.add_argument('--spirv-val', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/tests/xlate-geometry')
    args, rest = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = unittest.main(argv=[__file__] + rest, exit=False).result
    (args.out / 'result.json').write_text(json.dumps(dict(tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped),
        program_sha256=hashlib.sha256((ROOT / 'tools/xlate/Program.cs').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256((ROOT / 'tests/geometry_shader_fixture.py').read_bytes()).hexdigest(),
        translator_sha256=hashlib.sha256(args.xlate.read_bytes()).hexdigest(), gpu_executed=False), indent=2))
    raise SystemExit(not result.wasSuccessful())
