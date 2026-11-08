"""CPU-only ABI/reflection/control-flow checks for opt-in storage candidates.

Uses the original shaders in storage_candidates_fixture.py. This does not replace
the separate functional GPU check: descriptor resolution belongs to the host.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import unittest
from xml.sax.saxutils import escape

import storage_candidates_fixture as original

ROOT = Path(__file__).resolve().parents[1]


def encoded(ops):
    words = []
    for i in range(0, len(ops), 3):
        batch = ops[i:i + 3]
        words += [original.SCHED] + batch + [0] * (3 - len(batch))
    return struct.pack('<' + 'Q' * len(words), *words)


def packet(key, code, **changes):
    path = args.out / (key + '.bin')
    path.write_bytes(code)
    job = dict(key=key, stage='cs', code=str(path), debug=False,
               c1=(struct.pack('<Q', 32) + bytes(248)).hex(), local=[1, 1, 1], shared=0,
               c0=[0, 0, 0, 0], c0_version=2, c0_roots=[8, 0, 0, 0, 0, 0, 0, 0], sbuf_align=256)
    job.update(changes)
    return job


def instructions(data):
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203
    offset = 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 65535
        assert count and offset + count <= len(words)
        yield opcode, words[offset + 1:offset + count]
        offset += count


def evaluate_wide(bits, inline, memory_offset, input_pointer):
    """Evaluate only the original fixture's integer/address/copy instructions.

    All addresses are dictionary keys. Compare every initialized input/output
    word so wrong register overlap, a missing vector word or an extra write is
    visible without dereferencing host memory or running a shader.
    """
    output_pointer = 0x500000000
    cbuf = {(0, 4): input_pointer & 0xffffffff, (0, 5): input_pointer >> 32,
            (0, 8): output_pointer & 0xffffffff, (0, 9): output_pointer >> 32, (1, 0): 32, (1, 1): 0}
    memory = {base + offset: (0xa5000000 + offset) & 0xffffffff
              for base in (input_pointer, output_pointer) for offset in range(-32, 128, 4)}
    for i, value in enumerate(original.WIDE_VALUES[:bits // 32]):
        memory[input_pointer + 32 + memory_offset + i * 4] = value
    expected = dict(memory)
    for i, value in enumerate(original.WIDE_VALUES[:bits // 32]):
        expected[output_pointer + memory_offset + i * 4] = value
    registers, carry = {}, 0
    for op in original.wide_instructions(bits, inline, memory_offset):
        dst, a, prefix = op & 255, (op >> 8) & 255, op >> 51
        if op == original.EXIT:
            return memory, expected, tuple(registers[8 + i] for i in range(bits // 32))
        if prefix == 0x4C98000000000000 >> 51:
            registers[dst] = cbuf[((op >> 34) & 31, (op >> 20) & 0x3fff)]
        elif op >> 56 == 1:
            registers[dst] = (op >> 20) & 0xffffffff
        elif prefix in (0x4C10000000000000 >> 51, 0x5C10000000000000 >> 51):
            b = (cbuf[((op >> 34) & 31, (op >> 20) & 0x3fff)]
                 if prefix == 0x4C10000000000000 >> 51 else registers[(op >> 20) & 255])
            value = registers[a] + b + (carry if op & (1 << 43) else 0)
            registers[dst] = value & 0xffffffff
            if op & (1 << 47):
                carry = int(value > 0xffffffff)
        elif prefix in (0xEED0000000000000 >> 51, 0xEED8000000000000 >> 51):
            size = (op >> 48) & 7
            assert size in (5, 6) and dst == 8 and op & (1 << 45)
            immediate = (op >> 20) & 0xffffff
            if immediate & (1 << 23):
                immediate -= 1 << 24
            address = (registers[a] | registers[a + 1] << 32) + immediate
            assert address % (8 if size == 5 else 16) == 0
            for i in range(2 if size == 5 else 4):
                if prefix == 0xEED0000000000000 >> 51:
                    registers[dst + i] = memory[address + i * 4]
                else:
                    memory[address + i * 4] = registers[dst + i]
        else:
            raise AssertionError(f'unknown original vector instruction {op:016x}')
    raise AssertionError('original vector fixture has no EXIT')


def guarded_memory_blocks(data, storage_bindings):
    """Find actual guarded SSBO accesses, not a copy of candidate arithmetic.

    Every SSBO operation must be in an accepting block which returns, entered
    only by a real conditional branch whose dataflow includes OpArrayLength
    and the full high/low subtraction. Scalar OpSelect is permitted for flags;
    descriptor memory accesses must never be speculative select operands.
    """
    ops = list(instructions(data))
    bindings = {o[0]: o[2] for op, o in ops if op == 71 and o[1] == 33}
    buffers = {var for var, binding in bindings.items() if binding in storage_bindings}
    pointers = {o[1]: o[2] for op, o in ops if op == 65}
    definitions = {}
    for op, o in ops:
        if op in (43, 57, 61, 65, 68, 124, 128, 130, 167, 169, 170, 176, 194, 199):
            definitions[o[1]] = (op, o[2:])

    def closure(value, seen=None):
        seen = set() if seen is None else seen
        if value in seen or value not in definitions:
            return []
        seen.add(value)
        op, sources = definitions[value]
        if op == 43:
            return [op]  # The literal payload is not a result ID.
        # OpLoad uses a pointer. Its value may come from a function-local store;
        # high and low subtraction results are already explicit in the condition.
        return [op] + [x for source in sources for x in closure(source, seen)]

    functions, current, block = [], None, None
    for op, o in ops:
        if op == 54:
            current = {}; functions.append(current); block = None
        elif op == 248:
            block = o[0]; current[block] = []
        elif op == 56:
            current = None
        elif current is not None and block is not None:
            current[block].append((op, o))
    count = 0
    queried = {bindings[o[2]] for op, o in ops if op == 68}
    assert queried == storage_bindings, (queried, storage_bindings)
    for function in functions:
        incoming = {}
        for source, body in function.items():
            for op, o in body:
                if op == 250:
                    incoming.setdefault(o[1], []).append((source, o[0]))
                    incoming.setdefault(o[2], []).append((source, o[0]))
                elif op == 249:
                    incoming.setdefault(o[0], []).append((source, None))
        for label, body in function.items():
            memory = []
            for op, o in body:
                pointer = o[2] if op == 61 or 229 <= op <= 242 else o[0] if op == 62 else None
                if pointers.get(pointer) in buffers:
                    memory.append((op, o))
            if not memory:
                continue
            assert len(memory) == 1, 'a candidate can perform only one memory operation'
            assert body[-1][0] in (253, 254), 'a selected candidate must return before alternatives'
            guards = incoming.get(label, [])
            assert len(guards) == 1 and guards[0][1] is not None, 'SSBO access lacks a branch guard'
            condition = closure(guards[0][1])
            assert 68 in condition and 176 in condition and 170 in condition
            assert condition.count(130) >= 2, 'high-word subtraction/borrow was lost'
            count += 1
    return count


class CandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args.out.mkdir(parents=True, exist_ok=True)
        cls.env = {k: v for k, v in os.environ.items() if k.upper() in {
            'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA',
            'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMDATA', 'SYSTEMDRIVE'}}
        cls.env.update(DOTNET_ROOT=str(args.dotnet.parent), DOTNET_PROCESSOR_COUNT='2',
                       DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_GENERATE_ASPNET_CERTIFICATE='false',
                       DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1', DOTNET_NOLOGO='1',
                       DOTNET_CLI_HOME=str(args.out / 'dotnet-home'), MSBUILDDISABLENODEREUSE='1',
                       DOTNET_CLI_USE_MSBUILD_SERVER='0')
        cls.command = [str(args.dotnet), str(args.xlate)] if args.xlate.suffix == '.dll' else [str(args.xlate)]

    def translate(self, jobs, valid=True):
        run = subprocess.run(self.command + ['--serve'], env=self.env,
                             input=''.join(json.dumps(dict(job=j, out=str(args.out))) + '\n' for j in jobs),
                             capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        (args.out / (jobs[0]['key'] + '.service.log')).write_text(run.stdout + run.stderr)
        replies = run.stdout.splitlines()
        self.assertEqual(len(replies), len(jobs), run.stdout + run.stderr)
        for job, reply in zip(jobs, replies):
            if valid:
                self.assertEqual(reply, 'ok ' + job['key'])
                validation = subprocess.run([str(args.spirv_val), '--target-env', 'vulkan1.3',
                                             str(args.out / (job['key'] + '.spv'))],
                                            capture_output=True, text=True, timeout=30)
                self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
            else:
                self.assertTrue(reply.startswith('fail ' + job['key'] + ' '), reply)
                self.assertFalse((args.out / (job['key'] + '.res.json')).exists())
        return replies

    def reflection(self, key):
        return json.loads((args.out / (key + '.res.json')).read_text())

    def test_default_and_explicit_zero_keep_legacy_code(self):
        jobs = [packet('default', original.shader_bytes(True)),
                packet('zero', original.shader_bytes(True), storage_candidates_version=0)]
        self.translate(jobs)
        self.assertEqual((args.out / 'default.spv').read_bytes(), (args.out / 'zero.spv').read_bytes())
        for job in jobs:
            res = self.reflection(job['key'])
            self.assertEqual(res['storage_candidates_version'], 0)
            self.assertTrue(all(not (x['flags'] & 0x80) for x in res['sbufs']))
            self.assertFalse(any(op == 68 for op, _ in instructions((args.out / (job['key'] + '.spv')).read_bytes())))
        if args.baseline:
            run = subprocess.run([str(args.baseline), '--serve'], env=self.env,
                                 input=json.dumps(dict(job=dict(jobs[0], key='baseline'), out=str(args.out))) + '\n',
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.stdout.strip(), 'ok baseline', run.stdout + run.stderr)
            self.assertEqual((args.out / 'baseline.spv').read_bytes(), (args.out / 'default.spv').read_bytes())

    def test_both_orientations_emit_ordered_bounded_candidates(self):
        jobs = [packet('c0_pointer', original.shader_bytes(True), storage_candidates_version=1),
                packet('c1_pointer', original.c1_pointer_bytes(True), storage_candidates_version=1,
                       c1=(struct.pack('<Q', 0x600001000) + bytes(248)).hex())]
        self.translate(jobs)
        for job, preferred in zip(jobs, ((1, 0), (0, 4))):
            res = self.reflection(job['key'])
            self.assertEqual(res['storage_candidates_version'], 1)
            self.assertEqual((res['sbufs'][0]['cb_slot'], res['sbufs'][0]['cb_offset']), preferred)
            self.assertEqual({(x['cb_slot'], x['cb_offset'], x['flags']) for x in res['sbufs']},
                             {(1, 0, 0x80), (0, 4, 0x80), (0, 8, 0x81)})
            self.assertEqual(guarded_memory_blocks((args.out / (job['key'] + '.spv')).read_bytes(),
                                                  {x['binding'] for x in res['sbufs']}), 3)

    def test_c1_only_and_unknown_c1_do_not_get_folded(self):
        jobs = [packet('only_c1', original.c1_pointer_bytes(False), storage_candidates_version=1,
                       c1=(struct.pack('<Q', 0x600001000) + bytes(248)).hex()),
                packet('unknown_c1', original.shader_bytes(True), storage_candidates_version=1),
                packet('short_c1', original.shader_bytes(True), storage_candidates_version=1, c1='20' + '00' * 6)]
        del jobs[1]['c1']
        self.translate(jobs)
        self.assertEqual({(x['cb_slot'], x['cb_offset']) for x in self.reflection('only_c1')['sbufs']}, {(1, 0), (0, 8)})
        for key in ('unknown_c1', 'short_c1'):
            res = self.reflection(key)
            c1 = next(x for x in res['cbufs'] if x['slot'] == 1)
            self.assertTrue(original.loads_uniform_word_zero((args.out / (key + '.spv')).read_bytes(), c1['binding']))
            self.assertIn((1, 0), {(x['cb_slot'], x['cb_offset']) for x in res['sbufs']})

    def test_descriptor_base_alignment_is_an_explicit_contract(self):
        jobs = [packet('alignment_' + str(alignment), original.shader_bytes(True),
                       storage_candidates_version=1, sbuf_align=alignment) for alignment in (4, 16, 256)]
        exact = packet('alignment_exact', original.shader_bytes(True), storage_candidates_version=1)
        del exact['sbuf_align']
        self.translate([exact] + jobs)
        for job in [exact] + jobs:
            constants = {o[2] for op, o in instructions((args.out / (job['key'] + '.spv')).read_bytes())
                         if op == 43 and len(o) == 3}
            if job is exact:
                self.assertNotIn((-16) & 0xffffffff, constants,
                                 'the accessor default must not round an exact descriptor base')
            else:
                self.assertIn((-job['sbuf_align']) & 0xffffffff, constants)

    def test_unknown_versions_and_non_nvn2_compute_fail_closed(self):
        jobs = [packet('invalid_version_' + str(i), original.shader_bytes(True), storage_candidates_version=value)
                for i, value in enumerate((-1, 2, 1.0, True, '1', None))]
        jobs += [packet('wrong_stage', original.shader_bytes(True), storage_candidates_version=1, stage='ps'),
                 packet('old_roots', original.shader_bytes(True), storage_candidates_version=1, c0_version=1),
                 packet('no_roots', original.shader_bytes(True), storage_candidates_version=1)]
        del jobs[-1]['c0']
        self.translate(jobs, valid=False)

    def test_unsupported_widths_fail_before_scalarization(self):
        jobs = []
        # The 16-bit store is admitted by design (see test_halfword_store_u16_* below). Everything else of a width
        # other than B32/B64/B128 (8-bit loads and stores, 16-bit loads, unaligned 128-bit) still fails closed.
        for key, store, size in (('load8', False, 0), ('store8', True, 0),
                                 ('load16', False, 2), ('unaligned128', False, 7)):
            ops = original.instructions(True)
            for i, op in enumerate(ops):
                if op >> 51 == (0xEED8000000000000 if store else 0xEED0000000000000) >> 51:
                    ops[i] = (op & ~(7 << 48)) | (size << 48)
            jobs.append(packet(key, encoded(ops), storage_candidates_version=1))
        replies = self.translate(jobs, valid=False)
        self.assertTrue(all('naturally aligned B32/B64/B128' in x for x in replies), replies)

    def test_halfword_store_u16_is_admitted_with_a_guarded_compare_exchange(self):
        ops = original.instructions(True)
        for i, op in enumerate(ops):
            if op >> 51 == 0xEED8000000000000 >> 51:
                ops[i] = (op & ~(7 << 48)) | (2 << 48)   # size field 2: LsSize2.U16
        jobs = [packet('store16-ok', encoded(ops), storage_candidates_version=1)]
        self.translate(jobs)   # reply 'ok <key>' and spirv-val --target-env vulkan1.3 passes
        disasm = args.spirv_val.with_name(args.spirv_val.name.replace('spirv-val', 'spirv-dis'))
        self.assertTrue(disasm.is_file(), f'spirv-dis is missing next to spirv-val: {disasm}')
        dis = subprocess.run([str(disasm), str(args.out / (jobs[0]['key'] + '.spv'))],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(dis.returncode, 0, dis.stderr)
        self.assertIn('OpAtomicCompareExchange', dis.stdout, 'a guarded halfword store must use a compare-exchange')

    def _compare_exchange_sites(self, key):
        disasm = args.spirv_val.with_name(args.spirv_val.name.replace('spirv-val', 'spirv-dis'))
        dis = subprocess.run([str(disasm), str(args.out / (key + '.spv'))], capture_output=True, text=True, timeout=30)
        self.assertEqual(dis.returncode, 0, dis.stderr)
        return dis.stdout.count('OpAtomicCompareExchange')

    def test_packed_half_red_translates_as_a_guarded_compare_exchange(self):
        # RED.E.ADD.F16x2.FTZ.RN [R0], R6: RedSize (bits 20-22) = 4 (F16x2FtzRn), RedOp (23-25) = 0 (Add), data R6 (0-7).
        ops = original.instructions(True)
        load = ops.index(original.global_memory(False, 2, 0))
        red = 0xEBF8000000000000 | original.PT | 6 | (4 << 20)
        ops[load:load + 1] = [original.movi(6, 0x3c00), red]
        job = packet('packed_half_red', encoded(ops), storage_candidates_version=1)
        self.translate([job])   # reply 'ok <key>' and spirv-val --target-env vulkan1.3 passes
        res = self.reflection('packed_half_red')
        self.assertTrue(all(x['flags'] == 0x81 for x in res['sbufs']))
        self.assertGreaterEqual(self._compare_exchange_sites('packed_half_red'), 1)

    def test_crossing_word_halfword_store_has_both_compare_exchange_paths(self):
        # The 16-bit store is guarded: a single-word compare-exchange and a two-word one for halfwords starting at byte 3.
        ops = original.instructions(True)
        for i, op in enumerate(ops):
            if op >> 51 == 0xEED8000000000000 >> 51:
                ops[i] = (op & ~(7 << 48)) | (2 << 48)   # size field 2: LsSize2.U16
        self.translate([packet('crossing_store16', encoded(ops), storage_candidates_version=1)])
        self.assertGreaterEqual(self._compare_exchange_sites('crossing_store16'), 3)

    def test_original_wide_register_values_and_canaries(self):
        for bits in (64, 128):
            last_start_pointer = (0x300000000 - bits // 8) - 32
            for inline in (False, True):
                for pointer, offset in ((0x200001000, 0), (last_start_pointer, 0),
                                        (0x2fffffff0, 0), (0x2ffffffc0, 32),
                                        (0x200001000, -16), (0x2fffffff0, -32)):
                    with self.subTest(bits=bits, inline=inline, pointer=hex(pointer), immediate=offset):
                        actual, expected, values = evaluate_wide(bits, inline, offset, pointer)
                        self.assertEqual(actual, expected, 'every copied word and surrounding canary must agree')
                        self.assertEqual(values, original.WIDE_VALUES[:bits // 32])

    def test_aligned_wide_vectors_emit_every_guarded_word(self):
        jobs = [packet(f'wide{bits}_{inline}_{offset}', original.wide_shader_bytes(bits, inline, offset),
                       storage_candidates_version=1)
                for bits in (64, 128) for inline in (False, True) for offset in (0, 32, -16, -32)]
        self.translate(jobs)
        for job in jobs:
            bits = int(job['key'].split('_')[0][4:])
            inline = job['key'].split('_')[1] == 'True'
            res = self.reflection(job['key'])
            spv = (args.out / (job['key'] + '.spv')).read_bytes()
            bases = {(x['cb_slot'], x['cb_offset'], x['flags']) for x in res['sbufs']}
            expected = {(0, 4, 0x80), (0, 8, 0x81)} | ({(1, 0, 0x80)} if inline else set())
            self.assertEqual(bases, expected)
            self.assertEqual(sum(op == 57 for op, _ in instructions(spv)), 2 * (bits // 32),
                             'each data word needs one guarded load and one guarded store call')
            self.assertEqual(guarded_memory_blocks(spv, {x['binding'] for x in res['sbufs']}), 3 if inline else 2)

    def test_wide_default_preserves_legacy_translation(self):
        jobs = [packet(f'legacy_wide{bits}_{offset}_{version}', original.wide_shader_bytes(bits, True, offset),
                       storage_candidates_version=version)
                for bits in (64, 128) for offset in (32, -16, -32) for version in (None, 0)]
        for job in jobs:
            if job['storage_candidates_version'] is None:
                del job['storage_candidates_version']
        self.translate(jobs)
        for bits in (64, 128):
            for offset in (32, -16, -32):
                missing, zero = f'legacy_wide{bits}_{offset}_None', f'legacy_wide{bits}_{offset}_0'
                self.assertEqual((args.out / (missing + '.spv')).read_bytes(), (args.out / (zero + '.spv')).read_bytes())
                self.assertTrue(all(not (r['flags'] & 0x80) for r in self.reflection(missing)['sbufs']))
                if args.baseline:
                    job = dict(next(j for j in jobs if j['key'] == missing), key=f'baseline_wide{bits}_{offset}')
                    run = subprocess.run([str(args.baseline), '--serve'], env=self.env,
                                         input=json.dumps(dict(job=job, out=str(args.out))) + '\n',
                                         capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.stdout.strip(), 'ok ' + job['key'], run.stdout + run.stderr)
                    self.assertEqual((args.out / (job['key'] + '.spv')).read_bytes(), (args.out / (missing + '.spv')).read_bytes())

    def test_unsupported_atomic_width_and_cas_fail_explicitly(self):
        ops = original.instructions(True)
        load = ops.index(original.global_memory(False, 2, 0))
        atom64 = 0xED00000000000000 | (1 << 48) | (2 << 49) | (6 << 20) | original.PT | 2
        cas = 0xEEF0000000000000 | (6 << 20) | original.PT | 2
        jobs = []
        for key, atomic in (('atomic64', atom64), ('atomic_cas', cas)):
            body = list(ops)
            body[load:load + 1] = [original.movi(6, 1), original.movi(7, 0), atomic]
            jobs.append(packet(key, encoded(body), storage_candidates_version=1))
        replies = self.translate(jobs, valid=False)
        self.assertIn('32-bit integer atomics', replies[0])
        self.assertIn('AtomCas', replies[1])

    def test_integer_atomic_has_branch_and_single_commit(self):
        ops = original.instructions(True)
        load = ops.index(original.global_memory(False, 2, 0))
        atomic = 0xED00000000000000 | (1 << 48) | (6 << 20) | original.PT | 2  # ATOM.ADD.U32.E R2,[R0],R6
        ops[load:load + 1] = [original.movi(6, 1), atomic]
        job = packet('atomic', encoded(ops), storage_candidates_version=1)
        self.translate([job])
        res = self.reflection('atomic')
        self.assertTrue(all(x['flags'] == 0x81 for x in res['sbufs']))
        self.assertEqual(guarded_memory_blocks((args.out / 'atomic.spv').read_bytes(),
                                              {x['binding'] for x in res['sbufs']}), 3)

    def test_descriptor_limit_rejects_instead_of_truncating(self):
        jobs = []
        for count in (31, 32):
            ops = [original.movc(4, 0, 192), original.movc(5, 0, 193)]
            for index in range(count):
                ops += [original.movc(0, 0, 4 + 4 * index), original.movc(1, 0, 5 + 4 * index),
                        original.global_memory(False, 2, 0), original.global_memory(True, 2, 4)]
            ops += [original.EXIT]
            jobs.append(packet('resources_' + str(count), encoded(ops), storage_candidates_version=1,
                               c0_roots=[64, 64, 64, 0, 0, 0, 0, 0]))
        self.translate(jobs[:1])
        self.assertEqual(len(self.reflection('resources_31')['sbufs']), 32)
        self.assertIn('descriptor resource limit', self.translate(jobs[1:], valid=False)[0])

    def test_guarded_and_legacy_resource_identity_and_backend_gate(self):
        work = args.out / 'resource-probe'
        work.mkdir(exist_ok=True)
        project = work / 'probe.csproj'
        shader_project = args.ryu_src / 'src/Ryujinx.Graphics.Shader/Ryujinx.Graphics.Shader.csproj'
        source = ROOT / 'tests/storage_candidates_resource_probe.cs'
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType>'
                           '<TargetFramework>net10.0</TargetFramework><EnableDefaultCompileItems>false</EnableDefaultCompileItems>'
                           '</PropertyGroup><ItemGroup><Compile Include="' + escape(str(source)) + '" />'
                           '<ProjectReference Include="' + escape(str(shader_project)) + '" /></ItemGroup></Project>')
        run = subprocess.run([str(args.dotnet), 'build', str(project), '-c', 'Release', '--disable-build-servers',
                              '-m:2', '-nr:false', '-p:UseSharedCompilation=false'],
                             env=self.env, capture_output=True, text=True, timeout=120)
        (work / 'build.log').write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        run = subprocess.run([str(args.dotnet), str(work / 'bin/Release/net10.0/probe.dll')],
                             env=self.env, capture_output=True, text=True, timeout=30)
        (work / 'run.log').write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('resource/capability probe:', run.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xlate', type=Path, required=True, help='native exe or managed totk_xlate.dll')
    parser.add_argument('--dotnet', type=Path, required=True)
    parser.add_argument('--ryu-src', type=Path, required=True, help='patched source copy used to build the translator')
    parser.add_argument('--spirv-val', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='optional default translator for exact SPIR-V parity')
    parser.add_argument('--out', type=Path, default=ROOT / 'build/tests/storage-candidates')
    args = parser.parse_args()
    args.out = args.out.resolve()
    unittest.main(argv=[__file__], verbosity=2)
