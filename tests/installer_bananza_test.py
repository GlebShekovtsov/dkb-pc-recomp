"""Folder installer safety/resume tests with synthetic files and tiny stand-in build tools; no game or GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from installer import bananza as b
from installer.events import Reporter
from installer.guilogic import Catalog, Form
from installer.layout import Layout

sys.path.insert(0, str(ROOT / 'tests'))
from nup_extract_check import build as build_nup  # noqa: E402  synthetic NUP containers


def tree(folder):
    """A folder as the {name: bytes | tree} that build_nup takes."""
    return {p.name: tree(p) if p.is_dir() else p.read_bytes() for p in sorted(folder.iterdir())}


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        base = Path(self.work.name)
        self.root, self.exefs, self.romfs = base / 'source', base / 'dump exefs', base / 'dump romfs'
        self.dest, self.temp = base / 'installed [game]', base / 'temporary'
        self.calls = []
        for p in (self.root, self.exefs, self.romfs):
            p.mkdir()
        for name in ('main', 'sdk', 'rtld', 'main.npdm'):
            (self.exefs / name).write_bytes(('synthetic ' + name).encode())
        (self.romfs / 'folder').mkdir()
        (self.romfs / 'folder' / 'café.bin').write_bytes(b'synthetic resource' * 23)
        (self.romfs / 'other.bin').write_bytes(bytes(range(251)))
        for name in ('tools/data', 'recomp', 'runtime', 'llvm/bin', 'deps/include', 'deps/lib', 'bin'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        hashes = {p.name: b.sha256(p) for p in self.exefs.iterdir()}
        (self.root / 'tools/data/expected-dkb100.json').write_text(json.dumps({'exefs_sha256': hashes}))
        (self.root / 'runtime/nvn2_lower.cpp').write_text('// synthetic fixture')
        (self.root / 'recomp/patches-dkb100.txt').write_text('# synthetic patch table')
        (self.root / 'tools/build_xlate.py').write_text("def identity(): return 'synthetic-translator-id'\n")
        shutil.copyfile(ROOT / 'tools/dkb_pack_romfs.py', self.root / 'tools/dkb_pack_romfs.py')
        shutil.copyfile(ROOT / 'tools/dkb_shader_census.py', self.root / 'tools/dkb_shader_census.py')
        shutil.copyfile(ROOT / 'tools/extract_nup.py', self.root / 'tools/extract_nup.py')
        (self.root / 'tools/romfs_index.py').write_text('# synthetic fixture')
        (self.root / 'recomp/gen.py').write_text('''
import json, pathlib, sys
a=sys.argv; out=pathlib.Path(a[a.index('--out')+1])
assert a[a.index('--modules')+1]=='main,sdk'
assert pathlib.Path(a[a.index('--patches')+1]).name=='patches-dkb100.txt'
out.mkdir(parents=True,exist_ok=True)
(out/'layout.json').write_text('{}'); (out/'patches.h').write_text('// synthetic')
for name in ('main','sdk'):
    (out/name).mkdir(exist_ok=True)
    (out/name/(name+'_0000.c')).write_text('// synthetic fixture, no game code')
    (out/name/'manifest.json').write_text(json.dumps({'chunks':[name+'_0000.c'],'stats':{'patched_sites':1}}))
''')
        (self.root / 'tools/build_game.py').write_text('''
import pathlib, sys
a=sys.argv; out=pathlib.Path(a[a.index('--out')+1]); out.mkdir(parents=True,exist_ok=True)
assert a[a.index('--version')+1]=='dkb100' and '--runtime-lib' not in a
assert '--full-build' in a and pathlib.Path(a[a.index('--expected')+1]).name=='expected-dkb100.json'
(out/('dkb.exe' if sys.platform=='win32' else 'dkb')).write_bytes(b'synthetic executable')
(out/'romfs-index.bin').write_bytes(b'synthetic index')
(out/'obj').mkdir(exist_ok=True)
(out/'obj/guest.o').write_bytes(b'synthetic object')
''')
        for name in ('clang' + b.EXE, 'ninja' + b.EXE):
            (self.root / ('llvm/bin' if name.startswith('clang') else 'bin') / name).write_bytes(b'synthetic tool')
        translator = self.root / 'bin' / ('totk_xlate' + b.EXE)
        translator.write_bytes(b'synthetic translator')
        translator.with_suffix('.id').write_text('synthetic-translator-id\n')
        dlls = []
        if os.name == 'nt':
            for name in ('SDL3.dll', 'libopus-0.dll'):
                p = self.root / 'bin' / name
                p.write_bytes(b'synthetic runtime library')
                dlls.append(p)
        self.layout = Layout(root=self.root, dev=True, release='test', python=Path(sys.executable),
                             tools=self.root/'tools', recomp=self.root/'recomp', llvm=self.root/'llvm',
                             deps=self.root/'deps', runtime=self.root/'runtime', runtime_lib=self.root/'unused.a',
                             ninja=self.root/'bin'/('ninja'+b.EXE), hactool=self.root/'unused', xlate=translator,
                             dlls=tuple(dlls), temp=self.temp)
        self.options = SimpleNamespace(base=None, update=None, keys=None, jobs=2, lang='en-US', exefs=self.exefs,
                                       romfs=self.romfs, dest=self.dest, temp=self.temp, root=self.root, verify=False,
                                       package=self.root, toolchain=None, deps=None, ninja=None, dll_dir=None,
                                       translator=translator, dotnet=None)

    def run_install(self, fail=None):
        original = b.run_tool
        def tool(ctx, step, command, env=None, cwd=None, name=None):
            self.calls.append((step, list(map(str, command))))
            if fail:
                fail(ctx, step, command)
            if step == 'dkb_shaders':   # the census and the translation run for real only in a full install
                if str(command[-1]) == '--translate-shaders':
                    (ctx.dest / 'shader-cache').mkdir(exist_ok=True)
                return
            if step == 'dkb_verify':
                return
            return original(ctx, step, command, env, cwd, name)
        output = io.StringIO()
        with patch.object(b.layouts, 'find', return_value=self.layout), patch.object(b.Check, 'host'), \
                patch.object(b, 'run_tool', side_effect=tool), contextlib.redirect_stdout(output):
            code = b.main(self.options)
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def host_warnings(self, **extra):
        host = dict(x86_64_v3=True, loader=True, devices=[{'name': 'gpu', 'api': [1, 3, 0]}], **extra)
        out = io.StringIO()
        ctx = SimpleNamespace(temp=self.temp, layout=self.layout, reporter=Reporter(out=out))
        with patch.object(b, 'run_tool'), patch.object(b.subprocess, 'run',
                                                       return_value=SimpleNamespace(stdout=json.dumps(host), returncode=0)):
            b.Check().host(ctx)   # a warning must not raise
        return [json.loads(line)['id'] for line in out.getvalue().splitlines() if json.loads(line)['type'] == 'warning']

    def test_low_commit_limit_warns_and_continues(self):
        self.assertEqual(self.host_warnings(commit_limit_gib=17.9), ['warn.low_commit_limit'])

    def test_enough_or_unknown_commit_limit_does_not_warn(self):
        self.assertEqual(self.host_warnings(commit_limit_gib=30.0), [])
        self.assertEqual(self.host_warnings(), [])

    def test_invalid_base_writes_nothing(self):
        (self.exefs / 'main').write_bytes(b'unsupported update')
        code, events = self.run_install()
        self.assertEqual(code, 1)
        self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.exefs_invalid')
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.temp.exists())

    def test_unrelated_destination_is_preserved(self):
        self.dest.mkdir()
        (self.dest / 'keep.txt').write_text('untouched')
        self.assertEqual(self.run_install()[0], 1)
        self.assertEqual((self.dest / 'keep.txt').read_text(), 'untouched')
        self.assertFalse(self.temp.exists())

    def test_overlap_and_unsupported_update_are_rejected(self):
        self.options.dest = self.exefs / 'output'
        self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.options.dest.exists())
        self.options.dest = self.dest
        self.options.update = self.root / 'update.nsp'
        self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.dest.exists())

    def test_unowned_temp_is_preserved(self):
        tag = b.hashlib.sha256(str(self.dest).lower().encode()).hexdigest()[:12]
        target = self.temp / f'dkb-{self.dest.name}-{tag}'
        target.mkdir(parents=True)
        (target / 'keep.txt').write_text('untouched')
        self.assertEqual(self.run_install()[0], 1)
        self.assertEqual((target / 'keep.txt').read_text(), 'untouched')
        self.assertFalse(self.dest.exists())

    def test_fresh_portable_install_and_resume(self):
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        state = b.statefile.load(self.dest, b.STATE)
        self.assertEqual(state['game'], 'dkb100')
        self.assertEqual(set(state['steps']), {'dkb_data','dkb_translator','lift','compile','dkb_shaders','dkb_verify'})
        self.assertEqual((self.dest/'exefs/main').read_bytes(), (self.exefs/'main').read_bytes())
        self.assertTrue((self.dest/('dkb'+b.EXE)).is_file())
        self.assertFalse((self.dest/'build.ninja').exists())
        self.assertFalse((self.dest/'totk-install.json').exists())
        self.calls.clear()
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual(self.calls, [])
        self.assertEqual({e['step'] for e in events if e['type']=='step' and e['state']=='skipped'},
                         {'dkb_data','dkb_translator','lift','compile','dkb_shaders','dkb_verify'})

    def test_library_warm_failure_does_not_fail_the_step(self):
        def crash(ctx, step, command):
            if str(command[-1]) == '--warm-shaders':
                raise b.InstallError('dkb.tool_failed', step=step, code=3221225477, log='warm.log')
        code, events = self.run_install(crash)
        self.assertEqual(code, 0, events)
        self.assertIn('warn.dkb_libraries_failed', [e.get('id') for e in events if e['type'] == 'warning'])
        record = b.statefile.load(self.dest, b.STATE)['steps']['dkb_shaders']
        self.assertTrue(record['prepared'])
        self.assertFalse(record['libraries'])
        self.calls.clear()
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual(self.calls, [], 'a retry must not run the crashing library build again')

    def test_library_warm_path_rejection_fails_the_step(self):
        def reject(ctx, step, command):
            if str(command[-1]) == '--warm-shaders':
                raise b.InstallError('dkb.path_link', path='warm.log')
        code, events = self.run_install(reject)
        self.assertEqual(code, 1)
        self.assertNotIn('dkb_shaders', b.statefile.load(self.dest, b.STATE)['steps'])

    def test_translation_failure_fails_the_step(self):
        def fail(ctx, step, command):
            if str(command[-1]) == '--translate-shaders':
                raise b.InstallError('dkb.tool_failed', step=step, code=1, log='translate.log')
        code, events = self.run_install(fail)
        self.assertEqual(code, 1)
        self.assertNotIn('dkb_shaders', b.statefile.load(self.dest, b.STATE)['steps'])
        self.assertNotIn('--warm-shaders', [c[-1] for _, c in self.calls])

    def test_cancel_before_compile_resumes_without_redoing_pack(self):
        def cancel(ctx, step, command):
            if step == 'compile':
                raise KeyboardInterrupt
        self.assertEqual(self.run_install(cancel)[0], 3)
        self.assertNotIn('compile', b.statefile.load(self.dest, b.STATE)['steps'])
        self.calls.clear()
        self.assertEqual(self.run_install()[0], 0)
        self.assertNotIn('dkb_data', [step for step, _ in self.calls])
        self.assertNotIn('lift', [step for step, _ in self.calls])

    def test_resume_recovers_sources_and_build_tools_from_state(self):
        self.assertEqual(self.run_install()[0], 0)
        self.options.exefs = self.options.romfs = self.options.translator = self.options.package = None
        self.calls.clear()
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual(self.calls, [])

    def test_concurrent_setup_lock_and_release(self):
        self.temp.mkdir()
        first = b.lock_temp(self.temp)
        try:
            with self.assertRaises(b.InstallError) as error:
                b.lock_temp(self.temp)
            self.assertEqual(error.exception.id, 'dkb.busy')
        finally:
            first.close()
        b.lock_temp(self.temp).close()

    def test_failure_before_first_checkpoint_resumes_owned_empty_destination(self):
        with patch.object(b.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            self.assertEqual(self.run_install()[0], 1)
        self.assertFalse((self.dest/b.STATE).exists())
        self.assertEqual(self.run_install()[0], 0)

    def test_destination_lock_also_covers_different_workspaces(self):
        self.assertEqual(self.run_install()[0], 0)
        lock = b.lock_temp(self.dest)
        try:
            self.options.temp = self.temp.parent / 'different workspace'
            code, events = self.run_install()
            self.assertEqual(code, 1)
            self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.busy')
        finally:
            lock.close()

    def cancelled_workspace(self):
        def cancel(ctx, step, command):
            if step == 'compile':
                raise KeyboardInterrupt
        self.assertEqual(self.run_install(cancel)[0], 3)
        return next(self.temp.glob('dkb-*'))

    def test_owned_workspace_directory_redirects_rejected_before_writers(self):
        workspace = self.cancelled_workspace()
        outside = Path(self.work.name) / 'outside workspace'
        outside.mkdir()
        (outside/'keep.txt').write_text('untouched')
        for relative in ('gen', 'build', 'xlate', 'logs', 'data', 'data/romfs',
                         'build/obj/runtime', 'xlate/ryujinx/src'):
            with self.subTest(relative=relative):
                link = workspace/relative
                if link.exists():
                    shutil.rmtree(link)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(outside, target_is_directory=True)
                self.calls.clear()
                try:
                    code, events = self.run_install()
                    self.assertEqual(code, 1, events)
                    self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.path_link')
                    self.assertEqual(self.calls, [], 'a writer started before rejecting the redirected workspace')
                    self.assertEqual({p.name for p in outside.iterdir()}, {'keep.txt'})
                    self.assertEqual((outside/'keep.txt').read_text(), 'untouched')
                finally:
                    link.unlink()

    def test_owned_workspace_leaf_and_owner_links_rejected_before_writers(self):
        workspace = self.cancelled_workspace()
        outside = Path(self.work.name) / 'outside-file.txt'
        outside.write_text('untouched')
        for relative in ('gen/main/main_0000.c', 'logs/compile.log', b.OWNER):
            with self.subTest(relative=relative):
                link = workspace/relative
                old = link.read_bytes() if link.exists() else None
                if link.exists():
                    link.unlink()
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(outside)
                self.calls.clear()
                try:
                    code, events = self.run_install()
                    self.assertEqual(code, 1, events)
                    self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.path_link')
                    self.assertEqual(self.calls, [])
                    self.assertEqual(outside.read_text(), 'untouched')
                finally:
                    link.unlink()
                    if old is not None:
                        link.write_bytes(old)

    def test_managed_destination_redirect_rejected_before_writers(self):
        self.cancelled_workspace()
        outside = Path(self.work.name) / 'outside-exefs'
        shutil.move(self.dest/'exefs', outside)
        (self.dest/'exefs').symlink_to(outside, target_is_directory=True)
        self.calls.clear()
        code, events = self.run_install()
        self.assertEqual(code, 1, events)
        self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.path_link')
        self.assertEqual(self.calls, [])

    def test_subprocess_writer_rechecks_workspace_before_opening_log_or_spawning(self):
        workspace = self.cancelled_workspace()
        outside = Path(self.work.name)/'outside-writer'
        outside.mkdir()
        (workspace/'build').symlink_to(outside, target_is_directory=True)
        ctx = SimpleNamespace(temp=workspace, dest=self.dest, layout=self.layout, reporter=Reporter(out=io.StringIO()))
        with patch.object(b.subprocess, 'Popen') as child:
            with self.assertRaises(b.InstallError) as error:
                b.run_tool(ctx, 'compile', [sys.executable, '-c', 'raise AssertionError()'])
            self.assertEqual(error.exception.id, 'dkb.path_link')
            child.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_native_directory_junction_is_rejected_without_following_it(self):
        workspace = self.cancelled_workspace()
        outside = Path(self.work.name)/'outside-junction'
        outside.mkdir()
        (outside/'keep.txt').write_text('untouched')
        junction = workspace/'build'
        if os.name == 'nt':
            literal = lambda p: "'" + str(p).replace("'", "''") + "'"
            command = 'New-Item -ItemType Junction -Path ' + literal(junction) + ' -Target ' + literal(outside)
            subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                           check=True, capture_output=True)
            self.assertTrue(junction.is_junction())
        else:
            junction.symlink_to(outside, target_is_directory=True)
        self.calls.clear()
        try:
            code, events = self.run_install()
            self.assertEqual(code, 1, events)
            self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.path_link')
            self.assertEqual(self.calls, [])
            self.assertEqual({p.name for p in outside.iterdir()}, {'keep.txt'})
        finally:
            if os.name == 'nt':
                junction.rmdir()
            else:
                junction.unlink()

    def test_changed_source_invalidates_data_and_downstream(self):
        self.assertEqual(self.run_install()[0], 0)
        (self.romfs/'other.bin').write_bytes(b'a changed source')
        self.calls.clear()
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual([step for step, _ in self.calls], ['dkb_data','lift','compile','dkb_shaders','dkb_shaders','dkb_shaders','dkb_verify'])

    def test_source_mutation_during_pack_is_not_installed(self):
        def mutate(ctx, step, command):
            if step == 'dkb_data':
                (self.romfs/'other.bin').write_bytes(b'changed after preflight')
        code, events = self.run_install(mutate)
        self.assertEqual(code, 1)
        self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.input_changed')
        self.assertFalse((self.dest/'romfs/romfs.bin').exists())

    def test_corrupt_installed_pack_is_repaired_with_verify(self):
        self.assertEqual(self.run_install()[0], 0)
        image = self.dest/'romfs/romfs.bin'
        with image.open('r+b') as f:
            f.write(b'bad')
        self.options.verify = True
        self.calls.clear()
        self.assertEqual(self.run_install()[0], 0)
        self.assertIn('dkb_data', [step for step, _ in self.calls])
        b.pack_manifest(self.dest/'romfs')

    def test_profile_survives_rebuild(self):
        self.assertEqual(self.run_install()[0], 0)
        (self.dest/'profile').mkdir()
        (self.dest/'profile/save.bin').write_bytes(b'synthetic user save')
        (self.root/'runtime/nvn2_lower.cpp').write_text('// changed synthetic runtime')
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual((self.dest/'profile/save.bin').read_bytes(), b'synthetic user save')

    def test_success_retains_generated_code_and_objects(self):
        self.update_fixture()   # only the update keeps generated code and objects
        self.assertEqual(self.run_install()[0], 0)
        workspace = next(self.temp.glob('dkb-*'))
        self.assertTrue((workspace/'gen/main/main_0000.c').is_file())
        self.assertEqual((workspace/'build/obj/guest.o').read_bytes(), b'synthetic object')
        self.assertFalse((workspace/'data').exists())

    def test_base_install_removes_generated_code_and_objects(self):
        self.assertEqual(self.run_install()[0], 0)
        workspace = next(self.temp.glob('dkb-*'))
        self.assertFalse((workspace/'gen').exists() or (workspace/'build').exists())

    def test_runtime_change_reuses_generated_code_and_assets(self):
        self.update_fixture()   # only the update keeps generated code and objects
        self.assertEqual(self.run_install()[0], 0)
        workspace = next(self.temp.glob('dkb-*'))
        generated = workspace/'gen/main/main_0000.c'
        timestamp = generated.stat().st_mtime_ns
        (self.root/'runtime/nvn2_lower.cpp').write_text('// changed synthetic runtime')
        self.calls.clear()
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual([step for step, _ in self.calls],
                         ['compile', 'dkb_shaders', 'dkb_shaders', 'dkb_shaders', 'dkb_verify'])
        self.assertEqual(generated.stat().st_mtime_ns, timestamp)

    def update_fixture(self):
        hashes = {}
        for p in self.exefs.iterdir():
            p.write_bytes(b'updated ' + p.read_bytes())
            hashes[p.name] = b.sha256(p)
        (self.root / 'tools/data/expected-dkb458752.json').write_text(json.dumps({'exefs_sha256': hashes}))
        (self.root / 'recomp/patches-dkb458752.txt').write_text('# no controller proxy in this version')
        for relative in ('recomp/gen.py', 'tools/build_game.py'):
            p = self.root / relative
            text = p.read_text().replace('dkb100', 'dkb458752')
            if relative == 'recomp/gen.py':
                text = text.replace("'patched_sites':1", "'patched_sites':0")
            p.write_text(text)

    def test_update_has_separate_identity_and_no_base_patch_requirement(self):
        self.update_fixture()
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual(b.statefile.load(self.dest, b.STATE)['game'], 'dkb458752')
        self.assertEqual(json.loads((self.dest / b.DEST_OWNER).read_text())['game'], 'dkb458752')
        self.calls.clear()
        self.options.exefs = self.options.romfs = None
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual(self.calls, [])

    def test_target_switch_cannot_overwrite_existing_base_install(self):
        self.assertEqual(self.run_install()[0], 0)
        receipt = (self.dest / b.STATE).read_bytes()
        self.update_fixture()
        self.calls.clear()
        self.assertEqual(self.run_install()[0], 1)
        self.assertEqual(self.calls, [])
        self.assertEqual((self.dest / b.STATE).read_bytes(), receipt)

    def test_base_still_requires_controller_patch(self):
        p = self.root / 'recomp/gen.py'
        p.write_text(p.read_text().replace("'patched_sites':1", "'patched_sites':0"))
        code, events = self.run_install()
        self.assertEqual(code, 1)
        self.assertEqual(next(e['id'] for e in events if e['type'] == 'error'), 'dkb.patches_missing')

    def test_mixed_update_exefs_is_rejected_before_writers(self):
        original_sdk = (self.exefs / 'sdk').read_bytes()
        self.update_fixture()
        (self.exefs / 'sdk').write_bytes(original_sdk)
        self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.temp.exists())

    def test_missing_generated_output_is_regenerated_when_rebuilding(self):
        self.update_fixture()   # only the update keeps generated code and objects
        self.assertEqual(self.run_install()[0], 0)
        workspace = next(self.temp.glob('dkb-*'))
        (workspace/'gen/main/main_0000.c').unlink()
        (self.root/'runtime/nvn2_lower.cpp').write_text('// changed synthetic runtime')
        self.calls.clear()
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual([step for step, _ in self.calls],
                         ['lift', 'compile', 'dkb_shaders', 'dkb_shaders', 'dkb_shaders', 'dkb_verify'])

    def test_corrupt_generated_output_is_regenerated_when_rebuilding(self):
        self.update_fixture()   # only the update keeps generated code and objects
        self.assertEqual(self.run_install()[0], 0)
        workspace = next(self.temp.glob('dkb-*'))
        (workspace/'gen/main/main_0000.c').write_bytes(b'corrupt')
        (self.root/'runtime/nvn2_lower.cpp').write_text('// changed synthetic runtime')
        self.calls.clear()
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual([step for step, _ in self.calls],
                         ['lift', 'compile', 'dkb_shaders', 'dkb_shaders', 'dkb_shaders', 'dkb_verify'])

    def test_already_packed_input_is_verified_and_copied(self):
        packed = self.root / 'packed input'
        subprocess.run([sys.executable, str(ROOT/'tools/dkb_pack_romfs.py'), '--dir', str(self.romfs),
                        '--out', str(packed)], check=True, capture_output=True)
        self.options.romfs = packed
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual(b.sha256(packed/'romfs.bin'), b.sha256(self.dest/'romfs/romfs.bin'))
        self.assertNotIn('dkb_data', [step for step, _ in self.calls])

    def test_packed_input_corruption_and_traversal_are_rejected(self):
        packed = self.root / 'packed input'
        subprocess.run([sys.executable, str(ROOT/'tools/dkb_pack_romfs.py'), '--dir', str(self.romfs),
                        '--out', str(packed)], check=True, capture_output=True)
        self.options.romfs = packed
        original = (packed/'index.json').read_text()
        m = json.loads(original)
        m['files'][0]['path'] = '../outside'
        (packed/'index.json').write_text(json.dumps(m))
        self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.dest.exists())
        (packed/'index.json').write_text(original)
        (packed/'romfs.bin').write_bytes(b'corrupt')
        self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.temp.exists())

    def test_negative_first_packed_offset_is_rejected_before_writes(self):
        packed = self.root / 'packed input'
        subprocess.run([sys.executable, str(ROOT/'tools/dkb_pack_romfs.py'), '--dir', str(self.romfs),
                        '--out', str(packed)], check=True, capture_output=True)
        manifest = json.loads((packed/'index.json').read_text())
        manifest['files'][0]['offset'] = -1
        (packed/'index.json').write_text(json.dumps(manifest))
        self.options.romfs = packed
        self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.temp.exists())

    TITLE = 0x040074A01BF12000

    def write_nup(self, update=None, name='game.nup'):
        """A container of the fixture's base folders and, optionally, an update package {'exefs': ..., 'romfs': ...}."""
        packages = [(0, self.TITLE, 0, {'exefs': tree(self.exefs), 'romfs': tree(self.romfs)})]
        if update is not None:
            packages.append((1, self.TITLE | 0x800, 458752, update))
        path = Path(self.work.name) / name
        path.write_bytes(build_nup(packages))
        self.options.exefs = self.options.romfs = None
        self.options.nup = path
        return path

    def loose_pack(self, folder):
        out = Path(self.work.name) / 'reference pack'
        shutil.rmtree(out, ignore_errors=True)
        subprocess.run([sys.executable, str(ROOT/'tools/dkb_pack_romfs.py'), '--dir', str(folder), '--out', str(out)],
                       check=True, capture_output=True)
        return out

    def test_nup_base_installs_the_same_data_as_its_folders(self):
        path = self.write_nup()
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        state = b.statefile.load(self.dest, b.STATE)
        self.assertEqual((state['game'], state['sources']), ('dkb100', {'nup': str(path.resolve()), 'nup_package': 'latest'}))
        reference = self.loose_pack(self.romfs)
        self.assertEqual(b.sha256(self.dest/'romfs/romfs.bin'), b.sha256(reference/'romfs.bin'))
        self.assertEqual(json.loads((self.dest/'romfs/index.json').read_text())['files'],
                         json.loads((reference/'index.json').read_text())['files'])
        for name in ('main', 'sdk', 'rtld', 'main.npdm'):
            self.assertEqual((self.dest/'exefs'/name).read_bytes(), (self.exefs/name).read_bytes())
        extract = next(c for step, c in self.calls if step == 'dkb_data')
        self.assertIn('--pack', extract)
        self.assertEqual(extract[extract.index('--select') + 1], 'latest')
        # Resume with the destination alone reuses the recorded container.
        self.calls.clear()
        self.options.nup = None
        self.assertEqual(self.run_install()[0], 0)
        self.assertEqual(self.calls, [])

    def test_nup_with_update_builds_the_update_with_merged_romfs(self):
        base_romfs = tree(self.romfs)
        base_exefs = tree(self.exefs)
        self.update_fixture()   # the fixture's ExeFS folder now holds the update's files and hashes
        update = {'exefs': tree(self.exefs), 'romfs': {'other.bin': b'patched by the update', 'added.bin': b'new'}}
        for name, data in base_exefs.items():
            (self.exefs / name).write_bytes(data)
        self.write_nup(update)
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual(b.statefile.load(self.dest, b.STATE)['game'], 'dkb458752')
        self.assertEqual((self.dest/'exefs/main').read_bytes(), update['exefs']['main'])
        merged = Path(self.work.name) / 'merged romfs'
        (merged / 'folder').mkdir(parents=True)
        (merged / 'folder' / 'café.bin').write_bytes(base_romfs['folder']['café.bin'])
        (merged / 'other.bin').write_bytes(b'patched by the update')
        (merged / 'added.bin').write_bytes(b'new')
        self.assertEqual(b.sha256(self.dest/'romfs/romfs.bin'), b.sha256(self.loose_pack(merged)/'romfs.bin'))

    def test_nup_unsupported_update_is_refused_and_base_can_be_chosen(self):
        update = {'exefs': {'main': b'a future update', 'main.npdm': b'npdm'}, 'romfs': {'other.bin': b'changed'}}
        self.write_nup(update)
        code, events = self.run_install()
        self.assertEqual(code, 1)
        error = next(e for e in events if e['type'] == 'error')
        self.assertEqual((error['id'], error['params']['title_version']), ('dkb.nup_version', 458752))
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.temp.exists())
        self.options.nup_package = 'base'
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual(b.sha256(self.dest/'romfs/romfs.bin'), b.sha256(self.loose_pack(self.romfs)/'romfs.bin'))

    def test_nup_with_mismatched_exefs_writes_nothing(self):
        (self.exefs / 'sdk').write_bytes(b'another build')
        self.write_nup()
        code, events = self.run_install()
        self.assertEqual(code, 1)
        self.assertEqual(next(e['id'] for e in events if e['type'] == 'error'), 'dkb.exefs_invalid')
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.temp.exists())

    def test_nup_and_folders_together_and_bad_container_are_refused(self):
        path = self.write_nup()
        self.options.exefs = self.exefs
        code, events = self.run_install()
        self.assertEqual(next(e['id'] for e in events if e['type'] == 'error'), 'dkb.nup_and_folders')
        self.options.exefs = None
        # A package choice with folders and no .nup is refused too, not ignored.
        self.options.nup, self.options.nup_package, self.options.exefs = None, 'base', self.exefs
        code, events = self.run_install()
        self.assertEqual(next(e['id'] for e in events if e['type'] == 'error'), 'dkb.nup_and_folders')
        self.options.nup, self.options.nup_package, self.options.exefs = path, None, None
        path.write_bytes(b'PFS0' + bytes(0x200))
        code, events = self.run_install()
        self.assertEqual(next(e['id'] for e in events if e['type'] == 'error'), 'dkb.nup_invalid')
        self.options.nup = path.with_name('missing.nup')
        code, events = self.run_install()
        self.assertEqual(next(e['id'] for e in events if e['type'] == 'error'), 'dkb.need_nup')
        self.assertFalse(self.dest.exists())

    def test_nup_inside_destination_is_refused(self):
        self.dest.mkdir()
        self.write_nup()
        moved = self.dest / 'game.nup'
        shutil.move(self.options.nup, moved)
        self.options.nup = moved
        code, events = self.run_install()
        self.assertEqual(code, 1)
        self.assertIn(next(e['id'] for e in events if e['type'] == 'error'), ('dkb.path_overlap', 'dkb.dest_unrelated'))

    def test_corrupt_nup_file_stops_before_installing_data(self):
        path = self.write_nup()
        data = bytearray(path.read_bytes())
        data[-20] ^= 0xFF   # inside a RomFS extent: the table of contents still checks out
        path.write_bytes(bytes(data))
        code, events = self.run_install()
        self.assertEqual(code, 1)
        error = next(e for e in events if e['type'] == 'error')
        self.assertEqual((error['id'], error.get('step')), ('dkb.tool_failed', 'dkb_data'))
        self.assertFalse((self.dest/'romfs/romfs.bin').exists())
        self.assertFalse((self.dest/'exefs').exists())

    def test_form_takes_a_nup_instead_of_folders(self):
        path = self.write_nup()
        form = Form(game='bananza', dest=str(self.dest), nup=str(path))
        self.assertEqual(form.problems(), [])
        command = form.command('python')
        self.assertEqual(command[command.index('--nup') + 1], str(path))
        self.assertNotIn('--exefs', command)
        self.assertEqual(Form(game='bananza', dest=str(self.dest), nup=str(path), exefs=str(self.exefs)).problems(),
                         ['dkb.nup_and_folders'])
        self.assertEqual(Form(game='bananza', dest=str(self.dest), nup=str(path) + '.missing').problems(),
                         ['dkb.need_nup'])
        self.assertEqual(Form(game='bananza', dest=str(self.dest)).problems(), ['dkb.need_exefs', 'dkb.need_romfs'])

    def stale_tools(self):
        """The receipt of a finished install, with tool paths of a release folder that was deleted since."""
        self.assertEqual(self.run_install()[0], 0)
        state = b.statefile.load(self.dest, b.STATE)
        gone = Path(self.work.name) / 'deleted release'
        state['build_tools'] = {'toolchain': str(gone / 'llvm'), 'deps': str(gone / 'runtime/deps'),
                                'ninja': str(gone / 'bin' / ('ninja' + b.EXE)), 'dll_dir': str(gone / 'bin'),
                                'dotnet': str(gone / 'dotnet')}
        b.statefile.save(self.dest, state, b.STATE)
        self.calls.clear()
        self.options.exefs = self.options.romfs = None

    def test_tool_paths_of_a_deleted_release_are_not_reused(self):
        self.stale_tools()
        self.options.package = None
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual(b.statefile.load(self.dest, b.STATE)['build_tools']['ninja'], str(self.layout.ninja.resolve()))

    def test_a_release_uses_its_own_tools_over_recorded_ones(self):
        self.stale_tools()
        other = Path(self.work.name) / 'older release'
        shutil.copytree(self.root / 'bin', other / 'bin')
        state = b.statefile.load(self.dest, b.STATE)
        state['build_tools']['ninja'] = str(other / 'bin' / ('ninja' + b.EXE))
        b.statefile.save(self.dest, state, b.STATE)
        self.options.package = None
        self.layout = b.replace(self.layout, dev=False)
        code, events = self.run_install()
        self.assertEqual(code, 0, events)
        self.assertEqual(b.statefile.load(self.dest, b.STATE)['build_tools']['ninja'], str(self.layout.ninja.resolve()))

    def test_renamed_translator_is_installed_under_runtime_name(self):
        renamed = self.root/'bin/custom-translator.exe'
        shutil.copyfile(self.options.translator, renamed)
        shutil.copyfile(self.options.translator.with_suffix('.id'), renamed.with_suffix('.id'))
        self.options.translator = renamed
        self.assertEqual(self.run_install()[0], 0)
        self.assertTrue((self.dest/('totk_xlate'+b.EXE)).is_file())
        self.assertTrue((self.dest/'totk_xlate.id').is_file())
        self.assertFalse((self.dest/renamed.name).exists())

    def test_wrong_translator_identity_is_rejected_before_writes(self):
        self.options.translator.with_suffix('.id').write_text('different build')
        self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.temp.exists())

    def test_source_translator_build_keeps_build_servers_out_of_owned_workspace(self):
        # A real child observes the scoped settings; the caller's build environment stays untouched.
        builder = self.root/'tools/build_xlate.py'
        builder.write_text('''
import json, os, pathlib, sys
a=sys.argv; out=pathlib.Path(a[a.index('--out')+1]); out.mkdir(parents=True)
(out/('totk_xlate.exe' if sys.platform=='win32' else 'totk_xlate')).write_bytes(b'synthetic translator')
(out/'totk_xlate.id').write_text('synthetic-translator-id\\n')
(out/'child-environment.json').write_text(json.dumps(dict(os.environ)))
''')
        self.temp.mkdir()
        ctx = SimpleNamespace(temp=self.temp, dest=self.dest, layout=self.layout, options=self.options,
                              reporter=Reporter(), info={'translator':None, 'identity':'synthetic-translator-id',
                                                         'dotnet':Path(sys.executable)})
        caller = dict(os.environ)
        with contextlib.redirect_stdout(io.StringIO()):
            record = b.Translator().run(ctx)
        child = json.loads((self.temp/'xlate/child-environment.json').read_text())
        self.assertEqual(child.get('MSBUILDDISABLENODEREUSE'), '1')
        self.assertEqual(child.get('DOTNET_CLI_USE_MSBUILD_SERVER'), '0')
        # Windows treats environment names case-insensitively, including imported MSBuild properties.
        properties = {key.lower():value for key,value in child.items()}
        self.assertEqual(properties.get('usesharedcompilation'), 'false')
        self.assertEqual(record['sha256'], b.sha256(self.dest/('totk_xlate'+b.EXE)))
        self.assertEqual(dict(os.environ), caller)

    def test_linked_source_is_rejected_before_writes(self):
        actual = b.linked
        with patch.object(b, 'linked', side_effect=lambda p: p == self.exefs/'main' or actual(p)):
            self.assertEqual(self.run_install()[0], 1)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.temp.exists())

    def test_bananza_form_and_catalog(self):
        form = Form(game='bananza', exefs=str(self.exefs), romfs=str(self.romfs), dest=str(self.dest))
        self.assertEqual(form.problems(), [])
        command = form.command(sys.executable, tools={'jobs':2,'translator':self.options.translator})
        self.assertIn('--exefs', command)
        self.assertNotIn('--keys', command)
        self.assertNotIn('--update', command)
        self.assertIn('dkb.exe' if os.name=='nt' else 'dkb', Catalog('en','bananza')('ui.done_text', dest='here'))
        self.assertEqual(Catalog('en','totk')('ui.title'), 'TotK Setup')

    def test_native_gui_finish_fixtures_cover_all_real_steps(self):
        from installer.gui import samples, SAMPLE_OK, SAMPLE_RUN
        from installer.guilogic import Model, DKB_STEP_ORDER, DONE
        model = Model()
        for event in samples(SAMPLE_OK, 'bananza'):
            model.apply(event)
        model.finish(0)
        self.assertEqual(tuple(model.steps), DKB_STEP_ORDER)
        self.assertEqual(set(model.states.values()), {DONE})
        self.assertEqual(model.dest, 'Z:/selftest/Bananza')
        cancelled = Model()
        for event in samples(SAMPLE_RUN[:3], 'bananza'):
            cancelled.apply(event)
        cancelled.finish(3)
        self.assertEqual(tuple(cancelled.steps), DKB_STEP_ORDER)
        self.assertEqual(cancelled.dest, 'Z:/selftest/Bananza')

    def test_python312_cli_does_not_import_container_extractor(self):
        code = "import installer.cli,installer.bananza,sys; assert 'extract' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dest_only_resume_uses_bananza_dispatch_on_python312(self):
        self.assertEqual(self.run_install()[0], 0)
        result = subprocess.run([sys.executable, '-m', 'installer', '--dest', str(self.dest)],
                                cwd=ROOT, capture_output=True, text=True)
        # This checkout's pinned real hashes reject the synthetic fixture, after selecting the correct pipeline.
        self.assertEqual(result.returncode, 1, result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(next(e['id'] for e in events if e['type']=='error'), 'dkb.exefs_invalid')

    def test_catalogs_cover_same_messages_and_placeholders(self):
        import re
        english, russian = Catalog('en').english, Catalog('ru').texts
        self.assertEqual(set(english), set(russian))
        for key in english:
            self.assertEqual(sorted(re.findall(r'\{\w+\}', english[key])),
                             sorted(re.findall(r'\{\w+\}', russian[key])), key)


if __name__ == '__main__':
    unittest.main()
