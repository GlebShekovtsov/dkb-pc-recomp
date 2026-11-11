"""Validated Bananza installs from ExeFS and RomFS folders or a .nup container. No NCA extraction or keys.

The existing runner owns step events and atomic checkpoints. Game-derived temporary files stay in a separately
marked task directory; only verified files enter the portable destination. No prebuilt TotK runtime is reused.

A .nup (the container Eglantine and Eden load) is read with tools/extract_nup.py: preflight reads only its table of
contents, which records every file's size and SHA-256, so the target is chosen and the ExeFS checked before anything
is written; the Data step then extracts the ExeFS and packs the RomFS straight from the container.
"""
import hashlib
import copy
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from . import layout as layouts
from . import state as statefile
from . import ini
from .events import Reporter
from .runner import Context, InstallError, Step, run_steps

STATE = 'dkb-install.json'
OWNER = '.dkb-setup-owner.json'
DEST_OWNER = '.dkb-install-owner.json'
GAME = 'dkb100'
TARGETS = (GAME, 'dkb458752')   # 1.0.0 and 3.2.0 (title version 458752); tools/data/expected-<target>.json
NUP_SELECT = ('latest', 'base')
EXE = '.exe' if os.name == 'nt' else ''
CHUNK = 8 << 20
COMMIT_WARN_GIB = 24   # the game commits about 14 GB; README asks for 16 GB of RAM plus a large page file


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode() + b'\0')
    return h.hexdigest()


def linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def contains(parent, child):
    return child == parent or parent in child.parents


def ensure_plain(path):
    if any(linked(p) for p in (path, *path.parents)):
        raise InstallError('dkb.path_link', path=str(path))


def plain_tree(root):
    """Reject existing redirects without descending through them, before a tool can reuse the workspace."""
    ensure_plain(root)
    if not root.is_dir():
        return
    def failed(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        for name in dirs + files:
            p = Path(directory) / name
            if linked(p):
                raise InstallError('dkb.path_link', path=str(p))


def plain_destination(dest, dlls=()):
    # profile/ and caches belong to the player and are never traversed or written by setup.
    files = (STATE, STATE + '.tmp', DEST_OWNER, '.dkb-setup.lock', 'dkb.ini', 'dkb.ini.tmp', 'totk.ini',
             'totk.ini.tmp', 'romfs-index.bin', 'seed-nvn2.bin', 'dkb' + EXE, 'totk_xlate' + EXE, 'totk_xlate.id', *(p.name for p in dlls))
    for name in ('exefs', 'romfs', *files, *(name + '.setup-part' for name in files)):
        plain_tree(dest / name)


def plain_outputs(ctx):
    plain_tree(ctx.temp)
    plain_destination(ctx.dest, ctx.layout.dlls)


def tree_files(root):
    """Same file selection as dkb_pack_romfs; refuse links instead of following data outside the selected folder."""
    result = []
    for directory, dirs, files in os.walk(root):
        for name in dirs + files:
            p = Path(directory) / name
            if linked(p):
                raise InstallError('dkb.path_link', path=str(p))
        result += [Path(directory) / n for n in files if not n.startswith('.')]
    return sorted(result, key=lambda p: p.relative_to(root).as_posix().encode())


def inventory(root, reporter):
    files = tree_files(root)
    if not files:
        raise InstallError('dkb.romfs_empty', path=str(root))
    result = []
    for i, p in enumerate(files):
        before = p.stat()
        value = sha256(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise InstallError('dkb.input_changed', path=str(p))
        result.append({'path': p.relative_to(root).as_posix(), 'size': before.st_size, 'sha256': value})
        reporter.progress('check', 'dkb_hash', i + 1, len(files), 'files')
    return result


def pack_manifest(folder):
    try:
        m = json.loads((folder / 'index.json').read_text(encoding='utf-8'))
        if m.get('status') != 'pass' or not isinstance(m['files'], list) or not m['files']:
            raise ValueError('missing verified file index')
        size, previous = m['romfs_size'], -1
        paths = set()
        for e in m['files']:
            p = Path(e['path'])
            if (not e['path'] or p.is_absolute() or p.drive or '..' in p.parts or '\\' in e['path'] or '\0' in e['path'] or
                    e['path'] in paths or e['offset'] < 0 or e['offset'] < previous or e['size'] < 0 or
                    e['offset'] + e['size'] > size):
                raise ValueError('invalid file range or path')
            paths.add(e['path'])
            previous = e['offset'] + e['size']
        image = folder / 'romfs.bin'
        if image.stat().st_size != size or sha256(image) != m['romfs_sha256']:
            raise ValueError('packed image checksum mismatch')
        return m
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise InstallError('dkb.pack_invalid', path=str(folder), detail=str(e)) from None


def expected_exefs(layout, exefs, game=GAME):
    expected = json.loads((layout.tools / 'data' / f'expected-{game}.json').read_text(encoding='utf-8'))
    hashes = expected['exefs_sha256']
    for name, value in hashes.items():
        p = exefs / name
        ensure_plain(p)
        if not p.is_file() or sha256(p) != value:
            raise InstallError('dkb.exefs_invalid', path=str(p))
    if any(exefs.glob('subsdk*')):
        raise InstallError('dkb.update_unsupported')
    return hashes


def select_target(layout, exefs):
    main = exefs / 'main'
    ensure_plain(main)
    if main.is_file():
        value = sha256(main)
        for game in TARGETS:
            path = layout.tools / 'data' / f'expected-{game}.json'
            if path.is_file():
                expected = json.loads(path.read_text(encoding='utf-8'))
                if expected['exefs_sha256']['main'] == value:
                    return game
    raise InstallError('dkb.exefs_invalid', path=str(main))


def tool_layout(layout, options):
    package = (options.package or layout.root).resolve()
    llvm = (options.toolchain or (package / 'llvm' if (package / 'llvm').is_dir() else layout.llvm)).resolve()
    deps = options.deps or (package / 'runtime' / 'deps' if (package / 'runtime' / 'deps').is_dir() else layout.deps)
    ninja = options.ninja or (package / 'bin' / ('ninja' + EXE))
    if not ninja.is_file():
        ninja = layout.ninja if layout.ninja.is_file() else Path(shutil.which('ninja') or ninja)
    dll_dir = options.dll_dir
    if dll_dir is None:
        dll_dir = deps / 'bin' if deps and (deps / 'bin' / 'SDL3.dll').is_file() else package / 'bin'
    dlls = tuple(sorted(dll_dir.glob('*.dll'))) if os.name == 'nt' else layout.dlls
    return replace(layout, llvm=llvm, deps=deps.resolve() if deps else None, ninja=ninja.resolve(), dlls=dlls)


def source_key(layout, *folders):
    values = []
    for folder in folders:
        base = layout.root / folder
        for p in sorted(base.rglob('*')) if base.is_dir() else [base]:
            if p.is_file() and '__pycache__' not in p.parts:
                values.append((p.relative_to(layout.root).as_posix(), sha256(p)))
    return digest(values)


def tree_key(base):
    return digest([(p.relative_to(base).as_posix(), sha256(p)) for p in sorted(base.rglob('*'))
                   if p.is_file() and '__pycache__' not in p.parts])


def tool_module(layout, name):
    spec = importlib.util.spec_from_file_location('_dkb_' + name.removesuffix('.py'), layout.tools / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def translator_identity(layout):
    return tool_module(layout, 'build_xlate.py').identity()


def nup_source(layout, path, select):
    """What a .nup install takes from the container, from its table of contents alone: the target (by the recorded
    SHA-256 of main), the ExeFS hashes checked against the target's, and the RomFS file list (path, size, SHA-256) in
    the format and order inventory() gives for a loose folder. Nothing is decompressed or written."""
    tool = tool_module(layout, 'extract_nup.py')
    try:
        container = tool.Nup(path)
    except (tool.NupError, OSError, ValueError, UnicodeDecodeError) as e:
        raise InstallError('dkb.nup_invalid', path=str(path), detail=str(e)) from None
    try:
        plan = tool.install_plan(container, select)
        exefs = {'/'.join(parts): idx for parts, idx in tool.merged_entries(container, [plan['exefs']], 'exefs')}
        romfs = tool.packed_entries(tool.merged_entries(container, plan['romfs'], 'romfs'))
        package = plan['exefs']
        version = package.display_version or str(package.version)
        recorded = lambda name: container.files[exefs[name]][4].hex() if name in exefs else None
        game = next((g for g in TARGETS if (layout.tools / 'data' / f'expected-{g}.json').is_file() and
                     json.loads((layout.tools / 'data' / f'expected-{g}.json').read_text(encoding='utf-8'))
                     ['exefs_sha256']['main'] == recorded('main')), None)
        if game is None:
            raise InstallError('dkb.nup_version', path=str(path), version=version, title_version=package.version)
        hashes = json.loads((layout.tools / 'data' / f'expected-{game}.json').read_text(encoding='utf-8'))['exefs_sha256']
        for name, value in hashes.items():
            if recorded(name) != value:
                raise InstallError('dkb.exefs_invalid', path=f'{path} (exefs/{name})')
        if any(name.startswith('subsdk') for name in exefs):
            raise InstallError('dkb.update_unsupported')
        files = [{'path': '/'.join(parts), 'size': container.files[idx][1], 'sha256': container.files[idx][4].hex()}
                 for parts, idx in romfs]
        if not files:
            raise InstallError('dkb.romfs_empty', path=f'{path} (romfs)')
        compressed = any(container.files[i][5] for i in (*exefs.values(), *(idx for _, idx in romfs)))
    except tool.NupError as e:
        raise InstallError('dkb.nup_invalid', path=str(path), detail=str(e)) from None
    finally:
        container.close()
    if compressed:
        # The Data step decompresses in layout.python, which can differ from the Python running this check.
        probe = subprocess.run([str(layout.python), str(layout.tools / 'extract_nup.py'), '--check-zstd'],
                               capture_output=True, text=True)
        if probe.returncode:
            raise InstallError('dkb.nup_zstd', path=str(path))
    return game, hashes, files


def preflight(layout, options, dest, temp, rep):
    if options.base or options.update or options.keys:
        raise InstallError('dkb.update_unsupported')
    if options.jobs < 1:
        raise InstallError('dkb.jobs')
    if options.lang and ini.canonical_language(options.lang) not in ini.LANGUAGES:
        raise InstallError('ui.need_lang')
    ensure_plain(dest / STATE)
    old = statefile.load(dest, STATE)
    if old is not None and (old.get('game') not in TARGETS or old.get('destination') != str(dest)):
        raise InstallError('dkb.dest_unrelated', path=str(dest))
    nup_arg = getattr(options, 'nup', None)
    # A package choice only means something for a .nup, so it counts as asking for one next to folders too.
    if (nup_arg or getattr(options, 'nup_package', None)) and (options.exefs or options.romfs):
        raise InstallError('dkb.nup_and_folders')
    recorded = old['sources'] if old else {}
    if not nup_arg and not options.exefs and not options.romfs and recorded.get('nup'):
        nup_arg = Path(recorded['nup'])
    select = getattr(options, 'nup_package', None) or (recorded.get('nup_package') if nup_arg else None) or 'latest'
    if select not in NUP_SELECT:
        raise InstallError('dkb.nup_invalid', path=str(nup_arg), detail=f'unknown package choice {select}')
    if nup_arg:
        for p in (nup_arg, dest, temp):
            ensure_plain(p.absolute())
        nup = nup_arg.resolve()
        if not nup.is_file():
            raise InstallError('dkb.need_nup')
        rep.progress('check', 'dkb_base')
        game, hashes, files = nup_source(layout, nup, select)
        inputs, sources = (nup,), {'nup': str(nup), 'nup_package': select}
    else:
        exefs_arg = options.exefs or (Path(recorded['exefs']) if recorded.get('exefs') else None)
        romfs_arg = options.romfs or (Path(recorded['romfs']) if recorded.get('romfs') else None)
        if not exefs_arg or not romfs_arg:
            raise InstallError('dkb.need_folders')
        for p in (exefs_arg, romfs_arg, dest, temp):
            ensure_plain(p.absolute())
        exefs, romfs = exefs_arg.resolve(), romfs_arg.resolve()
        if not exefs.is_dir() or not romfs.is_dir():
            raise InstallError('dkb.need_folders')
        game = select_target(layout, exefs)
        inputs, sources = (exefs, romfs), {'exefs': str(exefs), 'romfs': str(romfs)}
    if old is not None and old['game'] != game:
        raise InstallError('dkb.dest_unrelated', path=str(dest))
    if dest.exists() and not dest.is_dir():
        raise InstallError('dkb.dest_unrelated', path=str(dest))
    for a, b in ((dest, temp), *((out, source) for out in (dest, temp) for source in inputs)):
        if contains(a, b) or contains(b, a):
            raise InstallError('dkb.path_overlap', path=str(a), other=str(b))
    if dest.exists() and old is None and any(dest.iterdir()):
        try:
            ensure_plain(dest / DEST_OWNER)
            partial = json.loads((dest / DEST_OWNER).read_text(encoding='utf-8'))
            ours = partial == {'game': game, 'destination': str(dest)}
            ours &= {p.name for p in dest.iterdir()} <= {DEST_OWNER, '.dkb-setup.lock'}
        except (OSError, ValueError):
            ours = False
        if not ours:
            raise InstallError('dkb.dest_unrelated', path=str(dest))
    if nuget_needs_fallback(temp) and any(c.isspace() for c in tempfile.gettempdir()):
        raise InstallError('dkb.temp_space', path=str(temp))
    owner = {'game': game, 'destination': str(dest), 'temporary': str(temp)}
    for p in (temp / OWNER, dest / STATE, dest / (STATE + '.tmp'), dest / DEST_OWNER):
        ensure_plain(p)
    if temp.exists() and any(temp.iterdir()):
        try:
            actual = json.loads((temp / OWNER).read_text(encoding='utf-8'))
        except (ValueError, OSError):
            actual = None
        if actual != owner:
            raise InstallError('dkb.temp_unrelated', path=str(temp))
    plain_tree(temp)
    plain_destination(dest, layout.dlls)
    # Reject mixed-version inputs before any writer can replace a checkpoint or installed file.
    # A .nup's ExeFS was checked against its table of contents above, and its file list has the identity of a loose
    # folder holding the same files.
    packed = False
    if not nup_arg:
        rep.progress('check', 'dkb_base')
        hashes = expected_exefs(layout, exefs, game)
        packed = (romfs / 'index.json').is_file() and (romfs / 'romfs.bin').is_file()
    if packed:
        ensure_plain(romfs / 'index.json')
        ensure_plain(romfs / 'romfs.bin')
        manifest = pack_manifest(romfs)
        files = None
        romfs_id = digest(manifest)
        romfs_size = manifest['romfs_size']
    else:
        files = files if nup_arg else inventory(romfs, rep)
        romfs_id = digest(files)
        romfs_size = sum(e['size'] + 15 for e in files)
    missing = [p for p in (layout.clang, layout.ninja, layout.recomp / 'gen.py',
                           layout.recomp / f'patches-{game}.txt', layout.tools / 'dkb_pack_romfs.py',
                           *([layout.tools / 'extract_nup.py'] if nup_arg else []),
                           layout.tools / 'dkb_shader_census.py',
                           layout.runtime / 'nvn2_lower.cpp', layout.tools / 'data' / f'expected-{game}.json')
               if not p.is_file()]
    if layout.deps:
        missing += [p for p in (layout.deps / 'include', layout.deps / 'lib') if not p.is_dir()]
    if os.name == 'nt':
        names = {p.name for p in layout.dlls}
        if not {'SDL3.dll', 'libopus-0.dll'} <= names:
            raise InstallError('dkb.dlls_missing')
    if missing:
        raise InstallError('dkb.tools_missing', paths='; '.join(map(str, missing)))
    identity = translator_identity(layout)
    candidates = [options.translator] if options.translator else [layout.xlate, dest / ('totk_xlate' + EXE)]
    translator = None
    for candidate in candidates:
        if candidate.is_file() and candidate.with_suffix('.id').is_file():
            if candidate.with_suffix('.id').read_text(encoding='ascii').strip() == identity:
                translator = candidate.resolve()
                break
    if options.translator and translator is None:
        raise InstallError('dkb.translator_mismatch')
    dotnet = options.dotnet or layout.root / 'toolchain' / 'dotnet' / ('dotnet' + EXE)
    if not dotnet.is_file():
        dotnet = Path(shutil.which('dotnet') or dotnet)
    if not translator:
        proc = subprocess.run([str(dotnet), '--list-sdks'], capture_output=True, text=True) if dotnet.is_file() else None
        if proc is None or proc.returncode or not any(line.startswith('10.') for line in proc.stdout.splitlines()):
            raise InstallError('dkb.dotnet_missing')
    return dict(old=old, game=game, sources=sources, hashes=hashes, files=files,
                romfs_id=romfs_id, romfs_size=romfs_size, owner=owner, translator=translator, identity=identity,
                dotnet=dotnet, packed=packed,
                pipeline=digest(*(sha256(Path(__file__).with_name(n)) for n in ('bananza.py', 'runner.py', 'state.py'))))


class BananzaContext(Context):
    def save(self):
        plain_outputs(self)
        ensure_plain(self.dest / STATE)
        ensure_plain(self.dest / (STATE + '.tmp'))
        self.dest.mkdir(parents=True, exist_ok=True)
        statefile.save(self.dest, self.state, STATE)


def owned_dir(ctx, path):
    ensure_plain(path)
    if not contains(ctx.temp, path) or path == ctx.temp:
        raise InstallError('dkb.path_overlap', path=str(path), other=str(ctx.temp))
    if json.loads((ctx.temp / OWNER).read_text(encoding='utf-8')) != ctx.info['owner']:
        raise InstallError('dkb.temp_unrelated', path=str(ctx.temp))
    if path.exists():
        plain_tree(path)
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(src, dst):
    ensure_plain(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(dst.name + '.setup-part')
    ensure_plain(temporary)
    try:
        shutil.copyfile(src, temporary)
        try:
            shutil.copymode(src, temporary)  # keeps the executable bit
        except OSError:
            pass
        os.replace(temporary, dst)
    except PermissionError:
        raise InstallError('compile.exe_in_use', path=str(dst)) from None


def nuget_needs_fallback(temp):
    """NativeAOT hands its native libraries to clang unquoted on POSIX, so NuGet must live where no path has a space."""
    return os.name != 'nt' and any(c.isspace() for c in str(temp / 'nuget'))


def lock_temp(temp):
    """An OS-held lock, released even after a crashed process; no PID guessing or stale-lock deletion."""
    path = temp / '.dkb-setup.lock'
    ensure_plain(path)
    stream = path.open('a+b')
    if path.stat().st_size == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise InstallError('dkb.busy', path=str(temp)) from None
    return stream


def run_tool(ctx, step, command, env=None, cwd=None, name=None):
    plain_outputs(ctx)
    log = ctx.temp / 'logs' / f'{name or step}.log'
    ensure_plain(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    ctx.reporter.progress(step, 'dkb_run', force=True)
    with log.open('w', encoding='utf-8') as out:
        try:
            proc = subprocess.Popen(list(map(str, command)), cwd=cwd or ctx.layout.root, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', **(env or {})),
                                    start_new_session=os.name != 'nt')
        except OSError as e:
            if getattr(e, 'winerror', None) != 740:   # ERROR_ELEVATION_REQUIRED
                raise
            ctx.reporter.log(f'{step}: Windows requires elevation to start {command[0]}')
            raise InstallError('dkb.elevation_required', exe=Path(str(command[0])).name) from None
        try:
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code:
        raise InstallError('dkb.tool_failed', step=step, code=code, log=str(log))
    plain_outputs(ctx)


class Check(Step):
    id, always = 'check', True

    def run(self, ctx):
        disk = lambda path: next(p for p in (path, *path.parents) if p.exists())
        dest_disk, temp_disk = disk(ctx.dest), disk(ctx.temp)
        free, temp_free = shutil.disk_usage(dest_disk).free, shutil.disk_usage(temp_disk).free
        same_disk = dest_disk.stat().st_dev == temp_disk.stat().st_dev
        temp_need = ctx.info['romfs_size'] + (10 << 30)
        need = ctx.info['romfs_size'] + (3 << 30) + (temp_need if same_disk else 0)
        ctx.reporter.emit('space', dest=str(ctx.dest), need=need, free=free, temp=str(ctx.temp),
                          temp_need=temp_need, temp_free=temp_free, same_disk=same_disk)
        if free < need or temp_free < temp_need:
            raise InstallError('dkb.space', need=round(need / (1 << 30), 1))
        ctx.temp.mkdir(parents=True, exist_ok=True)
        self.host(ctx)
        ctx.verify_kept(self.id)

    def host(self, ctx):
        exe = ctx.temp / ('hostcheck' + EXE)
        command = [ctx.layout.clang, '-O1', ctx.layout.root / 'installer' / 'hostcheck.c', '-o', exe]
        if ctx.layout.deps:
            command += ['-I', ctx.layout.deps / 'include']
        if os.name != 'nt':
            command += ['-ldl']
        run_tool(ctx, self.id, command)
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        try:
            host = json.loads(result.stdout)
        except ValueError:
            raise InstallError('check.host_probe_failed', code=result.returncode) from None
        if not host['x86_64_v3']:
            raise InstallError('check.cpu_no_avx2')
        if not host['loader']:
            raise InstallError('check.vulkan_missing')
        if not any(tuple(d['api'][:2]) >= (1, 3) for d in host['devices']):
            raise InstallError('check.vulkan_no_device')
        commit = host.get('commit_limit_gib')   # Windows only; the game commits about 14 GB, so leave room for the system
        if commit is not None and commit < COMMIT_WARN_GIB:
            ctx.reporter.warning('warn.low_commit_limit', commit=f'{commit:.0f}', need=COMMIT_WARN_GIB)


class Data(Step):
    id = 'dkb_data'

    def key(self, ctx):
        parts = [ctx.info['hashes'], ctx.info['romfs_id'], ctx.info['pipeline'],
                 sha256(ctx.layout.tools / 'dkb_pack_romfs.py')]
        if 'nup' in ctx.info['sources']:   # folder installs keep the key they had before .nup input existed
            parts += [sha256(ctx.layout.tools / 'extract_nup.py'), ctx.info['sources']['nup_package']]
        return digest(*parts)

    def current(self, ctx, record):
        try:
            expected_exefs(ctx.layout, ctx.dest / 'exefs', ctx.info['game'])
            image = ctx.dest / 'romfs' / 'romfs.bin'
            return (image.stat().st_size == record['size'] and image.stat().st_mtime_ns == record['mtime_ns'] and
                    sha256(ctx.dest / 'romfs' / 'index.json') == record['index_sha256'])
        except (OSError, InstallError, KeyError):
            return False

    def verify(self, ctx, record):
        try:
            return pack_manifest(ctx.dest / 'romfs')['romfs_sha256'] == record['sha256']
        except InstallError:
            return False

    def run(self, ctx):
        stage = ctx.temp / 'data'
        owned_dir(ctx, stage)
        sources = ctx.info['sources']
        if 'nup' in sources:
            # Every extracted file is checked against the container's SHA-256 record; the packed image is read back
            # below against the preflight list taken from the same records.
            romfs = Path(sources['nup'])
            run_tool(ctx, self.id, [ctx.layout.python, ctx.layout.tools / 'extract_nup.py', romfs,
                                     '--select', sources['nup_package'], '--pack', '--out', stage])
        else:
            exefs = Path(sources['exefs'])
            for name in ctx.info['hashes']:
                copy_file(exefs / name, stage / 'exefs' / name)
            romfs = Path(sources['romfs'])
        expected_exefs(ctx.layout, stage / 'exefs', ctx.info['game'])
        if ctx.info['packed']:
            for name in ('romfs.bin', 'index.json'):
                copy_file(romfs / name, stage / 'romfs' / name)
        elif 'nup' not in sources:
            run_tool(ctx, self.id, [ctx.layout.python, ctx.layout.tools / 'dkb_pack_romfs.py', '--dir', romfs,
                                     '--out', stage / 'romfs'])
        manifest = pack_manifest(stage / 'romfs')
        if ctx.info['packed'] and digest(manifest) != ctx.info['romfs_id']:
            raise InstallError('dkb.input_changed', path=str(romfs))
        if ctx.info['files'] is not None:
            # Compare every packed file with the preflight content snapshot; no NCA authenticity is claimed.
            files = ctx.info['files']
            if [(e['path'], e['size']) for e in manifest['files']] != [(e['path'], e['size']) for e in files]:
                raise InstallError('dkb.input_changed', path=str(romfs))
            with (stage / 'romfs' / 'romfs.bin').open('rb') as packed:
                for entry, source in zip(manifest['files'], files):
                    packed.seek(entry['offset'])
                    h, remaining = hashlib.sha256(), entry['size']
                    while remaining:
                        block = packed.read(min(CHUNK, remaining))
                        if not block:
                            raise InstallError('dkb.pack_invalid', path=str(romfs), detail='short file')
                        h.update(block)
                        remaining -= len(block)
                    if h.hexdigest() != source['sha256']:
                        raise InstallError('dkb.input_changed', path=str(romfs / source['path']))
        # Final names are replaced only after the complete stage has passed its checks.
        for name in ctx.info['hashes']:
            copy_file(stage / 'exefs' / name, ctx.dest / 'exefs' / name)
        for name in ('romfs.bin', 'index.json'):
            copy_file(stage / 'romfs' / name, ctx.dest / 'romfs' / name)
        image = ctx.dest / 'romfs' / 'romfs.bin'
        return {'sha256': manifest['romfs_sha256'], 'size': image.stat().st_size, 'mtime_ns': image.stat().st_mtime_ns,
                'index_sha256': sha256(ctx.dest / 'romfs' / 'index.json')}


class Translator(Step):
    id = 'dkb_translator'

    def key(self, ctx):
        return digest(ctx.info['identity'], sha256(ctx.info['translator']) if ctx.info['translator'] else 'build')

    def current(self, ctx, record):
        exe = ctx.dest / ('totk_xlate' + EXE)
        return (exe.is_file() and sha256(exe) == record.get('sha256') and exe.with_suffix('.id').is_file() and
                exe.with_suffix('.id').read_text(encoding='ascii').strip() == ctx.info['identity'])

    def run(self, ctx):
        source = ctx.info['translator']
        if source is None:
            out = ctx.temp / 'xlate'
            nuget = Path(tempfile.mkdtemp(prefix='dkb-nuget-')) if nuget_needs_fallback(ctx.temp) else ctx.temp / 'nuget'
            env = {'PATH': str(ctx.info['dotnet'].parent) + os.pathsep + os.environ.get('PATH', ''),
                   'DOTNET_ROOT': str(ctx.info['dotnet'].parent), 'DOTNET_CLI_HOME': str(ctx.temp / 'dotnet-home'),
                   'NUGET_PACKAGES': str(nuget),
                   'DOTNET_PROCESSOR_COUNT': str(ctx.options.jobs),
                   # Reusable MSBuild nodes retain NuGet task DLLs after publish and block Windows cleanup.
                   # These settings affect only this build, not the owner's other .NET work.
                   'MSBUILDDISABLENODEREUSE': '1', 'DOTNET_CLI_USE_MSBUILD_SERVER': '0',
                   'UseSharedCompilation': 'false'}
            command = [ctx.layout.python, ctx.layout.tools / 'build_xlate.py', '--out', out]
            if os.name != 'nt':
                command += ['--llvm', ctx.layout.llvm]
                if ctx.layout.sysroot:
                    command += ['--sysroot', ctx.layout.sysroot]
            try:
                run_tool(ctx, self.id, command, env)
            finally:
                if nuget.parent != ctx.temp:
                    shutil.rmtree(nuget, ignore_errors=True)
            source = out / ('totk_xlate' + EXE)
        if source.with_suffix('.id').read_text(encoding='ascii').strip() != ctx.info['identity']:
            raise InstallError('dkb.translator_mismatch')
        target = ctx.dest / ('totk_xlate' + EXE)
        copy_file(source, target)
        copy_file(source.with_suffix('.id'), target.with_suffix('.id'))
        return {'sha256': sha256(target)}


class Lift(Step):
    id, transient = 'lift', True

    def key(self, ctx):
        return digest(ctx.keys['dkb_data'], source_key(ctx.layout, 'recomp'))

    def current(self, ctx, record):
        gen = ctx.temp / 'gen'
        try:
            for name in ('main', 'sdk'):
                m = json.loads((gen / name / 'manifest.json').read_text(encoding='utf-8'))
                if not m['chunks'] or not all((gen / name / chunk).is_file() for chunk in m['chunks']):
                    return False
            return ((gen / 'layout.json').is_file() and (gen / 'patches.h').is_file() and
                    record.get('outputs') == tree_key(gen))
        except (OSError, KeyError, ValueError):
            return False

    def run(self, ctx):
        gen = ctx.temp / 'gen'
        plain_outputs(ctx)
        gen.mkdir(parents=True, exist_ok=True)
        run_tool(ctx, self.id, [ctx.layout.python, ctx.layout.recomp / 'gen.py', '--exefs', ctx.dest / 'exefs',
                                 '--out', gen, '--modules', 'main,sdk', '--jobs', ctx.options.jobs,
                                 '--patches', ctx.layout.recomp / f"patches-{ctx.info['game']}.txt"])
        manifest = json.loads((gen / 'main' / 'manifest.json').read_text(encoding='utf-8'))
        if ctx.info['game'] == GAME and manifest.get('stats', {}).get('patched_sites', 0) < 1:
            raise InstallError('dkb.patches_missing')
        return {'outputs': tree_key(gen)}


class Compile(Step):
    id = 'compile'

    def key(self, ctx):
        return digest(ctx.keys['lift'], source_key(ctx.layout, 'runtime', 'tools/build_game.py', 'tools/romfs_index.py'),
                      str(ctx.layout.llvm), str(ctx.layout.deps),
                      sha256(ctx.layout.clang), sha256(ctx.layout.ninja),
                      tree_key(ctx.layout.deps) if ctx.layout.deps else 'system-libraries',
                      [(str(p), sha256(p)) for p in ctx.layout.dlls])

    def current(self, ctx, record):
        exe = ctx.dest / ('dkb' + EXE)
        try:
            return (exe.is_file() and sha256(exe) == record.get('sha256') and
                    sha256(ctx.dest / 'romfs-index.bin') == record.get('index_sha256') and
                    all(sha256(ctx.dest / p.name) == sha256(p) for p in ctx.layout.dlls))
        except OSError:
            return False

    def run(self, ctx):
        out = ctx.temp / 'build'
        game = ctx.info['game']
        args = [ctx.layout.python, ctx.layout.tools / 'build_game.py', '--version', game, '--expected',
                ctx.layout.tools / 'data' / f'expected-{game}.json', '--gen', ctx.temp / 'gen', '--out', out,
                '--modules', 'main,sdk', '--exefs', ctx.dest / 'exefs', '--romfs', ctx.dest / 'romfs',
                '--profile', ctx.dest / 'profile', '--runtime', ctx.layout.runtime, '--toolchain', ctx.layout.llvm,
                '--ninja', ctx.layout.ninja, '--jobs', ctx.options.jobs, '--full-build']
        if ctx.layout.deps:
            args += ['--deps', ctx.layout.deps]
        if ctx.layout.sysroot:
            args += ['--sysroot', ctx.layout.sysroot]
        run_tool(ctx, self.id, args)
        for p in (out / ('dkb' + EXE), out / 'romfs-index.bin', *ctx.layout.dlls):
            copy_file(p, ctx.dest / p.name)
        return {'sha256': sha256(ctx.dest / ('dkb' + EXE)), 'index_sha256': sha256(ctx.dest / 'romfs-index.bin')}


class Shaders(Step):
    """Translate every shader stage in the game's files into the shader cache, so no draw waits for a first translation.

    tools/dkb_shader_census.py lists the stages (RomFS) into seed-nvn2.bin and dkb.exe --translate-shaders lowers and
    translates them. The cache belongs to the player and survives later steps; the 187 MiB pack is removed again."""
    id = 'dkb_shaders'

    def key(self, ctx):
        return digest(ctx.keys['compile'], ctx.keys['dkb_translator'], ctx.keys['dkb_data'],
                      sha256(ctx.layout.tools / 'dkb_shader_census.py'))

    def current(self, ctx, record):
        return record.get('prepared') is True and (ctx.dest / 'shader-cache').exists()

    def run(self, ctx):
        pack = ctx.dest / 'seed-nvn2.bin'
        run_tool(ctx, self.id, [ctx.layout.python, ctx.layout.tools / 'dkb_shader_census.py', '--version', ctx.info['game'],
                                '--romfs', ctx.dest / 'romfs', '--out', ctx.dest], name=f'{self.id}-census')
        try:
            run_tool(ctx, self.id, [ctx.dest / ('dkb' + EXE), '--translate-shaders'], cwd=ctx.dest)
        finally:
            pack.unlink(missing_ok=True)
        # The driver pipeline libraries are an optimisation: a driver that crashes building them (nvgpucomp64.dll on
        # NVIDIA 621.01) must not fail the step, or every retry would crash again.
        libraries = True
        try:
            run_tool(ctx, self.id, [ctx.dest / ('dkb' + EXE), '--warm-shaders'], cwd=ctx.dest,
                     name=f'{self.id}-warm')
        except InstallError as error:
            if error.id != 'dkb.tool_failed':   # a path-safety rejection still stops the install
                raise
            # On Windows Ctrl+C the child can exit 0xC000013A before the parent handler raises; that is recorded as
            # libraries:false here, which is harmless.
            libraries = False
            ctx.reporter.warning('warn.dkb_libraries_failed', code=error.params.get('code'),
                                 log=str(error.params.get('log')))
        return {'prepared': True, 'libraries': libraries}


class Prepare(Step):
    id = 'dkb_verify'

    def key(self, ctx):
        return digest(ctx.keys['compile'], ctx.keys['dkb_translator'], ctx.state['lang'])

    def current(self, ctx, record):
        return record.get('selftest') is True

    def run(self, ctx):
        for name in ('dkb.ini', 'dkb.ini.tmp', 'totk.ini'):
            ensure_plain(ctx.dest / name)
        ini.write(ctx.dest, ctx.state['lang'], ctx.layout.ini_example, 'dkb.ini', fallback='totk.ini')
        run_tool(ctx, self.id, [ctx.dest / ('dkb' + EXE), '--selftest'])
        return {'selftest': True}


class Cleanup(Step):
    id, always = 'cleanup', True

    def run(self, ctx):
        # Update 458752 keeps generated C and objects so a second install does not relift or recompile.
        keep = ctx.info['game'] != GAME
        for name in ('data', 'xlate', 'dotnet-home', 'nuget') if keep else ('data', 'gen', 'build', 'xlate', 'dotnet-home', 'nuget'):
            p = ctx.temp / name
            if p.exists():
                owned_dir(ctx, p)
                p.rmdir()


def all_steps():
    return [Check(), Data(), Translator(), Lift(), Compile(), Shaders(), Prepare(), Cleanup()]


def cancel(*_):
    raise KeyboardInterrupt


def write_owner(path, value):
    ensure_plain(path)
    try:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream)
    except FileExistsError:
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise InstallError('dkb.temp_unrelated', path=str(path.parent)) from None


def main(options):
    rep = Reporter()
    locks = []
    started = time.monotonic()
    try:
        for name in ('SIGBREAK', 'SIGTERM'):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), cancel)
        ensure_plain(options.dest.absolute())
        dest = options.dest.resolve()
        ensure_plain(dest / STATE)
        previous = statefile.load(dest, STATE)
        options = copy.copy(options)
        found = layouts.find(options.root)
        # A resume reuses the tool paths the last run recorded, for a checkout run with its own --toolchain or --deps.
        # A release carries its tools, and the folder of the release that made the install may be gone or older (a new
        # release unpacked elsewhere): its own tools win there, and so does an explicit --package. A recorded path
        # that no longer exists is never reused.
        if (previous and previous.get('game') in TARGETS and previous.get('destination') == str(dest) and found.dev and
                options.package is None):
            for name, value in previous.get('build_tools', {}).items():
                if (name in ('toolchain', 'deps', 'ninja', 'dll_dir', 'dotnet') and getattr(options, name) is None and
                        value and Path(value).exists()):
                    setattr(options, name, Path(value))
        layout = tool_layout(found, options)
        raw_base = (options.temp or layout.temp or dest.parent / '.dkb-setup').absolute()
        ensure_plain(raw_base)
        base = raw_base.resolve()
        tag = hashlib.sha256(str(dest).lower().encode()).hexdigest()[:12]
        # The work folder's name goes onto compiler and MSBuild command lines that split on spaces (issue 134): keep
        # only characters that need no quoting. The tag already tells two destinations apart; a name that was safe
        # keeps its folder, so an interrupted setup still resumes.
        safe = ''.join(ch if ch.isascii() and (ch.isalnum() or ch in '._-') else '_' for ch in dest.name)
        temp = base / f'dkb-{safe}-{tag}'
        info = preflight(layout, options, dest, temp, rep)
        # All read-only validation succeeded. Logs and owned output may now be created.
        temp.mkdir(parents=True, exist_ok=True)
        write_owner(temp / OWNER, info['owner'])
        locks.append(lock_temp(temp))
        dest.mkdir(parents=True, exist_ok=True)
        write_owner(dest / DEST_OWNER, {'game': info['game'], 'destination': str(dest)})
        locks.append(lock_temp(dest))
        plain_tree(temp)
        plain_destination(dest, layout.dlls)
        log = temp / 'logs' / time.strftime('install-%Y%m%d-%H%M%S.log')
        ensure_plain(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        rep.log_file = log.open('a', encoding='utf-8', buffering=1)
        rep.log_path = log
        ctx = BananzaContext(layout, dest, temp, rep, options, state=info['old'], verify=options.verify)
        ctx.info = info
        ctx.state.update(game=info['game'], destination=str(dest), sources=info['sources'])
        ctx.state['build_tools'] = {'toolchain': str(layout.llvm), 'deps': str(layout.deps) if layout.deps else None,
                                    'ninja': str(layout.ninja),
                                    'dll_dir': str(layout.dlls[0].parent) if layout.dlls else None,
                                    'dotnet': str(info['dotnet'])}
        ctx.state['lang'] = ini.canonical_language(options.lang or ctx.state.get('lang') or 'en-US')
        steps = all_steps()
        rep.emit('start', release=layout.release, mode=ctx.mode, dev=layout.dev, dest=str(dest), log=str(log),
                 game=info['game'], game_exe='dkb' + EXE, steps=[s.id for s in steps])
        result = run_steps(steps, ctx)
    except InstallError as e:
        rep.error(e.id, e.params)
        result = 'failed'
    except statefile.StateError as e:
        rep.error(e.id, {'path': str(e.path)})
        result = 'failed'
    except KeyboardInterrupt:
        result = 'cancelled'
    except (OSError, ValueError, KeyError, TypeError, ImportError) as e:
        rep.error('internal', {'exception': type(e).__name__, 'detail': str(e)})
        result = 'failed'
    rep.emit('end', result=result, seconds=round(time.monotonic() - started, 1))
    rep.close()
    for lock in locks:
        lock.close()
    return {'ok': 0, 'failed': 1, 'cancelled': 3}[result]
