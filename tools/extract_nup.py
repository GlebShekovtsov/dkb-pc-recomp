"""Extract the ExeFS and RomFS folders of a NUP (NX Package) container, the single file Eglantine/Eden loads.

A NUP holds loose ExeFS and RomFS trees for a base program, its updates and add-on content, with every file raw or
zstd-compressed and listed with its SHA-256. The Bananza build takes those trees as folders (README, "Building from
source"), so this tool writes them out:

    .venv-py314\\Scripts\\python tools\\extract_nup.py <game>.nup --out build\\extracted\\dkb100-dump
    .venv-py314\\Scripts\\python tools\\extract_nup.py <game>.nup --list

It writes <out>\\exefs and <out>\\romfs, the folders `recomp\\gen.py --exefs` and `tools\\dkb_pack_romfs.py --dir` take.
The output is game data: keep it under build/. Every extracted file is checked against the SHA-256 in the container
(--no-verify skips that), and the table of contents against the one in the superblock.

By default the base package is extracted. --package picks another package by index (see --list); --overlay-update
also lays the newest update package over the base.

--select latest|base extracts what Eglantine's loader (core/loader/nup.cpp, AppLoader_NUP) runs: with latest, the
ExeFS of the newest update package (or of the base when the update has none) and the base RomFS with the update's
RomFS laid over it (NupContainer::MergeShadowDirs); a standalone update without a base gives its own trees. base takes
the base package alone. --pack then writes the RomFS as the romfs.bin and index.json of tools/dkb_pack_romfs.py
straight from the container, without the loose copy; the Bananza installer extracts .nup input this way:

    python tools\\extract_nup.py <game>.nup --select latest --pack --out build\\extracted\\dkb458752


zstd comes from `compression.zstd` (Python 3.14), the `zstandard` module, or a `zstd` program on PATH, in that order.
Everything else is the standard library.

The layout is Eglantine's core/file_sys/nup_container.cpp (GPL-3.0-or-later); this is an independent reader of the
same format, with the same checks.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SUPER_SIZE = 0x80
TOC_HEADER_SIZE = 0x30
PACKAGE_SIZE = 0x40
DIR_SIZE = 0x18
FILE_SIZE = 0x50
TOC_MAGIC = 0x01434F54
NO_DIR = 0xFFFFFFFF
CODEC_RAW, CODEC_ZSTD = 0, 1
KINDS = {0: 'base', 1: 'update', 2: 'addon'}
CHUNK = 4 << 20
ALIGN = 16   # tools/dkb_pack_romfs.py

META_NPDM, META_DISPLAY_VERSION, META_TITLE_NAME = 1, 4, 5


class NupError(Exception):
    pass


class Package:
    def __init__(self, index, rec):
        (self.title_id, self.root_dir, self.exefs_dir, self.romfs_dir, self.version, self.kind, self.flags,
         _, _, self.meta_offset, self.meta_size) = rec
        self.index = index
        self.title = ''
        self.display_version = ''

    @property
    def kind_name(self):
        return KINDS.get(self.kind, f'kind {self.kind}')


class Nup:
    """A parsed NUP container. Only the superblock, the table of contents and the meta blocks are read."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, 'rb')
        try:
            self._open()
        except BaseException:
            self.f.close()
            raise

    def _open(self):
        size = os.fstat(self.f.fileno()).st_size
        sb = self.f.read(SUPER_SIZE)
        if len(sb) < SUPER_SIZE or sb[:4] != b'NUP\x01':
            raise NupError('not a NUP container (missing "NUP\\x01" magic)')
        (header_size, _version, self.toc_offset, self.toc_size, toc_sha, self.blob_offset, self.file_size,
         package_count, flags) = struct.unpack_from('<HHQQ32sQQII', sb, 4)
        if header_size < SUPER_SIZE or flags != 0:
            raise NupError('unsupported NUP superblock')
        if (self.file_size != size or self.toc_offset < SUPER_SIZE or
                self.blob_offset < self.toc_offset + self.toc_size or self.blob_offset % 16 or
                self.blob_offset > self.file_size):
            raise NupError('the NUP superblock disagrees with the file (truncated or modified?)')
        self.f.seek(self.toc_offset)
        toc = self.f.read(self.toc_size)
        if len(toc) != self.toc_size or hashlib.sha256(toc).digest() != toc_sha:
            raise NupError('NUP table of contents SHA-256 mismatch: the file is corrupt or was modified')
        self._parse_toc(toc, package_count)
        for pkg in self.packages:
            self._parse_meta(pkg)

    def close(self):
        self.f.close()

    def _parse_toc(self, toc, package_count):
        if len(toc) < TOC_HEADER_SIZE:
            raise NupError('NUP table of contents too short')
        magic, count, dir_count, file_count, pool_off, pool_size, _ = struct.unpack_from('<IIQQQQQ', toc, 0)
        if magic != TOC_MAGIC or count == 0 or count != package_count:
            raise NupError('bad NUP table of contents header')
        needed = TOC_HEADER_SIZE + count * PACKAGE_SIZE + dir_count * DIR_SIZE + file_count * FILE_SIZE
        if pool_off < needed or pool_off + pool_size > len(toc) or (pool_off - needed) % 4:
            raise NupError('bad NUP table of contents layout')
        pool = toc[pool_off:pool_off + pool_size]

        def name_at(off, length):
            if off + length > pool_size:
                raise NupError('NUP name outside the string pool')
            return pool[off:off + length].decode('utf-8')

        pos = TOC_HEADER_SIZE
        self.packages = []
        for i in range(count):
            title_id, root, exefs, romfs, version, kind, flags, r1, r2, meta_off, meta_size = struct.unpack_from(
                '<QIIIIBBHIQI', toc, pos)
            pos += PACKAGE_SIZE
            if kind > 2 or (flags & ~1) or root >= dir_count or \
                    (exefs != NO_DIR and exefs >= dir_count) or (romfs != NO_DIR and romfs >= dir_count):
                raise NupError(f'bad NUP package record {i}')
            if not (meta_size and meta_off and meta_off + meta_size <= self.file_size):
                meta_off = meta_size = 0
            self.packages.append(Package(i, (title_id, root, exefs, romfs, version, kind, flags, r1, r2,
                                             meta_off, meta_size)))

        self.dirs = []   # (name, first_child, child_count)
        children_bytes = pool_off - needed
        for i in range(dir_count):
            name_off, name_len, first, n, _ = struct.unpack_from('<IIIIQ', toc, pos)
            pos += DIR_SIZE
            if first + n > children_bytes // 4:
                raise NupError(f'bad NUP directory record {i}')
            self.dirs.append((name_at(name_off, name_len), first, n))

        self.files = []  # (name, size, extent_offset, extent_size, sha256, codec)
        for i in range(file_count):
            name_off, name_len, size, ext_off, ext_size, sha, codec, codec_flags, _, _ = struct.unpack_from(
                '<IIQQQ32sBBH12s', toc, pos)
            pos += FILE_SIZE
            if codec > CODEC_ZSTD or codec_flags != 0 or (size == 0 and (ext_off or ext_size)) or \
                    (size != 0 and (ext_size == 0 or ext_off < self.blob_offset or ext_off % 16 or
                                    ext_off + ext_size > self.file_size)):
                raise NupError(f'bad NUP file record {i}')
            self.files.append((name_at(name_off, name_len), size, ext_off, ext_size, sha, codec))

        self.children = list(struct.unpack_from(f'<{children_bytes // 4}I', toc, needed))
        if any(c >= dir_count + file_count for c in self.children):
            raise NupError('bad NUP child index')

    def _parse_meta(self, pkg):
        if not pkg.meta_offset:
            return
        self.f.seek(pkg.meta_offset)
        block = self.f.read(pkg.meta_size)
        pos = 0
        while pos + 8 <= len(block):
            typ, _, size = struct.unpack_from('<HHI', block, pos)
            pos += 8
            if pos + size > len(block):
                raise NupError('bad NUP meta block')
            payload = block[pos:pos + size]
            pos += size
            if typ == META_TITLE_NAME:
                pkg.title = payload.decode('utf-8', 'replace')
            elif typ == META_DISPLAY_VERSION:
                pkg.display_version = payload.decode('utf-8', 'replace')

    def walk(self, dir_index, prefix=(), path=()):
        """Yield (relative path parts, file index) for every file under a directory node. `path` holds the directories
        being visited, so a container whose directories point back at an ancestor is refused instead of recursing."""
        if dir_index in path:
            raise NupError('NUP directory cycle')
        path += (dir_index,)
        _, first, n = self.dirs[dir_index]
        for slot in range(first, first + n):
            child = self.children[slot]
            if child >= len(self.dirs):
                name = self.files[child - len(self.dirs)][0]
                yield prefix + (safe_name(name),), child - len(self.dirs)
            else:
                yield from self.walk(child, prefix + (safe_name(self.dirs[child][0]),), path)

    def package_files(self, pkg, which):
        dir_index = pkg.exefs_dir if which == 'exefs' else pkg.romfs_dir
        return [] if dir_index == NO_DIR else list(self.walk(dir_index))


def safe_name(name):
    """A container is a file someone handed you: refuse names that would leave the output folder."""
    if not name or name in ('.', '..') or any(c in name for c in '/\\:\0'):
        raise NupError(f'unsafe name in the container: {name!r}')
    return name


class Zstd:
    """Streaming zstd decompression from whichever backend exists."""

    def __init__(self):
        self.kind = None
        try:
            from compression import zstd as _z   # Python 3.14
            self._mod = _z
            self.kind = 'compression.zstd'
        except ImportError:
            try:
                import zstandard as _z
                self._mod = _z
                self.kind = 'zstandard'
            except ImportError:
                self.exe = shutil.which('zstd')
                if self.exe is None and os.path.exists(r'C:\msys64\usr\bin\zstd.exe'):
                    self.exe = r'C:\msys64\usr\bin\zstd.exe'
                if self.exe:
                    self.kind = 'zstd program'

    def stream(self, src, nbytes):
        """Yield decompressed chunks of the next `nbytes` of the file object `src`."""
        if self.kind is None:
            raise NupError('this container has zstd files: use Python 3.14 (compression.zstd), pip install '
                           'zstandard, or put a zstd program on PATH')
        if self.kind == 'zstd program':
            raw = _read_exact(src, nbytes)
            proc = subprocess.run([self.exe, '-d', '-c', '-q'], input=raw, capture_output=True)
            if proc.returncode:
                raise NupError('zstd failed: ' + proc.stderr.decode(errors='replace').strip())
            yield proc.stdout
            return
        dec = self._mod.ZstdDecompressor() if self.kind == 'compression.zstd' else \
            self._mod.ZstdDecompressor().decompressobj()
        left = nbytes
        while left:
            chunk = src.read(min(CHUNK, left))
            if not chunk:
                raise NupError('the container ends inside a file extent')
            left -= len(chunk)
            out = dec.decompress(chunk)
            if out:
                yield out


def _read_exact(src, n):
    data = src.read(n)
    if len(data) != n:
        raise NupError('the container ends inside a file extent')
    return data


def file_chunks(nup, zstd, idx, verify):
    """Yield the contents of file `idx` in chunks; after the last one, check its size and, with verify, its SHA-256.
    A failed check raises before the generator finishes, so a caller publishes nothing it has not seen checked."""
    name, size, ext_off, ext_size, sha, codec = nup.files[idx]
    h = hashlib.sha256()
    written = 0
    if size:
        nup.f.seek(ext_off)
        if codec == CODEC_RAW:
            left = size
            while left:
                chunk = nup.f.read(min(CHUNK, left))
                if not chunk:
                    raise NupError(f'{name}: the container ends inside the file')
                left -= len(chunk)
                h.update(chunk)
                written += len(chunk)
                yield chunk
        else:
            for chunk in zstd.stream(nup.f, ext_size):
                h.update(chunk)
                written += len(chunk)
                yield chunk
    if written != size:
        raise NupError(f'{name}: expected {size} bytes, got {written}')
    if verify and h.digest() != sha:
        raise NupError(f'{name}: SHA-256 differs from the container record (corrupt container)')


def extract_file(nup, zstd, idx, dest, verify):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    written = 0
    try:
        # Exclusive creation avoids existing links, stale partials and legitimate
        # archive entries named <destination>.part. Keep replacement on one filesystem.
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.nup-', suffix='.part',
                                         dir=dest.parent, delete=False) as out:
            tmp = Path(out.name)
            for chunk in file_chunks(nup, zstd, idx, verify):
                out.write(chunk)
                written += len(chunk)
        os.replace(tmp, dest)
        return written
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def extract_tree(nup, zstd, pkg, which, out_dir, verify, quiet, entries=None):
    if entries is None:
        entries = nup.package_files(pkg, which)
    total = sum(nup.files[i][1] for _, i in entries)
    done = 0
    skipped = 0
    t0 = time.time()
    root = out_dir.resolve()
    for n, (parts, idx) in enumerate(entries, 1):
        dest = (out_dir.joinpath(*parts)).resolve()
        if root not in dest.parents:
            raise NupError(f'refusing to write outside the output folder: {"/".join(parts)}')
        size = nup.files[idx][1]
        if dest.exists() and dest.stat().st_size == size and not (verify and size and _hash(dest) != nup.files[idx][4]):
            skipped += 1
            done += size
            continue
        done += extract_file(nup, zstd, idx, dest, verify)
        if not quiet and (n % 200 == 0 or n == len(entries)):
            rate = done / max(time.time() - t0, 1e-6) / 1e6
            print(f'  {which}: {n}/{len(entries)} files, {done / 1e9:.2f} of {total / 1e9:.2f} GB ({rate:.0f} MB/s)')
    return len(entries), total, skipped


def merged_entries(nup, packages, which):
    """(path parts, file index) of the tree `which` of each package in turn, every later package laid over the earlier
    ones as Eglantine's NupContainer::MergeShadowDirs does: a file replaces the file at the same path, folders merge.
    Sorted by the UTF-8 bytes of the path, the order tools/dkb_pack_romfs.py packs in. A path that is a file in one
    package and a folder in another has no single meaning, so it is refused."""
    merged = {}
    for pkg in packages:
        for parts, idx in nup.package_files(pkg, which):
            merged[parts] = idx
    folders = {parts[:i] for parts in merged for i in range(1, len(parts))}
    clash = sorted(folders & merged.keys())
    if clash:
        raise NupError(f'{"/".join(clash[0])} is a file in one package and a folder in another')
    return sorted(merged.items(), key=lambda e: '/'.join(e[0]).encode())


def install_plan(nup, select='latest'):
    """The packages whose trees make the program, as Eglantine's AppLoader_NUP picks them: a dict with 'exefs' (the
    package whose ExeFS is used whole), 'romfs' (the packages whose RomFS trees are laid over each other, base first)
    and 'update' (the update package taken, or None). select is 'latest' or 'base'."""
    if select not in ('latest', 'base'):
        raise NupError(f'--select takes latest or base, not {select!r}')
    base = next((p for p in nup.packages if p.kind == 0), None)
    updates = [p for p in nup.packages if p.kind == 1]
    update = max(updates, key=lambda p: p.version) if updates and select == 'latest' else None
    if base is not None and base.exefs_dir != NO_DIR:
        exefs = update if update is not None and update.exefs_dir != NO_DIR else base
        romfs = [p for p in (base, update) if p is not None and p.romfs_dir != NO_DIR]
    elif update is not None and update.exefs_dir != NO_DIR:
        exefs, romfs = update, [update] if update.romfs_dir != NO_DIR else []   # a standalone update
    else:
        raise NupError('no base package with an ExeFS in the container' if select == 'base' else
                       'no package with an ExeFS in the container (see --list)')
    return {'exefs': exefs, 'romfs': romfs, 'update': update}


def packed_entries(entries):
    """The entries tools/dkb_pack_romfs.py would pack from the same loose tree: it skips files whose name starts with
    a dot (the .listing files dump tools leave)."""
    return [(parts, idx) for parts, idx in entries if not parts[-1].startswith('.')]


def pack_romfs(nup, zstd, entries, out_dir, verify, quiet):
    """Write out_dir/romfs.bin and index.json in the layout of tools/dkb_pack_romfs.py (files in path order, each
    aligned to 16 bytes) straight from the container: the same image a loose extraction packed would give, without
    the 19 GB loose copy. The image goes to romfs.bin.part and is renamed once every file passed its checks; the index
    is written last, so a folder with an index.json holds a whole image."""
    entries = packed_entries(entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    image, index = out_dir / 'romfs.bin', out_dir / 'index.json'
    index.unlink(missing_ok=True)
    part = out_dir / 'romfs.bin.part'
    total = sum(nup.files[i][1] for _, i in entries)
    sha = hashlib.sha256()
    files, off, t0 = [], 0, time.time()
    try:
        with open(part, 'wb') as dst:
            for n, (parts, idx) in enumerate(entries, 1):
                pad = -off % ALIGN
                if pad:
                    dst.write(b'\0' * pad)
                    sha.update(b'\0' * pad)
                    off += pad
                size = 0
                for chunk in file_chunks(nup, zstd, idx, verify):
                    dst.write(chunk)
                    sha.update(chunk)
                    size += len(chunk)
                files.append({'path': '/'.join(parts), 'offset': off, 'size': size})
                off += size
                if not quiet and (n % 2000 == 0 or n == len(entries)):
                    rate = off / max(time.time() - t0, 1e-6) / 1e6
                    print(f'  romfs: {n}/{len(entries)} files, {off / 1e9:.2f} of {total / 1e9:.2f} GB '
                          f'({rate:.0f} MB/s)', flush=True)
        os.replace(part, image)
    finally:
        part.unlink(missing_ok=True)
    # 'pass' as tools/dkb_pack_romfs.py writes it: the image is whole and its SHA-256 recorded. 'source' says what the
    # files were checked against: the container's per-file SHA-256 records, not an NCA hash tree.
    source = ('packed from a NUP container; every file checked against its SHA-256 record' if verify else
              'packed from a NUP container; file SHA-256 records not checked')
    manifest = {'status': 'pass', 'source': source, 'romfs_size': off, 'romfs_sha256': sha.hexdigest(), 'files': files}
    index.write_text(json.dumps(manifest), encoding='utf-8')
    return len(files), off


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                return h.digest()
            h.update(chunk)


def list_packages(nup):
    print(f'{nup.path.name}: {len(nup.packages)} package(s), {len(nup.files)} files, '
          f'{nup.file_size / 1e9:.2f} GB')
    for pkg in nup.packages:
        exe = nup.package_files(pkg, 'exefs')
        rom = nup.package_files(pkg, 'romfs')
        print(f'  [{pkg.index}] {pkg.kind_name:6} title {pkg.title_id:016x} version {pkg.version} '
              f'{pkg.display_version or "-"}{"  (standalone update)" if pkg.flags & 1 else ""}  '
              f'"{pkg.title or "-"}"')
        print(f'        exefs: {len(exe)} files, {sum(nup.files[i][1] for _, i in exe) / 1e6:.1f} MB   '
              f'romfs: {len(rom)} files, {sum(nup.files[i][1] for _, i in rom) / 1e9:.2f} GB')


def pick(nup, which):
    if which == 'base':
        for pkg in nup.packages:
            if pkg.kind == 0:
                return pkg
        raise NupError('no base package in the container (see --list)')
    if which == 'update':
        best = [p for p in nup.packages if p.kind == 1]
        if not best:
            raise NupError('no update package in the container (see --list)')
        return max(best, key=lambda p: p.version)
    try:
        return nup.packages[int(which)]
    except (ValueError, IndexError):
        raise NupError(f'--package takes base, update or an index from --list, not {which!r}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('nup', type=Path, nargs='?', help='the .nup file')
    ap.add_argument('--out', type=Path, help='output folder: <out>\\exefs and <out>\\romfs are written')
    ap.add_argument('--list', action='store_true', help='list the packages and exit')
    ap.add_argument('--package', default='base', help='base (default), update, or an index from --list')
    ap.add_argument('--overlay-update', action='store_true',
                    help='after the chosen package, extract the newest update package over it')
    ap.add_argument('--only', choices=('exefs', 'romfs'), help='extract just one of the two folders')
    ap.add_argument('--select', choices=('latest', 'base'),
                    help="what Eglantine runs: latest (the newest update's ExeFS, base RomFS with the update's laid "
                         "over it) or base; replaces --package and --overlay-update")
    ap.add_argument('--pack', action='store_true',
                    help='write the RomFS as <out>\\romfs\\romfs.bin and index.json (tools/dkb_pack_romfs.py layout) '
                         'instead of loose files')
    ap.add_argument('--check-zstd', action='store_true',
                    help='exit 0 when a zstd backend is available, 1 otherwise, and do nothing else')
    ap.add_argument('--no-verify', action='store_true', help='skip the per-file SHA-256 check')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    if args.check_zstd:
        kind = Zstd().kind
        print(kind or 'no zstd backend')
        return 0 if kind else 1
    if args.nup is None:
        ap.error('the .nup file is required')
    if args.pack and not args.select:
        ap.error('--pack needs --select')
    try:
        nup = Nup(args.nup)
        if args.list or args.out is None:
            list_packages(nup)
            if not args.list:
                print('\nGive --out <folder> to extract.')
            return 0
        zstd = Zstd()
        if args.select:
            plan = install_plan(nup, args.select)
            trees = (args.only,) if args.only else ('exefs', 'romfs')
            for which in trees:
                pkgs = [plan['exefs']] if which == 'exefs' else plan['romfs']
                names = '+'.join(f'[{p.index}] {p.kind_name} {p.display_version or p.version}' for p in pkgs) or 'none'
                print(f'{which}: package {names} to {args.out}')
                entries = merged_entries(nup, pkgs, which)
                if which == 'romfs' and args.pack:
                    count, total = pack_romfs(nup, zstd, entries, args.out / 'romfs', not args.no_verify, args.quiet)
                    print(f'  romfs: {count} files packed, {total / 1e9:.2f} GB')
                else:
                    count, total, skipped = extract_tree(nup, zstd, None, which, args.out / which,
                                                         not args.no_verify, args.quiet, entries)
                    print(f'  {which}: {count} files, {total / 1e9:.2f} GB'
                          f'{f" ({skipped} already there)" if skipped else ""}')
            return 0
        packages = [pick(nup, args.package)]
        if args.overlay_update:
            packages.append(pick(nup, 'update'))
        for pkg in packages:
            print(f'extracting package [{pkg.index}] {pkg.kind_name} {pkg.title_id:016x} '
                  f'{pkg.display_version or pkg.version} to {args.out}')
            for which in ((args.only,) if args.only else ('exefs', 'romfs')):
                count, total, skipped = extract_tree(nup, zstd, pkg, which, args.out / which,
                                                     not args.no_verify, args.quiet)
                print(f'  {which}: {count} files, {total / 1e9:.2f} GB'
                      f'{f" ({skipped} already there)" if skipped else ""}')
        return 0
    except NupError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
