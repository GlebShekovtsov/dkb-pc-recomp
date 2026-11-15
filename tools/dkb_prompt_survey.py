"""Survey Bananza's RomFS for the resources that draw the button prompts, before any PlayStation prompt overlay is made.

Bananza does not ask the system for the Nintendo Extension shared font (no nn::pl request in the logs), so its button
icons are in its own files. This lists, without copying any of their bytes:

- every font (OpenType or TrueType, plain or in the Switch's XORed BFOTF / BFTTF wrapping, and BFFNT): its path and the
  private-use code points it maps (U+E000 to U+F8FF), where button glyphs live;
- every texture of a BNTX in the UI archives whose name looks like a button (Btn, Key, Pad, Stick, ...), with its
  format and size;
- the private-use characters the message files (MSBT) use, and the most used inline tags (group and type), with a few
  message labels each, since a prompt in a sentence is either a private-use glyph or a tag;
- the layout panes and texture references (BFLYT) whose names look like a button.

The input is the install's RomFS folder (romfs.bin and index.json, as the setup writes it). The report goes to
build/prompt-survey/ (or --out under build/): report.txt to read and report.json. It holds names and counts only.

    python\\python.exe tools\\dkb_prompt_survey.py --romfs Z:\\Games\\dkb\\romfs

Runs under Python 3.14 (compression.zstd, as the release's python does) or with the zstandard package.
"""
import argparse
import collections
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUTTONISH = re.compile(r'(?i)(btn|button|key(?!frame)|npad|pad|stick|trigger|ctrl|controller|command|cmd|operation|guide|'
                       r'glyph|icon_?[abxylrz](?![a-z])|plus|minus|capture|joy|shoulder|bumper)')
PUA = range(0xE000, 0xF900)
SCANNED = re.compile(r'(?i)(layout|font|ui/|mals|message|msg|system/resource|tex/ui|icon)')

try:
    from compression import zstd as _zstd

    def frame_dict_id(data):
        return _zstd.get_frame_info(data).dictionary_id

    def make_dict(raw):
        return _zstd.ZstdDict(raw)

    def decompress(data, dictionary):
        return _zstd.decompress(data, zstd_dict=dictionary)
except ImportError:  # Python before 3.14: the zstandard package
    import zstandard as _zs

    def frame_dict_id(data):
        return _zs.get_frame_parameters(data).dict_id

    def make_dict(raw):
        return _zs.ZstdCompressionDict(raw)

    def decompress(data, dictionary):
        return _zs.ZstdDecompressor(dict_data=dictionary).decompress(data, max_output_size=1 << 30)


def sarc_members(data):
    """(name, bytes) of each member of a little-endian SARC; nothing for anything else."""
    if data[:4] != b'SARC' or data[6:8] != b'\xff\xfe':
        return
    header, total, start = struct.unpack_from('<HxxII', data, 4)
    if total > len(data) or data[header:header + 4] != b'SFAT':
        return
    count = struct.unpack_from('<H', data, header + 6)[0]
    names = header + 12 + count * 16
    if data[names:names + 4] != b'SFNT':
        return
    names += 8
    for i in range(count):
        _, attr, lo, hi = struct.unpack_from('<4I', data, header + 12 + i * 16)
        name = data[names + (attr & 0xffffff) * 4:].split(b'\0', 1)[0].decode('utf-8', 'replace') if attr >> 24 else f'#{i}'
        if lo <= hi <= len(data) - start:
            yield name, data[start + lo:start + hi]


def bntx_textures(data):
    """(name, format code, width, height, mips, layers) of each texture of the BNTX at data[0:]."""
    if data[:4] != b'BNTX' or len(data) < 0x38:
        return
    size = struct.unpack_from('<I', data, 0x1C)[0]
    data = data[:size] if 0x38 <= size <= len(data) else data
    count, table = struct.unpack_from('<IQ', data, 0x24)
    if count > 100000 or table + count * 8 > len(data):
        return
    for k in range(count):
        info = struct.unpack_from('<Q', data, table + 8 * k)[0]
        if info + 0x78 > len(data) or data[info:info + 4] != b'BRTI':
            continue
        mips = struct.unpack_from('<H', data, info + 0x16)[0]
        fmt = struct.unpack_from('<I', data, info + 0x1C)[0]
        width, height, _, layers = struct.unpack_from('<4I', data, info + 0x24)
        name_ptr = struct.unpack_from('<Q', data, info + 0x60)[0]
        if name_ptr + 2 > len(data):
            continue
        n = struct.unpack_from('<H', data, name_ptr)[0]
        yield data[name_ptr + 2:name_ptr + 2 + n].decode('utf-8', 'replace'), fmt, width, height, mips, layers


def sfnt_cmap_pua(font):
    """The private-use code points an OpenType / TrueType font maps (formats 4 and 12), and the outline kind."""
    if font[:4] not in (b'OTTO', b'\0\1\0\0', b'true'):
        return None
    tables = {}
    for k in range(struct.unpack_from('>H', font, 4)[0]):
        tag, _, offset, length = struct.unpack_from('>4sIII', font, 12 + 16 * k)
        tables[tag] = (offset, length)
    if b'cmap' not in tables:
        return None
    cmap, found = tables[b'cmap'][0], set()
    for k in range(struct.unpack_from('>H', font, cmap + 2)[0]):
        _, _, offset = struct.unpack_from('>HHI', font, cmap + 4 + 8 * k)
        p = cmap + offset
        fmt = struct.unpack_from('>H', font, p)[0]
        if fmt == 4:
            seg = struct.unpack_from('>H', font, p + 6)[0] // 2
            for s in range(seg):
                end = struct.unpack_from('>H', font, p + 14 + 2 * s)[0]
                start = struct.unpack_from('>H', font, p + 16 + 2 * seg + 2 * s)[0]
                found.update(c for c in range(max(start, PUA.start), min(end, PUA.stop - 1) + 1))
        elif fmt == 12:
            for g in range(struct.unpack_from('>I', font, p + 12)[0]):
                start, end, _ = struct.unpack_from('>III', font, p + 16 + 12 * g)
                found.update(c for c in range(max(start, PUA.start), min(end, PUA.stop - 1) + 1))
    found.discard(0xFFFF)
    return {'outlines': 'CFF' if b'CFF ' in tables else 'TrueType' if b'glyf' in tables else '?',
            'glyphs': struct.unpack_from('>H', font, tables[b'maxp'][0] + 4)[0] if b'maxp' in tables else None,
            'pua': sorted(found)}


def sfnt_plain(data):
    """The plain OpenType / TrueType bytes of a font member (unwrapping a BFOTF / BFTTF), or None."""
    if data[:4] in (b'OTTO', b'\0\1\0\0', b'true'):
        return bytes(data)
    if len(data) > 16:
        key = bytes(x ^ y for x, y in zip(data[4:8], struct.pack('>I', len(data) - 8)))
        if bytes(x ^ key[i % 4] for i, x in enumerate(data[8:12])) in (b'OTTO', b'\0\1\0\0', b'true'):
            return bytes(x ^ key[i % 4] for i, x in enumerate(data[8:]))
    return None


def font_info(data):
    """Font facts of a member, or None: a plain sfnt, a BFOTF / BFTTF (8-byte header, then every 4-byte word XORed with
    one key, which the size field gives away), or a BFFNT (bitmap sheets, listed only)."""
    if data[:4] in (b'FFNT', b'CFNT', b'RFNT'):
        return {'kind': 'BFFNT', 'pua': []}
    plain = sfnt_cmap_pua(data)
    if plain:
        return {'kind': 'sfnt', **plain}
    if len(data) > 16:
        key = bytes(x ^ y for x, y in zip(data[4:8], struct.pack('>I', len(data) - 8)))
        head = bytes(x ^ key[i % 4] for i, x in enumerate(data[8:12]))
        if head in (b'OTTO', b'\0\1\0\0', b'true'):
            body = bytes(x ^ key[i % 4] for i, x in enumerate(data[8:]))
            info = sfnt_cmap_pua(body)
            if info:
                return {'kind': 'BFOTF' if head == b'OTTO' else 'BFTTF', **info}
    return None


def msbt_strings(data):
    """(label, list of str chars and (group, type) tag tuples) of each message of a UTF-16 MSBT."""
    if data[:8] != b'MsgStdBn' or data[0x0C] != 1:
        return
    big = data[8:10] == b'\xfe\xff'
    e = '>' if big else '<'
    sections = struct.unpack_from(e + 'H', data, 0x0E)[0]
    pos, labels, texts = 0x20, {}, []
    for _ in range(sections):
        if pos + 16 > len(data):
            break
        magic, size = data[pos:pos + 4], struct.unpack_from(e + 'I', data, pos + 4)[0]
        body = pos + 16
        if magic == b'LBL1':
            for k in range(struct.unpack_from(e + 'I', data, body)[0]):
                n, off = struct.unpack_from(e + 'II', data, body + 4 + 8 * k)
                p = body + off
                for _ in range(n):
                    ln = data[p]
                    name = data[p + 1:p + 1 + ln].decode('utf-8', 'replace')
                    labels[struct.unpack_from(e + 'I', data, p + 1 + ln)[0]] = name
                    p += 1 + ln + 4
        elif magic == b'TXT2':
            count = struct.unpack_from(e + 'I', data, body)[0]
            offs = [struct.unpack_from(e + 'I', data, body + 4 + 4 * k)[0] for k in range(count)] + [size]
            for k in range(count):
                p, end, items = body + offs[k], body + offs[k + 1], []
                while p + 2 <= end:
                    c = struct.unpack_from(e + 'H', data, p)[0]
                    if c == 0x0E and p + 8 <= end:
                        group, typ, plen = struct.unpack_from(e + 'HHH', data, p + 2)
                        items.append((group, typ))
                        p += 8 + plen
                        continue
                    if c == 0x0F and p + 6 <= end:
                        p += 6
                        continue
                    if c == 0:
                        break
                    items.append(chr(c))
                    p += 2
                texts.append(items)
        pos = body + (size + 15) // 16 * 16
    for k, items in enumerate(texts):
        yield labels.get(k, f'#{k}'), items


def bflyt_names(data):
    """(texture names, font names, pane names) of a BFLYT."""
    if data[:4] != b'FLYT':
        return [], [], []
    e = '>' if data[4:6] == b'\xfe\xff' else '<'
    head = struct.unpack_from(e + 'H', data, 0x0C)[0]
    count = struct.unpack_from(e + 'H', data, 0x10)[0]
    pos, textures, fonts, panes = head, [], [], []
    for _ in range(count):
        if pos + 8 > len(data):
            break
        magic, size = data[pos:pos + 4], struct.unpack_from(e + 'I', data, pos + 4)[0]
        if size < 8:
            break
        if magic in (b'txl1', b'fnl1'):
            n = struct.unpack_from(e + 'H', data, pos + 8)[0]
            base = pos + 12
            names = [data[base + struct.unpack_from(e + 'I', data, base + 4 * k)[0]:].split(b'\0', 1)[0]
                     .decode('utf-8', 'replace') for k in range(n)]
            (textures if magic == b'txl1' else fonts).extend(names)
        elif magic in (b'pan1', b'pic1', b'txt1', b'wnd1', b'bnd1', b'prt1', b'scr1', b'ali1'):
            panes.append((magic.decode(), data[pos + 12:pos + 36].split(b'\0', 1)[0].decode('utf-8', 'replace')))
        pos += size
    return textures, fonts, panes


class Survey:
    def __init__(self):
        self.fonts, self.textures, self.layout_refs = [], [], []
        self.pua_use = collections.Counter()
        self.pua_labels = collections.defaultdict(list)
        self.tag_use = collections.Counter()
        self.tag_labels = collections.defaultdict(list)
        self.kinds = collections.Counter()
        self.errors = []
        self.export, self.exported = None, {}   # --export-font: member name part -> {sha256: (path, plain bytes)}

    def visit(self, path, data, depth=0):
        name = path.rsplit('/', 1)[-1].lower()
        if data[:4] == b'\x28\xb5\x2f\xfd':   # a zstd frame inside an archive
            return
        if data[:4] == b'SARC' and depth < 3:
            self.kinds['SARC'] += 1
            for member, body in sarc_members(data):
                self.visit(f'{path}:{member}', body, depth + 1)
            return
        if data[:4] == b'BNTX':
            self.kinds['BNTX'] += 1
            for tex in bntx_textures(data):
                if BUTTONISH.search(tex[0]):
                    self.textures.append((path, *tex))
            return
        if data[:4] == b'FLYT':
            self.kinds['BFLYT'] += 1
            textures, fonts, panes = bflyt_names(data)
            hits = sorted({t for t in textures if BUTTONISH.search(t)}) + [f'{k}:{p}' for k, p in panes if BUTTONISH.search(p)]
            if hits or fonts:
                self.layout_refs.append((path, fonts, hits))
            return
        if data[:8] == b'MsgStdBn':
            self.kinds['MSBT'] += 1
            for label, items in msbt_strings(data):
                for it in items:
                    if isinstance(it, tuple):
                        self.tag_use[it] += 1
                        if len(self.tag_labels[it]) < 4:
                            self.tag_labels[it].append(f'{path}#{label}')
                    elif ord(it) in PUA:
                        self.pua_use[ord(it)] += 1
                        if len(self.pua_labels[ord(it)]) < 4:
                            self.pua_labels[ord(it)].append(f'{path}#{label}')
            return
        if name.endswith(('.bfotf', '.bfttf', '.otf', '.ttf', '.bffnt')) or data[:4] in (b'OTTO', b'FFNT'):
            info = font_info(data)
            self.kinds['font'] += 1
            self.fonts.append((path, info))
            plain = sfnt_plain(data) if self.export and self.export.lower() in name else None
            if plain:
                self.exported.setdefault(hashlib.sha256(plain).hexdigest(), (path, plain))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--romfs', type=Path, required=True, help='the install\'s romfs folder (romfs.bin and index.json)')
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'prompt-survey')
    ap.add_argument('--all', action='store_true', help='scan every file, not only the UI, font and message folders')
    ap.add_argument('--export-font', metavar='NAME', help='also write the fonts whose member name contains NAME as plain '
                    '.otf / .ttf files into the output folder, one per distinct copy (game data: keep it under build/)')
    args = ap.parse_args()
    out = args.out.resolve()
    if 'build' not in out.parts:
        ap.error('the report stays in a folder under build/')
    index = json.loads((args.romfs / 'index.json').read_text(encoding='utf-8'))
    if index.get('status') != 'pass':
        ap.error('the RomFS index has not passed its extraction check')
    files = index['files']
    folders = collections.Counter()
    for f in files:
        folders[f['path'].split('/', 1)[0]] += 1
    survey = Survey()
    survey.export = args.export_font
    with (args.romfs / 'romfs.bin').open('rb') as image:
        def read(row):
            image.seek(row['offset'])
            return image.read(row['size'])

        dictionaries = {}
        for row in files:
            if row['path'].lower().endswith('zsdic.pack.zs'):
                pack = decompress(read(row), None)
                for _, member in sarc_members(pack):
                    if member[:4] == b'\x37\xa4\x30\xec':
                        dictionaries[struct.unpack_from('<I', member, 4)[0]] = make_dict(member)
        chosen = [r for r in files if args.all or SCANNED.search(r['path'])]
        for n, row in enumerate(chosen):
            path = row['path']
            try:
                data = read(row)
                if path.endswith('.zs'):
                    data = decompress(data, dictionaries.get(frame_dict_id(data)))
                survey.visit(path, data)
            except Exception as ex:   # one bad file must not hide the rest
                survey.errors.append(f'{path}: {type(ex).__name__}: {ex}')
            if n % 500 == 0:
                print(f'{n}/{len(chosen)} files', file=sys.stderr)

    lines = [f'RomFS: {len(files)} files, {len(chosen)} scanned ({"all" if args.all else "UI, font and message folders"})',
             'folders: ' + ', '.join(f'{k} {v}' for k, v in sorted(folders.items())),
             'parsed: ' + ', '.join(f'{k} {v}' for k, v in sorted(survey.kinds.items())), '', '== fonts']
    for path, info in survey.fonts:
        if not info:
            lines.append(f'{path}: not read')
            continue
        pua = info.get('pua', [])
        ranges, start = [], None
        for i, c in enumerate(pua):
            if start is None:
                start = c
            if i + 1 == len(pua) or pua[i + 1] != c + 1:
                ranges.append(f'U+{start:04X}' + (f'-U+{c:04X}' if c != start else ''))
                start = None
        lines.append(f'{path}: {info["kind"]} {info.get("outlines", "")} glyphs {info.get("glyphs")} '
                     f'private use {len(pua)}: {" ".join(ranges[:40])}')
    lines += ['', '== private-use characters in messages (code point: uses; first labels)']
    for c, n in survey.pua_use.most_common():
        lines.append(f'U+{c:04X}: {n}; {"; ".join(survey.pua_labels[c])}')
    lines += ['', '== inline tags in messages (group, type: uses; first labels), top 60']
    for t, n in survey.tag_use.most_common(60):
        lines.append(f'{t[0]},{t[1]}: {n}; {"; ".join(survey.tag_labels[t])}')
    lines += ['', '== button-like textures (archive:member, name, format, width x height, mips, layers)']
    for path, name, fmt, w, h, mips, layers in survey.textures:
        lines.append(f'{path}  {name}  {fmt:#06x}  {w}x{h}  mips {mips}  layers {layers}')
    lines += ['', '== layouts with button-like textures or panes (fonts used; matches)']
    for path, fonts, hits in survey.layout_refs:
        if hits:
            lines.append(f'{path}  fonts {",".join(fonts)}  {" ".join(hits[:30])}{" ..." if len(hits) > 30 else ""}')
    lines += ['', '== fonts named by layouts']
    named = collections.Counter(f for _, fonts, _ in survey.layout_refs for f in fonts)
    lines += [f'{f}: {n} layouts' for f, n in named.most_common()]
    lines += ['', f'== errors ({len(survey.errors)})'] + survey.errors[:50]
    out.mkdir(parents=True, exist_ok=True)
    if args.export_font:
        lines += ['', f'== exported fonts matching {args.export_font!r} (one file per distinct copy)']
        for k, (digest, (path, plain)) in enumerate(sorted(survey.exported.items(), key=lambda kv: kv[1][0])):
            stem = re.sub(r'\W+', '_', path.rsplit(':', 1)[-1].rsplit('/', 1)[-1].rsplit('.', 1)[0])
            fname = f'{stem}-{k}.{"otf" if plain[:4] == b"OTTO" else "ttf"}'
            (out / fname).write_bytes(plain)
            same = [p for p, info in survey.fonts if p.rsplit(':', 1)[-1] == path.rsplit(':', 1)[-1]]
            lines.append(f'{fname}: from {path}, sha256 {digest[:16]}; {len(same)} copies of that member in all')
    (out / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'report.json').write_text(json.dumps({
        'fonts': survey.fonts, 'textures': survey.textures, 'layouts': survey.layout_refs,
        'pua_use': {f'U+{c:04X}': n for c, n in survey.pua_use.items()},
        'tags': {f'{g},{t}': n for (g, t), n in survey.tag_use.items()}, 'errors': survey.errors}, indent=1),
        encoding='utf-8')
    print(f'report: {out / "report.txt"}')


if __name__ == '__main__':
    main()
