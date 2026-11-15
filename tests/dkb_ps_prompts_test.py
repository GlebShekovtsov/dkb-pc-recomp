"""tools/dkb_ps_prompts.py on a synthetic font, made here: no game file is read.

The font is a CFF OpenType font with the code points of Bananza's button font (scft/NinDev-HardKey.bfotf): each glyph a
plate (a large counter-clockwise contour) with a cut-out (a small clockwise one), ZL and ZR with two cut-outs. One glyph
is written with stem hints, a hintmask, hlineto / vlineto and hvcurveto, so the charstring reader is exercised beyond
what the tool writes. The checks:

- the face buttons keep their plate and get the symbol for the layout (contour counts: cross 1, circle, square and
  triangle 2), in the right winding (cut-outs clockwise);
- L / R / ZL / ZR keep their plate and get the L or R of ZL / ZR (both of the R's contours) and a digit;
- with --layout xbox the diamonds trade charstrings, with nintendo they stay; every other glyph is byte-identical;
- the rebuilt font's table checksums and head.checkSumAdjustment are right, and a font run through twice reads the same;
- end to end (with compression.zstd or zstandard): a RomFS with two Font archives, a SARC member beside the font that
  stays byte-identical, the XOR-wrapped BFOTF, and the manifest's hashes.
"""
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dkb_ps_prompts as T  # noqa: E402

CODES = [0xE0AB, 0xE0AC, 0xE0AD, 0xE0AE, 0xE0CB, 0xE0CE, 0xE0E0, 0xE0E1, 0xE0E2, 0xE0E3, 0xE0E4, 0xE0E5, 0xE0E6,
         0xE0E7, 0xE0EF, 0xE0F0, 0xE100]
PLATE = [(0, -570), (2635, -570), (2635, 2065), (0, 2065)]   # counter-clockwise


def rect(x0, y0, x1, y1, ccw):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return T.poly(pts if ccw else pts[::-1])


def glyph_contours(code):
    plate = T.poly(PLATE)
    if code in (0xE0E6, 0xE0E7):   # Z and L (or R with its counter)
        inner = [rect(483, 160, 1426, 1168, False), rect(1570, 160, 2288, 1168, False)]
        if code == 0xE0E7:
            inner.append(rect(1700, 700, 2000, 1000, True))
        return [plate] + inner
    k = CODES.index(code)
    return [plate, rect(800 + 10 * k, 300, 1800, 1300, False)]


def hinted_charstring(width):
    """The plate as hlineto / vlineto and a cut-out with an hvcurveto, after stem hints and a hintmask."""
    n = T.num
    out = n(width) + n(0) + n(100) + n(500) + n(100) + b'\x01'   # width, two hstems (hstem)
    out += n(0) + n(200) + b'\x03'                               # one vstem
    out += b'\x13' + b'\xe0'                                     # hintmask: 3 stems, one byte
    out += n(0) + n(-570) + b'\x15'                              # rmoveto (0, -570)
    out += n(2635) + n(2635) + n(-2635) + b'\x06'                # hlineto: +x, then +y, then -x
    out += n(1000) + n(300) + b'\x15'                            # rmoveto to (1000, 2065 - ... ) relative
    out += n(400) + n(-200) + n(-300) + n(-500) + b'\x1f'        # hvcurveto: one curve
    out += n(-100) + n(100) + b'\x05'                            # rlineto back
    return out + b'\x0e'


def build_font(codes):
    """A minimal CFF OpenType font: glyph 0 .notdef, then one glyph per code point."""
    nominal, advance = 2740, 2635
    charstrings = [T.encode_charstring(None, [])]
    for code in codes:
        if code == 0xE100:
            charstrings.append(hinted_charstring(advance - nominal))
        else:
            charstrings.append(T.encode_charstring(advance - nominal, glyph_contours(code)))
    nglyphs = len(charstrings)
    header = bytes([1, 0, 4, 2])
    names = T.write_index([b'Synthetic-Buttons'])
    strings = T.write_index([])
    gsubrs = T.write_index([])
    charset = bytes([0]) + b''.join(struct.pack('>H', 391 + i) for i in range(nglyphs - 1))
    private = T.num(nominal) + bytes([21]) + T.num(2635) + bytes([20])   # nominalWidthX, defaultWidthX

    def top(cs_off, ch_off, pv_off):
        return (T.int5(cs_off) + bytes([15]) + T.int5(ch_off) + bytes([17]) +
                T.int5(len(private)) + T.int5(pv_off) + bytes([18]))
    top_len = len(T.write_index([top(0, 0, 0)]))
    base = len(header) + len(names) + top_len + len(strings) + len(gsubrs)
    cs_index = T.write_index(charstrings)
    cff = (header + names + T.write_index([top(base, base + len(charset), base + len(charset) + len(cs_index))]) +
           strings + gsubrs + charset + cs_index + private)
    # cmap format 4: one segment per code point, then 0xFFFF
    segs = [(c, c, (i + 1 - c) & 0xffff) for i, c in enumerate(codes)] + [(0xFFFF, 0xFFFF, 1)]
    n = len(segs)
    f4 = struct.pack('>HHHHHHH', 4, 0, 0, 2 * n, 0, 0, 0)
    f4 += b''.join(struct.pack('>H', e) for _, e, _ in segs) + b'\0\0'
    f4 += b''.join(struct.pack('>H', s) for s, _, _ in segs)
    f4 += b''.join(struct.pack('>H', d) for _, _, d in segs) + bytes(2 * n)
    f4 = f4[:2] + struct.pack('>H', len(f4)) + f4[4:]
    cmap = struct.pack('>HH', 0, 1) + struct.pack('>HHI', 3, 1, 12) + f4
    head = bytearray(54)
    struct.pack_into('>IIII', head, 0, 0x10000, 0x10000, 0, 0x5F0F3CF5)
    struct.pack_into('>H', head, 18, 2048)
    maxp = struct.pack('>IH', 0x5000, nglyphs)
    tables = {b'CFF ': cff, b'cmap': cmap, b'head': bytes(head), b'maxp': maxp}
    tags = sorted(tables)
    out = bytearray(b'OTTO' + struct.pack('>HHHH', len(tags), 0, 0, 0) + bytes(16 * len(tags)))
    for k, tag in enumerate(tags):
        out += bytes(-len(out) % 4)
        struct.pack_into('>4sIII', out, 12 + 16 * k, tag, 0, len(out), len(tables[tag]))
        out += tables[tag]
    return T.sfnt_build(bytes(out), tables)


def read(font):
    tables = T.sfnt_tables(font)
    cff = T.Cff(tables[b'CFF '])
    gid = {c: T.cmap_lookup(tables[b'cmap'], c) for c in CODES}
    return tables, cff, gid


class PsPromptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = build_font(CODES)
        cls.tables, cls.cff, cls.gid = read(cls.font)

    def test_reader_handles_hints_and_short_operators(self):
        width, contours = self.cff.glyph(self.gid[0xE100])
        self.assertEqual(width, 2635 - 2740)
        self.assertEqual(len(contours), 2)
        self.assertEqual([op for op, _ in contours[0]], ['M', 'L', 'L', 'L'])
        self.assertEqual(contours[0][2][1], (2635, 2065))
        self.assertEqual([op for op, _ in contours[1]], ['M', 'C', 'L'])

    def check_layout(self, layout):
        out = T.ps_font(self.font, layout)
        tables, cff, gid = read(out)
        self.assertEqual(T.checksum(out), 0xb1b0afba)   # head.checkSumAdjustment makes the whole font sum to this
        for tag, data in tables.items():
            n = struct.unpack_from('>H', out, 4)[0]
            entry = next(12 + 16 * k for k in range(n) if out[12 + 16 * k:16 + 16 * k] == tag)
            if tag != b'head':
                self.assertEqual(struct.unpack_from('>I', out, entry + 4)[0], T.checksum(data), tag)
        counts = {'cross': 1, 'circle': 2, 'square': 2, 'triangle': 2}
        for code, sym in T.FACE[layout].items():
            width, cs = cff.glyph(gid[code])
            self.assertEqual(width, 2635 - 2740)
            self.assertEqual(len(cs), 1 + counts[sym], (layout, hex(code)))
            self.assertEqual(T.bbox(cs[0]), (0, 2635, -570, 2065))          # the plate, as it was
            self.assertGreater(T.area([a if op != 'C' else a[-1] for op, a in cs[0]]), 0)
            self.assertLess(T.area([a if op != 'C' else a[-1] for op, a in cs[1]]), 0)   # a cut-out
        for code, (_, which) in T.LABELS.items():
            _, cs = cff.glyph(gid[code])
            letter = 3 if code in (0xE0E5, 0xE0E7) else 2   # the R comes with its counter
            self.assertEqual(len(cs), 1 + letter - 1 + 1, hex(code))
            self.assertEqual(T.bbox(cs[0]), (0, 2635, -570, 2065))
        for code in (0xE0EF, 0xE0F0):
            _, cs = cff.glyph(gid[code])
            self.assertEqual(len(cs), 4)
        swapped = {c: b for a, b in T.DIAMOND_SWAPS[layout] for c, b in ((a, b), (b, a))}
        changed = set(T.FACE[layout]) | set(T.LABELS) | {0xE0EF, 0xE0F0}
        for code in CODES:
            if code in changed:
                continue
            want = self.cff.charstrings[self.gid[swapped.get(code, code)]]
            self.assertEqual(cff.charstrings[gid[code]], want, (layout, hex(code)))
        self.assertEqual(cff.charstrings[0], self.cff.charstrings[0])
        self.assertEqual(T.ps_font(self.font, layout), out)   # the same every run
        return out

    def test_xbox_layout(self):
        self.check_layout('xbox')

    def test_nintendo_layout(self):
        self.check_layout('nintendo')

    def test_generate_end_to_end(self):
        try:
            T.zstd_compress(b'probe')
        except Exception as ex:   # pragma: no cover
            self.skipTest(f'no zstd: {ex}')
        key = b'\x5a\x3c\x91\x07'
        plain = b'\x1a\x87\x9b\xd9' + struct.pack('>I', len(self.font)) + self.font
        wrapped = plain[:4] + bytes(x ^ key[i % 4] for i, x in enumerate(plain[4:]))
        neighbour = bytes(range(256)) * 3
        sarc = make_sarc([('scft/Other.bfotf', neighbour), (T.FONT_MEMBER, wrapped)])
        with tempfile.TemporaryDirectory() as work:
            romfs = Path(work) / 'romfs'
            romfs.mkdir()
            image, rows = b'', []
            for path in ('Font/Font.Nin_Ounce_NVN2.bfarc.zs', 'Font/Font_KRko.Nin_Ounce_NVN2.bfarc.zs', 'Model/x.bin'):
                raw = T.zstd_compress(sarc) if path.startswith('Font/') else b'not a font'
                rows.append({'path': path, 'offset': len(image), 'size': len(raw)})
                image += raw
            (romfs / 'romfs.bin').write_bytes(image)
            (romfs / 'index.json').write_text(json.dumps({'status': 'pass', 'files': rows}))
            out = Path(work) / 'out'
            T.generate(romfs, out, 'xbox')
            lines = [line.split('\t') for line in (out / 'manifest.tsv').read_text().splitlines()[1:]]
            self.assertEqual([line[0] for line in lines], [r['path'] for r in rows[:2]])
            for path, size, digest, original in lines:
                data = (out / path).read_bytes()
                self.assertEqual((int(size), digest), (len(data), hashlib.sha256(data).hexdigest()))
                row = next(r for r in rows if r['path'] == path)
                self.assertEqual(original, hashlib.sha256(image[row['offset']:row['offset'] + row['size']]).hexdigest())
                members = {n: (lo, hi) for n, _, lo, hi in T.sarc_nodes(T.zstd_decompress(data))}
                archive = T.zstd_decompress(data)
                self.assertEqual(archive[slice(*members['scft/Other.bfotf'])], neighbour)
                k, font = T.bfotf_unwrap(archive[slice(*members[T.FONT_MEMBER])])
                self.assertEqual(k, key)
                self.assertEqual(font, T.ps_font(self.font, 'xbox'))


def make_sarc(members):
    names, offs = b'', []
    for name, _ in members:
        offs.append(len(names))
        names += name.encode() + b'\0'
        names += bytes(-len(names) % 4)
    start = 0x14 + 12 + 16 * len(members) + 8 + len(names)
    start += -start % 0x100
    data, entries = b'', b''
    for (_, body), no in zip(members, offs):
        data += bytes(-len(data) % 8)
        lo = len(data)
        data += body
        entries += struct.pack('<4I', 0, 1 << 24 | no // 4, lo, len(data))
    head = bytearray(0x14)
    head[0:4] = b'SARC'
    struct.pack_into('<H', head, 4, 0x14)
    head[6:8] = b'\xff\xfe'
    struct.pack_into('<II', head, 8, start + len(data), start)
    pre = bytes(head) + b'SFAT' + struct.pack('<HHI', 12, len(members), 0x65) + entries + b'SFNT' + \
        struct.pack('<HH', 8, 0) + names
    return pre + bytes(start - len(pre)) + data


if __name__ == '__main__':
    unittest.main()
