"""Build the PlayStation button prompt overlay for Bananza from the user's RomFS (runtime/romfs.c overlay_init loads it
in addition to the other overlays when a DualShock 4 or DualSense pad is connected at start, or pad_prompts says so).

Bananza draws every button prompt with one font, scft/NinDev-HardKey.bfotf, the same in each Font/Font*.bfarc.zs
archive (found with tools/dkb_prompt_survey.py). Each glyph is an irregular black plate with the button's letter cut
out of it. This keeps each plate as the game draws it and replaces only the cut-out:

- A, B, X, Y (U+E0E0 to U+E0E3): a cross, circle, square or triangle, which button depending on --layout;
- L, R, ZL, ZR (U+E0E4 to U+E0E7): L1, R1, L2, R2, the game's own L and R letters (taken from its ZL and ZR glyphs)
  with a 1 or a 2;
- + and - (U+E0EF, U+E0F0): the Options bars and the Create rays.

With --layout xbox (pad_layout auto with a Sony pad, or xbox) the button that does Switch A is the bottom one, so A
gets the cross, B the circle, X the square and Y the triangle, and the face-button diamonds that point at one button
(U+E0AB to U+E0AE, U+E0CB and U+E0CE) trade places so they point at it. With --layout nintendo (pad_layout nintendo or
xbox_prompts: Switch A on the right button) A gets the circle, B the cross, X the triangle and Y the square, and the
diamonds stay.

The symbols are original drawings, made here from straight strokes and arcs. Only the plates and the L and R letters
come from the user's own font, at install time. Output: <out>/Font/Font*.Nin_Ounce_NVN2.bfarc.zs and
<out>/manifest.tsv (path, size, SHA-256, original SHA-256), under build/ or an installer workspace. Runs under the
release's Python 3.14 (compression.zstd) or with the zstandard package. No third-party font library is needed.
"""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

try:
    from compression import zstd as _zstd

    def _dict_id(data):
        return _zstd.get_frame_info(data).dictionary_id

    def _make_dict(raw):
        return _zstd.ZstdDict(raw)

    def zstd_decompress(data, dictionary=None):
        return _zstd.decompress(data, zstd_dict=dictionary)

    def zstd_compress(data):
        return _zstd.compress(data, level=9)
except ImportError:  # Python before 3.14: the zstandard package, if there is one (the font editing needs neither)
    try:
        import zstandard as _zs
    except ImportError:
        _zs = None

    def _dict_id(data):
        return _zs.get_frame_parameters(data).dict_id

    def _make_dict(raw):
        return _zs.ZstdCompressionDict(raw)

    def zstd_decompress(data, dictionary=None):
        if _zs is None:
            raise RuntimeError('zstd needs Python 3.14 (compression.zstd) or the zstandard package')
        return _zs.ZstdDecompressor(dict_data=dictionary).decompress(data, max_output_size=1 << 30)

    def zstd_compress(data):
        if _zs is None:
            raise RuntimeError('zstd needs Python 3.14 (compression.zstd) or the zstandard package')
        return _zs.ZstdCompressor(level=9).compress(data)

FONT_MEMBER = 'scft/NinDev-HardKey.bfotf'
FONT_ARCHIVE = re.compile(r'Font/Font(_\w+)?\.Nin_Ounce_NVN2\.bfarc\.zs')

# ---------------------------------------------------------------------------------------------------- the symbols
# Font units, y up. The letter area of the A/B/X/Y plates is about x 640..2000, y 40..1450.
CX, CY = 1318, 745
STROKE = 310


def jitter(pts, amount=18, seed=1):
    """Moves each vertex a little, the same way every run, so the edges are not machine-perfect, like the plates'."""
    out = []
    for i, (x, y) in enumerate(pts):
        a = math.sin((i + 1) * 12.9898 * seed) * 43758.5453
        b = math.sin((i + 1) * 78.233 * seed) * 12345.678
        out.append((x + (a - math.floor(a) - .5) * 2 * amount, y + (b - math.floor(b) - .5) * 2 * amount))
    return out


def area(pts):
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts))) / 2


def wind(pts, ccw):
    """pts in counter-clockwise order (ccw) or clockwise. The plates run counter-clockwise; a cut-out clockwise."""
    return pts if (area(pts) > 0) == ccw else pts[::-1]


def rot(pts, deg, cx=CX, cy=CY):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]


def poly(pts):
    """A closed contour of straight edges: [('M', p), ('L', p), ...]."""
    return [('M', pts[0])] + [('L', p) for p in pts[1:]]


def cross(size=760, w=STROKE):
    h = w / 2
    plus = [(CX - h, CY + size), (CX + h, CY + size), (CX + h, CY + h), (CX + size, CY + h), (CX + size, CY - h),
            (CX + h, CY - h), (CX + h, CY - size), (CX - h, CY - size), (CX - h, CY - h), (CX - size, CY - h),
            (CX - size, CY + h), (CX - h, CY + h)]
    return [poly(wind(jitter(rot(plus, 45), seed=3), False))]


def square(half=580, w=290):
    o = [(CX - half, CY - half), (CX + half, CY - half), (CX + half, CY + half), (CX - half, CY + half)]
    i = [(CX - half + w, CY - half + w), (CX + half - w, CY - half + w), (CX + half - w, CY + half - w),
         (CX - half + w, CY + half - w)]
    return [poly(wind(jitter(rot(o, 3), seed=5), False)), poly(wind(jitter(rot(i, 3), 12, seed=7), True))]


def triangle(r=900, w=255):
    cy = CY - 90   # the centroid sits low, so the triangle looks centred in the plate
    ri = r - 2 * w   # an inset of w on each side of an equilateral triangle takes 2w off its circumradius
    o = [(CX + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in (90, 210, 330)]
    i = [(CX + ri * math.cos(math.radians(a)), cy + ri * math.sin(math.radians(a))) for a in (90, 210, 330)]
    return [poly(wind(jitter(o, 20, seed=9), False)), poly(wind(jitter(i, 10, seed=11), True))]


def circle_contour(r, ccw):
    """A circle of four cubic arcs."""
    k = 4 / 3 * math.tan(math.pi / 8)
    pts = []
    for j in range(4):
        a0, a1 = math.pi / 2 * j, math.pi / 2 * (j + 1)
        p0 = (CX + r * math.cos(a0), CY + r * math.sin(a0))
        p3 = (CX + r * math.cos(a1), CY + r * math.sin(a1))
        pts.append((p0, (p0[0] - k * r * math.sin(a0), p0[1] + k * r * math.cos(a0)),
                    (p3[0] + k * r * math.sin(a1), p3[1] - k * r * math.cos(a1)), p3))
    if not ccw:
        pts = [(p3, p2, p1, p0) for p0, p1, p2, p3 in reversed(pts)]
    return [('M', pts[0][0])] + [('C', (p1, p2, p3)) for _, p1, p2, p3 in pts]


def circle(r=680, w=275):
    return [circle_contour(r, False), circle_contour(r - w, True)]


# Digits for L1 / R1 / L2 / R2, in a box 600 wide and 1008 tall (the height of the L of the game's ZL glyph).
DIGIT_1 = [(250, 0), (530, 0), (530, 1008), (320, 1008), (70, 870), (70, 640), (250, 730)]
DIGIT_2 = [(30, 0), (600, 0), (600, 240), (330, 240), (505, 455), (580, 590), (595, 740), (560, 880), (460, 975),
           (305, 1008), (150, 978), (55, 890), (18, 755), (262, 728), (282, 790), (312, 805), (345, 785), (352, 735),
           (322, 660), (30, 270)]


def digit(which, x0, y0):
    pts = DIGIT_1 if which == 1 else DIGIT_2
    return [poly(wind(jitter([(x0 + x, y0 + y) for x, y in pts], 10, seed=13 + which), False))]


def options_bars(n=3, length=1050, thick=210, gap=120):
    out, top = [], CY + (n * thick + (n - 1) * gap) / 2
    for k in range(n):
        y1 = top - k * (thick + gap)
        y0 = y1 - thick
        bar = [(CX - length / 2, y0), (CX + length / 2, y0), (CX + length / 2, y1), (CX - length / 2, y1)]
        out.append(poly(wind(jitter(bar, 14, seed=17 + k), False)))
    return out


def create_rays(thick=190):
    out = []
    for k, (deg, start, length) in enumerate(((140, 330, 430), (90, 300, 520), (40, 330, 430))):
        a = math.radians(deg)
        ux, uy = math.cos(a), math.sin(a)
        nx, ny = -uy, ux
        base = (CX + ux * start, CY - 420 + uy * start)
        tip = (base[0] + ux * length, base[1] + uy * length)
        h = thick / 2
        bar = [(base[0] + nx * h, base[1] + ny * h), (tip[0] + nx * h, tip[1] + ny * h),
               (tip[0] - nx * h, tip[1] - ny * h), (base[0] - nx * h, base[1] - ny * h)]
        out.append(poly(wind(jitter(bar, 12, seed=23 + k), False)))
    return out


SYMBOLS = {'cross': cross, 'circle': circle, 'square': square, 'triangle': triangle}
FACE = {'xbox': {0xE0E0: 'cross', 0xE0E1: 'circle', 0xE0E2: 'square', 0xE0E3: 'triangle'},
        'nintendo': {0xE0E0: 'circle', 0xE0E1: 'cross', 0xE0E2: 'triangle', 0xE0E3: 'square'}}
# Diamonds that point at one face button: A (right), B (bottom), X (top), Y (left), and two that point at a pair.
DIAMOND_SWAPS = {'xbox': ((0xE0AB, 0xE0AC), (0xE0AD, 0xE0AE), (0xE0CB, 0xE0CE)), 'nintendo': ()}
LABELS = {0xE0E4: (0xE0E6, 1), 0xE0E5: (0xE0E7, 1), 0xE0E6: (0xE0E6, 2), 0xE0E7: (0xE0E7, 2)}   # glyph: (letter source, digit)

# ---------------------------------------------------------------------------------------------------- CFF reading


def read_index(data, pos):
    """([bytes of each item], position after the INDEX)."""
    count = struct.unpack_from('>H', data, pos)[0]
    if not count:
        return [], pos + 2
    size = data[pos + 2]
    offs = [int.from_bytes(data[pos + 3 + i * size:pos + 3 + (i + 1) * size], 'big') for i in range(count + 1)]
    first = pos + 2 + (count + 1) * size
    return [bytes(data[first + offs[i]:first + offs[i + 1]]) for i in range(count)], first + offs[-1]


def write_index(items):
    if not items:
        return b'\0\0'
    offs, total = [1], 1
    for it in items:
        total += len(it)
        offs.append(total)
    size = 1 if total < 0x100 else 2 if total < 0x10000 else 3 if total < 0x1000000 else 4
    return (struct.pack('>HB', len(items), size) + b''.join(o.to_bytes(size, 'big') for o in offs) + b''.join(items))


def read_dict(data):
    """[(operator, [operands], raw operand bytes)] of a DICT."""
    out, operands, start, pos = [], [], 0, 0
    while pos < len(data):
        b = data[pos]
        if b <= 21:
            op = (12, data[pos + 1]) if b == 12 else b
            end = pos + (2 if b == 12 else 1)
            out.append((op, operands, data[start:pos]))
            operands, start, pos = [], end, end
        elif b == 28:
            operands.append(struct.unpack_from('>h', data, pos + 1)[0]); pos += 3
        elif b == 29:
            operands.append(struct.unpack_from('>i', data, pos + 1)[0]); pos += 5
        elif b == 30:
            pos += 1
            while data[pos] & 0x0f != 0x0f and data[pos] & 0xf0 != 0xf0:
                pos += 1
            pos += 1
            operands.append(None)   # a real number: kept as its raw bytes
        elif b <= 246:
            operands.append(b - 139); pos += 1
        elif b <= 250:
            operands.append((b - 247) * 256 + data[pos + 1] + 108); pos += 2
        else:
            operands.append(-(b - 251) * 256 - data[pos + 1] - 108); pos += 2
    return out


def dict_op_bytes(op):
    return bytes(op) if isinstance(op, tuple) else bytes([op])


def int5(v):
    return b'\x1d' + struct.pack('>i', v)


def charstring_contours(code, subrs, gsubrs):
    """Interprets a Type 2 charstring: (width operand or None, [contour]), each contour [('M', p), ('L', p),
    ('C', (p1, p2, p3)), ...] in absolute font units."""
    def bias(n):
        return 107 if n < 1240 else 1131 if n < 33900 else 32768
    stack, contours, cur = [], [], None
    x = y = 0.0
    nstems = 0
    width = None
    first = [True]

    def take_width(expected_even):
        nonlocal width
        if first[0]:
            first[0] = False
            if (len(stack) % 2 == 1) == expected_even:
                width = stack.pop(0)

    def move(dx, dy):
        nonlocal x, y, cur
        x, y = x + dx, y + dy
        cur = [('M', (x, y))]
        contours.append(cur)

    def line(dx, dy):
        nonlocal x, y
        x, y = x + dx, y + dy
        cur.append(('L', (x, y)))

    def curve(a, b, c, d, e, f):
        nonlocal x, y
        p1 = (x + a, y + b)
        p2 = (p1[0] + c, p1[1] + d)
        p3 = (p2[0] + e, p2[1] + f)
        x, y = p3
        cur.append(('C', (p1, p2, p3)))

    def run(prog):
        nonlocal nstems
        pos = 0
        while pos < len(prog):
            b = prog[pos]
            pos += 1
            if b >= 32 or b == 28 or b == 255:
                if b == 28:
                    stack.append(struct.unpack_from('>h', prog, pos)[0]); pos += 2
                elif b == 255:
                    stack.append(struct.unpack_from('>i', prog, pos)[0] / 65536); pos += 4
                elif b <= 246:
                    stack.append(b - 139)
                elif b <= 250:
                    stack.append((b - 247) * 256 + prog[pos] + 108); pos += 1
                else:
                    stack.append(-(b - 251) * 256 - prog[pos] - 108); pos += 1
                continue
            if b in (1, 3, 18, 23):             # hstem vstem hstemhm vstemhm
                take_width(True)
                nstems += len(stack) // 2
                stack.clear()
            elif b in (19, 20):                 # hintmask cntrmask
                take_width(True)
                nstems += len(stack) // 2
                stack.clear()
                pos += (nstems + 7) // 8
            elif b == 21:                       # rmoveto
                take_width(True)
                move(stack[-2], stack[-1]); stack.clear()
            elif b == 22:                       # hmoveto
                take_width(False)
                move(stack[-1], 0); stack.clear()
            elif b == 4:                        # vmoveto
                take_width(False)
                move(0, stack[-1]); stack.clear()
            elif b == 5:                        # rlineto
                for i in range(0, len(stack) - 1, 2):
                    line(stack[i], stack[i + 1])
                stack.clear()
            elif b in (6, 7):                   # hlineto vlineto
                horiz = b == 6
                for v in stack:
                    line(v, 0) if horiz else line(0, v)
                    horiz = not horiz
                stack.clear()
            elif b == 8:                        # rrcurveto
                for i in range(0, len(stack) - 5, 6):
                    curve(*stack[i:i + 6])
                stack.clear()
            elif b == 24:                       # rcurveline
                n = len(stack) - 2
                for i in range(0, n, 6):
                    curve(*stack[i:i + 6])
                line(stack[-2], stack[-1]); stack.clear()
            elif b == 25:                       # rlinecurve
                n = len(stack) - 6
                for i in range(0, n, 2):
                    line(stack[i], stack[i + 1])
                curve(*stack[-6:]); stack.clear()
            elif b in (26, 27):                 # vvcurveto hhcurveto
                s = list(stack)
                first_off = s.pop(0) if len(s) % 4 == 1 else 0
                for i in range(0, len(s), 4):
                    a, c, d, e = s[i:i + 4]
                    if b == 26:
                        curve(first_off if i == 0 else 0, a, c, d, 0, e)
                    else:
                        curve(a, first_off if i == 0 else 0, c, d, e, 0)
                stack.clear()
            elif b in (30, 31):                 # vhcurveto hvcurveto
                s = list(stack)
                horiz = b == 31
                i = 0
                while i + 4 <= len(s):
                    last = i + 8 > len(s)
                    extra = s[i + 4] if last and i + 5 == len(s) else 0
                    if horiz:
                        curve(s[i], 0, s[i + 1], s[i + 2], extra, s[i + 3])
                    else:
                        curve(0, s[i], s[i + 1], s[i + 2], s[i + 3], extra)
                    horiz = not horiz
                    i += 4
                stack.clear()
            elif b == 10:                       # callsubr
                n = int(stack.pop()) + bias(len(subrs))
                run(subrs[n])
            elif b == 29:                       # callgsubr
                n = int(stack.pop()) + bias(len(gsubrs))
                run(gsubrs[n])
            elif b == 11:                       # return
                return
            elif b == 14:                       # endchar
                take_width(False)
                stack.clear()
                return 'end'
            elif b == 12:
                raise ValueError(f'charstring operator 12 {prog[pos]} (flex or arithmetic) is not handled')
            else:
                raise ValueError(f'charstring operator {b} is not handled')
        return None

    run(code)
    return width, contours


def num(v):
    v = int(round(v))
    return bytes([v + 139]) if -107 <= v <= 107 else struct.pack('>bh', 28, v)


def encode_charstring(width, contours):
    """A Type 2 charstring of rmoveto, rlineto and rrcurveto, with the width operand when given."""
    out, x, y, first = bytearray(), 0, 0, True
    for c in contours:
        for op, a in c:
            if op == 'M':
                px, py = round(a[0]), round(a[1])
                if first and width is not None:
                    out += num(width)
                first = False
                out += num(px - x) + num(py - y) + b'\x15'
                x, y = px, py
            elif op == 'L':
                px, py = round(a[0]), round(a[1])
                out += num(px - x) + num(py - y) + b'\x05'
                x, y = px, py
            else:
                for p in a:
                    px, py = round(p[0]), round(p[1])
                    out += num(px - x) + num(py - y)
                    x, y = px, py
                out += b'\x08'
    if first and width is not None:
        out += num(width)
    return bytes(out + b'\x0e')


def bbox(c):
    pts = [p for op, a in c for p in ((a,) if op != 'C' else a)]
    return min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts)


def charset_length(data, pos, nglyphs):
    fmt = data[pos]
    if fmt == 0:
        return 1 + 2 * (nglyphs - 1)
    covered, p = 1, pos + 1
    while covered < nglyphs:
        left = data[p + 2] if fmt == 1 else struct.unpack_from('>H', data, p + 2)[0]
        p += 3 if fmt == 1 else 4
        covered += left + 1
    return p - pos


class Cff:
    """A non-CID CFF table whose CharStrings can be replaced and the table written again."""

    def __init__(self, data):
        self.data = data
        hdr = data[2]
        self.header = data[:hdr]
        names, pos = read_index(data, hdr)
        self.name_index = data[hdr:pos]
        tops, pos2 = read_index(data, pos)
        if len(tops) != 1:
            raise ValueError('one font per CFF expected')
        self.top = read_dict(tops[0])
        strings_start = pos2
        _, pos3 = read_index(data, pos2)
        self.strings = data[strings_start:pos3]
        self.gsubrs, pos4 = read_index(data, pos3)
        self.gsubr_bytes = data[pos3:pos4]
        ops = {op: operands for op, operands, _ in self.top}
        if (12, 30) in ops:
            raise ValueError('CID-keyed CFF')
        if ops.get(16, [0])[0] not in (0, 1):
            raise ValueError('a custom CFF Encoding is not handled')
        self.charstrings, _ = read_index(data, ops[17][0])
        psize, poff = ops[18]
        private = read_dict(data[poff:poff + psize])
        pops = {op: operands for op, operands, _ in private}
        self.nominal = pops.get(21, [0])[0]
        self.default_width = pops.get(20, [0])[0]
        self.subrs = []
        end = poff + psize
        if 19 in pops:
            self.subrs, end = read_index(data, poff + pops[19][0])
        self.private_block = data[poff:end]   # the Private DICT and its Subrs, which it addresses relative to itself
        self.private_size = psize
        cs_at = ops[15][0] if 15 in ops else 0
        self.charset = data[cs_at:cs_at + charset_length(data, cs_at, len(self.charstrings))] if cs_at > 2 else None
        self.charset_id = cs_at if cs_at <= 2 else None

    def glyph(self, gid):
        return charstring_contours(self.charstrings[gid], self.subrs, self.gsubrs)

    def build(self):
        def top_bytes(charset_off, cs_off, priv_off):
            out = bytearray()
            for op, operands, raw in self.top:
                if op == 15:
                    raw = int5(charset_off) if self.charset is not None else raw
                elif op == 17:
                    raw = int5(cs_off)
                elif op == 18:
                    raw = int5(self.private_size) + int5(priv_off)
                out += raw + dict_op_bytes(op)
            return bytes(out)
        charstrings = write_index(self.charstrings)
        top_len = len(write_index([top_bytes(0, 0, 0)]))   # every offset is written in 5 bytes, so the size is fixed
        base = len(self.header) + len(self.name_index) + top_len + len(self.strings) + len(self.gsubr_bytes)
        charset_off = base
        cs_off = charset_off + (len(self.charset) if self.charset is not None else 0)
        priv_off = cs_off + len(charstrings)
        top = write_index([top_bytes(charset_off, cs_off, priv_off)])
        assert len(top) == top_len
        return (bytes(self.header) + bytes(self.name_index) + top + bytes(self.strings) + bytes(self.gsubr_bytes) +
                (bytes(self.charset) if self.charset is not None else b'') + charstrings + bytes(self.private_block))

# ---------------------------------------------------------------------------------------------------- the font


def sfnt_tables(font):
    out = {}
    for k in range(struct.unpack_from('>H', font, 4)[0]):
        tag, _, offset, length = struct.unpack_from('>4sIII', font, 12 + 16 * k)
        out[tag] = font[offset:offset + length]
    return out


def checksum(data):
    data = bytes(data) + bytes(-len(data) % 4)
    return sum(struct.unpack(f'>{len(data) // 4}I', data)) & 0xffffffff


def sfnt_build(font, tables):
    """The font with its tables replaced (same set and directory order), laid out again with checksums."""
    n = struct.unpack_from('>H', font, 4)[0]
    tags = [struct.unpack_from('>4s', font, 12 + 16 * k)[0] for k in range(n)]
    order = sorted(tags, key=lambda t: struct.unpack_from('>I', font, 12 + 16 * tags.index(t) + 8)[0])
    head = bytearray(tables[b'head'])
    struct.pack_into('>I', head, 8, 0)
    tables = {**tables, b'head': bytes(head)}
    out = bytearray(font[:12 + 16 * n])
    offsets = {}
    for t in order:
        out += bytes(-len(out) % 4)
        offsets[t] = len(out)
        out += tables[t]
    out += bytes(-len(out) % 4)
    for k, t in enumerate(tags):
        struct.pack_into('>4sIII', out, 12 + 16 * k, t, checksum(tables[t]), offsets[t], len(tables[t]))
    struct.pack_into('>I', out, offsets[b'head'] + 8, (0xb1b0afba - checksum(out)) & 0xffffffff)
    return bytes(out)


def cmap_lookup(cmap, code):
    for k in range(struct.unpack_from('>H', cmap, 2)[0]):
        platform, encoding, offset = struct.unpack_from('>HHI', cmap, 4 + 8 * k)
        p = offset
        if (platform, encoding) != (3, 1) or struct.unpack_from('>H', cmap, p)[0] != 4:
            continue
        seg = struct.unpack_from('>H', cmap, p + 6)[0] // 2
        for s in range(seg):
            end = struct.unpack_from('>H', cmap, p + 14 + 2 * s)[0]
            start = struct.unpack_from('>H', cmap, p + 16 + 2 * seg + 2 * s)[0]
            if not start <= code <= end:
                continue
            delta = struct.unpack_from('>h', cmap, p + 16 + 4 * seg + 2 * s)[0]
            at = p + 16 + 6 * seg + 2 * s
            ro = struct.unpack_from('>H', cmap, at)[0]
            glyph = code if not ro else struct.unpack_from('>H', cmap, at + ro + 2 * (code - start))[0]
            return (glyph + delta) & 0xffff if glyph else 0
    raise ValueError(f'U+{code:04X} is not in the font')


def plate_of(contours):
    """The plate: the widest contour of a glyph."""
    return max(contours, key=lambda c: bbox(c)[1] - bbox(c)[0])


def shifted(c, dx):
    return [(op, (a[0] + dx, a[1]) if op != 'C' else tuple((p[0] + dx, p[1]) for p in a)) for op, a in c]


def ps_font(font, layout):
    """The plain OpenType HardKey font with the PlayStation glyphs."""
    tables = sfnt_tables(font)
    cff = Cff(tables[b'CFF '])
    gid = {code: cmap_lookup(tables[b'cmap'], code) for code in (*FACE[layout], *LABELS, 0xE0EF, 0xE0F0,
                                                                   *(c for pair in DIAMOND_SWAPS[layout] for c in pair))}
    original = {code: cff.glyph(g) for code, g in gid.items()}
    new = {}
    for code, sym in FACE[layout].items():
        width, cs = original[code]
        new[code] = encode_charstring(width, [plate_of(cs)] + SYMBOLS[sym]())
    for code, (source, which) in LABELS.items():
        width, cs = original[code]
        plate = plate_of(cs)
        _, src = original[source]
        inner = [c for c in src if c is not plate_of(src)]
        z = min(inner, key=lambda c: bbox(c)[0])
        letter = [c for c in inner if bbox(c)[0] > bbox(z)[1] - 40]   # the L or R, with the R's counter
        lx0 = min(bbox(c)[0] for c in letter)
        lx1 = max(bbox(c)[1] for c in letter)
        ly0 = min(bbox(c)[2] for c in letter)
        gap, dw = 90, 600 * (0.82 if which == 1 else 1.0)
        px0, px1 = bbox(plate)[0], bbox(plate)[1]
        start = (px0 + px1) / 2 - ((lx1 - lx0) + gap + dw) / 2
        moved = [shifted(c, start - lx0) for c in letter]
        new[code] = encode_charstring(width, [plate] + moved +
                                      digit(which, start + (lx1 - lx0) + gap - (60 if which == 1 else 0), ly0))
    for code, art in ((0xE0EF, options_bars), (0xE0F0, create_rays)):
        width, cs = original[code]
        new[code] = encode_charstring(width, [plate_of(cs)] + art())
    for a, b in DIAMOND_SWAPS[layout]:
        new[a], new[b] = cff.charstrings[gid[b]], cff.charstrings[gid[a]]
    for code, cs in new.items():
        cff.charstrings[gid[code]] = cs
    return sfnt_build(font, {**tables, b'CFF ': cff.build()})

# ---------------------------------------------------------------------------------------------------- the archives


def bfotf_unwrap(data):
    key = bytes(x ^ y for x, y in zip(data[4:8], struct.pack('>I', len(data) - 8)))
    return key, bytes(x ^ key[i % 4] for i, x in enumerate(data[8:]))


def bfotf_wrap(head4, key, font):
    plain = bytes(head4) + struct.pack('>I', len(font)) + font
    return plain[:4] + bytes(x ^ key[i % 4] for i, x in enumerate(plain[4:]))


def sarc_nodes(data):
    """(name, node offset, absolute start, end) of every SARC member, in SFAT order (as tools/make_romfs_patches.py)."""
    assert data[:4] == b'SARC' and data[6:8] == b'\xff\xfe'
    header, size, start = struct.unpack_from('<HxxII', data, 4)
    assert size == len(data) and data[header:header + 4] == b'SFAT'
    count = struct.unpack_from('<H', data, header + 6)[0]
    names = header + 12 + count * 16
    assert data[names:names + 4] == b'SFNT'
    names += 8
    out = []
    for i in range(count):
        node = header + 12 + i * 16
        _, attribute, lo, hi = struct.unpack_from('<4I', data, node)
        offset = names + (attribute & 0xffffff) * 4
        out.append((data[offset:data.index(0, offset, start)].decode(), node, start + lo, start + hi))
    return out


def sarc_replace(data, new):
    """The SARC with the members named in new replaced, laid out again in their old order, each at least as aligned as
    before (as tools/make_romfs_patches.py)."""
    start = struct.unpack_from('<I', data, 12)[0]
    out = bytearray(data[:start])
    for name, node, lo, hi in sorted(sarc_nodes(data), key=lambda n: n[2]):
        align = min(lo & -lo, 0x2000) if lo else 0x2000
        out += bytes(-len(out) % align)
        body = new.get(name, data[lo:hi])
        struct.pack_into('<II', out, node + 8, len(out) - start, len(out) - start + len(body))
        out += body
    struct.pack_into('<I', out, 8, len(out))
    return bytes(out)


def generate(romfs, out, layout):
    index = json.loads((romfs / 'index.json').read_text(encoding='utf-8'))
    if index.get('status') != 'pass':
        raise ValueError('the RomFS index has not passed its extraction check')
    files = index['files']
    lines = ['# path\tsize\tsha256\toriginal_sha256 (tools/dkb_ps_prompts.py --layout ' + layout + ')']
    outputs, made = [], {}
    with (romfs / 'romfs.bin').open('rb') as image:
        def read(row):
            image.seek(row['offset'])
            raw = image.read(row['size'])
            if len(raw) != row['size'] or ('sha256' in row and hashlib.sha256(raw).hexdigest() != row['sha256']):
                raise ValueError(f'{row["path"]}: packed resource checksum mismatch')
            return raw
        dictionaries = {}
        for row in files:
            if row['path'].lower().endswith('zsdic.pack.zs'):
                pack = zstd_decompress(read(row))
                for _, _, lo, hi in sarc_nodes(pack):
                    member = pack[lo:hi]
                    if member[:4] == b'\x37\xa4\x30\xec':
                        dictionaries[struct.unpack_from('<I', member, 4)[0]] = _make_dict(member)
        archives = [r for r in files if FONT_ARCHIVE.fullmatch(r['path'])]
        if not archives:
            raise ValueError('no Font/Font*.Nin_Ounce_NVN2.bfarc.zs in the RomFS')
        for row in archives:
            raw = read(row)
            sarc = zstd_decompress(raw, dictionaries.get(_dict_id(raw)))
            members = {name: (lo, hi) for name, _, lo, hi in sarc_nodes(sarc)}
            if FONT_MEMBER not in members:
                raise ValueError(f'{row["path"]} has no {FONT_MEMBER}')
            lo, hi = members[FONT_MEMBER]
            wrapped = sarc[lo:hi]
            digest = hashlib.sha256(wrapped).hexdigest()
            if digest not in made:   # the five archives hold the same font: make it once
                key, font = bfotf_unwrap(wrapped)
                if font[:4] != b'OTTO':
                    raise ValueError(f'{row["path"]}:{FONT_MEMBER} is not a CFF OpenType font')
                made[digest] = bfotf_wrap(wrapped[:4], key, ps_font(font, layout))
            patched = zstd_compress(sarc_replace(sarc, {FONT_MEMBER: made[digest]}))
            outputs.append((row['path'], raw, patched))
    for path, raw, patched in outputs:
        dest = out / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(patched)
        lines.append('\t'.join((path, str(len(patched)), hashlib.sha256(patched).hexdigest(),
                                hashlib.sha256(raw).hexdigest())))
    temporary = out / 'manifest.tsv.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(out / 'manifest.tsv')   # the manifest last: an interrupted run leaves no valid overlay
    print(f'PlayStation prompts ({layout} layout): {len(outputs)} font archives in {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--romfs', type=Path, required=True, help="the install's romfs folder (romfs.bin and index.json)")
    ap.add_argument('--out', type=Path, required=True, help='the overlay folder, under build/ or an installer workspace')
    ap.add_argument('--layout', choices=('xbox', 'nintendo'), default='xbox',
                    help='xbox: Switch A on the bottom button (cross); nintendo: on the right one (circle)')
    ap.add_argument('--installer-workspace', type=Path, help='owned installer workspace containing build staging')
    args = ap.parse_args()
    build = (Path(__file__).resolve().parents[1] / 'build').resolve()
    if args.installer_workspace:
        workspace = args.installer_workspace.resolve()
        owner = json.loads((workspace / '.dkb-setup-owner.json').read_text(encoding='utf-8'))
        if Path(owner['temporary']).resolve() != workspace or owner['game'] not in ('dkb100', 'dkb458752'):
            ap.error('invalid installer workspace owner')
        build = workspace / 'build'
    out = args.out.resolve()
    if not out.is_relative_to(build) or out == build:
        ap.error('game-derived overlays must be in a dedicated directory under build/')
    generate(args.romfs, out, args.layout)


if __name__ == '__main__':
    main()
