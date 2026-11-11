"""Code patch table for the generator: guest instructions replaced in the lifted C, switchable at run time.

Lifted code cannot be byte-patched when the game runs, so a patch is applied when the C is generated. A patch file
(gen.py --patches) has two kinds of lines; `#` starts a comment, fields are separated by whitespace:

  set  <name> <env> <default>               a patch set: a run-time switch <env>=0/1, <default> when it is unset
  site <set> <module> <address> <original> <replacement> [<replacement> ...]

  spin <module> <head> <back-edge> [<back-edge> ...]

  guard <set> <module> <address> <target> <kind> x<n> <original> [st32 x<b> <offset> <value>]
                                            before the instruction at <address> (whose word <original> is checked),
                                            when the set is on and the runtime's check rt_guard_<kind>(x<n>) fails,
                                            go to <target> in the same function, after storing the 32-bit <value>
                                            at x<b> + <offset> (signed hex) when the st32 action is given

<address> is a module offset (hex), <original> the instruction word the module has there (checked; a mismatch stops
the generator), each <replacement> an instruction word (hex) or `@page:x<n>`, which is `adrp x<n>, <patch page>`
encoded for the site's address. The patch page is the first 4 KiB page after main's image (layout.py leaves the rest
of the 2 MiB-aligned module slot free); the runtime commits it and owns its contents. Words in use, by byte offset:
+0..15 the frame-rate step (runtime/frame_rate.c), +16..55 the output, swapchain and scene sizes
(runtime/resolution.c res_init), +56..59 the camera's aspect ratio (runtime/resolution.c res_init, set aspect), +60..63
the field-of-view factor (runtime/resolution.c res_fov_live, set fov), +0x100..0x113 the Bananza display
size and aspect (runtime/dkb_wide.c), +0x200..0x24f the five Bananza mouse-camera host dispatch slots
(runtime/dkb_camera.c), and +0x250..0x25f the Bananza 3.2.0 radial blur lifetime slot (runtime/frame_rate.c
blur_frames). A new set takes words outside these.

The replacement words are lifted at the site's address in place of the original instruction, and the generated C reads
`if (rt_patch_on[<set>]) { replacement } else { original }`: one byte load and a branch per executed site. Replacement
words must not branch except a final `bl` (lifted with the site's return address); the original must not be a branch
other than `bl`. So the function's control flow, labels and jump tables are the original's. The generator writes
patches.h (set names, switches, the page address) next to layout.h for the runtime.

A `guard` line adds a check before an instruction and a branch to an instruction of the same function, for game code
that would follow a bad pointer: the check is a runtime function of one register (runtime/patches.c rt_guard_<kind>),
and the branch is an edge of the function's flow like any other, so the register and flag plans see it. The guarded
instruction must be an unpatched op that writes no flags and does not escape (no rt_svc/rt_brk/rt_udf call): the guard
branches away before it, so its register and flag state at the target is the one of the instruction before it. The
generator refuses a guarded instruction that is not. The optional st32 action writes guest memory on the failing path
only, for game state the target would otherwise read stale (a cached key, say).

A `spin` line names a spin wait found by tools/spin_waits.py: a loop with head <head> that only another thread can end,
and the direct branches back to it (module offsets). Each back-edge (checked: a b, b.cond, cbz / cbnz or tbz / tbnz to
<head>) is lifted as `rt_spin_gate(c, <head>); goto <head>`, which keeps every register and only adds time
(runtime/kernel.c kern_spin_gate).
"""
import re
from pathlib import Path

PAGE = 0x1000
GUARD_KINDS = ('ptr', 'vcall')   # runtime/patches.c rt_guard_ptr, rt_guard_vcall
REG = re.compile(r'x([0-9]|[12][0-9]|30)')


class PatchError(ValueError):
    pass


def adrp(pc, target, rd):
    imm = (target >> 12) - (pc >> 12)
    if not -(1 << 20) <= imm < (1 << 20):
        raise PatchError(f'{pc:#x}: patch page {target:#x} out of ADRP range')
    imm &= 0x1FFFFF
    return 0x90000000 | (imm & 3) << 29 | (imm >> 2) << 5 | rd


class Table:
    """sets: [(name, env, default)]; sites: {module: {offset: (set index, [word | ('page', reg)], original)}}; page."""

    def __init__(self, sets=(), sites=None, page=0, spins=None, guards=None):
        self.sets = list(sets)
        self.sites = sites or {}
        self.page = page
        self.spins = spins or {}   # {module: {back-edge offset: head offset}}
        self.guards = guards or {}   # {module: {offset: (set index, target offset, kind, register, original, action)}}

    def guards_for(self, name, base):
        """{absolute address: (set index, absolute target, kind, register, action)}; action is None or
        ('st32', base register, signed offset, value)."""
        return {base + o: (k, base + t, kind, r, act) for o, (k, t, kind, r, _, act) in self.guards.get(name, {}).items()}

    def spins_for(self, name, base):
        """{absolute back-edge address: absolute head address}."""
        return {base + b: base + h for b, h in self.spins.get(name, {}).items()}

    def for_module(self, name, base):
        """{absolute address: (set index, [words])} with @page resolved."""
        out = {}
        for off, (k, repl, _) in self.sites.get(name, {}).items():
            a = base + off
            out[a] = (k, [adrp(a, self.page, w[1]) if isinstance(w, tuple) else w for w in repl])
        return out


def load(path, mods):
    """Table from a patch file; mods is layout() output [(name, base, size)]."""
    sets, index, sites, spins, guards = [], {}, {}, {}, {}
    bases = {n: (b, s) for n, b, s in mods}
    for no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        f = raw.split('#', 1)[0].split()
        if not f:
            continue
        where = f'{path}:{no}'
        if f[0] == 'set' and len(f) == 4:
            if f[1] in index:
                raise PatchError(f'{where}: set {f[1]} defined twice')
            index[f[1]] = len(sets)
            sets.append((f[1], f[2], int(f[3])))
        elif f[0] == 'site' and len(f) >= 6:
            if f[1] not in index:
                raise PatchError(f'{where}: unknown set {f[1]}')
            if f[2] not in bases:
                raise PatchError(f'{where}: module {f[2]} not in the layout')
            off, orig = int(f[3], 16), int(f[4], 16)
            repl = []
            for w in f[5:]:
                if w.startswith('@page:x'):
                    repl.append(('page', int(w[7:])))
                else:
                    repl.append(int(w, 16))
            mod = sites.setdefault(f[2], {})
            if off in mod:
                raise PatchError(f'{where}: {f[2]}+{off:#x} patched twice')
            mod[off] = (index[f[1]], repl, orig)
        elif f[0] == 'guard' and len(f) in (8, 12):
            if f[1] not in index:
                raise PatchError(f'{where}: unknown set {f[1]}')
            if f[2] not in bases:
                raise PatchError(f'{where}: module {f[2]} not in the layout')
            if f[5] not in GUARD_KINDS or not REG.fullmatch(f[6]):
                raise PatchError(f'{where}: guard kind {f[5]} or register {f[6]} is not known')
            off, target, orig = int(f[3], 16), int(f[4], 16), int(f[7], 16)
            if off % 4 or target % 4:
                raise PatchError(f'{where}: guard address {off:#x} or target {target:#x} is not 4-aligned')
            act = None
            if len(f) == 12:
                if f[8] != 'st32' or not REG.fullmatch(f[9]):
                    raise PatchError(f'{where}: guard action {f[8]} {f[9]} is not st32 x<n>')
                act = ('st32', int(f[9][1:]), int(f[10], 16), int(f[11], 16) & 0xffffffff)
            mod = guards.setdefault(f[2], {})
            if off in mod:
                raise PatchError(f'{where}: {f[2]}+{off:#x} guarded twice')
            mod[off] = (index[f[1]], target, f[5], int(f[6][1:]), orig, act)
        elif f[0] == 'spin' and len(f) >= 4:
            if f[1] not in bases:
                raise PatchError(f'{where}: module {f[1]} not in the layout')
            mod, head = spins.setdefault(f[1], {}), int(f[2], 16)
            for b in f[3:]:
                b = int(b, 16)
                if b in mod:
                    raise PatchError(f'{where}: {f[1]}+{b:#x} is a back-edge twice')
                mod[b] = head
        else:
            raise PatchError(f'{where}: cannot parse {raw!r}')
    base, size = bases['main'][0], bases['main'][1]
    page = (base + size + PAGE - 1) & ~(PAGE - 1)
    nxt = min((b for _, b, _ in mods if b > base), default=None)
    if nxt is not None and page + PAGE > nxt:
        raise PatchError(f'no room for the patch page after main ({page:#x})')
    return Table(sets, sites, page, spins, guards)


def check(table, mod, name):
    """Stops on a site whose original word differs from the module's."""
    for off, (_, _, want) in table.sites.get(name, {}).items():
        a = mod.base + off
        if not mod.in_text(a):
            raise PatchError(f'{name}+{off:#x}: patch site outside .text')
        have = mod.word(a)
        if have != want:
            raise PatchError(f'{name}+{off:#x}: module has {have:08x}, the patch expects {want:08x}')
    for off, (_, target, _, _, want, _) in table.guards.get(name, {}).items():
        if not mod.in_text(mod.base + off) or not mod.in_text(mod.base + target):
            raise PatchError(f'{name}+{off:#x}: guard or its target {name}+{target:#x} outside .text')
        have = mod.word(mod.base + off)
        if have != want:
            raise PatchError(f'{name}+{off:#x}: module has {have:08x}, the guard expects {want:08x}')
    for b, head in table.spins.get(name, {}).items():
        a = mod.base + b
        if not mod.in_text(a) or branch_target(mod.word(a), a) != mod.base + head:
            raise PatchError(f'{name}+{b:#x}: not a direct branch to the spin wait head {name}+{head:#x}')


def branch_target(w, a):
    """Target of the direct branch w at a (b, b.cond, cbz / cbnz, tbz / tbnz), or None."""
    if w >> 26 == 0x05:
        return a + ((((w & 0x3FFFFFF) ^ 0x2000000) - 0x2000000) << 2)
    if w & 0xFF000010 == 0x54000000 or w & 0x7E000000 == 0x34000000:
        return a + ((((w >> 5) & 0x7FFFF) ^ 0x40000) - 0x40000 << 2)
    if w & 0x7E000000 == 0x36000000:
        return a + ((((w >> 5) & 0x3FFF) ^ 0x2000) - 0x2000 << 2)
    return None


def header_text(table, marker):
    lines = [f'// {marker}: code patch sets, generated by recomp/gen.py --patches. Do not edit or commit.',
             '#pragma once', '', f'#define RT_PATCH_COUNT {len(table.sets)}',
             f'#define RT_PATCH_PAGE {table.page:#x}ull',
             '// X(index, name, env switch, default)',
             '#define RT_PATCH_SETS(X) ' + ' '.join(f'X({i}, {n}, "{e}", {d})' for i, (n, e, d) in
                                                    enumerate(table.sets)), '']
    return '\n'.join(lines)
