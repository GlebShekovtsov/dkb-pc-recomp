"""Point a copy of a Bananza save at another stage, so one save reaches every area (a warp without code patches).

    python tools/save_warp.py --stages                         the stages your RomFS has (StageData/<name>.alsarc.zs)
    python tools/save_warp.py --show tutorial-cave             where that private save is
    python tools/save_warp.py tutorial-cave hill HillLevelStage   build/saves/hill/: tutorial-cave moved to HillLevelStage

tools/scenario.py does the same on each run's profile copy when a scenario has `warp = "<stage>"` or with --warp.

What it changes. The current location of a save slot (<profile>/save/<user>/<slot>/progress.sav) is three fixed 64-byte,
zero-padded text fields in a row: the checkpoint the player last touched ("<guid>_CheckPointTmp"), the location path
("Develop/<stage>/Location") and the stage name ("<stage>"); caption.sav (the slot's preview) repeats the location
path. The tool writes the new stage into the location and stage fields of both and clears the checkpoint, which names
an object of the old stage, so the game starts the new one at its default entrance. Everything else in the save stays:
progress, items and unlocks are those of the source save, so a stage the source save has not reached may still behave
as unvisited, and a stage that needs a story flag may not load the way playing would.

It refuses unless the layout is exactly that (one location record, its stage field equal to the stage in its path, a
checkpoint field before it), the new stage exists in the RomFS index (when the index is present) and the name fits
its field, so it never writes into a format it does not recognise. Measured on Bananza 1.0.0 saves. Standard library only.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVES = ROOT / 'build' / 'saves'
ROMFS_INDEX = ROOT / 'build' / 'extracted' / 'dkb100' / 'romfs' / 'index.json'
FIELD = 64
LOCATION = re.compile(rb'Develop/([A-Za-z0-9_]{1,40})/Location\x00')


class SaveError(Exception):
    pass


def stages(index=ROMFS_INDEX):
    """The stage names the RomFS index lists (StageData/<name>.alsarc.zs), or None without an index."""
    if not index.is_file():
        return None
    files = json.loads(index.read_text(encoding='utf-8'))['files']
    return sorted({f['path'][len('StageData/'):].split('.')[0] for f in files
                   if f['path'].startswith('StageData/') and f['path'].count('/') == 1})


def _field(data, at):
    raw = bytes(data[at:at + FIELD])
    return raw.split(b'\0', 1)[0].decode('ascii', 'replace'), raw


def _put(data, at, text):
    b = text.encode('ascii')
    if len(b) >= FIELD:
        raise SaveError(f'"{text}" does not fit a {FIELD}-byte field')
    data[at:at + FIELD] = b + bytes(FIELD - len(b))


def location(progress):
    """(offset of the location field, stage, checkpoint) of a progress.sav's bytes; SaveError unless the layout holds."""
    found = [m for m in LOCATION.finditer(progress) if m.start() % 4 == 0]
    if len(found) != 1:
        raise SaveError(f'expected one "Develop/<stage>/Location" record, found {len(found)}')
    at = found[0].start()
    stage = found[0].group(1).decode('ascii')
    if at < FIELD or at + 2 * FIELD > len(progress):
        raise SaveError('the location record is too close to the start or the end of the file')
    loc, raw = _field(progress, at)
    if raw[len(loc):].strip(b'\0'):
        raise SaveError('the location field is not zero-padded')
    name, raw = _field(progress, at + FIELD)
    if name != stage or raw[len(name):].strip(b'\0'):
        raise SaveError(f'the stage field after the location is "{name}", not "{stage}"')
    checkpoint, raw = _field(progress, at - FIELD)
    if raw[len(checkpoint):].strip(b'\0') or (checkpoint and not checkpoint.endswith('_CheckPointTmp')):
        raise SaveError(f'the field before the location is not a checkpoint ("{checkpoint[:40]}")')
    return at, stage, checkpoint


def slots(profile):
    """[(user folder, slot folder)] with a progress.sav under profile/save/."""
    out = []
    for user in sorted((profile / 'save').glob('*')):
        for slot in sorted(user.glob('*')):
            if (slot / 'progress.sav').is_file():
                out.append((user, slot))
    return out


def warp_profile(profile, stage, slot='0', keep_checkpoint=False, known=None):
    """Point the save slot of a profile folder (account.uid, save/) at stage, in place. Returns the old stage."""
    profile = Path(profile)
    if known is None:
        known = stages()
    if known is not None and stage not in known:
        close = [s for s in known if stage.lower() in s.lower()][:8]
        raise SaveError(f'no stage "{stage}" in the RomFS' + (f' (did you mean {", ".join(close)}?)' if close else
                                                              '; tools/save_warp.py --stages lists them'))
    found = [(u, s) for u, s in slots(profile) if s.name == slot]
    if len(found) != 1:
        raise SaveError(f'expected one save slot {slot} under {profile / "save"}, found {len(found)}')
    user, sdir = found[0]
    progress = bytearray((sdir / 'progress.sav').read_bytes())
    at, old, _ = location(progress)
    _put(progress, at, f'Develop/{stage}/Location')
    _put(progress, at + FIELD, stage)
    if not keep_checkpoint:
        _put(progress, at - FIELD, '')
    caption_path = user / 'caption.sav'
    caption = None
    if caption_path.is_file():
        caption = bytearray(caption_path.read_bytes())
        old_loc = f'Develop/{old}/Location\0'.encode('ascii')
        hits = [m.start() for m in re.finditer(re.escape(old_loc), caption)]
        if len(hits) != 1:
            raise SaveError(f'expected the location once in caption.sav, found it {len(hits)} times')
        _put(caption, hits[0], f'Develop/{stage}/Location')
    (sdir / 'progress.sav').write_bytes(progress)
    if caption is not None:
        caption_path.write_bytes(caption)
    return old


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', nargs='?', help='a save under build/saves/')
    ap.add_argument('dest', nargs='?', help='the new save under build/saves/')
    ap.add_argument('stage', nargs='?', help='the stage to start in')
    ap.add_argument('--slot', default='0')
    ap.add_argument('--keep-checkpoint', action='store_true', help='keep the old checkpoint (it names an old object)')
    ap.add_argument('--stages', action='store_true', help='list the stages of the RomFS')
    ap.add_argument('--show', metavar='SAVE', help="show a save's current location")
    a = ap.parse_args(argv)
    if a.stages:
        s = stages()
        if s is None:
            print(f'no RomFS index at {ROMFS_INDEX}')
            return 2
        print('\n'.join(s))
        return 0
    try:
        if a.show:
            for user, slot in slots(SAVES / a.show):
                at, stage, checkpoint = location((slot / 'progress.sav').read_bytes())
                print(f'{a.show} slot {slot.name}: stage {stage}, checkpoint {checkpoint or "(none)"}')
            return 0
        if not (a.source and a.dest and a.stage):
            ap.error('give SOURCE DEST STAGE, or --stages, or --show')
        for name in (a.source, a.dest):   # a plain folder name under build/saves/, never a path out of it
            if not name or Path(name).name != name or name in ('.', '..'):
                raise SaveError(f'"{name}" is not a save name: give a folder name under {SAVES}')
        src, dst = SAVES / a.source, SAVES / a.dest
        if not (src / 'save').is_dir():
            raise SaveError(f'no save at {src}')
        if dst.exists():
            raise SaveError(f'{dst} exists; remove it or pick another name')
        shutil.copytree(src, dst)
        try:
            old = warp_profile(dst, a.stage, a.slot, a.keep_checkpoint)
        except Exception:
            shutil.rmtree(dst, ignore_errors=True)
            raise
        print(f'{a.dest}: slot {a.slot} moved from {old} to {a.stage}'
              + ('' if a.keep_checkpoint else ' (checkpoint cleared)'))
        return 0
    except SaveError as e:
        print(f'save_warp: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
