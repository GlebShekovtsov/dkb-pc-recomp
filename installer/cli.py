"""Command line: python -m installer --nup FILE --dest DIR [--package DIR] (Bananza 1.0.0 or 3.2.0)
              or python -m installer --exefs DIR --romfs DIR --dest DIR [--package DIR]

Legacy TotK: --base FILE --update FILE --keys FILE --lang CODE --dest DIR [--root DIR]
                                     [--temp DIR] [--verify]

A first install needs every option. On a folder with totk-install.json (resume after a cancel or failure, or update
with a newer release) only --dest is needed: --lang defaults to the recorded language, and the game files are needed
only when the extraction has to run again. Prints JSON events on stdout (events.py). Exit code: 0 done, 1 failed,
2 bad command line, 3 cancelled (Ctrl+C or Ctrl+Break).
"""
import argparse
import hashlib
import signal
import sys
import time
from pathlib import Path

from . import layout as layouts
from . import state as statefile
from .events import Reporter
from .runner import Context, run_steps

EXIT = {'ok': 0, 'failed': 1, 'cancelled': 3}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        Reporter().error('cli.usage', {'detail': message})
        sys.exit(2)


# The temp folder inside the install folder when the layout names none (the Linux release); the check step does not
# count it when it requires an empty install folder
TEMP_IN_DEST = '.setup-temp'


def temp_folder(base, dest):
    """One temp folder per install folder, so two installs never share generated files."""
    tag = hashlib.sha256(str(dest).lower().encode()).hexdigest()[:8]
    return base / f'{dest.name or "root"}-{tag}'


def cancel(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    ap = Parser(prog='python -m installer', description=__doc__.splitlines()[0])
    ap.add_argument('--base', type=Path, help='base game, XCI or NSP')
    ap.add_argument('--update', type=Path, help='the update NSP: 1.1.1, 1.2.1 or 1.4.3')
    ap.add_argument('--keys', type=Path, help='prod.keys')
    ap.add_argument('--lang', help='game language, a Switch language code (en-US, en-GB, ja, de, ru, ...)')
    ap.add_argument('--dest', type=Path, required=True, help='install folder')
    ap.add_argument('--root', type=Path, help='release folder or repository (default: the folder above installer/)')
    ap.add_argument('--temp', type=Path, help='temp folder (default: temp/ of the release on Windows, .setup-temp in '
                                              'the install folder on Linux, build/install-temp in a repository)')
    ap.add_argument('--verify', action='store_true', help='hash every kept file, also those this release made')
    ap.add_argument('--game', choices=('bananza', 'totk'), help='game profile (.nup and folder inputs select Bananza)')
    ap.add_argument('--nup', type=Path, help='Bananza .nup container (base 1.0.0, with or without update 3.2.0)')
    ap.add_argument('--nup-package', choices=('latest', 'base'),
                    help='what to install from the .nup: latest (default: the newest update, as Eglantine runs it) '
                         'or base (1.0.0 even when the container also holds an update)')
    ap.add_argument('--exefs', type=Path, help='Bananza 1.0.0 or 3.2.0 ExeFS folder')
    ap.add_argument('--romfs', type=Path, help='Bananza loose RomFS folder or packed romfs.bin/index.json folder')
    ap.add_argument('--package', type=Path, help='tool package with llvm/, runtime/deps/ and bin/')
    ap.add_argument('--toolchain', type=Path, help='LLVM toolchain directory')
    ap.add_argument('--deps', type=Path, help='build dependency directory with include/ and lib/')
    ap.add_argument('--ninja', type=Path, help='Ninja executable')
    ap.add_argument('--dll-dir', type=Path, help='runtime DLL directory (Windows)')
    ap.add_argument('--translator', type=Path, help='prebuilt translator, with matching .id beside it')
    ap.add_argument('--dotnet', type=Path, help='portable .NET 10 dotnet executable, when building the translator')
    ap.add_argument('--jobs', type=int, default=6, help='Bananza worker limit (default 6)')
    args = ap.parse_args(argv)

    if (args.game == 'bananza' or args.nup or args.nup_package or args.exefs or args.romfs or
            (args.game is None and (args.dest / 'dkb-install.json').is_file())):
        if args.game == 'totk':
            ap.error('--nup/--exefs/--romfs require --game bananza')
        from .bananza import main as bananza_main
        return bananza_main(args)

    from .steps import all_steps   # legacy extraction needs Python 3.14; folder installs do not import it

    # The window cancels with Ctrl+Break to the installer's process group (CREATE_NEW_PROCESS_GROUP), which also reaches
    # the tools it runs; Python's default for it ends the process without recording anything.
    for name in ('SIGBREAK', 'SIGTERM'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), cancel)
    layout = layouts.find(args.root)
    dest = args.dest.resolve()
    base = args.temp or layout.temp
    temp = temp_folder(base.resolve(), dest) if base else dest / TEMP_IN_DEST
    log = temp / 'logs' / time.strftime('install-%Y%m%d-%H%M%S.log')
    rep = Reporter(log_path=log)
    t0 = time.monotonic()
    try:
        state = statefile.load(dest)
    except statefile.StateError as e:
        rep.error(e.id, {'path': str(e.path)})
        rep.emit('end', result='failed', seconds=0)
        return 1
    ctx = Context(layout, dest, temp, rep, args, state=state, verify=args.verify)
    steps = all_steps()
    rep.emit('start', release=layout.release, mode=ctx.mode, dev=layout.dev, dest=str(dest), log=str(log),
             steps=[s.id for s in steps])
    result = run_steps(steps, ctx)
    rep.emit('end', result=result, seconds=round(time.monotonic() - t0, 1))
    rep.close()
    return EXIT[result]
