"""Assemble the Bananza source package for Linux or macOS (docs/installer.md, "Bananza Linux and macOS packages").

    python tools/make_release_posix.py --platform linux [--out build/release/DKB-Setup-<version>-linux] [--pack]
    python tools/make_release_posix.py --platform macos --pack

The folder holds every runtime source, the lifter, the translator's sources and the CLI installer, but no game data and no
binaries: the player already has the system packages, and setup.sh / setup.command (README-SETUP.txt) check them, create a
Python environment and run `python -m installer --game bananza`, which builds everything on their machine. The folder is
the Windows one of tools/make_release.py --game dkb minus the toolchain and the launcher, so this script reuses its source
copy, its version string and its check_content pass. Needs only Python 3.12+ and git.
"""
import argparse
import io
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_release as mr  # noqa: E402

ROOT = mr.ROOT
ARCH = {'linux': 'x86_64', 'macos': 'arm64'}
SETUP = {'linux': 'setup.sh', 'macos': 'setup.command'}
# Extra tools/ files on top of make_release.py's lists: the installer's pip requirements (setup script) and fetch_refs.sh,
# which build_xlate.py reads for its MacNativeGames pin. hactool is not needed: the Bananza installer extracts with its own scripts.
EXTRA_TOOLS = ['requirements-installer.txt', 'fetch_refs.sh']
# Licence texts of third-party code that the sources carry, collected in licenses/ as in the Windows folder.
LICENSES = {'runtime/shaders/ffx/LICENSE.txt': 'FidelityFX-FSR-LICENSE.txt',
            'runtime/third_party/imgui/LICENSE.txt': 'imgui-LICENSE.txt',
            'runtime/ui/fonts/LICENSE-Roboto.txt': 'Roboto-LICENSE.txt',
            'tools/xlate/LICENSE': 'totk_xlate-LICENSE.txt'}

README_SETUP = """\
DKB Recomp setup for {label}

This folder builds Donkey Kong Bananza (base game 1.0.0) from your own dump. It holds the sources and the command-line
installer only: no game data, no prebuilt binaries. Status: {status}

Requirements
{requirements}

Run
  {setup} --check
      Lists what is missing and the command that installs it; changes nothing. Exit code 0 when all is present.
  {setup} --exefs DIR --romfs DIR --dest DIR
      Creates .venv with uv (Python 3.12), then runs: python -m installer --game bananza <your arguments>
      --exefs: the ExeFS folder of the 1.0.0 base game. --romfs: its RomFS folder. --dest: where the game is built.
      Further options (--toolchain, --deps, --ninja, --translator, --dotnet, --jobs) are passed through; see
      `.venv/bin/python -m installer --help`.
  {setup} --source PATH --dest DIR
      The same with your copy as one path: a .nup file, or a folder with exefs/ and romfs/ in it. RecompLauncher runs
      it this way from recomplauncher.json, with --prepare and --dest set to the game/ folder beside this file.

The script never uses sudo and never installs system packages. The build compiles the whole runtime and the generated
code on your machine and takes a long time. Keep your keys and dumps outside this folder.
"""
STATUS = {'linux': 'preview. Building from source is verified on AMD and NVIDIA GPUs; long play sessions are not.',
          'macos': 'preview. It boots to the mode select on Apple Silicon with MoltenVK; the destructible terrain is missing '
                   'and the game is not verified beyond that.'}
REQUIREMENTS = {
    'linux': ('  Linux 6.7 or later (userfaultfd write protection), x86-64, a Vulkan 1.3 driver with VK_EXT_external_memory_host.\n'
              '  clang, lld, ninja, SDL3, Opus, libvpx, zstd, Vulkan headers and loader, spirv-tools, .NET 10 SDK, uv, git.\n'
              '  setup.sh --check prints the apt, dnf and pacman commands for what is missing.'),
    'macos': ('  macOS 14.4 or later on Apple Silicon (arm64), MoltenVK 1.4 or later.\n'
              '  Homebrew: llvm ninja sdl3 molten-vk vulkan-headers vulkan-loader opus libvpx zstd, .NET 10 SDK, uv, git.'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--platform', choices=sorted(ARCH), required=True)
    ap.add_argument('--out', type=Path, help='folder (replaced); default build/release/DKB-Setup-<version>-<platform>')
    ap.add_argument('--pack', action='store_true', help='also write the archive beside the folder: '
                    'DKB-Setup-<version>-linux-x86_64.tar.gz or DKB-Setup-<version>-macos-arm64.zip (the usual format of each)')
    args = ap.parse_args()
    plat = args.platform
    ver = mr.version()
    name = f'DKB-Setup-{ver}-{plat}'
    out = (args.out or mr.MAIN_ROOT / 'build' / 'release' / name).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    mr.GAME = 'dkb'
    mr.build_sources(out)   # recomp/, tools/ (+ xlate, vendor/ryujinx), data/, installer/, dkb.ini.example, notices, VERSION
    mr.runtime_lib(out, None)   # every runtime source, headers and dkb.ico
    for f in EXTRA_TOOLS:
        shutil.copy2(ROOT / 'tools' / f, out / 'tools' / f)
    (out / 'licenses').mkdir()
    for src, dst in LICENSES.items():
        shutil.copy2(ROOT / src, out / 'licenses' / dst)
    for lic in sorted(ROOT.glob('LICENSE*')):
        shutil.copy2(lic, out / lic.name)
    shutil.copy2(ROOT / 'tools' / 'release' / 'setup.sh', out / SETUP[plat])   # one script, named setup.command on macOS
    # RecompLauncher's profile: it asks for the copy, runs the setup with --prepare and starts play.sh (docs/installer.md)
    shutil.copy2(ROOT / 'tools' / 'release' / 'recomplauncher.json', out / 'recomplauncher.json')
    shutil.copy2(ROOT / 'tools' / 'release' / 'play.sh', out / 'play.sh')
    (out / 'README-SETUP.txt').write_text(
        README_SETUP.format(label=f'{plat} {ARCH[plat]}', status=STATUS[plat], requirements=REQUIREMENTS[plat],
                            setup='./' + SETUP[plat]), encoding='utf-8', newline='\n')
    for script in (SETUP[plat], 'play.sh'):   # the archives carry their own bits; this is for a folder used directly
        (out / script).chmod(0o755)
    mr.check_folder(out)
    if args.pack:
        if plat == 'linux':
            write_tar(out, out.parent / f'DKB-Setup-{ver}-{plat}-{ARCH[plat]}.tar.gz')
        else:
            write_zip(out, out.parent / f'DKB-Setup-{ver}-{plat}-{ARCH[plat]}.zip')


def is_exe(p):
    return p.name.startswith('setup.') or p.suffix in ('.sh', '.command')


def write_tar(out, path):
    """The folder as a .tar.gz, the usual Linux archive: it stores the modes itself (0755 on setup.* and the shell scripts)."""
    stamp = mr.commit_time()   # not a fixed time: a folder unpacked over an older one must not keep its byte code
    with tarfile.open(path, 'w:gz', compresslevel=6) as t:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                info = tarfile.TarInfo((Path(out.name) / p.relative_to(out)).as_posix())
                data = p.read_bytes()
                info.size, info.mode, info.mtime = len(data), 0o755 if is_exe(p) else 0o644, stamp
                info.uid = info.gid = 0
                info.uname = info.gname = 'root'
                t.addfile(info, io.BytesIO(data))
    print(f'{path}: {mr.mib(path.stat().st_size)}')


def write_zip(out, path):
    """The folder as a zip with the executable bit on setup.* and the shell scripts (unzip honours external_attr)."""
    date = mr.zip_date()   # the commit's time, as tools/make_release.py write_zip
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                info = zipfile.ZipInfo((Path(out.name) / p.relative_to(out)).as_posix(), date)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o755 if is_exe(p) else 0o644) << 16
                z.writestr(info, p.read_bytes(), compresslevel=6)
    print(f'{path}: {mr.mib(path.stat().st_size)}')


if __name__ == '__main__':
    main()
