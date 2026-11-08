"""Build the shader translator: build/xlate/totk_xlate.exe (Ryujinx 1.3.2 + patches, NativeAOT, SPIR-V for Vulkan).

Copies tools/vendor/ryujinx to build/xlate/ryujinx, applies the selected tools/xlate/patches there (tools/vendor stays
byte-identical), then runs `dotnet publish`. The default applies patches 0001-0013 and 0015-0023, preserving the fast floating
min/max/clamp operations. Patch 0015 makes runtime storage-candidate resolution available only to explicit version-1
NVN 2 compute requests; missing/version-0 requests retain the legacy pass. --defined-float-minmax also applies 0014's integer selection and numeric constant folding;
this adds shader ALU work and has no established game visual or frame-time benefit. Needs the .NET 10 SDK, the MSVC build
tools and vswhere.exe (docs/building.md). Run with .venv. On Linux it builds build/xlate/totk_xlate for linux-x64, which
NativeAOT links with the system's clang (no MSVC), and on macOS for osx-arm64 (osx-x64 on Intel), which needs
the Xcode command line tools for the linker; NativeAOT builds for the OS it runs on only.

Next to the exe it writes totk_xlate.id: the translator identity, a sha256 over everything the build reads (this script,
Program.cs, the csproj, the selected patches, the vendor tree). runtime/gfx_shader.c puts it into every shader cache key, so a
translator change re-keys the cache without a hand-made KEY_VERSION bump. --id-only writes it for the exe already built
(the caller vouches that the exe matches the sources and must use the same --defined-float-minmax setting).

  python tools/build_xlate.py [--defined-float-minmax] [--id-only] [--out DIR]

--out builds into DIR instead of build/xlate (a task build dir: totk.exe takes the translator next to it).
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "tools" / "vendor" / "ryujinx"
MNG_VENDOR = ROOT / "refs" / "MacNativeGames" / "vendor" / "ryujinx"
XLATE = ROOT / "tools" / "xlate"
BUILD = ROOT / "build" / "xlate"
WIN = os.name == "nt"
EXE = "totk_xlate.exe" if WIN else "totk_xlate"
# NativeAOT builds for the OS it runs on: the target RID is this host's.
# .NET runtime ids name the architecture arm64 / x64, while platform.machine() says arm64 / x86_64 on macOS.
RID = "win-x64" if WIN else ("osx-x64" if platform.machine() == "x86_64" else "osx-arm64") if sys.platform == "darwin" \
    else "linux-x64"
VSWHERE_DIR = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer"
DEFINED_FLOAT_MINMAX_PATCH = "0014-numeric-float-minmax-and-saturation.patch"


def tree_hashes(base: Path) -> dict[str, str]:
    """Hash every source file; src/*/obj and src/*/bin are build output (an IDE restore writes them, the vendor
    .gitignore ignores them and prepare_sources() drops them), so they are left out."""
    def build_output(rel: Path) -> bool:
        return len(rel.parts) > 2 and rel.parts[0] == "src" and rel.parts[2] in ("obj", "bin")

    return {
        p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.rglob("*"))
        if p.is_file() and not build_output(p.relative_to(base))
    }


def mng_pin() -> str:
    """The MacNativeGames commit tools/fetch_refs.sh pins."""
    for line in (ROOT / "tools" / "fetch_refs.sh").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) >= 4 and f[:2] == ["fetch", "MacNativeGames"]:   # a fifth word asks for the history
            return f[3]
    sys.exit("tools/fetch_refs.sh pins no MacNativeGames commit")


def pinned_blobs() -> dict[str, str]:
    """Git blob ids of MNG's vendor/ryujinx at the pinned commit. The checkout itself may be newer than the pin or
    carry CRLF line ends (core.autocrlf), so its files are not compared."""
    out = subprocess.run(["git", "-C", str(MNG_VENDOR.parent.parent), "ls-tree", "-r", mng_pin(), "vendor/ryujinx"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"git ls-tree of the MacNativeGames pin failed: {out.stderr.strip()}")
    blobs = {}
    for line in out.stdout.splitlines():
        meta, path = line.split("\t", 1)
        blobs[path.removeprefix("vendor/ryujinx/")] = meta.split()[2]
    return blobs


def blob_ids(base: Path) -> dict[str, str]:
    """Git blob ids of the source files tree_hashes() hashes."""
    ids = {}
    for rel in tree_hashes(base):
        data = (base / rel).read_bytes()
        ids[rel] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return ids


def verify_vendor() -> None:
    """tools/vendor/ryujinx must be byte-identical to MNG's copy at the pinned commit (when refs/ is present)."""
    if not MNG_VENDOR.is_dir():
        print("refs/MacNativeGames missing: vendor comparison skipped")
        return
    ours, theirs = blob_ids(VENDOR), pinned_blobs()
    if ours != theirs:
        diff = sorted(k for k in ours.keys() | theirs.keys() if ours.get(k) != theirs.get(k))
        sys.exit(f"tools/vendor/ryujinx differs from MNG in {len(diff)} files, first: {diff[:5]}")
    print(f"vendor identical to MNG: {len(ours)} files")


def selected_patches(*, defined_float_minmax: bool = False) -> list[Path]:
    patches = sorted((XLATE / "patches").glob("*.patch"))
    if len(patches) != 23:
        sys.exit(f"expected 23 patches, found {len(patches)}")
    optional = XLATE / "patches" / DEFINED_FLOAT_MINMAX_PATCH
    if optional not in patches:
        sys.exit(f"missing optional patch: {DEFINED_FLOAT_MINMAX_PATCH}")
    return patches if defined_float_minmax else [p for p in patches if p != optional]


def prepare_sources(*, defined_float_minmax: bool = False) -> Path:
    src = BUILD / "ryujinx"
    if BUILD.exists():
        shutil.rmtree(BUILD / "ryujinx", ignore_errors=True)
    shutil.copytree(VENDOR, src)
    for junk in list(src.glob("src/*/obj")) + list(src.glob("src/*/bin")):
        shutil.rmtree(junk)
    for patch in selected_patches(defined_float_minmax=defined_float_minmax):
        # git apply works outside a repository when given a directory; --check first so a failure names the patch.
        for extra in (["--check"], []):
            r = subprocess.run(
                ["git", "apply", *extra, "--whitespace=nowarn", "--directory", str(src.relative_to(BUILD)), str(patch)],
                cwd=BUILD, capture_output=True, text=True,
            )
            if r.returncode != 0:
                sys.exit(f"patch does not apply: {patch.name}\n{r.stdout}{r.stderr}")
        print(f"applied {patch.name}")
    return src


def identity(*, defined_float_minmax: bool = False) -> str:
    """sha256 over the translator's inputs, in a fixed order: path and content hash of each file."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), XLATE / "Program.cs", XLATE / "totk_xlate.csproj",
             *selected_patches(defined_float_minmax=defined_float_minmax)]
    for f in files:
        h.update(f"{f.name}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    for rel, digest in tree_hashes(VENDOR).items():
        h.update(f"vendor/{rel}\0{digest}\n".encode())
    return h.hexdigest()


def write_identity(exe: Path, *, defined_float_minmax: bool = False) -> None:
    ident = identity(defined_float_minmax=defined_float_minmax)
    exe.with_suffix(".id").write_text(ident + "\n", encoding="ascii")
    print(f"translator identity {ident[:16]} -> {exe.with_suffix('.id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id-only", action="store_true", help="write totk_xlate.id for the exe already built")
    parser.add_argument("--out", type=Path, help="build directory instead of build/xlate")
    parser.add_argument("--defined-float-minmax", action="store_true",
                        help="opt in to numeric NaN/signed-zero min/max/SAT via integer selection (more shader ALU work)")
    parser.add_argument("--llvm", type=Path, help="Linux: an LLVM directory whose bin/clang and lld link the exe")
    parser.add_argument("--sysroot", type=Path, help="Linux: link against this sysroot's glibc instead of the host's "
                                                     "(tools/make_release_linux.py)")
    a = parser.parse_args()
    if a.out:
        global BUILD
        BUILD = a.out.resolve()
    if a.id_only:
        write_identity(BUILD / EXE, defined_float_minmax=a.defined_float_minmax)
        return
    started = time.time()
    dotnet = shutil.which("dotnet")
    if not dotnet:
        sys.exit("missing: dotnet (.NET 10 SDK)")
    env = dict(os.environ)
    if WIN:
        if not (VSWHERE_DIR / "vswhere.exe").is_file() and not shutil.which("vswhere"):
            sys.exit(f"missing: vswhere.exe (expected in {VSWHERE_DIR}); NativeAOT cannot link without it")
        env["PATH"] = str(VSWHERE_DIR) + os.pathsep + env["PATH"]

    verify_vendor()
    src = prepare_sources(defined_float_minmax=a.defined_float_minmax)
    work = BUILD / "xlate"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for name in ("Program.cs", "totk_xlate.csproj"):
        shutil.copy2(XLATE / name, work / name)
    out = BUILD / "publish"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [
        dotnet, "publish", str(work / "totk_xlate.csproj"), "-c", "Release", "-r", RID,
        "-p:PublishAot=true", "-p:InvariantGlobalization=true", f"-p:RyuSrc={src}", "-o", str(out),
    ]
    if a.llvm:
        env["PATH"] = str(a.llvm.resolve() / "bin") + os.pathsep + env["PATH"]
        cmd += ["-p:CppCompilerAndLinker=clang", "-p:LinkerFlavor=lld"]
    if a.sysroot:
        cmd.append(f"-p:SysRoot={a.sysroot.resolve()}")
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit("dotnet publish failed")
    exe = BUILD / EXE
    shutil.copy2(out / EXE, exe)
    write_identity(exe, defined_float_minmax=a.defined_float_minmax)
    print(f"built {exe} ({exe.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
