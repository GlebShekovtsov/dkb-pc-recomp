#!/bin/sh
# Setup for the DKB Recomp source package (Linux: setup.sh, macOS: setup.command; the same file under two names).
#   --check            list the missing prerequisites with the command that installs them; exit 0 when all are present
#   anything else      create .venv with uv (Python 3.12), then run: python -m installer --game bananza "$@"
# It never uses sudo and never installs a system package. The package lists are those of docs/building.md in the repository.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
os=$(uname -s)
missing=0

miss() {
    echo "missing: $1"
    missing=1
}

have() {
    command -v "$1" >/dev/null 2>&1
}

# A header below the usual prefixes (Homebrew's lives in /opt/homebrew on Apple silicon, /usr/local on Intel).
have_header() {
    for dir in /usr/include /usr/local/include /opt/homebrew/include; do
        [ -f "$dir/$1" ] && return 0
    done
    return 1
}

# version_ge A B: dotted version A >= B (numeric fields; sort -V is missing from old macOS).
version_ge() {
    a1=${1%%.*}
    b1=${2%%.*}
    ra=${1#*.}
    rb=${2#*.}
    a2=${ra%%.*}
    b2=${rb%%.*}
    [ "$a1" -gt "$b1" ] && return 0
    [ "$a1" -lt "$b1" ] && return 1
    [ "$a2" -ge "$b2" ]
}

check() {
    if [ "$os" = Linux ]; then
        kernel=$(uname -r | sed 's/[^0-9.].*//')
        version_ge "$kernel" 6.7 || miss "Linux 6.7 or later (userfaultfd write protection); this is $(uname -r)"
        [ "$(uname -m)" = x86_64 ] || miss "an x86-64 machine; this is $(uname -m)"
    elif [ "$os" = Darwin ]; then
        [ "$(uname -m)" = arm64 ] || miss "Apple silicon (arm64); this is $(uname -m)"
        version_ge "$(sw_vers -productVersion)" 14.4 || miss "macOS 14.4 or later; this is $(sw_vers -productVersion)"
        PATH="/opt/homebrew/opt/llvm/bin:$PATH"   # Homebrew's LLVM is keg-only
    else
        miss "Linux or macOS; this is $os"
    fi
    have clang || miss "clang"
    # macOS links with the system linker: docs/building.md lists no lld for it, and Homebrew's llvm does not ship one
    [ "$os" = Darwin ] || have ld.lld || have lld || miss "lld"
    have ninja || miss "ninja"
    have_header SDL3/SDL.h || miss "SDL3 development files"
    have_header opus/opus.h || miss "Opus development files"
    have_header vpx/vpx_decoder.h || miss "libvpx development files"
    have_header zstd.h || miss "zstd development files"
    have_header vulkan/vulkan.h || miss "Vulkan headers"
    have spirv-as || miss "spirv-tools"
    if have dotnet; then
        dotnet --list-sdks | grep -q '^10\.' || miss ".NET 10 SDK (dotnet is here, but no 10.x SDK)"
    else
        miss ".NET 10 SDK"
    fi
    have uv || miss "uv"
    have git || miss "git"
    if [ "$missing" -ne 0 ]; then
        echo
        echo "Install what is missing, for example:"
        if [ "$os" = Darwin ]; then
            echo "  brew install llvm ninja sdl3 molten-vk vulkan-headers vulkan-loader opus libvpx zstd spirv-tools dotnet uv git"
        else
            echo "  Arch:   pacman -S --needed clang lld ninja make sdl3 opus libvpx zstd vulkan-headers vulkan-icd-loader \\"
            echo "              vulkan-validation-layers shaderc spirv-tools dotnet-sdk uv git"
            echo "  Fedora: dnf install --skip-unavailable clang lld ninja-build make SDL3-devel opus-devel libvpx-devel \\"
            echo "              libzstd-devel vulkan-headers vulkan-loader-devel vulkan-validation-layers vulkan-tools spirv-tools \\"
            echo "              libstdc++-static dotnet-sdk-10.0 uv git"
            echo "  Debian/Ubuntu: apt install clang lld ninja-build make libsdl3-dev libopus-dev libvpx-dev libzstd-dev \\"
            echo "              libvulkan-dev vulkan-validation-layers spirv-tools dotnet-sdk-10.0 git   (uv: https://docs.astral.sh/uv/)"
        fi
        return 1
    fi
    echo "all prerequisites found"
}

case "${1:-}" in
--check)
    check
    exit $?
    ;;
"")
    echo "usage: $0 --check"
    echo "       $0 --nup FILE --dest DIR [installer options]"
    echo "       $0 --exefs DIR --romfs DIR --dest DIR [installer options]"
    echo "--nup is your Donkey Kong Bananza .nup file (1.0.0, or 3.2.0 when the update is inside), or --exefs and --romfs"
    echo "are the folders of a 1.0.0 or 3.2.0 dump; --dest is where the game is built."
    echo "See README-SETUP.txt and: .venv/bin/python -m installer --help"
    exit 2
    ;;
esac

check || exit 1
cd "$here"
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r tools/requirements-installer.txt
exec .venv/bin/python -m installer --game bananza "$@"
